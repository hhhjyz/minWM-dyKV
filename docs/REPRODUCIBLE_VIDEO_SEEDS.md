# 视频级可复现 Seed 与初始噪声

## 1. 修改目的

旧推理入口只在进程启动时执行一次 `set_seed(SEED)`，随后不同 prompt 按运行顺序共同消耗
全局随机数流。这会导致：

- 完整重跑且 prompt 顺序完全相同时，初始噪声通常相同；
- 如果断点续跑跳过了已有视频，跳过的样本不再消耗随机数，后续重新生成的视频会发生 seed
  漂移；
- 一旦 case 清单、样本分片或运行顺序改变，就很难从日志确认两个视频是否真的从同一初始
  噪声开始。

这不适合 `baseline`、FOV、WorldKV pose 和不同压缩 case 的公平视频对比。

## 2. 当前 Seed 策略

每个视频使用独立且不包含 case 名称的 sample seed：

```text
sample_seed = (base_seed + prompt_index) mod 2^63
pipeline_seed = (sample_seed + 2^32) mod 2^63
seed_policy = base_seed_plus_prompt_index_v1
```

例如 `SEED=7` 时：

| prompt index | sample seed |
| ---: | ---: |
| 0 | 7 |
| 1 | 8 |
| 2 | 9 |
| 3 | 10 |

case 名称不会参与计算，因此同一 assignment 中相同 `prompt_index` 在以下 case 间使用相同
sample seed 和初始噪声：

```text
baseline
retrieval_no_compression
worldkv_pose_no_compression
yaw_intrinsics
predecessor_query_backfill
其他注册 case
```

不同 prompt index 仍使用不同噪声，避免同一批样本全部从完全相同的噪声张量开始。

## 3. 实现流程

对每个尚未生成的样本，`wan_inference.py` 执行：

1. 根据 `base_seed + prompt_index` 计算 `sample_seed`；
2. 在处理该样本前重置 Python、NumPy、PyTorch 和全部 CUDA RNG；
3. 建立只属于该样本的显式 `torch.Generator`；
4. 使用该 generator 生成初始 latent noise，不消耗进程全局 RNG；
5. 计算初始噪声前 2048 个固定位置的 SHA-256 指纹；
6. 进入 pipeline 前按独立的 `pipeline_seed` 重置全局 RNG，使 scheduler 内部
   `torch.randn_like` 等后续随机操作也不受之前生成或跳过样本影响，同时避免它与初始
   latent noise 使用同一随机序列前缀。

因此以下操作不会改变某个固定 `prompt_index` 的初始噪声：

- 改变 case 的运行顺序；
- 单独运行一个 case 或使用统一 runner 连续运行多个 case；
- 输出目录中已有其他视频并被跳过；
- 只删除并重新生成中间某一个视频；
- 单卡与 sequence-parallel peers 之间的进程随机状态差异。

## 4. 日志字段

`generation_manifest.jsonl` 的每一行现在记录：

```text
base_seed
sample_seed
pipeline_seed
seed_policy
initial_noise_fingerprint
```

生成成功的行具有 SHA-256 指纹；`skipped_exists` 行不重新创建大噪声张量，因此指纹为
`null`，但仍记录确定性的 `sample_seed`。dyKV 的 `dykv_summaries.jsonl` 顶层也记录相同
五个字段。

可以分别查看两个 case 的生成清单：

```bash
jq -r 'select(.status=="generated") |
  [.prompt_index,.sample_seed,.initial_noise_fingerprint] | @tsv' \
  output/compare/retrieval_no_compression/generation_manifest.jsonl

jq -r 'select(.status=="generated") |
  [.prompt_index,.sample_seed,.initial_noise_fingerprint] | @tsv' \
  output/compare/worldkv_pose_no_compression/generation_manifest.jsonl
```

相同 `prompt_index` 的 seed 和 fingerprint 应完全一致。如果 fingerprint 不同，应先停止
质量比较并检查输出帧数、分辨率、dtype、PyTorch/CUDA 环境或代码版本。

## 5. 运行方式

外部命令不需要新增参数，仍然只设置一个 `SEED`：

```bash
SEED=0 \
CASES=baseline,retrieval_no_compression,worldkv_pose_no_compression \
OUTPUT_ROOT=output/seed0_compare \
bash Wan21/scripts/inference/run_dykv_cases.sh
```

统一 runner 会把相同 `SEED` 传给每个 case，推理入口再按 prompt index 派生稳定 sample
seed。不要为不同 case 手工设置不同 `SEED`。

## 6. 保证范围与限制

当前保证的是：在相同代码、checkpoint、输入、输出形状、dtype 和软件/硬件环境下，对应视频
使用相同初始噪声，并让 pipeline 内部随机流从相同状态开始。

还需注意：

- 改变 prompt/assignment 的行顺序会改变 `prompt_index`，因此也会改变对应 sample seed；
- 改变 `NUM_OUTPUT_FRAMES` 或 I2V/T2V 模式会改变噪声张量形状；
- 某些 CUDA/FlashAttention 算子不保证跨 GPU 型号、PyTorch/CUDA 版本逐 bit 一致；即使
  初始 fingerprint 相同，跨环境最终视频也可能出现数值差异；
- 修复前生成的视频没有 `sample_seed` 和 fingerprint。正式新旧 case 对比应使用当前提交和
  全新输出目录重新生成，不能仅根据旧目录名中的 `seed0` 推断初始噪声严格一致。

本模块解决的是比较实验的随机初始条件，不改变 dyKV 检索、压缩、packing 或 RoPE 算法。
