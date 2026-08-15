# 实验总记录

本文档记录 minWM-dyKV 实验的统一规范。尚未验证的结果必须明确留空，不得用估计值填充。

## 统一环境

除非某项实验另有明确记录，所有生成、适配、测试与评测均使用 Conda 环境 `minwm-fa`：

```bash
conda activate minwm-fa
```

## 可复现性约定

每次运行必须记录：

- Git commit 与工作区是否干净；
- checkpoint 路径与校验和；
- prompt/用例清单及选中的 case ID；
- seed、输出 latent 帧数、分辨率和相机轨迹；
- GPU 型号/数量、PyTorch/CUDA 版本及显存峰值；
- 生成总耗时与检索耗时；
- dyKV 记忆预算及最终采用的内部布局；
- 输出目录与评测报告目录。

## 核心对比

| 运行编号 | Case | 方法 | 目的 | 状态 |
| --- | --- | --- | --- | --- |
| B0 | `baseline` | 固定 4 帧 sink + rolling local 16 | 质量/速度基线 | 待运行 |
| B1 | `retrieval_no_compression` | 不压缩的 dyKV | 单独分析检索收益 | 待运行 |
| B1-W | `worldkv_pose_no_compression` | WorldKV 平均位姿检索，不压缩 | 与 B1 单变量比较检索评分 | 待运行 |
| B2 | `fixed_novelty` | 固定内容压缩 dyKV | 与相机无关的压缩对照 | 待运行 |
| B3 | `yaw_intrinsics` | 当前-query 几何裁剪（E0 兼容默认） | 评估旧动态路径的记忆、速度与质量 | 待运行 |
| B4 | `packed_chunks` | 固定档位完整 chunk 扩容 | 评估更多完整历史覆盖 | 待运行 |
| B5 | `packed_chunks_latent` | 完整 chunk + latent 尾部补齐 | 评估余量补齐收益 | 待运行 |
| B6 | `retr8_compression_r050` | 8 源帧、固定 `r=1/2` | 固定覆盖下的压缩损失 | 待运行 |
| B7 | `retr12_compression_r050` | 12 源帧、固定 `r=1/2` | 同比例下扩充 chunk 的收益 | 待运行 |
| B8 | `retr16_compression_r033` | 16 源帧、固定 `r=1/3` | 与 8 帧不压缩严格等 token 对比 | 待运行 |
| B9 | `predecessor_chunks` | 前驱新增角域四档压缩，完整 chunk | 验证压缩依据从 query 改为前驱的影响 | 待运行 |
| B10 | `predecessor_chunks_latent` | B9 + 单 latent 尾部 | 验证剩余容量带来的历史覆盖收益 | 待运行 |
| B11 | `predecessor_query_backfill` | B10 + 当前 query coverage 回填 | 推荐完整方案；验证跨度与当前相关性 | 待运行 |

以上 case 均可由 `Wan21/scripts/inference/run_dykv_cases.sh` 统一运行。固定 FOV 与混合
FOV 消融已移除；FOV 路径统一使用相机内参，B1-W 则明确只使用 WorldKV 外参位姿得分。定义见
[`CASES_AND_RUNNER.md`](CASES_AND_RUNNER.md)。

所有实验固定保留最初 4 帧 sink。B0 使用 `4 + 16`，所有 dyKV case 使用连续的
`4 + 8 + 8` latent：4 帧 sink、8 帧 retrieval、
8 帧 local（4 帧 recent + 4 帧 current），正好覆盖 20 帧 RoPE 训练窗口。
动态空间压缩及完整消融矩阵见 [`DYNAMIC_SPATIAL_COMPRESSION.md`](DYNAMIC_SPATIAL_COMPRESSION.md)
和 [`ABLATIONS.md`](ABLATIONS.md)。
压缩后扩充历史覆盖、latent 尾部补齐和 frame-level RoPE slot folding 已按
[`DYNAMIC_RETRIEVAL_PACKING.md`](DYNAMIC_RETRIEVAL_PACKING.md) 实现。现有
`yaw_intrinsics` 仍保留“先选择 8 个原始 latent，再裁剪并允许 retrieval token 欠填”作为
E0；B4/B5 分别对应 E1/E2。
minWM-back 固定比例 A--D 对照由 B1/B6/B7/B8 构成，具体预算与适配差异见
[`FIXED_WORLDKV_CASES.md`](FIXED_WORLDKV_CASES.md)。
B9--B11 的压缩参考系、四档规则和 32 原子精确装箱见
[`PREDECESSOR_INCREMENTAL_COMPRESSION.md`](PREDECESSOR_INCREMENTAL_COMPRESSION.md)。
B1 与 B1-W 的唯一实验变量是 FOV overlap 或 WorldKV 平均位姿得分，见
[`WORLDKV_RETRIEVAL_ABLATION.md`](WORLDKV_RETRIEVAL_ABLATION.md)。

### 历史精度阻塞与重新运行要求

旧推理入口曾把相机 `viewmats/Ks` 量化为 BF16，已有 `predecessor_chunks` 典型样本因此
全部记录为 `predecessor_fixed_novelty_fallback` 和 `keep_tier=0.5`，最大只装入 4 个 chunk。
当前版本已让 dyKV 几何保持 FP32，并在 PRoPE 内部单独转换计算副本；对应回归测试已确认
`j*7` 进入 `predecessor_incremental_yaw` 的 `1/4` 档。旧产物仍只能视为链路冒烟，不能
登记为 B9--B11 四档结果。新实验必须使用全新输出目录，并以日志出现
`predecessor_incremental_yaw`、非空 `incremental_fov_ratios` 和预期四档为验收条件。
完整修复与判读方法见
[`RETRIEVAL_ROTATION_COMPRESSION_FLOW.md`](RETRIEVAL_ROTATION_COMPRESSION_FLOW.md)。

## 评测分组

1. 闭环相机路径：验证模型能否恢复之前观察过的内容。
2. 长单调路径：检查检索是否会破坏新视角的稳定性。
3. MBench 用例：报告基准原生指标及逐用例产物。
4. 资源画像：记录显存峰值、CPU 记忆库字节数、检索耗时和总延迟。

### `retrieval_no_compression` 典型样本日志诊断

对 `output/mbench_typical_4_crop_compare_seed0/retrieval_no_compression` 的 4 个典型样本进行
了尾部日志检查。四个样本共 80 次 retrieval event，全部使用完整 12480 token，且 80/80
次都选择两个时间相邻 chunk；最后一步全部选择 source starts `[4,8]`。没有发现 token
超预算、sink/source 重复或 RoPE 越界，但存在 recent 被早期历史替换、检索与 sink 冗余、
FOV 同分偏向更旧状态以及动态内容陈旧等策略风险。完整日志和结论见
[`RETRIEVAL_NO_COMPRESSION_DIAGNOSIS.md`](RETRIEVAL_NO_COMPRESSION_DIAGNOSIS.md)。该诊断不等于
B1 正式质量指标已经完成，核心结果表仍保持“待运行”。

## MBench 实验规范

- 使用 MBench-A 官方任务分配，并记录准确的 `samples.jsonl` 校验和。
- 10 秒/25 秒用例分别使用与 checkpoint 对齐的 40/100 个 latent 位姿；报告时同时列出
  解码后的 157/397 帧长度和官方目标 161/401 帧。
- 表中所有方法必须使用相同的用例分配、checkpoint、latent 长度、分辨率和 seed。
- 每种方法和每个 seed 分别注册为独立的 MBench `model_id`。
- 评测前先运行接口约定校验，并记录因缺少 DA3/VLM 产物而跳过的指标。

## 结果模板

| Commit | 运行编号 | 用例数 | Seed | 帧数 | 质量报告 | 显存峰值 | 耗时 | 备注 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 待运行 | B0 | 待运行 | 0 | 待运行 | 待运行 | 待运行 | 待运行 | 基线 |
| 待运行 | B1 | 待运行 | 0 | 待运行 | 待运行 | 待运行 | 待运行 | 仅检索 |
| 待运行 | B1-W | 待运行 | 0 | 待运行 | 待运行 | 待运行 | 待运行 | WorldKV 位姿检索 |
| 待运行 | B2 | 待运行 | 0 | 待运行 | 待运行 | 待运行 | 待运行 | 固定新颖性 |
| 待运行 | B3 | 待运行 | 0 | 待运行 | 待运行 | 待运行 | 待运行 | E0 兼容默认 |
| 待运行 | B4 | 待运行 | 0 | 待运行 | 待运行 | 待运行 | 待运行 | 完整 chunk 扩容 |
| 待运行 | B5 | 待运行 | 0 | 待运行 | 待运行 | 待运行 | 待运行 | latent 尾部补齐 |
| 待运行 | B6 | 待运行 | 0 | 待运行 | 待运行 | 待运行 | 待运行 | 8 帧固定 1/2 |
| 待运行 | B7 | 待运行 | 0 | 待运行 | 待运行 | 待运行 | 待运行 | 12 帧固定 1/2 |
| 待运行 | B8 | 待运行 | 0 | 待运行 | 待运行 | 待运行 | 待运行 | 16 帧固定 1/3 |
| 待运行 | B9 | 待运行 | 0 | 待运行 | 待运行 | 待运行 | 待运行 | 前驱四档完整 chunk |
| 待运行 | B10 | 待运行 | 0 | 待运行 | 待运行 | 待运行 | 待运行 | 加 latent 尾部 |
| 待运行 | B11 | 待运行 | 0 | 待运行 | 待运行 | 待运行 | 待运行 | 推荐 predecessor 完整方案 |

禁止用估计值替换“待运行”，表中只记录实测结果。

## 实现冒烟测试

### 当前 FP32 相机链路与 predecessor 冒烟

该记录验证本文档所在提交的 FP32 几何修复和连续 `4+8+8` predecessor 链路，不代表
MBench 分数。

| 字段 | 实测值 |
| --- | --- |
| 日期 / commit | 2026-08-14 / 本文档所在提交 |
| 环境 | `minwm-fa` |
| Checkpoint | Action2V DMD `model.pt` |
| Prompt / seed | 1 条合成的红色木屋闭环 prompt / 0 |
| 轨迹 | `j*10,l*10,n*3` |
| 长度 | 24 latent 帧 |
| 第 20 帧候选 / 选中 | 候选 block `[1,2,3]`；选中 source starts `[4,8,12]` |
| 几何路径 | 12/12 frame segment 为 `predecessor_incremental_yaw`，没有 fallback |
| 四档与预算 | 12/12 为 `1/4`；使用 12/32 atoms；每层 4680 retrieval token |
| 输出 | `/tmp/minwm_dykv_fp32_predecessor_smoke`（临时实现验收产物） |

### 历史冒烟

以下是历史冒烟记录，只验证当时提交的实现链路，不代表 MBench 分数，也不验证当前连续
`4+8+8`、动态装箱或 predecessor 路径。短闭环轨迹曾跨过第 20 帧启用边界。

| 字段 | 实测值 |
| --- | --- |
| 日期 / commit | 2026-08-13 / `d4dcdd2` |
| 环境 | `minwm-fa`；PyTorch 2.8.0+cu128；CUDA 12.8；FlashAttention 2.8.3.post1 |
| GPU | 1 × NVIDIA GeForce RTX 4090 (48 GB) |
| Checkpoint | Action2V DMD `model.pt`；SHA-256 `bdb947d45fb04513305492c2ee393d51d0621ec0e99fd312224f5d61a330aa77` |
| Prompt / seed | 1 条合成的红色木屋闭环 prompt / 0 |
| 轨迹 | `j*10,l*10,n*3` |
| 长度 / 输出 | 24 latent 帧 / 93 解码帧，832×480，16 fps |
| 生成状态 | 6/6 个因果块完成；已写出 MP4 和生成清单 |
| 结束时记忆库 | 6 个无损 CPU 块，共 6,900,940,800 字节 |
| 第 20 帧检索 | 候选 `[1,2,3]`；排序 `[1,3,2]`；选中起始帧 `[4,12]` |
| 压缩结果 | 每层 12,480 个原始选中 token → 7,800 个注意力 token |
| 检索耗时 | 2.6609 秒（选择、传输与压缩） |

运行输出位于 `/tmp/minwm_dykv_smoke`，不属于持久化基准产物。该冒烟测试未采集显存峰值，
因此不报告该数值。该记录对应 commit `d4dcdd2` 的旧三区域布局；切换到连续
`4 + 8 + 8` 布局后需要重新运行冒烟测试，不能将本行视为新布局的验证结果。正式
表中正式 MBench 结果仍保持“待运行”，直到完成对应实验。
