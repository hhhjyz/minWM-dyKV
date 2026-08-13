# dyKV 实验 Case 与统一 Runner

## 1. 设计原则

所有可直接运行的对照都注册在 `Wan21/dykv_cases.py`。公开接口只增加一个枚举参数
`--dykv-case`，每个 case 一次性确定压缩方式、检索 FOV 来源和裁剪 FOV 来源，不再暴露
彼此独立的内部开关。固定缓存布局始终是连续的 `sink 4 + retrieval 8 + local 8` latent。

## 2. 当前六个 Case

| Case | dyKV | 检索 FOV | 检索时压缩 | 裁剪 FOV | 用途 |
| --- | --- | --- | --- | --- | --- |
| `baseline` | 关闭 | — | — | — | 上游 local cache 基线 |
| `retrieval_no_compression` | 开启 | 相机 `K` | 不压缩 | — | 隔离长期检索收益和最大开销 |
| `fixed_novelty` | 开启 | 相机 `K` | 固定锚点 + 新颖性 | — | 与相机无关的压缩对照 |
| `yaw_fixed_fov` | 开启 | 固定 `60°×35°` | yaw 空间列裁剪 | 固定水平 `60°` | F0，复现固定角度假设 |
| `yaw_mixed_fov` | 开启 | 固定 `60°×35°` | yaw 空间列裁剪 | 相机 `K` | F1，隔离检索角度的影响 |
| `yaw_intrinsics` | 开启 | 相机 `K` | yaw 空间列裁剪 | 相机 `K` | F2，默认完整方法 |

`baseline` 必须不带 `--dykv`；其余 case 通过 `--dykv --dykv-case NAME` 启用。当前 minWM
默认归一化内参对应约 `89.424°×58.225°`，因此 F0 与 F2 是有效且差异明显的消融。
当历史归档或当前 query 缺少合法 `K` 时，内参检索回退到固定 `60°×35°`；动态裁剪因
缺失几何信息回退到固定新颖性压缩。

可随时列出注册表：

```bash
conda activate minwm-fa
LIST_CASES=1 bash Wan21/scripts/inference/run_dykv_cases.sh
```

## 3. 普通 Prompt 一键运行

默认顺序运行全部六组，并将同一输入、轨迹、seed 的结果保存到独立子目录：

```bash
conda activate minwm-fa
DATA_PATH=Wan21/prompts/demos.txt \
TRAJECTORY='j*10,l*10,n*3' \
NUM_OUTPUT_FRAMES=24 \
SEED=0 \
OUTPUT_ROOT=output/dykv_cases \
bash Wan21/scripts/inference/run_dykv_cases.sh
```

只运行部分 case 时使用逗号分隔的 `CASES`，例如：

```bash
CASES=baseline,retrieval_no_compression,yaw_intrinsics \
OUTPUT_ROOT=output/dykv_core \
bash Wan21/scripts/inference/run_dykv_cases.sh
```

Runner 会把结果写到 `OUTPUT_ROOT/{case}/`。生成参数仍可用
`CONFIG_PATH`、`CHECKPOINT_PATH`、`SP_SIZE` 等原有环境变量覆盖。执行前可设置
`DRY_RUN=1` 检查所有命令而不加载模型。

## 4. MBench 一键运行与打包

设置 `MBENCH_ROOT` 后，同一个 runner 会先转换一次用例，再为每个 case 生成并注册独立
的 MBench model：

```bash
conda activate minwm-fa
MBENCH_ROOT=/absolute/path/to/MBench-A-Setup \
ASSIGNMENTS=/absolute/path/to/samples.jsonl \
CASES=baseline,retrieval_no_compression,fixed_novelty,yaw_fixed_fov,yaw_mixed_fov,yaw_intrinsics \
NUM_OUTPUT_FRAMES=100 \
SEED=0 \
OUTPUT_ROOT=output/mbench_dykv_cases \
MODEL_PREFIX=minwm_dykv \
bash Wan21/scripts/inference/run_dykv_cases.sh
```

输出模型 ID 为 `minwm_dykv_{case}_seed0`。`SUBSETS`、`CONDITIONS`、`LIMIT` 和
`LINK_MODE` 与单组 MBench runner 的含义一致。正式对比必须保持 case 清单、checkpoint、
轨迹、输出长度和 seed 完全相同。
