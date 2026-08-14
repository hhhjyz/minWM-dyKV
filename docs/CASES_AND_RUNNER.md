# dyKV 实验 Case 与统一 Runner

## 1. 设计原则

所有可直接运行的对照都注册在 `Wan21/dykv_cases.py`。公开接口只增加一个枚举参数
`--dykv-case`，每个 case 一次性确定压缩方式、检索 FOV 来源和裁剪 FOV 来源，不再暴露
彼此独立的内部开关。所有 case 都固定保留最初 4 个 latent 作为 sink：baseline 使用
`fixed sink 4 + rolling local 16`，十三个 dyKV case 使用连续的
`fixed sink 4 + retrieval 8 + local 8`。

## 2. 当前十四个 Case

| Case | Sink | dyKV | 检索 FOV | 检索时压缩 | 裁剪 FOV | 用途 |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline` | 固定 4 | 关闭 | — | — | — | 无长期检索的固定 sink 基线 |
| `retrieval_no_compression` | 固定 4 | 开启 | 相机 `K` | 不压缩 | — | 隔离长期检索收益和最大开销 |
| `fixed_novelty` | 固定 4 | 开启 | 相机 `K` | 固定锚点 + 新颖性 | — | 与相机无关的压缩对照 |
| `yaw_fixed_fov` | 固定 4 | 开启 | 固定 `60°×35°` | yaw 空间列裁剪 | 固定水平 `60°` | F0，复现固定角度假设 |
| `yaw_mixed_fov` | 固定 4 | 开启 | 固定 `60°×35°` | yaw 空间列裁剪 | 相机 `K` | F1，隔离检索角度的影响 |
| `yaw_intrinsics` | 固定 4 | 开启 | 相机 `K` | yaw 空间列裁剪 | 相机 `K` | F2，默认完整方法 |
| `packed_chunks` | 固定 4 | 开启 | 相机 `K` | `{1,1/2,1/4}` 固定档位 | 相机 `K` | E1，32 原子预算内扩充完整 chunk |
| `packed_chunks_latent` | 固定 4 | 开启 | 相机 `K` | 固定档位 + 单 latent 尾部 | 相机 `K` | E2，完整 chunk 后继续补齐余量 |
| `predecessor_chunks` | 固定 4 | 开启 | 当前 query、相机 `K` | 相对前驱的 `{1/4,1/2,3/4,1}` 裁剪 | 相机 `K` | P0，仅装入完整 chunk |
| `predecessor_chunks_latent` | 固定 4 | 开启 | 当前 query、相机 `K` | 前驱四档裁剪 + latent 尾部 | 相机 `K` | P1，用单 latent 补齐余量 |
| `predecessor_query_backfill` | 固定 4 | 开启 | 当前 query、相机 `K` | P1 + query 可见列回填 | 相机 `K` | P2，完整前驱增量方案 |
| `retr8_compression_r050` | 固定 4 | 开启 | 相机 `K` | 2 chunk，anchor + `r=1/2` | — | minWM-back B：8 源帧压到 5 帧容量 |
| `retr12_compression_r050` | 固定 4 | 开启 | 相机 `K` | 3 chunk，anchor + `r=1/2` | — | minWM-back C：12 源帧压到 7.5 帧容量 |
| `retr16_compression_r033` | 固定 4 | 开启 | 相机 `K` | 4 chunk，anchor + `r=1/3` | — | minWM-back D：16 源帧压到 8 帧容量 |

`baseline` 必须不带 `--dykv`；其余 case 通过 `--dykv --dykv-case NAME` 启用。当前 minWM
默认归一化内参对应约 `89.424°×58.225°`，因此 F0 与 F2 是有效且差异明显的消融。
当历史归档或当前 query 缺少合法 `K` 时，内参检索回退到固定 `60°×35°`；动态裁剪因
缺失几何信息回退到固定新颖性压缩。

固定 WorldKV A--D 的预算公式、与旧实现的差异及运行方式见
[`FIXED_WORLDKV_CASES.md`](FIXED_WORLDKV_CASES.md)。
前驱压缩三个 case 的公式、退化路径、装箱与 RoPE 语义见
[`PREDECESSOR_INCREMENTAL_COMPRESSION.md`](PREDECESSOR_INCREMENTAL_COMPRESSION.md)。

可随时列出注册表：

```bash
conda activate minwm-fa
LIST_CASES=1 bash Wan21/scripts/inference/run_dykv_cases.sh
```

## 3. 普通 Prompt 一键运行

默认顺序运行全部十四组，并将同一输入、轨迹、seed 的结果保存到独立子目录：

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
CASES=yaw_intrinsics,packed_chunks,packed_chunks_latent \
OUTPUT_ROOT=output/dykv_core \
bash Wan21/scripts/inference/run_dykv_cases.sh
```

前驱增量方法的三组对照可使用：

```bash
CASES=predecessor_chunks,predecessor_chunks_latent,predecessor_query_backfill \
OUTPUT_ROOT=output/predecessor_cases \
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
CASES=baseline,yaw_intrinsics,packed_chunks,packed_chunks_latent \
NUM_OUTPUT_FRAMES=100 \
SEED=0 \
OUTPUT_ROOT=output/mbench_dykv_cases \
MODEL_PREFIX=minwm_dykv \
bash Wan21/scripts/inference/run_dykv_cases.sh
```

输出模型 ID 为 `minwm_dykv_{case}_seed0`。`SUBSETS`、`CONDITIONS`、`LIMIT` 和
`LINK_MODE` 与单组 MBench runner 的含义一致。正式对比必须保持 case 清单、checkpoint、
轨迹、输出长度和 seed 完全相同。
