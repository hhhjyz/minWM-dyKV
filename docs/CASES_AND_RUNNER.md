# dyKV 实验 Case 与统一 Runner

## 1. 设计原则

所有可直接运行的对照都注册在 `Wan21/dykv_cases.py`。公开接口只增加一个枚举参数
`--dykv-case`，每个 case 一次性确定压缩方式、检索 FOV 来源和裁剪 FOV 来源，不再暴露
彼此独立的内部开关。所有 case 都固定保留最初 4 个 latent 作为 sink，并使用映射到
`0~19` 的 tri-region RoPE：baseline 使用空 retrieval 的 `4+0+16`，十四个 dyKV case
使用连续的 `4+8+8`。

## 2. 当前十五个 Case

| Case | Sink | dyKV | 检索方法 | 检索时压缩 | 裁剪 FOV | 用途 |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline` | 固定 4 | 关闭 | — | — | — | `4+0+16` tri-region RoPE，无长期检索 |
| `retrieval_no_compression` | 固定 4 | 开启 | FOV overlap（相机 `K`） | 不压缩 | — | 隔离长期检索收益和最大开销 |
| `retrieval_no_compression_relevance_order` | 固定 4 | 开启 | 与上一行相同 | 不压缩 | — | 只把最高相关 chunk rebase 到更靠近 query 的 `8~11` |
| `worldkv_pose_no_compression` | 固定 4 | 开启 | WorldKV 平均 C2W 位姿 | 不压缩 | — | 与 `retrieval_no_compression` 构成只改变检索评分的消融 |
| `yaw_intrinsics` | 固定 4 | 开启 | 相机 `K` | 历史相对当前 query 的 yaw 空间列裁剪 | 相机 `K` | E0，兼容默认预设 |
| `packed_chunks` | 固定 4 | 开启 | 相机 `K` | `{1,1/2,1/4}` 固定档位 | 相机 `K` | E1，32 原子预算内扩充完整 chunk |
| `packed_chunks_latent` | 固定 4 | 开启 | 相机 `K` | 固定档位 + 单 latent 尾部 | 相机 `K` | E2，完整 chunk 后继续补齐余量 |
| `retr8_compression_r050` | 固定 4 | 开启 | 相机 `K` | 2 chunk，anchor + `r=1/2` | — | minWM-back B：8 源帧压到 5 帧容量 |
| `retr12_compression_r050` | 固定 4 | 开启 | 相机 `K` | 3 chunk，anchor + `r=1/2` | — | minWM-back C：12 源帧压到 7.5 帧容量 |
| `retr16_r033_slot_packed` | 固定 4 | 开启 | 相机 `K` | 与 D 相同，按 virtual slot 拼接 | — | D 的旧 slot-order 排列诊断 |
| `retr16_compression_r033` | 固定 4 | 开启 | 相机 `K` | 4 chunk，anchor + `r=1/3` | — | minWM-back D：16 源帧压到 8 帧容量 |
| `motion_novelty_slot_capped` | 固定 4 | 开启 | 相机 `K` | 连续 FOV token 数 + novelty，单槽受限 | — | flat 方法的 capped packing 消融 |
| `motion_novelty_unfilled` | 固定 4 | 开启 | 相机 `K` | 连续 FOV token 数 + novelty，flat 欠填 | — | A16 动态压缩基础方法 |
| `motion_novelty_backfill` | 固定 4 | 开启 | 相机 `K` | A16 基础计划 + 唯一 token 回填 | — | A17：测量真实额外信息的作用 |
| `motion_novelty_duplicate` | 固定 4 | 开启 | 相机 `K` | A16 基础计划 + 最高相关 chunk token 重复 | — | A18：满长度/attention 重加权诊断 |

`baseline` 必须不带 `--dykv`；其余 case 通过 `--dykv --dykv-case NAME` 启用。除明确用于
消融的 `worldkv_pose_no_compression` 外，现有检索和几何裁剪 case 都使用归一化相机内参
`K`，不再提供固定 FOV 或混合 FOV。WorldKV 位姿检索只使用外参，不使用 `K`；纯 yaw
裁剪仍需要 `K`。两种检索的公平适配和原仓库其余差异见
[`WORLDKV_RETRIEVAL_ABLATION.md`](WORLDKV_RETRIEVAL_ABLATION.md)。

为保持已有脚本兼容，单独设置 `DYKV=1` 仍默认运行 `yaw_intrinsics`。

固定 WorldKV A--D 的预算公式、与旧实现的差异及运行方式见
[`FIXED_WORLDKV_CASES.md`](FIXED_WORLDKV_CASES.md)。

`retr16_r033_slot_packed` 与 `retr16_compression_r033` 的检索、novelty token、segment 长度、
virtual slot 和 `8F` 预算完全一致，只分别按 virtual slot 与 source frame 排列 payload。
该 case 用于验证非 causal attention 下同步 K/V permutation 的等价性，不是新的压缩方法。

`retrieval_no_compression_relevance_order` 与 `retrieval_no_compression` 的候选、FOV 距离、
被选 chunk、token 数量和压缩模式完全一致。唯一变量是 chunk 的 retrieval RoPE 位置：原
case 按源时间排列，新 case 从 retrieval region 右侧向左按相关性分配，使最高相关 chunk
位于 `8~11`。实现与日志字段见
[`RELEVANCE_ORDERED_RETRIEVAL.md`](RELEVANCE_ORDERED_RETRIEVAL.md)。

### 连续比例 Case

四个 motion novelty case 均已注册并可由 runner 直接执行。它们都使用 chunk 内
anchor-relative 连续 FOV 新增比例决定每帧 token 数，再用 WorldKV novelty 决定具体 token；
区别是 capped 版本限制单个 virtual slot 不超过 `F`，A16 使用 flat 总预算并允许欠填，A17
补回尚未选择的唯一 token，A18 重复最高 query-relevance chunk 的基础 token。A17/A18
共享 A16 的候选排名、selected chunk、基础比例和基础索引，并对齐最终 token 数及 slot load。
完整实现契约见
[`MOTION_ADAPTIVE_NOVELTY_COMPRESSION.md`](MOTION_ADAPTIVE_NOVELTY_COMPRESSION.md)。

下一版 `motion_projected_unfilled` 计划用双向二维、多深度投影替换 sphere overlap，修正
forward translation 为零以及平移依赖球形体积的问题。该 case 当前没有注册，不能传给
runner；详细设计见 [`PROJECTED_MOTION_COMPRESSION.md`](PROJECTED_MOTION_COMPRESSION.md)。

可随时列出注册表：

```bash
conda activate minwm-fa
LIST_CASES=1 bash Wan21/scripts/inference/run_dykv_cases.sh
```

## 3. 普通 Prompt 一键运行

默认顺序运行全部十五组，并将同一输入、轨迹、seed 的结果保存到独立子目录：

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

只比较 WorldKV 原始位姿评分与 FOV 检索时运行：

```bash
CASES=retrieval_no_compression,worldkv_pose_no_compression \
OUTPUT_ROOT=output/retrieval_algorithm_ablation \
bash Wan21/scripts/inference/run_dykv_cases.sh
```

只比较源时间排列与相关性靠近 query 的 RoPE 排列：

```bash
CASES=retrieval_no_compression,retrieval_no_compression_relevance_order \
OUTPUT_ROOT=output/retrieval_order_ablation \
bash Wan21/scripts/inference/run_dykv_cases.sh
```

Runner 会把结果写到 `OUTPUT_ROOT/{case}/`。生成参数仍可用
`CONFIG_PATH`、`CHECKPOINT_PATH`、`SP_SIZE` 等原有环境变量覆盖。执行前可设置
`DRY_RUN=1` 检查所有命令而不加载模型。

所有 case 使用统一的视频级 seed 策略：相同 `SEED` 和 `prompt_index` 会得到相同初始噪声，
case 名称和跳过已有输出不会改变随机初始条件。生成清单会记录 sample seed 与噪声指纹，
详见 [`REPRODUCIBLE_VIDEO_SEEDS.md`](REPRODUCIBLE_VIDEO_SEEDS.md)。

生成清单同时记录 `tri_region_rope_layout`。当前 baseline 必须为 `[4,0,16]`，其余 case
必须为 `[4,8,8]`；旧 baseline 产物仍走普通 RoPE，不能与当前 case 做严格消融。

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
