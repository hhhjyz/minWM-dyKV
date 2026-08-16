# MBench 典型样本小规模对比

## 1. 官方清单概况

当前 `hy_worldplay/samples.jsonl` 有 547 个 MBench-A case，全部为 25 秒水平 yaw 往返：

| Subset | 左转再右转 | 右转再左转 | 合计 |
| --- | ---: | ---: | ---: |
| causal | 50 | 50 | 100 |
| environment | 104 | 123 | 227 |
| human | 60 | 60 | 120 |
| object | 59 | 41 | 100 |
| 合计 | 273 | 274 | 547 |

直接设置 `LIMIT=4` 会截取官方清单开头四个 causal case，不能代表四个 subset。因此项目
提供两个显式 assignment 清单：

- `Wan21/prompts/mbench_typical_4.jsonl`：每个 subset 一个前景/背景主体鲜明的样本，左右方向
  各两个；
- `Wan21/prompts/mbench_typical_8.jsonl`：以前四行为完整前缀，再为每个 subset 补齐相反
  方向。因此同一 `SEED` 下，typical-4 样本在两个清单中的 prompt index 和初始噪声一致。

## 2. 推荐的四样本集合

| Subset / 方向 | Sample ID | 场景及观察重点 |
| --- | --- | --- |
| causal / 左→右 | `a00327_00512` | 前景彩色水球、后方轮胎；观察水球颜色/数量、轮胎形状及返回后的不可逆状态一致性 |
| environment / 右→左 | `sample_214_f1fe1644` | 明亮展厅中的白色跑车与人物；观察车身轮廓、人物位置和背景结构恢复 |
| human / 右→左 | `mem_openhumanvid_7b09...de06` | 白色宇航服、头盔反光、蓝色地球与金属舱体；观察人物身份和高对比轮廓 |
| object / 左→右 | `sample_024_3575967d` | 黄色贵妃椅、黑白斑马纹地毯和花纹背景；观察颜色、纹理及相对布局 |

这是默认推荐集。四个样本覆盖四类评测内容并平衡左右运动，同时刻意选择了前景主体与
背景地标都容易辨认的首帧。由于 25 秒条件执行 yaw 往返，结尾返回初始视角后，可以直接
检查这些主体的身份、颜色、轮廓和相对位置是否恢复。所有方法均固定最初 4 帧 sink。
最小兼容性检查可比较 `baseline`、`retrieval_no_compression` 和
`yaw_intrinsics`，共生成 12 个视频；它只覆盖 E0 默认路径，不代表当前 predecessor
完整方法已经参与比较。

若只比较动态扩容机制，可将三组 case 改为
`yaw_intrinsics,packed_chunks,packed_chunks_latent`，仍只生成 12 个视频，分别对应
E0/E1/E2。

验证当前 predecessor 方案时，推荐比较
`baseline,retrieval_no_compression,yaw_intrinsics,predecessor_chunks_latent,predecessor_query_backfill`，
四个样本共生成 20 个视频。这样既保留无记忆、不压缩和 E0 对照，也能分离 latent 尾部与
query coverage 回填的增益。

## 3. 扩展的八样本集合

八样本集合在每个 subset 内补齐相反方向：

| Subset / 方向 | Sample ID | 场景及观察重点 |
| --- | --- | --- |
| causal / 右→左 | `a00533_00678` | 木工车床加工和飞散木屑；动态纹理与因果连续性 |
| environment / 左→右 | `sample_001_bcd37a7f` | 林间道路、树木和远处车辆；长结构与远景回访 |
| human / 左→右 | `mem_openhumanvid_079f...e2b7` | 窗边单人侧脸；补齐人物左转方向，观察脸部和光照稳定性 |
| object / 右→左 | `sample_194_a04d54a4` | 绿色房间、人物、条纹扶手椅和窗户；补齐物体右转方向 |

这组共 8×3=24 个核心对比视频，主要用于确认算法不存在明显的左右方向偏差。

## 4. 推荐运行命令

先进入项目和统一环境：

```bash
cd /data/zju-151/jiangyize/research/minWM-dyKV
conda activate minwm-fa
```

四样本、三种核心方法：

```bash
MBENCH_ROOT=/data/zju-151/jiangyize/research/datasets/MBench-Data/MBench-A \
ASSIGNMENTS="$PWD/Wan21/prompts/mbench_typical_4.jsonl" \
CASES=baseline,retrieval_no_compression,yaw_intrinsics \
NUM_OUTPUT_FRAMES=100 \
SEED=0 \
OUTPUT_ROOT=output/mbench_typical_4_salient_v2 \
MODEL_PREFIX=minwm_typical4_salient_v2 \
bash Wan21/scripts/inference/run_dykv_cases.sh
```

当前 predecessor 核心小规模对比：

```bash
MBENCH_ROOT=/data/zju-151/jiangyize/research/datasets/MBench-Data/MBench-A \
ASSIGNMENTS="$PWD/Wan21/prompts/mbench_typical_4.jsonl" \
CASES=baseline,retrieval_no_compression,yaw_intrinsics,predecessor_chunks_latent,predecessor_query_backfill \
NUM_OUTPUT_FRAMES=100 \
SEED=0 \
OUTPUT_ROOT=output/mbench_typical_4_salient_v2_predecessor \
MODEL_PREFIX=minwm_typical4_salient_v2_predecessor \
bash Wan21/scripts/inference/run_dykv_cases.sh
```

只比较 WorldKV 原始位姿检索与 FOV 检索：

```bash
MBENCH_ROOT=/data/zju-151/jiangyize/research/datasets/MBench-Data/MBench-A \
ASSIGNMENTS="$PWD/Wan21/prompts/mbench_typical_4.jsonl" \
CASES=retrieval_no_compression,worldkv_pose_no_compression \
NUM_OUTPUT_FRAMES=100 \
SEED=0 \
OUTPUT_ROOT=output/mbench_typical_4_salient_v2_retrieval_ablation \
MODEL_PREFIX=minwm_typical4_salient_v2_retrieval_ablation \
bash Wan21/scripts/inference/run_dykv_cases.sh
```

该命令生成 8 个视频；两个 case 只改变检索评分公式。实现边界与日志字段见
[`WORLDKV_RETRIEVAL_ABLATION.md`](WORLDKV_RETRIEVAL_ABLATION.md)。

将 `mbench_typical_4.jsonl` 替换为 `mbench_typical_8.jsonl` 即可运行八样本方向检查。

MBench 官方 25 秒条件必须使用 100 个 latent 帧，对应 397 个解码视频帧。若只是检查命令
和路径，可先添加 `DRY_RUN=1`；它不会加载模型或生成视频。

## 5. 输出位置

四样本三组兼容性检查生成到：

```text
output/mbench_typical_4_salient_v2/
├── baseline/
├── retrieval_no_compression/
└── yaw_intrinsics/
```

同时会在 MBench 数据集的 `models/` 下注册：

```text
minwm_typical4_salient_v2_baseline_seed0
minwm_typical4_salient_v2_retrieval_no_compression_seed0
minwm_typical4_salient_v2_yaw_intrinsics_seed0
```

定性对比时应固定 sample、轨迹和 seed，重点查看向外旋转阶段、最大偏航附近，以及返回原
视角后的主体身份、背景结构和细节是否恢复。
当前 runner 会按 `base_seed+prompt_index` 固定每个样本的初始噪声；比较前还应核对两个
case 的 `generation_manifest.jsonl` 中 sample seed 和初始噪声指纹一致。规则见
[`REPRODUCIBLE_VIDEO_SEEDS.md`](REPRODUCIBLE_VIDEO_SEEDS.md)。

旧版 typical-4 使用水珠托盘、池塘凉亭、窗边侧脸和绿色房间四个样本。assignment 已被
替换；为避免 runner 把旧 MP4 误判为可续跑结果，新一轮必须使用新的 `OUTPUT_ROOT` 和
`MODEL_PREFIX`，例如 `output/mbench_typical_4_salient_v2` 与 `minwm_typical4_salient_v2`。
