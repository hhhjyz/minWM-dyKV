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

- `Wan21/prompts/mbench_typical_4.jsonl`：每个 subset 一个样本，左右方向各两个；
- `Wan21/prompts/mbench_typical_8.jsonl`：每个 subset、每种方向各一个样本。

## 2. 推荐的四样本集合

| Subset / 方向 | Sample ID | 场景及观察重点 |
| --- | --- | --- |
| causal / 左→右 | `a00037_00237` | 轮胎压碎彩色水珠托盘；观察不可逆状态变化、散落物和返回视角一致性 |
| environment / 右→左 | `sample_166_5ff17586` | 池塘、折线路径和传统凉亭；观察远近结构、纹理和闭环恢复 |
| human / 左→右 | `mem_openhumanvid_079f...e2b7` | 窗边单人侧脸；观察人物身份、脸部和光照稳定性 |
| object / 右→左 | `sample_194_a04d54a4` | 绿色房间、条纹扶手椅、窗户和盆栽；观察固定物体布局和几何结构 |

这是默认推荐集。四个样本覆盖四类评测内容并平衡左右运动。所有方法均固定最初 4 帧
sink。最小兼容性检查可比较 `baseline`、`retrieval_no_compression` 和
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
| human / 右→左 | `mem_openhumanvid_8520...f1a6c` | 多人物复古办公室；人物身份和复杂背景布局 |
| object / 左→右 | `sample_007_7b421919` | 陈列丰富的书架；小物体、高频纹理和空间位置 |

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
OUTPUT_ROOT=output/mbench_typical_4 \
MODEL_PREFIX=minwm_typical4 \
bash Wan21/scripts/inference/run_dykv_cases.sh
```

如果只关心固定 FOV 与相机内参的差异：

```bash
MBENCH_ROOT=/data/zju-151/jiangyize/research/datasets/MBench-Data/MBench-A \
ASSIGNMENTS="$PWD/Wan21/prompts/mbench_typical_4.jsonl" \
CASES=yaw_fixed_fov,yaw_mixed_fov,yaw_intrinsics \
NUM_OUTPUT_FRAMES=100 \
SEED=0 \
OUTPUT_ROOT=output/mbench_typical_4_fov \
MODEL_PREFIX=minwm_typical4_fov \
bash Wan21/scripts/inference/run_dykv_cases.sh
```

当前 predecessor 核心小规模对比：

```bash
MBENCH_ROOT=/data/zju-151/jiangyize/research/datasets/MBench-Data/MBench-A \
ASSIGNMENTS="$PWD/Wan21/prompts/mbench_typical_4.jsonl" \
CASES=baseline,retrieval_no_compression,yaw_intrinsics,predecessor_chunks_latent,predecessor_query_backfill \
NUM_OUTPUT_FRAMES=100 \
SEED=0 \
OUTPUT_ROOT=output/mbench_typical_4_predecessor \
MODEL_PREFIX=minwm_typical4_predecessor \
bash Wan21/scripts/inference/run_dykv_cases.sh
```

将 `mbench_typical_4.jsonl` 替换为 `mbench_typical_8.jsonl` 即可运行八样本方向检查。

MBench 官方 25 秒条件必须使用 100 个 latent 帧，对应 397 个解码视频帧。若只是检查命令
和路径，可先添加 `DRY_RUN=1`；它不会加载模型或生成视频。

## 5. 输出位置

四样本三组兼容性检查生成到：

```text
output/mbench_typical_4/
├── baseline/
├── retrieval_no_compression/
└── yaw_intrinsics/
```

同时会在 MBench 数据集的 `models/` 下注册：

```text
minwm_typical4_baseline_seed0
minwm_typical4_retrieval_no_compression_seed0
minwm_typical4_yaw_intrinsics_seed0
```

定性对比时应固定 sample、轨迹和 seed，重点查看向外旋转阶段、最大偏航附近，以及返回原
视角后的主体身份、背景结构和细节是否恢复。
