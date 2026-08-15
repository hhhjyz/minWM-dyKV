# WorldKV 原始位姿检索与 FOV 检索消融

## 1. 结论

本项目参考的 WorldKV 工作树为 commit `046f6d1`。检查表明，WorldKV 与 minWM-dyKV
除了历史块打分算法外，缓存规模、chunk 大小、启用时机、选中块排列、RoPE 和压缩时机也
不完全一致。因此不能把 WorldKV 仓库直接跑出的结果与当前 FOV case 相比并把差异全部归因
于“检索算法”。

为得到单变量消融，当前项目新增：

```text
worldkv_pose_no_compression
```

它只移植 WorldKV 的平均相机位姿得分，其余部分全部复用 dyKV。应与
`retrieval_no_compression` 成对运行。

## 2. WorldKV 原始检索公式

WorldKV 对当前 chunk 和每个候选 chunk 分别计算平均绝对 C2W 位姿：

```text
t_bar = mean(frame translations)
R_bar = mean(frame rotation matrices)
```

随后计算：

```text
translation_squared_i = sum((t_i - t_query)^2)
rotation_radians_i = acos(clamp((trace(R_i^T R_query)-1)/2, -1, 1))

translation_normalized_i = translation_squared_i / max_j(translation_squared_j)
rotation_normalized_i = rotation_radians_i / max_j(rotation_radians_j)

distance_i = 0.5 * translation_normalized_i
           + 0.5 * rotation_normalized_i
```

如果某一项的候选最大值为 0，该项保持全 0。距离最小的历史 chunk 优先。

需要注意三个实现语义：

- 代码注释称 translation L2 distance，但实际计算的是**平方距离**，没有开平方；
- 平移和旋转分别按当前候选集合的最大值归一化，因此加入一个很远的候选可能改变其他块的
  相对综合分数；
- chunk 内旋转矩阵直接做算术平均，不会投影回 `SO(3)`，本次移植保持这一原始行为。

minWM 归档的是 W2C，因此 `dykv_worldkv.py` 先逐帧求逆得到 C2W，再执行上述公式。输入保持
FP32，避免重新引入旧 BF16 相机精度问题。

## 3. 与当前 FOV 检索的差异

| 项目 | WorldKV 原始位姿检索 | 当前 FOV 检索 |
| --- | --- | --- |
| 输入 | C2W 外参 | W2C 外参 + 相机内参 `K` |
| chunk 表示 | 所有帧平移/旋转矩阵的平均 | 当前 4 帧逐帧查询；历史取首帧和中间帧 |
| 距离 | 候选集合归一化的平移平方距离与旋转测地距离，各占 0.5 | 8192 个确定性三维探针的视锥 overlap，`1-overlap` |
| 是否使用真实 FOV | 否 | 是，由 `K` 推导 |
| 平移处理 | 直接比较相机中心 | 由有限半径三维探针近似共同可见空间 |
| 内容状态 | 不使用 | 不使用 |
| 分数稳定性 | 随候选集合最大值变化 | 每对 query/history 独立计算 |

两者都属于相机几何检索，不会检查人物、物体或不可逆世界状态是否已经改变。

## 4. 除检索算法之外是否一致

不一致。WorldKV 原仓库与当前项目还有以下实现差异：

| 环节 | WorldKV `046f6d1` | minWM-dyKV |
| --- | --- | --- |
| 模型与 chunk | LingBot-World-Fast，默认 3 latent/chunk | minWM Wan Action2V，4 latent/chunk |
| sink / local | 模型默认 sink 3；recent 固定 6 | sink 4；local 8（recent 4 + current 4） |
| retrieval 数量 | CLI 指定，必须为 3 的倍数 | case 固定；无压缩对照为 8 latent/2 chunks |
| 开始检索 | bank 和合法候选可用时 | `current_frame>=20` 后启用固定三区域布局 |
| 候选过滤 | 按 chunk ID 排除 sink 和 recent chunks | 按精确 frame 边界排除 sink 与仍在 local 的块 |
| KV 归档 | 干净 `t=0` 前向后归档，CPU/GPU 可选 | 干净前向后无损归档到 CPU |
| 选中块顺序 | `topk` 返回顺序直接拼接 | 得分选中后按历史时间顺序实例化 |
| RoPE | `retrieval_rope_correction` 是可选开关，默认关闭 | 始终 rebase 到连续 retrieval 位置 4--11 |
| 压缩时机 | 支持存储时压缩或检索时压缩 | CPU bank 无损，只做 retrieval-time compression |
| 注意力布局 | `sink | retrieval | recent`，具体长度随参数变化 | 固定连续 `4+8+8`，总计 20 latent |

两者共同点是：每个生成 chunk 规划一次检索、复用到各层和去噪 step；候选排除 sink/recent；
干净 KV 分层归档；最终以 `sink | retrieval | local` 形式进入注意力。

## 5. 公平消融 Case

| 固定项 | 两个 case 的共同配置 |
| --- | --- |
| sink | 固定最初 4 latent |
| retrieval 区域 | 8 latent / 12480 token |
| local | 8 latent（recent 4 + current 4） |
| 候选集合 | 相同的 evicted blocks |
| KV bank | 相同的无损 CPU 归档 |
| 压缩 | 均为 `none` |
| packing | 均为 `none` |
| 填充顺序 | 按历史时间升序 |
| RoPE | 相同地映射到 4--11 |
| 唯一变量 | `fov` 或 `worldkv_pose` 排名得分 |

日志公共字段 `retrieval_mode` 用于确认实际选择器。WorldKV case 还记录：

```text
worldkv_translation_squared
worldkv_rotation_degrees
worldkv_translation_normalized
worldkv_rotation_normalized
```

这些数组都与 `ranked_candidate_block_ids` 和 `distances` 对齐。

## 6. 轨迹级选择器回放

在不加载 checkpoint、只回放现有典型轨迹 `j*49,l*49,n*1` 的 20 次候选选择时，两种算法
有 19 次选择相同，仅最后一次不同：

```text
current_frame=96
FOV selected starts      = [4,8]
WorldKV selected starts  = [4,88]
```

这与此前日志诊断的可疑点相吻合：FOV 最后把早期 `[4,12)` 与固定 sink 拼成连续的初始历史，
WorldKV 位姿得分则在第二个位置选择了更近期的返回段。不过，这只是使用相机轨迹和候选边界
进行的选择器回放，不包含生成内容，不能提前当作 WorldKV 视频质量更好的证据。

## 7. 运行命令

### 普通 prompt

```bash
cd /data/zju-151/jiangyize/research/minWM-dyKV
conda activate minwm-fa

CASES=retrieval_no_compression,worldkv_pose_no_compression \
DATA_PATH=Wan21/prompts/demos.txt \
TRAJECTORY='j*10,l*10,n*3' \
NUM_OUTPUT_FRAMES=24 \
SEED=0 \
OUTPUT_ROOT=output/retrieval_algorithm_ablation \
bash Wan21/scripts/inference/run_dykv_cases.sh
```

### 四个典型 MBench 样本

```bash
cd /data/zju-151/jiangyize/research/minWM-dyKV
conda activate minwm-fa

MBENCH_ROOT=/data/zju-151/jiangyize/research/datasets/MBench-Data/MBench-A \
ASSIGNMENTS="$PWD/Wan21/prompts/mbench_typical_4.jsonl" \
CASES=retrieval_no_compression,worldkv_pose_no_compression \
NUM_OUTPUT_FRAMES=100 \
SEED=0 \
OUTPUT_ROOT=output/mbench_typical_4_retrieval_ablation \
MODEL_PREFIX=minwm_retrieval_ablation \
bash Wan21/scripts/inference/run_dykv_cases.sh
```

这会生成 4 样本 × 2 算法，共 8 个视频。正式比较必须使用相同 checkpoint、assignment、
轨迹、seed 和已有输出处理规则，并使用全新输出目录。当前推理会让对应 prompt index 跨
case 使用相同初始噪声；比较前应核对生成清单中的 sample seed 和噪声指纹，详见
[`REPRODUCIBLE_VIDEO_SEEDS.md`](REPRODUCIBLE_VIDEO_SEEDS.md)。

## 8. 判读建议

除视频质量外，至少比较：

- 每个 event 的 `selected_frame_starts` 和两个算法的选择分歧率；
- 选中块与 sink、local 的视角冗余；
- 返回阶段是否偏向最早历史或近期同视角状态；
- human/object/causal/environment 四类样本的差异；
- retrieval 耗时。FOV 有探针投影开销，WorldKV pose 只有小矩阵运算；
- MBench 指标与人工视频观察，不要只根据几何距离判断优劣。

当前工作树已完成实现、75/75 全量单元测试与双 case runner dry-run，尚未生成这 8 个正式对比
视频，实验结果保持“待运行”。
