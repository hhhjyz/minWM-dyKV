# dyKV 动态压缩消融实验

## 1. 实验目标

消融实验需要分别回答三个问题：长期 KV 检索是否有效、压缩是否真正减少注意力开销、
基于相机几何的“保留位置”是否比只调整 token 数量更重要。所有实验统一使用
`minwm-fa`、相同 checkpoint、相同 case 和相同 seed。所有 case 都使用 `0~19` 有界
tri-region RoPE：baseline 为 `4+0+16`，dyKV 为 `4+8+8`，后者具有 8 latent 帧物理
retrieval token 预算。非扩容方法的源帧预算也是 8；扩容方法允许覆盖更多源帧，但物理
token 仍不得超过 8 个完整 latent。

## 2. 当前可以直接运行的核心消融

核心对照已收敛为注册 case，可由统一 runner 直接运行：

| 编号 | Case | 方法 | 回答的问题 |
| --- | --- | --- | --- |
| A0 | `baseline` | `4+0+16` tri-region RoPE；固定 sink + rolling local 16 | 无长期记忆的统一 RoPE 基线 |
| A1 | `retrieval_no_compression` | 内参 FOV 检索，不压缩 retrieval KV | 检索本身带来的质量收益与最大开销 |
| A1-W | `worldkv_pose_no_compression` | WorldKV 平均位姿检索，不压缩 retrieval KV | 与 A1 隔离检索评分公式的影响 |
| A2 | `fixed_novelty` | 内参检索 + 固定锚点/新颖性压缩 | 与相机无关的内容压缩效果 |
| A3 | `yaw_intrinsics` | 内参检索 + 当前-query yaw/FOV 裁剪 | E0 兼容默认的质量/效率折中 |
| A11 | `packed_chunks` | 固定档位 + 完整 chunk 动态装箱 | 将压缩容量转换为更长历史覆盖 |
| A12 | `packed_chunks_latent` | A11 + 单 latent 尾部补齐 | 检查不能容纳完整 chunk 时的余量收益 |
| A13 | `retr8_compression_r050` | 8 源帧、anchor + `r=1/2` | 相同覆盖下固定压缩的信息损失 |
| A14 | `retr12_compression_r050` | 12 源帧、anchor + `r=1/2` | 相同压缩率下增加一个 chunk 的收益 |
| A15 | `retr16_compression_r033` | 16 源帧、anchor + `r=1/3` | 等 8 帧物理预算下覆盖两倍历史 |
为兼容已有命令，未指定 case 的 dyKV 推理默认 A3。A0--A3、A1-W、A11--A15 均可由
`run_dykv_cases.sh` 一键运行。

A1 与 A1-W 具有相同候选集合、缓存布局、token 预算、填充顺序和 RoPE，只分别使用 FOV
overlap 与 WorldKV 平均位姿得分。原 WorldKV 仓库中其他不一致项没有混入该消融，详见
[`WORLDKV_RETRIEVAL_ABLATION.md`](WORLDKV_RETRIEVAL_ABLATION.md)。

A1/A13/A14/A15 对应 minWM-back 的固定预算 A--D。A1 与 A15 物理 retrieval token
完全相同；A13 与 A14 使用相同 `r=1/2`。详细预算见
[`FIXED_WORLDKV_CASES.md`](FIXED_WORLDKV_CASES.md)。

### 公平比较方式

需要同时做两种口径，避免动态方法因 token 数不同而被错误解释：

1. **等历史覆盖**：A1/A2/A3 都先检索相同的 8 个原始 latent 帧，比较实际 attention
   token、耗时和质量。这是主实验。
2. **等 attention token**：根据 A3 的平均实际 token 数，为 A1/A2 构造相近计算量的
   离线对照。这是补充实验，用于分离“保留位置”与“token 数量”的影响。

## 3. 建议增加的机制消融

以下实验需要增加小型内部诊断策略，不能宣称当前已经可直接运行：

| 编号 | 对照设计 | 固定变量 | 目的 |
| --- | --- | --- | --- |
| A4 | 同数量随机列 | 使用 A3 的逐帧 token 数 | 判断收益是否来自正确空间位置，而不只是更少 token |
| A5 | 同数量新颖性 token | 使用 A3 的逐帧 token 数 | 比较相机几何位置与内容相似度选择 |
| A6 | 无方向比例裁剪 | 保留比例相同，但始终从中心/固定一侧裁剪 | 验证有向 yaw 决定左右位置的重要性 |
| A7 | query 首帧 / 中间帧 / 末帧 / 4 帧并集 | 其余相同 | 验证多 query mask 并集是否必要 |
| A9 | 无交集时丢弃块 / 每帧最少保留一列 | 其余相同 | 验证空块策略与稳定性 |
| A10 | 仅几何裁剪 / 几何后再做新颖性压缩 | 相同历史块 | 测试二阶段压缩的效率上界 |

A4--A7、A9--A10 只作为后续内部实验设计，避免重新制造大量公开超参数。原 A8 固定/
混合 FOV 消融已经删除：所有 FOV case 强制使用相机内参，不再将固定角度作为研究变量；
A1-W 是另一种不使用 FOV 的 WorldKV 外参位姿检索，不属于固定角度 FOV 回退。

## 4. 检索与压缩解耦消融

动态裁剪作用于“已经被选中的历史块”，因此还应确认收益不是 FOV 检索独自产生：

| 编号 | 历史块选择 | 压缩 | 目的 |
| --- | --- | --- | --- |
| R0 | 最近可用块 | 无压缩 | 非几何检索基线 |
| R1 | 最近可用块 | yaw/FOV 裁剪 | 单独观察压缩收益 |
| R2-W | WorldKV 平均位姿检索 | 无压缩 | 原检索评分对照，已实现为 A1-W |
| R2 | FOV 检索 | 无压缩 | 单独观察 FOV 检索收益，已实现为 A1 |
| R3 | FOV 检索 | 历史相对 query 的 yaw/FOV 裁剪 | E0 query-relative 完整组合 |

当前代码已直接支持 R2-W/R2/R3；R0/R1 需要增加内部的 recent-only 选择器。各组必须保持原始
retrieval 帧预算相同。

## 5. 运动类型与边界消融

### 官方 MBench 主实验

当前 547 个官方 case 全部是 25 秒 yaw 往返，适合按以下维度拆分：

- `left_then_right` 与 `right_then_left` 分别报告，检查左右方向偏差；
- `human`、`object`、`environment`、`causal` 分 subset 报告；
- 按相对 yaw 分桶：`0--15°`、`15--30°`、`30--60°`、`60--90°`、`>90°`；
- 分别统计离开阶段和返回阶段，检查闭环返回时记忆是否恢复。

### 补充边界轨迹

| 轨迹 | 预期行为 | 验证点 |
| --- | --- | --- |
| 0° 静止 | A3 保留全部；A16--A18 使用 1/4 novelty 安全下限 | 分别验证两种已注册语义 |
| 半个实际 FOV | 约保留一半水平角域 | 左右 mask 应互为镜像 |
| 一个实际 FOV | 无水平交集 | 历史块应从载荷中移除 |
| 360° | 回到等价朝向 | 角度回绕后应保留全部列 |
| 720° / 1080° | 多次周期回访 | 多周旋转不得累积角度误差 |
| 前后移动 | 固定压缩回退 | 第一版不错误应用 yaw 裁剪 |
| 横向平移 | 固定压缩回退 | 为后续三维投影版本建立基线 |

## 6. 指标与记录要求

每次实验至少记录：

- MBench 原生 entity/environment/causal 指标和 trigger coverage；
- 选中与实际实例化的历史块、逐帧相对 yaw、FOV 和保留列；
- 每层 raw/kept retrieval token、压缩比例和空块比例；
- retrieval 选择、CPU→GPU 传输、裁剪及总生成耗时；
- 峰值显存、CPU 无损记忆库字节数；
- 左/右方向、yaw 分桶和四个 subset 的分组结果。

主结论至少同时报告 A0、A1、A1-W、A3、A11--A15，不能只报告兼容默认或只报告新方法。
A4--A10 与 R0--R3 用来解释机制，不应在观察结果后只选择有利的子集报告。

动态压缩后用空余 token 容量装入更多 chunk，以及对应的 E0--E6 消融，见
[`DYNAMIC_RETRIEVAL_PACKING.md`](DYNAMIC_RETRIEVAL_PACKING.md)。
