# `retrieval_no_compression` 尾部检索日志诊断

## 1. 检查范围

本次检查针对最新一组完整典型样本输出：

```text
output/mbench_typical_4_crop_compare_seed0/retrieval_no_compression/
```

其中包含 4 个 prompt，每个视频 100 个 latent、seed 0；轨迹分别是
`j*49,l*49,n*1` 及其左右镜像。每个 prompt 从 `current_frame=20` 开始产生 20 次 retrieval
event。诊断读取 `dykv_summaries.jsonl`，并对照当前候选过滤、FOV 排序和三区域 RoPE 代码。

这些产物早于 baseline `4+0+16` tri-region RoPE 修改。下文的 no-compression 检索事件、
候选顺序和 token 预算诊断仍然成立，但旧 baseline 视频不能再作为当前统一 RoPE 消融的
质量对照；baseline 与 no-compression 必须在当前提交、相同 seed 和新输出目录下重新生成。

另外检查了两个更早输出：

```text
output/mbench_typical_4_all_cases/retrieval_no_compression/
output/mbench_typical_environment_all_cases/retrieval_no_compression/
```

三个输出的最后一步都选择源起始帧 `[4,8]`，不是单次运行偶然现象。

这些 no-compression 视频生成于相机 FP32 保真修复之前。该 case 不执行 yaw 压缩，旧 BF16
问题不会把它切换到 novelty fallback；FOV overlap 本身也会转为 FP32 计算。但位姿已经发生
过量化，正式复现实验仍应使用当前修复版本重跑。本文结论严格描述现有日志，不把旧结果
冒充修复后指标。

## 2. 最后六个 Chunk 的日志

下面以 prompt 0 的 `j*49,l*49,n*1` 为例。prompt 2 完全相同；prompt 1/3 是旋转方向
镜像，选择的 block 和 source frame 仍相同，距离只有确定性探针离散带来的微小差异。

| 当前 chunk | 当前 query yaw | FOV 排名前三 `(block:start:distance)` | 最终 source starts | 每层 retrieval token |
| ---: | --- | --- | --- | ---: |
| 76 | `-66,-63,-60,-57°` | `5:20:0.0410, 4:16:0.1175, 6:24:0.1476` | `[16,20]` | 12480 |
| 80 | `-54,-51,-48,-45°` | `4:16:0.0414, 3:12:0.1181, 5:20:0.1481` | `[12,16]` | 12480 |
| 84 | `-42,-39,-36,-33°` | `3:12:0.0415, 2:8:0.1170, 4:16:0.1490` | `[8,12]` | 12480 |
| 88 | `-30,-27,-24,-21°` | `2:8:0.0411, 1:4:0.1169, 3:12:0.1481` | `[4,8]` | 12480 |
| 92 | `-18,-15,-12,-9°` | `1:4:0.0410, 2:8:0.1478, 3:12:0.2810` | `[4,8]` | 12480 |
| 96 | `-6,-3,0,0°` | `1:4:0.1398, 2:8:0.2728, 22:88:0.2728` | `[4,8]` | 12480 |

最后一个 event 的关键原始字段为：

```text
current_frame:                 96
candidate_block_ids:           [1,2,3,...,22]
ranked_candidate_block_ids:    [1,2,22,3,21,4,20,...]
selected_block_ids:            [1,2]
selected_frame_starts:         [4,8]
materialized_frame_starts:     [4,8]
retrieved_tokens_per_layer:    12480
raw_tokens_per_layer:          12480
compression_modes:             [none, none]
packing_mode:                  none
```

统计四个 prompt 的全部 80 次 event：

```text
选择两个时间相邻 chunk：80 / 80
使用完整 8-latent retrieval token 预算：80 / 80
最后一步选择 source starts [4,8]：4 / 4
```

## 3. 实现不变量检查

日志没有显示 token 数、候选资格或 RoPE 越界错误：

| 检查项 | 结果 |
| --- | --- |
| sink `[0,4)` 是否又进入 retrieval | 否；候选从 block 1 开始 |
| recent/current 是否被重复检索 | 否；`evicted_candidates` 正确排除 local |
| 是否超过 8-latent retrieval 预算 | 否；始终为 12480 token |
| 是否发生了意外压缩 | 否；`compression_modes=[none,none]` |
| 是否选择 FOV 排名前两个合法 4-frame block | 是 |
| 是否按时间顺序 materialize | 是；例如排名 `[2,1]` 最终写成 `[1,2]` |
| 是否映射到 retrieval RoPE 4--11 | 是；两个完整 chunk 分别占 4--7 和 8--11 |

因此目前没有证据表明 `retrieval_no_compression` 存在“取错 tensor、超预算或 RoPE 重叠”
一类实现 bug。表现下降更像是检索目标与上下文分配策略的问题。

## 4. 发现的策略问题

### 4.1 Retrieval 替换了近期上下文，不是额外增加记忆

baseline 的注意力组成是：

```text
sink 4 + rolling local 16（tri-region 虚拟位置 0..19）
```

dyKV 的组成是：

```text
sink 0..3 + retrieval 8 + local 8
```

在 `current_frame=96`，可以近似理解为：

```text
baseline source : sink 0..3 + recent/current 84..99
baseline RoPE   : sink 0..3 + recent/current 4..19
dyKV source     : sink 0..3 + retrieved 4..11 + recent/current 92..99
dyKV RoPE       : sink 0..3 + retrieval 4..11 + recent/current 12..19
```

也就是说，no-compression case 用早期 `[4,12)` 替换了 baseline 中更连续的 `[84,92)`。
若早期内容不能弥补丢失的短期运动连续性，质量低于 baseline 是可能且合理的。

### 4.2 两个检索 Chunk 始终相邻，历史覆盖缺乏多样性

当前非扩容选择器只按 query FOV distance 排序，然后取够 8 个源 latent。连续 yaw 轨迹上，
相邻 chunk 的相机视角天然最接近，因此 80/80 次都选择了两个相邻块。

选择器没有以下机制：

- 两个已选 chunk 之间的视角或时间冗余惩罚；
- 与固定 sink 的重叠惩罚；
- 与当前 local/recent 的重叠惩罚；
- 对更长时间跨度或不同历史区域的覆盖奖励。

因此 8 帧 retrieval 容量往往用于两个高度相似的相邻块，而不是两个互补历史片段。

### 4.3 回到初始方向时，Retrieval 与 Sink 形成早期连续 12 帧

最后三步均选择 `[4,8]`，而 sink 已固定保存 `[0,4)`。最终注意力中的长期部分变成：

```text
sink [0,4) + retrieval [4,12)
```

这相当于重新放入视频最初连续 12 个 latent。它没有重复同一个 source frame，但在视角和
内容上高度相邻，可能过度放大初始状态，而不是提供互补的长期记忆。

### 4.4 FOV 同分时优先更老的 Chunk

当前排序键是：

```python
(distance, frame_start)
```

所以 distance 相同时，`frame_start` 更小的旧块优先。在最后一步：

```text
block 2 / start 8  / distance 0.2727743
block 22 / start 88 / distance 0.2727743
```

两者相机视角对称且得分完全相同，选择器取了 start 8，而不是 start 88。对于静态世界，旧块
可能有长期一致性价值；对于轮胎压碎物体等因果/动态 prompt，旧块包含的是过时物体状态，
可能与当前生成冲突。

### 4.5 FOV 检索只看相机，不看内容状态

相同轨迹的 prompt 0 和 prompt 2 得到完全相同的检索决策，说明当前检索与 prompt、物体
运动和生成内容无关。这符合代码设计，但会带来一个明显风险：

```text
相机方向相同 ≠ 世界状态仍然相同
```

对于静态环境，早期同视角 KV 可能有帮助；对于人、物体和 causal case，旧 KV 可能恢复
已经变化的姿态或物体状态，从而比只保留连续近期上下文的 baseline 更差。

## 5. 结论

当前日志中没有发现 retrieval tensor、token budget 或 RoPE rebase 的直接错误。更值得怀疑
的是以下组合效应：

1. dyKV 把 baseline 的 8 帧近期上下文换成 8 帧历史；
2. 两个历史 chunk 100% 时间相邻，缺乏检索多样性；
3. 返回初始方向时，检索内容紧邻 sink，长期预算存在冗余；
4. 同分时优先旧状态，动态场景容易引入时序冲突；
5. 几何检索不判断历史内容是否已经过时。

因此，“只检索不压缩不如 baseline”不意味着压缩一定是质量来源，也不表示 KV/RoPE 实现
必然出错；它首先说明**当前 FOV top-2 策略选出的历史未必比被替换掉的 8 帧 recent 更有
价值**。

## 6. 建议的下一步消融

在修改正式选择器之前，建议依次验证：

1. **同分新优先**：把 FOV 同分决胜从旧块改为新块，观察动态 case 是否改善；
2. **时间分散约束**：两个完整 chunk 之间至少间隔一个或多个 chunk；
3. **sink/local 边际覆盖**：惩罚与 sink 和 local 已覆盖视场高度重叠的候选；
4. **MMR 式多样性选择**：综合 query 相关性与候选间冗余，而不是独立 top-2；
5. **按 subset 报告**：静态 environment 与 human/object/causal 分开，验证陈旧内容冲突；
6. **同 local 预算对照**：增加不使用 retrieval、但同样只保留 local 8 的诊断 case，分离
   “减少 recent”与“加入历史”两个因素。

上述是诊断建议，本次没有擅自修改正式检索策略。任何一项实现后都应注册成独立 case，
保持 seed、轨迹、8-latent retrieval 物理预算和 `4+8+8` RoPE 布局一致。

当前已新增 `worldkv_pose_no_compression`，用于在上述相同预算下把 FOV overlap 替换为
WorldKV 平均位姿得分。它不会自动解决内容陈旧或候选多样性问题，但可以确认表现下降是否
主要来自 FOV 评分。仅回放 `j*49,l*49,n*1` 相机轨迹时，20 次 event 中有 19 次选择相同，
最后一步由 FOV 的 `[4,8]` 变为 WorldKV 的 `[4,88]`；这仍需实际视频验证。公平适配见
[`WORLDKV_RETRIEVAL_ABLATION.md`](WORLDKV_RETRIEVAL_ABLATION.md)。
