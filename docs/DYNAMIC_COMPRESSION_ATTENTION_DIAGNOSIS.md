# 动态压缩注意力分布诊断

本文记录 `retrieval_no_compression` 与 `motion_novelty_backfill` 的注意力分布对比，目标是区分
“retrieval region 没有被模型使用”“压缩后保留内容不合适”和“token 数量或布局改变造成的
attention 重分配”这三类问题。

## 1. 结论摘要

现有 40-latent loop-closure capture 表明，动态压缩的主要问题不是 retrieval attention 全局消失，
而是它在 layer、生成时刻和 retrieval 物理 token 窗口之间变得不稳定。部分中后层对 retrieval 的
attention 降低，同时 sink attention 增加；部分浅层又会出现 retrieval attention 上升。因此，
只观察整个 retrieval region 的 attention mass 不足以判断压缩是否有效。

目前最值得优先验证的问题为：

1. motion novelty 保留“与 anchor 差异最大”的 token，但 loop closure 更需要“能与当前 query
   重新匹配”的共享结构，novelty 和 retrieval usefulness 并不等价；
2. novelty cosine 使用缓存中的 RoPE 后 key，相似度同时包含内容差异和时间位置相位差；
3. motion-based ratio 决定每个 source frame 保留多少 token，但没有约束各 source frame 在压缩后
   是否仍具有稳定、均衡的 query 可访问性；
4. backfill 可以填满 `8F` 预算，但补回顺序仍由 novelty 排序主导，只解决 token 数量，不保证
   补回的信息与当前 query 相关；
5. 当前 attention capture 以 `F=1560` token 为单位切分拼接后的 retrieval token stream。对于
   `motion_novelty_backfill`，图中的 8 个 retrieval “frame”既不是 source frame，也不一定是
   virtual slot，而是连续的 8 个物理 `F` 窗口；窗口边界可以切开 source-frame segment 和
   virtual-slot 边界。

## 2. 已有 loop-closure attention 证据

已有输出位于：

```text
output/attn_capture/
├── retrieval_no_compression_honest/
└── motion_novelty_backfill_honest/
```

两组实验使用相同 prompt、`k*19,i*19,k*1` 轨迹、seed 0 和 honest retrieval RoPE。KV 布局为：

```text
sink(4F) | retrieval(8F) | local(8F)
```

### 2.1 Retrieval 总 attention 并未持续消失

以 layer 15 为例：

| Current frame | No compression | Motion backfill | 差值 |
| ---: | ---: | ---: | ---: |
| 24 | 16.68% | 16.50% | -0.18 pp |
| 28 | 19.69% | 22.03% | +2.34 pp |
| 32 | 22.57% | 20.61% | -1.96 pp |
| 36 | 20.92% | 18.57% | -2.35 pp |
| 40 | 19.03% | 16.60% | -2.43 pp |

动态压缩有时增加、有时降低 retrieval attention，符号会随生成时刻变化。layer 0 在 current
frame 40 甚至从 19.32% 增加到 23.82%，而 layer 5/10/15/20/25/29 均降低。因此不能将质量
下降简单归因于 retrieval region 被模型整体忽略。

### 2.2 Retrieval 物理窗口 attention 变得不均匀

在 layer 15、current frame 36，8 个连续 retrieval `F` 窗口的 attention 为：

```text
no compression:
0.0265, 0.0239, 0.0278, 0.0316, 0.0252, 0.0193, 0.0251, 0.0299

motion backfill:
0.0284, 0.0379, 0.0335, 0.0255, 0.0224, 0.0123, 0.0126, 0.0131
```

无压缩 case 的 8 个窗口也对应 8 个完整 source frame；motion case 的窗口则可能包含一个完整
anchor、多个压缩 segment，或跨 source-frame/virtual-slot 边界的拼接。后 3 个窗口 attention
较低只能证明压缩布局后的 token 区间利用率不均衡，尚不能直接证明某三个 source frame 被忽略。

### 2.3 Attention 会向 sink 回退

layer 15、current frame 36 的区域分配为：

| Region | No compression | Motion backfill | 差值 |
| --- | ---: | ---: | ---: |
| Sink | 12.73% | 15.14% | +2.41 pp |
| Retrieval | 20.92% | 18.57% | -2.35 pp |
| Local | 66.34% | 66.30% | -0.04 pp |

retrieval 减少的 mass 主要流向 sink，而非 local。这与“压缩后的 retrieval key 较难匹配，模型
退回稳定 sink 上下文”的解释一致，但仅凭 attention mass 还不能证明因果关系。

## 3. 统计口径的限制

当前 capture 的 `frame_attention[h,qf,kf]` 会先对一个 `F` 长度 token 区间求 attention mass，
再对 query token 平均。它没有记录每个物理 `F` 窗口内部的 source-frame/virtual-slot segment
边界，也没有输出每个保留 token 的统计。因此存在以下限制：

- 无法区分“保留 token 少但每个 token attention 高”和“保留 token 本身无用”；
- 无法把跨 slot/跨 source-frame 拼接后的 attention 精确归因到 source frame；
- 对 head 和 query frame 求平均后，少数关键 head 的变化可能被掩盖；
- 本次 DMD capture 的所有记录均为 `call_count=0`，即每个 generation block/layer 的单次
  snapshot，不是多个 diffusion step 的平均；
- 已有 capture 只有一个 loop-closure 样本，不能代表 MBench 的场景和运动分布；
- no-compression 检索 8 个完整 source frame，motion case 可从更多 source frame 压缩到 `8F`，
  二者不是“候选历史完全相同、只改变压缩”的严格对照。

后续 capture 应增加 token-normalized attention、attention entropy、head variance，以及
`source_frame_id / virtual_slot_id / segment_token_range` 映射。

## 4. MBench 同视频配对实验

本次选择当前 `mbench_typical_8.jsonl` 的第一条样本：

| 字段 | 值 |
| --- | --- |
| Subset | `causal` |
| Sample ID | `a00533_00678` |
| Condition | `right_then_left_25s` |
| 内容 | 木工车床、旋转木材、手套和飞散木屑 |
| Trajectory | `l*49,j*49,n*1` |
| Latent frames | 100 |
| Seed | 0 |

对比 case 为 `retrieval_no_compression` 和 `motion_novelty_backfill`，两者均使用默认
`fixed_slot` retrieval RoPE。两者必须使用相同 prompt、trajectory、sample seed 和
initial-noise fingerprint。输出目录为：

```text
output/attn_capture_mbench_typical1_seed0/
├── retrieval_no_compression/
└── motion_novelty_backfill/
```

需要分析的观测点包括：

1. outward yaw、中点最大偏航、return yaw 和返回初始视角附近的 region attention；
2. 各 layer 的 retrieval mass 差值及其符号是否一致；
3. 8 个连续 retrieval `F` 窗口的分布、熵和最大/最小比，并结合日志重建其 source/slot 构成；
4. retrieval mass 减少后流向 sink 还是 local；
5. motion case 的实际 source frame、保留 token 数和 virtual-slot load，判断 attention 异常是否
   与极低 keep ratio、跨 source-frame 拼接或 backfill 集中位置对应。

## 5. MBench 结果

### 5.1 产物与配对一致性

两组实验均已完成，分别生成 397 个解码视频帧（100 latent）和 140 条 attention record。两边的
`sample_seed=0`，initial-noise fingerprint 均为：

```text
bf81d251ab6add150e8f41712aaf1d336d57fddee961bc56fc002a21fe5e8faf
```

因此这是相同 prompt、轨迹和初始噪声下的配对比较。主要产物为：

```text
output/attn_capture_mbench_typical1_seed0/
├── retrieval_no_compression/
│   ├── *.mp4
│   ├── attention_capture_00000.json
│   └── dykv_summaries.jsonl
├── motion_novelty_backfill/
│   ├── *.mp4
│   ├── attention_capture_00000.json
│   └── dykv_summaries.jsonl
├── attention_comparison_metrics.json
├── region_attention_delta_heatmap.png
├── retrieval_attention_over_time.png
├── retrieval_window_attention_heatmap.png
├── video_contact_sheet.png
└── video_side_by_side.mp4
```

### 5.2 Retrieval attention 存在明确的阶段性符号翻转

下表对 7 个 capture layer 和对应时间点共同求平均。`pp` 表示百分点，current end frame 52
覆盖轨迹从向右旋转切换为向左返回的转向 block。

| 阶段 | No compression retrieval | Motion backfill retrieval | Retrieval 差值 | Sink 差值 | Local 差值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Outward，24–48 | 13.47% | 11.93% | **-1.54 pp** | +0.05 pp | +1.49 pp |
| Turnaround，52 | 12.64% | 10.96% | **-1.68 pp** | -0.15 pp | +1.83 pp |
| Return，56–100 | 8.72% | 9.53% | **+0.81 pp** | -0.11 pp | -0.71 pp |
| Final，96–100 | 7.97% | 10.24% | **+2.27 pp** | -0.18 pp | -2.09 pp |

在 outward 和 turnaround 阶段，motion case 丢失的 retrieval attention 基本转移到 local，而不是
sink；在返回末段则反过来从 local 转移到 retrieval。140 个 layer/time 配对点中 48.6% 的
retrieval 差值为负，所有点的平均差值只有 -0.13 pp。这个接近零的总平均会掩盖非常明显的
阶段性变化。

中层最敏感：

| Current end frame | Layer 10 | Layer 15 | Layer 20 | Layer 25 |
| ---: | ---: | ---: | ---: | ---: |
| 48，outward 末段 | -3.08 pp | **-4.80 pp** | -1.76 pp | -3.30 pp |
| 52，turnaround | -3.45 pp | **-4.13 pp** | -1.35 pp | -2.70 pp |
| 100，返回末段 | +3.29 pp | **+7.44 pp** | +1.78 pp | +3.75 pp |

因此 MBench 结果进一步确认：动态压缩不是简单地让 retrieval attention 全程降低，而是改变了
不同阶段可被 query 匹配的历史构成。

### 5.3 返回末段 attention 上升，不代表长期记忆更好

在 retrieval event 96，也就是 attention current end frame 100：

```text
retrieval_no_compression selected frame starts:
[4, 8]

motion_novelty_backfill selected frame starts:
[4, 8, 88, 12, 84, 48]

motion materialized source order:
[4, 8, 12, 48, 84, 88]  # 每项是四帧 chunk start
```

no-compression 的 `8F` 全部来自早期完整帧 4–11；motion case 则同时包含早期帧、转向附近的
frame 48 chunk，以及刚刚离开 local region 的 frame 84–91。layer 15、current end frame 100
中，motion case 的 W5–W7 attention 为：

```text
0.0245 + 0.0397 + 0.0524 = 0.1166
```

占 motion retrieval attention 0.1772 的 65.8%。这些物理窗口主要由 source frame 48、84–91
组成。相比之下，no-compression 的 W5–W7 只占其 retrieval attention 的 39.5%。

所以返回末段的 attention 上升主要说明 motion case 把更多“近期刚 evict 的历史”放进 retrieval
region，并不证明压缩后的早期 loop-closure memory 更有效。这也是当前对比不是严格
compression-only ablation 的具体表现：两种 case 的 source history 集合不同。

### 5.4 中层 retrieval attention 在压缩后更集中

将每个时刻 retrieval 内 8 个物理 `F` 窗口归一化后，计算 normalized entropy 和 coefficient
of variation（CV）：

| Layer | No-comp entropy | Motion entropy | No-comp CV | Motion CV |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.974 | **0.947** | 0.323 | **0.486** |
| 15 | 0.985 | **0.952** | 0.240 | **0.455** |
| 20 | 0.983 | **0.961** | 0.253 | **0.408** |
| 25 | 0.979 | **0.954** | 0.294 | **0.448** |

motion case 的 entropy 更低、CV 更高，说明中层 attention 更集中在少数 token 区间。到 current
end frame 100，layer 15 的 entropy 从 no-compression 的 0.994 降到 0.907，CV 从 0.161
升到 0.664。current end frame 96 时 motion CV 甚至达到 0.759。

这支持“压缩后只有少数拼接区间容易匹配”的判断。但由于物理窗口混合多个 segment，它仍不能
确定究竟是哪一个 source frame 或哪一类 token 获得 attention。

### 5.5 Flat source-order packing 使当前 attention 横轴难以解释

`motion_novelty_backfill` 使用 `flat_source_ordered`。KV 按 source frame 顺序连续拼接，但每个
segment 的 RoPE 又由 `virtual_slot_id` 决定。物理 `F` 窗口和 virtual slot 没有一一对应关系。

以 attention current end frame 52 的部分窗口为例：

```text
W4: source frame 35 / virtual slot 7:   39 tokens
    source frame 36 / virtual slot 8: 1521 tokens

W5: source frame 36 / virtual slot 8:   39 tokens
    source frame 37 / virtual slot 9:  104 tokens
    source frame 38 / virtual slot 9:  179 tokens
    source frame 39 / virtual slot 9:  250 tokens
    source frame 40 / virtual slot 9:  988 tokens
```

同一个 source-frame anchor 会被物理窗口切开，同一个物理窗口又可能混合多个 source frame 和
多个 RoPE slot。日志中的 virtual-slot load 最高达到 2093 token，大于一个 `F=1560`，这是
flat layout 允许的行为，但会产生两个潜在问题：

1. 多个 source frame 被压到相同 temporal RoPE position，可能形成 temporal phase collision；
2. 现有按 `F` 统计的 attention 图无法区分这种 RoPE collision 与 token 内容选择的影响。

因此需要新增 segment-aware capture，直接按
`source_frame_id / token range / virtual_slot_id / selection_kind` 聚合 attention。

### 5.6 纯 yaw 下“动态比例”几乎退化为固定的 chunk 内模板

该 MBench 轨迹以恒定 3°/latent 的速度旋转。每个正常四帧 chunk 的 keep ratio 均为：

```text
anchor: 100%
latent 1: 6.67%
latent 2: 11.41%
latent 3: 16.03%
```

这个比例会在每个 chunk 重复。因此对于恒速 `j/l` yaw，当前 motion ratio 并不会随视频时刻或
场景内容自适应，只会形成固定的 chunk 内压缩模板。此前关于 WASD/IJKL 恒速控制会让多个
latent 压缩率相同或周期性重复的担忧，在这次真实日志中得到了确认。

当选中 5 个 chunk 时：

```text
base tokens     = 10465
backfill tokens = 2015
full anchors    = 5 * 1560 = 7800，占最终 8F 的 62.5%
```

当选中 6 个 chunk 时：

```text
base tokens     = 12233
backfill tokens = 247
full anchors    = 6 * 1560 = 9360，占最终 8F 的 75.0%
```

候选历史增多后，retrieval budget 主要被完整 anchor 占据，所有非-anchor latent 只能分享剩余
25%。此时 backfill 只增加 247 个 token，已经不是决定最终内容的主要机制；“选择多少个完整
anchor”比连续 keep ratio 更强地控制实际预算。

### 5.7 Camera novelty 会把动态场景误判为冗余

最明确的问题出现在轨迹转向处。source chunk 48 包含 frame 48–51；frame 50 返回到与 anchor
frame 48 相同的相机位姿，因此 projected overlap 为 1，keep ratio 为 0，最终 frame 50 没有任何
segment，甚至不会出现在 `source_frame_ids` 中：

```text
source frame:    48, 49, 50, 51
keep ratio:      1.0, 0.0667, 0.0, 0.0667
base tokens:     1560, 104, 0, 104
```

但该样本包含高速旋转的木材、移动的手和持续飞散的木屑。相机回到同一位姿不代表世界状态和
latent 内容相同。当前算法只根据 camera geometry 决定 token 数量，会把：

```text
camera-view novelty = 0
```

错误地等价为：

```text
visual / temporal novelty = 0
```

这对 MBench causal、人类动作和物体运动场景都是系统性风险，不只是该样本的偶发现象。非-anchor
frame 至少需要一个内容/时间变化下限，不能仅凭相机 overlap 被压到 0。

### 5.8 Backfill 分配仍然高度不均衡

`_reference_lengths` 按 retrieval ranking 顺序逐 chunk 消耗剩余预算，并在一个 chunk 的三个
non-anchor frame 间按容量分配。它不是在所有已选 frame 上做全局 query-aware 分配。

因此：

- 5-chunk 阶段的 2015 个 backfill token 主要集中在最高排名 chunk；
- 6-chunk 阶段只有 247 个 backfill token，通常分成约 87/82/78；
- 其他大多数 non-anchor frame 仍只保留约 104/179/250 个 base token。

填满 `8F` 解决了 underfill，但没有解决 source-frame 平衡和 query relevance。attention 在返回末段
集中到近期 frame 的压缩窗口，也说明“token 更多”与“query 更愿意访问”并不是同一件事。

### 5.9 运行开销明显增加

同一视频的 retrieval event 统计为：

| Case | 平均 event 时间 | 20 个 event 总时间 |
| --- | ---: | ---: |
| No compression | 1.14 s | 22.73 s |
| Motion backfill | 9.38 s | 187.50 s |

motion planning/materialization 的 event 开销约为 8.25 倍；包含模型加载、attention capture 和
解码后的整条生成时间约为 161 s 对 428 s。当前动态压缩不仅有质量风险，也尚未带来推理时间
收益。

### 5.10 视频差异只能说明轨迹分叉，不能说明优劣

两组视频在 retrieval 生效后快速分叉。相互比较的 PSNR/SSIM 在 outward post-retrieval 阶段约为
21.13/0.700，在 turnaround 约为 14.33/0.507，在 return 约为 13.75/0.512。该数值仅衡量两种
生成路径的差异，不是对 ground truth 的质量指标，不能据此判断哪个 case 更好。定性 contact
sheet 同样只能用于定位分叉时刻。

## 6. 当前问题优先级

| 优先级 | 问题 | 本次证据 |
| --- | --- | --- |
| P0 | Camera novelty 不等于 visual/temporal novelty | 转向处 frame 50 被压到 0 token，但场景持续运动 |
| P0 | 两个 case 检索的 source history 不同 | 返回末段 motion 的高 attention 主要来自 frame 84–91 |
| P0 | Novelty 使用 RoPE 后 layer-0 key | 内容相似度混入 temporal phase，代码路径已确认 |
| P1 | 完整 anchor 占据 62.5%–75% 的 `8F` 预算 | 5/6-chunk 实际 token 日志 |
| P1 | 中层 attention 集中在少数 retrieval 窗口 | layer 10–25 entropy 降低、CV 上升 |
| P1 | Flat packing 的物理窗口、source frame、virtual slot 不对齐 | source/slot/token range 重建结果 |
| P1 | Backfill 只填数量，分配不具备全局 query awareness | 2015 或 247 token 集中到最高排名 chunk |
| P2 | motion planning 开销过高 | event 时间为 no-compression 的 8.25 倍 |

## 7. 建议的下一步消融

1. **严格 compression-only 对照**：固定相同 selected block/source frame、相同 retrieval order 和
   相同 RoPE mode，只改变 full KV 与 compressed KV；
2. **内容变化下限**：将 camera keep ratio 与 pre-RoPE content change 联合，并为每个 non-anchor
   设置非零最低预算；
3. **去除 RoPE 污染**：使用 pre-RoPE key 或显式消除 temporal phase 后再计算 novelty；
4. **Anchor budget 消融**：限制完整 anchor 总预算，避免 chunk 数越多、anchor 占比越高；
5. **全局 backfill**：在所有已选 non-anchor frame 上根据 query relevance、content novelty 和
   source-frame coverage 联合分配剩余 token；
6. **Segment-aware attention capture**：直接输出 source-frame segment 的 total attention、
   per-token attention、entropy、head variance 和 selection kind；
7. **RoPE 布局对照**：在同一 MBench 样本上配对比较 `fixed_slot` 与 `honest`，隔离 temporal
   phase collision 和 token selection 的影响。

## 8. 第一阶段优化实现（2026-08-20）

本轮把最优先的三个问题拆成可逐项归因的 case。三者共享 FOV retrieval、原始 KV/source-frame
顺序、flat 跨 slot 拼接、完整 anchor，以及固定的有效预算：最多选排名最靠前的 4 个合法
chunk；每个 chunk 保留 `2F`，因此历史充足时严格为 `4 × 2F = 8F`。其中 4 个 anchor 固定
占 `4F`，另外 `4F` 在 12 个 non-anchor latent 间连续按比例分配，不做档位量化。

| case | chunk / 总预算 | non-anchor 数量分配 | token novelty 排序 | 单一新增变量 |
| --- | --- | --- | --- | --- |
| `motion_alloc_cam_4chunk` | 最多 4 / 每 chunk `2F` | projected camera score | cached RoPE K | 固定 4-chunk 与公平 `8F` |
| `motion_alloc_cam_content_4chunk` | 同上 | `max(camera, content)` | cached RoPE K | 修复静止相机动态内容误压缩 |
| `motion_alloc_cam_content_prerope_4chunk` | 同上 | 同上 | layer-0 pre-RoPE K | 去掉 temporal RoPE 对排序的污染 |

### 8.1 固定预算分配

相机分数仍使用已经验证的双向二维多深度投影出界比例。12 个 non-anchor 的整数 token 数通过
带单帧 `F` 上限的确定性 proportional water-filling 得到；最大余数规则处理整数舍入，保证总和
严格等于 `4F`。若所有分数均为 0，则在所有 non-anchor 间均分，而不是把动态预算丢掉。历史
不足 4 chunk 时，每个有效 chunk 仍使用 `2F`，不会为了填满物理 `8F` 重复或虚构历史 token。

### 8.2 动态内容补偿

内容分数使用 layer-0、对应空间 token 的 anchor-relative V cosine distance：

```text
q_content(i) = mean_tokens clamp((1 - cosine(V_i, V_anchor)) / 2, 0, 1)
q_alloc(i)   = max(q_camera(i), q_content(i))
```

V 不经过 RoPE，因此这一修改可与下一步的 K novelty 去污染独立比较。这里的内容分数决定“分给
该 latent 多少 token”，并不直接生成空间裁剪 mask；实际保留位置仍由 novelty 排序决定。

### 8.3 pre-RoPE novelty

最终 case 只在 transformer layer 0 的 live cache 中增加 `content_k`，在
`causal_rope_apply` 之前写入归一化 K，并与 K/V 使用相同 rolling slice。归档时仅额外复制这一层
descriptor 到 CPU bank；30 层实际 attention K/V 以及 materialization 路径不变。这样排序使用
无 temporal phase 的 layer-0 K，而模型 attention 仍使用正常 RoPE K。

新增事件诊断字段为 `motion_camera_scores`、`motion_content_scores`、
`motion_allocation_scores`；summary/manifest 记录 `motion_allocation_mode` 与
`novelty_feature_mode`。单元测试分别覆盖：固定选择 4 chunk 且精确 `8F`、静止相机动态帧获得
更多 token、改变 cached RoPE K 不再改变 pre-RoPE novelty order。

### 8.4 20s × 30 的最小判别实验

主实验应使用相同 prompt、trajectory、seed 和模型权重，依次运行：

1. `retr16_compression_r033`：固定 4 chunk / 约 `8F` 的旧 WorldKV 基线；
2. `motion_alloc_cam_4chunk`：判断公平预算下 camera allocation 本身是否有效；
3. `motion_alloc_cam_content_4chunk`：只判断内容动态补偿；
4. `motion_alloc_cam_content_prerope_4chunk`：只判断 pre-RoPE novelty。

报告 30 个样本的 paired delta，而不只报告各 case 均值：pose/closure PSNR、SSIM、LPIPS 的
mean、median、bootstrap 95% CI、胜率，并按静态场景运动、动态物体、纯旋转、平移/混合运动分组。
先核对每个 retrieval event 的 selected source blocks 和总 token 数；如果输入状态不一致，PSNR
差异不能归因到当前单一变量。

该实验已经完成。完整均值、paired bootstrap CI、运动分组和后续设计见
[`MOTION_ALLOC_20S_ABLATION_RESULTS.md`](MOTION_ALLOC_20S_ABLATION_RESULTS.md)。结果不支持将
camera+content 或 completely pre-RoPE K 直接设为默认；尤其 completely pre-RoPE 同时移除了
H/W spatial RoPE，在 rotation-only 与 mixed-motion 上显著降低 pose PSNR。
