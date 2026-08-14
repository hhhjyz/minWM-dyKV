# 动态压缩后的 Retrieval Region 扩容方案

> 状态：P1--P4 已实现并通过单元测试，真实 checkpoint 冒烟（P5）待运行。
> `yaw_intrinsics` 保留原来的 E0 行为；`packed_chunks` 和
> `packed_chunks_latent` 分别实现 E1、E2，便于直接公平对照。

本文件描述的是“历史块相对当前 query”的第一版动态装箱。后续新增的“当前 query 只负责
检索、历史块相对前驱 chunk 压缩”、`3/4` 档精确装箱及 query coverage 回填已实现为独立
case，见 [`PREDECESSOR_INCREMENTAL_COMPRESSION.md`](PREDECESSOR_INCREMENTAL_COMPRESSION.md)。

## 实现入口

- `Wan21/pipeline/dykv_packing.py`：固定档位、32 原子背包、完整 chunk 优先、latent 尾部
  补齐、虚拟槽位分配和逐层物化；
- `Wan21/pipeline/dykv_runtime.py`：对全部 FOV 排序候选规划一次，并让所有层复用相同计划；
- `Wan21/wan/modules/dykv_rope.py`：按 `source_frame_ids + frame_token_lengths +
  virtual_slot_ids` 对每个 frame segment 独立执行 time-RoPE rebase；
- `Wan21/dykv_cases.py`：注册 E1/E2，旧 case 均不改变行为。

## 1. 目标

当前 dyKV 先按 8 个原始 latent 帧的预算选择两个四帧 chunk，再对这两个 chunk 做动态
空间裁剪。裁剪后实际 attention token 往往少于 retrieval region 的理论容量，但选择器
不会继续补入历史信息。因此，压缩节省的 token 只减少了计算量，没有转换成更长的历史
覆盖。

本方案希望在不改变现有连续 `sink 4 + retrieval 8 + local 8` RoPE 训练范围的前提下：

1. 将 retrieval region 固定解释为 **8 个完整 latent 的 token 容量**，即
   `B = 8 × F`，其中 `F=1560` 是一个 latent 的空间 token 数；
2. 动态压缩后的 chunk 按实际 token 成本装入该预算；
3. 完整 chunk 无法继续装入时，允许以单个 latent 为单位补齐尾部；
4. 不再用 chunk 的源帧数推断 RoPE 占位，而是显式记录每个源 latent 的虚拟槽位；
5. CPU KV bank 继续无损保存完整历史，只在 retrieval materialize 阶段裁剪。

推荐首先实现“固定压缩档位 + 整 chunk 优先 + latent 尾部补齐”，而不是直接实现任意长度
token 的连续背包。固定档位更容易控制 attention 长度、RoPE 槽位和消融变量。

## 2. 现有实现与 minWM-back 的启示

### 2.1 当前 minWM-dyKV

当前流程是：

```text
FOV 排序全部候选
  → 按原始帧数选择至 8 latent（通常两个四帧 chunk）
  → 对选中 chunk 做精确 yaw/FOV 列裁剪
  → 实例化 retrieval K/V
```

`materialize` 已支持不同 chunk 产生不同 `chunk_token_lengths`，但
`compose_tri_region` 仍要求 `sum(chunk_frame_counts) <= 8`。每个四帧 chunk 无论压缩到
多少 token，都会消耗四个 retrieval RoPE 位置，所以不能靠现有元数据放入第三个 chunk。

### 2.2 minWM-back

旧实现包含两个可复用思路：

- `allocate_dynamic_retrieval_keep_per_frame` 能在固定 token budget 下，为预先选中的完整
  chunk 分配不同压缩量；
- `latent_frame` granularity 能从无损 block 中单独提取一个历史 latent。

但旧实现仍有三个限制：

- 动态预算只在已经选中的 chunk 之间分配，不会因压缩后空闲而继续选择新 chunk；
- chunk retrieval 和 latent-frame retrieval 是互斥模式，没有“完整 chunk + 尾部 latent”
  的混合载荷；
- 三区域 RoPE 仍主要根据源 chunk 的帧跨度分配位置，未解决多个压缩源帧共享有限虚拟
  retrieval 槽位的问题。

因此本方案复用其“固定预算”和“latent frame 引用”思想，但重新设计选择器、载荷元数据
和 RoPE 映射。

## 3. 固定压缩档位

### 3.1 从相机几何得到原始保留比例

继续使用当前 `build_yaw_crop_plan` 的几何计算。对每个历史 latent，先求其与当前四帧
query FOV 并集对应的可见列数：

```text
rho_t = visible_columns_t / latent_width
```

四帧 chunk 的几何保留比例取保守值：

```text
rho_chunk = max(rho_0, rho_1, rho_2, rho_3)
```

使用最大值可以避免 chunk 内某一帧仍高度相关，却因为其他帧重叠较小而被过度压缩。四帧
chunk 内相机步长较小时，各帧比例通常接近。

### 3.2 第一版量化档位

第一版只使用三种非零压缩率：

| `rho_chunk` | 固定保留率 `q` | 每帧 token | 四帧 chunk token | 等价容量 |
| --- | ---: | ---: | ---: | ---: |
| `rho >= 0.75` | `1.00` | `1560` | `6240` | 4 latent |
| `0.375 <= rho < 0.75` | `0.50` | `780` | `3120` | 2 latent |
| `0 < rho < 0.375` | `0.25` | `390` | `1560` | 1 latent |
| `rho = 0` | 丢弃 | 0 | 0 | 0 |

Wan latent 空间网格为 `30×52`，宽度 52 可以被 2 和 4 整除，因此 50%/25% 分别对应
固定的 26/13 列，不需要 token 舍入。

阈值是第一版内部预设，不作为新的公开连续超参数。predecessor 系列后来增加了
`{1, 3/4, 1/2, 1/4}`，但它同时改变了压缩参考系，不能直接当作仅增加 `3/4` 档的单变量
E4 结论；严格 E4 仍需在同一 query-relative 参考系下比较。

对于相同内参、单个 query 位姿和纯 yaw，可用下式直观理解档位与角度的关系：

```text
rho ≈ max(0, 1 - |delta_yaw| / horizontal_FOV)
```

以当前推理入口由 `K(fx=0.5,cx=0.5)` 推导出的 `90°` 水平 FOV 为例：

| 绝对 yaw 差（近似） | 几何重叠 | 固定保留率 |
| --- | ---: | ---: |
| `0°--22.4°` | `>=0.75` | `1.00` |
| `22.4°--55.9°` | `0.375--0.75` | `0.50` |
| `55.9°--89.4°` | `0--0.375` | `0.25` |
| `>=89.4°` | 0 | 丢弃 |

真实实现仍以投影列 mask 为准，因为当前 query 包含四个连续位姿，其 FOV 并集会比单个位姿
更宽；上表只用于解释压缩率，不另写一套角度判断逻辑。

### 3.3 固定列数下如何选择位置

当前算法先得到有方向的 FOV 可见列。量化后，按以下顺序选出恰好 `q×52` 列：

1. 优先选择位于历史/当前 FOV 交集内的列；
2. 交集列过多时，按其到当前 query 光轴的角距离从小到大截取；
3. 交集列不足时，从交集边界向外扩张最近的列，形成小范围安全边界；
4. 左右 yaw 使用相反方向的列排序，保持镜像性；
5. 平移、pitch 或 roll 继续走固定新颖性回退，但回退结果也必须量化到同一档位；非法或
   缺失内参不会使用固定视场，而是在检索入口报错或跳过历史块。

这样每个 chunk 只有三种固定物理大小，同时仍保留当前算法的方向性空间位置。

## 4. Token Budget 下的历史装箱

### 4.1 预算单位

固定 retrieval token budget：

```text
B = memory_frames × frame_tokens = 8 × 1560 = 12480 tokens/layer
```

因此 attention 总上限保持为
`sink 4×1560 + retrieval 8×1560 + local 8×1560 = 31200 token/layer`，不会因覆盖更多
源历史而超过原来的 20-latent 物理 token 上限。

以 `390=1560/4` token 作为最小预算原子。retrieval 总预算是 32 个原子，每个 RoPE
槽位最多容纳 4 个原子。三个档位下，一个四帧 chunk 的成本分别为：

```text
q=1.00 → 6240 tokens → 16 个原子 → 4 个 RoPE 槽
q=0.50 → 3120 tokens →  8 个原子 → 2 个 RoPE 槽
q=0.25 → 1560 tokens →  4 个原子 → 1 个 RoPE 槽
```

因此，同样 12480 token 最多可表示：

- 2 个未压缩 chunk，即 8 个源 latent；
- 4 个半压缩 chunk，即 16 个源 latent；
- 8 个四分之一压缩 chunk，即 32 个源 latent；
- 或三种大小的混合组合。

例如：

```text
1 个 full chunk + 1 个 half chunk + 2 个 quarter chunk
= 16 + 8 + 4 + 4 个预算原子
= 32 个预算原子，使用 8 个 RoPE 槽，覆盖 16 个源 latent
```

### 4.2 选择流程

当前选择器需要从“先选两块再压缩”改为：

```text
1. 对全部 evicted candidate 做 FOV 排序
2. 为全部可用 candidate 预计算量化 crop plan 和 token cost
3. 在 32 个预算原子内选择完整 chunk
4. 若剩余容量放不下下一个完整 chunk，从候选中选择单个 latent 补齐
5. 按规划出的 virtual slot 顺序实例化最终载荷
```

完整 chunk 选择可使用只有 32 个离散预算原子的小型 0/1 动态规划：

```text
maximize sum(utility_i)
subject to sum(cost_i) <= 32
```

第一版 utility 建议使用：

```text
utility_i = FOV_similarity_i × sqrt(q_i)
```

`sqrt(q)` 会适度奖励信息更完整的 chunk，但不会像直接使用 `q` 那样完全抵消小 chunk
带来的历史覆盖收益。相同 utility 时，依次优先：FOV 距离更小、时间更新、block ID 更小，
保证确定性。

不建议直接按 `similarity / cost` 贪心，因为它会过度偏好非常小、但只与当前视角擦边的
quarter chunk。

### 4.3 Latent 尾部补齐

完整 chunk 选择完成后，若剩余 `R` 个预算原子不能装入下一个完整 chunk，则从未选中的
高分 chunk 中提取单个源 latent：

- 尾部 latent 使用自己的 `rho_t` 量化档位，不继承 parent chunk 的 `max(rho_t)` 档位；
- `q=1` 的单帧成本为 4 个原子；
- `q=0.5` 的单帧成本为 2 个原子；
- `q=0.25` 的单帧成本为 1 个原子；
- 优先选择该 chunk 中与 query FOV 距离最小的 latent；
- 同一源 latent 不能与已选完整 chunk 重复；
- 单帧引用直接切片无损 bank，再应用对应 crop plan，不在 bank 中建立重复副本。

第一版只允许以 390-token 原子补齐，不做任意 token 数的强行填充。完整 chunk 一定占用
整数个 RoPE 槽；尾部 frame segment 再用 `4→2→1` 原子大小的 first-fit decreasing 装入
剩余槽，每槽容量为 4 个原子，segment 不跨槽切分。候选组合不足时允许安全欠填，不能
为了填满而复制 token 或选择零 FOV 重叠的历史。

## 5. 可变 Retrieval 的 RoPE Rebase

### 5.1 为什么现有设计不够

当前 payload 只有：

```text
src_frame_ids
chunk_frame_counts
chunk_token_lengths
```

RoPE 由 `chunk_frame_counts` 推断每个 chunk 占几个虚拟时间位置。因此 quarter chunk 虽然
只有 1560 token，仍占四个位置；两个 chunk 后位置 4--11 已用完。

新的设计必须明确分离：

- **物理 token 长度**：影响 attention 计算和显存；
- **源 latent 时间**：用于移除已经编码在 K 上的原始时间 RoPE；
- **目标虚拟槽位**：用于将该 token 放入固定的 4--11 retrieval 范围。

### 5.2 帧级 segment 元数据

materialize 后每层使用同一份帧级元数据：

```text
source_frame_ids:    [4,   5,   6,   7,   20,  21,  22,  23,  ...]
frame_token_lengths: [780, 780, 780, 780, 390, 390, 390, 390, ...]
virtual_slot_ids:    [4,   4,   5,   5,   6,   6,   6,   6,   ...]
parent_block_ids:    [1,   1,   1,   1,   5,   5,   5,   5,   ...]
```

`chunk_token_lengths` 仍可保留用于诊断，但 RoPE 不再从 chunk 长度推断时间跨度。

### 5.3 Slot folding

retrieval 仍只有虚拟位置 4--11。压缩后允许多个源 latent 共享一个虚拟时间槽：

| 压缩率 | 每帧 token | 一个槽容纳的源 latent 数 | 四帧 chunk 使用槽数 |
| --- | ---: | ---: | ---: |
| 1.00 | 1560 | 1 | 4 |
| 0.50 | 780 | 2 | 2 |
| 0.25 | 390 | 4 | 1 |

同槽内不同源帧保留各自的空间 RoPE，只把时序 RoPE 映射到同一目标位置。对每个 frame
segment 单独执行：

```text
K_rebased = shift_roped_time(
    K_segment,
    target_virtual_slot - source_frame_id
)
```

不能对整个四帧 chunk 只乘一次统一 delta，因为折叠后四个源帧可能映射到一个或两个目标
位置，源帧之间不再保持原来的时间偏移。

### 槽位分配规则

1. 先将选中的完整 chunk 按源时间排序，并各自占用整数个槽；
2. 每个 full chunk 连续使用四个槽；half chunk 每两帧共享一槽；quarter chunk 四帧共享
   一槽；
3. 尾部 latent 按 segment 大小降序、FOV 分数降序、源时间升序装入剩余槽；segment
   不跨槽，每个槽的 token 总数不得超过 1560；
4. `virtual_slot_ids` 必须非递减并落在 `[4, 11]`；
5. 最终物理 K/V 按虚拟槽排序，同一个槽内按源时间排序，保证输出确定；
6. sink 固定为 0--3，recent 固定为 12--15，current/query 固定为 16--19，完全不改动。

这种设计仍保持连续 `4+8+8` 的训练位置范围，只是在 retrieval 的一个时间位置中放入多个
空间裁剪后的历史视图。

这里的 4--11 应理解为 **有序 retrieval memory slot**，而不是对真实历史时间间隔的精确
重建。完整 chunk 仍保持内部源时间顺序；尾部 latent 放在完整 chunk 之后并按上述确定规则
装箱。源帧的真实 ID 始终保留在元数据中用于 time-RoPE 去编码，但目标位置表达的是压缩后
的 memory rank。这个语义变化必须通过 E3/E6 单独验证，不能默认认为共享时间槽无损。

### 5.4 必须验证的 RoPE 不变量

- `sum(frame_token_lengths) <= 8×1560`，且必须是 390 的整数倍；
- 每个虚拟 retrieval 槽的 token 总数 `<=1560`；
- 每个 token 恰好属于一个源 frame segment；
- `virtual_slot_ids` 只在 4--11 内，且与 local 12--19 不重叠；
- K 的时间通道按 segment 重映射，空间高/宽通道逐 bit 保持不变；
- materialize 与 rebase 不修改 CPU bank 或在线 cache；
- 所有层使用完全相同的 segment 边界、token 索引和 slot 映射；
- 同一 retrieval payload 在多次去噪调用中不会累积 RoPE 旋转。

## 6. 建议的数据结构

引入两层只读计划对象：

```text
PackedFramePlan
  block_index
  source_frame_id
  frame_offset
  token_indices
  token_count
  keep_tier
  virtual_slot_id

PackedRetrievalPlan
  frames[]
  selected_full_blocks[]
  selected_tail_frames[]
  token_budget
  used_tokens
  used_virtual_slots
```

运行时只生成一次 `PackedRetrievalPlan`，每个 transformer layer 根据同一计划 gather 自己的
K/V。这样可以避免每层重复几何计算，也保证跨层索引一致。

原有 payload 增加：

```text
source_frame_ids
frame_token_lengths
virtual_slot_ids
parent_block_ids
selection_kinds        # full_chunk / tail_latent
keep_tiers
```

`chunk_frame_counts` 不能再作为 RoPE 占位依据，只保留向后兼容诊断或在旧 case 中使用。

## 7. 分阶段实施计划

### 阶段 P1：量化 crop plan（已完成）

- 在当前精确 yaw crop 之上增加 `{1, 1/2, 1/4, 0}` 档位；
- 保证每个四帧 chunk 的四个 latent 使用相同档位；
- 增加左右镜像、0°、半 FOV、边缘重叠和零重叠测试；
- 记录 `raw_overlap_ratio`、`quantized_keep_tier`、实际列数和 token cost。

### 阶段 P2：固定预算完整 chunk 装箱（已完成）

- FOV 选择器返回完整排序，不再提前截断到 8 个原始帧；
- 为全部候选预计算 cost/utility；
- 用 32 原子离散 DP 选择完整 chunk；
- 验证 total tokens 不超过 12480，并覆盖 2/4/8 chunk 极端情况。

### 阶段 P3：frame-level RoPE slot folding（已完成）

- payload 改为帧级 segment 元数据；
- `compose_tri_region` 按 segment 单独做 time shift；
- 增加半压缩两帧共享槽、quarter 四帧共享槽、混合档位和多次调用不累积的测试；
- 保持 sink/local/query 位置完全不变。

### 阶段 P4：latent 尾部补齐（已完成）

- 增加 bank frame reference，不复制 bank 数据；
- 完整 chunk 无法装入时，按帧级 FOV 分数补齐剩余槽；
- 验证不重复选择、源时间顺序、跨 block frame 引用和不足候选时的安全欠填充。

### 阶段 P5：真实 checkpoint 冒烟与典型样本对比（待运行）

- 先跑 24 latent 的左右往返轨迹，确认第三个以上历史 chunk 被实例化；
- 再使用 `mbench_typical_4.jsonl` 比较当前方法与扩容方法；
- 检查 attention token 恒定上限、GPU 峰值、检索耗时和返回视角恢复。

## 8. 必要消融

| Case | 选择与压缩 | 目的 |
| --- | --- | --- |
| E0 | 当前：先选 8 源 latent，再精确裁剪，允许欠填 | 现有动态压缩基线 |
| E1 | 固定档位，只装完整 chunk | 验证压缩节省量是否能转换为历史覆盖 |
| E2 | 固定档位，完整 chunk + latent 尾部补齐 | 验证尾部补齐收益 |
| E3 | E2，但禁用 slot sharing，只允许 8 个源 latent | 分离“更多历史”与“固定档位”影响 |
| E4 | `{1,1/2,1/4}` 对比 `{1,3/4,1/2,1/4}` | 压缩档位数量消融 |
| E5 | utility 用 FOV similarity / similarity×sqrt(q) / similarity÷cost | 选择目标消融 |
| E6 | 每槽最多 1/2/4 个源 latent | 检查时间槽碰撞对质量的影响 |

当前可直接运行 E0、E1、E2：

```bash
conda activate minwm-fa
CASES=yaw_intrinsics,packed_chunks,packed_chunks_latent \
OUTPUT_ROOT=output/dykv_packing_e0_e2 \
bash Wan21/scripts/inference/run_dykv_cases.sh
```

E3、E5、E6 仍是后续机制消融。`3/4` 分箱代码已在 predecessor 路径实现并测试，但由于
参考系同时变化，严格的 query-relative E4 实验仍不得标记为完成。

所有实验固定总 retrieval token 上限为 12480，保持 checkpoint、MBench case、seed、轨迹和
`4+8+8` RoPE 边界一致。报告时必须同时给出：源历史 latent 数、完整 chunk 数、尾部 latent
数、各档位分布、每槽 segment 数、实际 token、生成耗时和质量。

## 9. 风险与决策

### 主要风险

- 多个源 latent 共享一个时间位置，可能造成动态内容的时间歧义；
- quarter chunk 虽覆盖更长历史，但每帧只有 25% 空间信息，过多低质量历史可能稀释注意力；
- 按 token 容量扩大历史后，FOV selector 的微小误差会影响更多 chunk；
- 不同 query 的实际候选数不同，无法保证每次都恰好填满 12480 token；
- 当前 PRoPE 分支仍只使用 local cache。若未来让 PRoPE 也检索，不能直接复用时序 slot
  folding，必须单独设计相机重编码。

### 推荐决策

第一版采用固定 `{1, 1/2, 1/4}` 档位、8×1560 token 上限、每槽最多 1560 token、chunk
内部源时间有序的 slot folding，并实现 latent 尾部补齐。该设计同时满足：

- 压缩后能够覆盖更多历史；
- retrieval region 物理 token 不超过原始 8 latent；
- RoPE 仍严格落在 4--11；
- chunk 大小和槽位映射只有少数规则，便于理解和测试；
- 不需要修改无损 CPU bank，也不增加公开连续超参数。

当前另有 predecessor P0--P2 路径实现四档装箱与 coverage 回填，见对应文档。E0--E3 和
P0--P2 仍需在相同典型 MBench 样本上验证 slot sharing 的质量影响；单元测试通过不能替代
视频实验结论。

## 10. 暂不采用的替代方案

- **扩大 retrieval RoPE 范围到 12 以后**：会与 local/query 冲突，或超出 20 帧训练
  范围，不采用；
- **压缩 chunk 仍占四个 RoPE 位置**：实现简单，但最多仍只能选择两个 chunk，无法实现
  本模块目标；
- **整个压缩 chunk 只使用一个统一 delta**：half/quarter chunk 内的帧需要折叠到不同或
  相同槽位，统一 delta 无法正确表达，必须按 frame segment rebase；
- **保留任意精确比例并使用小数 RoPE 位置**：需要扩展频率生成和插值，且 attention 长度
  种类过多。第一版先用 `{1, 1/2, 1/4}` 的整数折叠关系；
- **只按 similarity/cost 贪心**：容易让大量低重叠 quarter chunk 淹没少数高质量记忆，
  第一版使用离散 DP 并在消融中比较。
