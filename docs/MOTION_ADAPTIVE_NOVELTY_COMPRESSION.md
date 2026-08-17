# 连续 FOV 比例驱动的 WorldKV Novelty 压缩

> 状态：设计完成，尚未实现、尚未注册为可运行 case，也没有 checkpoint 或 MBench 结果。
> 本文中的 case 名称、日志字段和验收条件是后续实现契约，不能在实现前写入当前 case
> 注册表或默认 runner。

## 1. 目标

本方案把几何与内容选择严格解耦：相机几何只回答每个非 anchor latent 应保留多少 token，
WorldKV 的 anchor/novelty 相似度只回答具体保留哪些 token。几何结果不再直接生成水平列、
矩形区域或其他空间裁剪 mask。

每个四 latent chunk 固定使用第 0 帧作为完整 anchor。后三帧分别相对 anchor 计算连续 FOV
新增比例，并据此得到不同的连续保留率。被逐出的 CPU KV bank 始终无损；比例计算、token
选择、packing、补回或重复都只发生在 retrieval materialization 阶段。

方案规划三个 case：

| 编号 | 计划 case | 剩余预算处理 | 研究问题 |
| --- | --- | --- | --- |
| A16 | `motion_novelty_unfilled` | 没有完整压缩 chunk 可加入后允许欠填 | 连续几何比例与 novelty 选择本身是否有效 |
| A17 | `motion_novelty_backfill` | 补回已选 chunk 中尚未选择的唯一 token | 欠填造成的质量损失能否由真实额外信息恢复 |
| A18 | `motion_novelty_duplicate` | 重复最高 query-relevance chunk 的已选源 token | 填满带来的收益是否只是注意力重加权 |

A18 是诊断消融，不是推荐的正式压缩方法。重复 token 会改变 softmax 中对应内容的有效权重，
必须与补回新 token 的 A17 分开解释。

## 2. 术语与固定预算

本文只使用 `keep_ratio` 表示保留比例：

```text
overlap_ratio     = 当前 latent 的视野被 anchor 覆盖的比例
compression_ratio = overlap_ratio
keep_ratio        = 1 - overlap_ratio
```

Wan 当前分辨率下每个 latent 有 `F=1560` 个 token。retrieval region 固定为 8 个虚拟时间
槽，每槽至多 `F` 个 token：

```text
slot_count       = 8
slot_capacity    = F
retrieval_budget = 8F = 12480
virtual_slots    = 4,5,...,11
```

比例不量化为 `1/4、1/2、3/4、1`，也不设置最低保留率。唯一不可避免的离散化是把浮点比例
转换为整数 token 数。

## 3. 共同的基础计划

### 3.1 Retrieval 与压缩参考系

历史 chunk 是否相关仍由当前 query 的 FOV retrieval 决定。对某个被排序的历史 chunk
`C=(P0,P1,P2,P3)`，压缩只比较 chunk 内部：

```text
P0 = 完整 anchor
Pi = 第 i 个非 anchor latent，i=1,2,3
```

因此需要区分两种相似度：

- `retrieval_similarity(C,Q)`：历史 chunk 相对当前 query 的相关性，用于 chunk 排序与
  backfill/duplicate 优先级；
- `anchor_token_similarity(i,j)`：latent `i` 的 token `j` 相对 anchor K centroid 的余弦
  相似度，用于 WorldKV novelty token 排序。

二者不能混用。A17/A18 中“最高相似度 chunk”始终指最高 `retrieval_similarity` 的已选
chunk，而不是 anchor-token cosine 最大的 chunk。

### 3.2 连续 FOV 保留率

对每个非 anchor latent，调用已有的确定性 FOV overlap，并把该 latent 放在分母一侧：

```text
o_i = FOVOverlap(current=Pi, historical=P0, current_K=Ki, historical_K=K0)
q_i = clamp(1 - o_i, 0, 1)
n_i = min(F, max(0, ceil(q_i * F)))
```

`o_i` 估计 `Pi` 当前视野中被 anchor 视野覆盖的比例，所以 `1-o_i` 是该 latent 相对 anchor
的新增视野比例。FOV overlap 同时响应旋转、位移和内参变化，但位移效果仍依赖确定性探针
使用的有限半径；当前 `fov_radius=8` 是隐含的场景尺度假设，不等价于真实深度感知投影。

当相机不动时允许 `q_i=0` 和 `n_i=0`。这意味着静止相机下后三帧可能全部被删除，只保留
anchor；动态物体造成的内容变化不会反映在几何比例中。这是本方法需要通过实验验证的核心
风险，不能暗中用固定 25% 或 50% 下限掩盖。

几何缺失、内参非法或 overlap 非有限值时，候选 chunk 应被标记为
`motion_geometry_invalid` 并跳过，不静默回退到固定比例。这样才能保持三个 case 的方法
语义一致。跳过数量必须进入日志。

### 3.3 WorldKV novelty 完整排序

基础实现沿用当前 minWM-dyKV 固定 WorldKV case 的层共享选择方式：使用无损 bank 中第 0 层
K 计算一次 token 顺序，所有 Transformer 层复用同一组索引。这样 A16--A18 只改变连续保留
率和剩余预算策略，不同时引入“逐层独立索引”变量。

对 chunk 的 anchor：

```text
centroid = mean(K_layer0[P0], spatial_tokens)
```

对非 anchor latent `Pi` 的每个 token：

```text
score_i[j] = cosine(K_layer0[Pi,j], centroid)
novelty_order_i = argsort(score_i, ascending=True)
base_indices_i = novelty_order_i[:n_i]
omitted_indices_i = novelty_order_i[n_i:]
```

较低 cosine 被视为相对 anchor 更 novel。物化前把选中的空间索引恢复为升序；同一层内 K/V
必须使用完全相同的索引。完整 anchor 使用 `0..F-1`。

基础 chunk 成本为：

```text
base_cost(C) = F + n_1 + n_2 + n_3
```

### 3.4 相关性优先的历史选择

候选按当前 query FOV similarity 从高到低排序，时间起点只作为确定性 tie-break。随后逐个
构造基础压缩计划：

```python
selected = []
for candidate in ranked_candidates:
    base = build_base_motion_plan(candidate)
    if base is invalid:
        continue
    if packable(selected + [base], slots=8, capacity=F):
        selected.append(base)
```

选择目标是 relevance-first，而不是单纯最大化 token 利用率或 chunk 数量：高相关候选先
获得进入预算的机会；放不下一个候选时继续检查后续更小的候选。扫描结束后冻结
`selected_block_ids`，A16、A17、A18 都不得再改变该集合。

## 4. 任意 token 长度的 packing

每个完整 anchor 独占一个 virtual slot。非 anchor frame segment 长度可以是 `0..F` 中的
任意整数；长度为 0 的 segment 不物化。一个非空 frame segment 只能映射到一个 virtual
slot，不拆成多个时间位置；多个 segment 可以共享一个 slot，但每槽总长度不得超过 `F`。

第一版使用确定性 best-fit decreasing：

1. 按源时间顺序为完整 anchor 保留独立 slot；
2. 其余 segment 按 `token_count` 降序排列，以 source frame ID 作为 tie-break；
3. 每个 segment 放入“放入后剩余空间最小”的可用 slot；
4. 没有可用 slot 时创建新 slot；
5. 使用超过 8 个 slot 即不可装入。

连续长度下仅检查 `sum(tokens)<=8F` 不够，因为可能产生 bin fragmentation。正式 planner
需要返回 `slot_token_loads`，并在 materialization 前再次断言：

```text
0 <= slot_load <= F
sum(slot_loads) == materialized_retrieval_tokens <= 8F
virtual_slot_id in [4,11]
```

如果 best-fit decreasing 在真实日志中频繁留下“总容量足够但无法装入”的碎片，再将可装入
检查替换为针对 8 个小 bin 的确定性回溯搜索；第一版不同时实现两套公开策略。

## 5. A16：允许欠填

`motion_novelty_unfilled` 只物化共同基础计划：

```text
anchor                  -> 全部 token
non-anchor frame i      -> base_indices_i
omitted_indices_i       -> 不使用
remaining retrieval     -> 空，不补零
```

扫描完候选且没有更多完整基础 chunk 可装入后立即停止。实际 retrieval token 数可以小于
12480；未使用的虚拟位置不创建 padding K/V，也不参与 attention。

A16 回答“连续几何比例本身是否足够”。它的低计算量是方法结果，不得在报告中与满预算 case
只比较质量而忽略延迟和实际 token 数。

## 6. A17：补回未选中的唯一 token

`motion_novelty_backfill` 使用与 A16 相同的 `selected_block_ids` 和基础索引。在没有其他完整
基础 chunk 可装入后，才从已选 chunk 的 `omitted_indices` 补回真实、尚未出现过的源 token。

目标长度定义为：

```text
unique_raw_tokens = 4F * number_of_selected_chunks
fill_target = min(8F, unique_raw_tokens)
```

通常选择至少两个 chunk 后 `fill_target=8F`。候选不足导致 `unique_raw_tokens<8F` 时，A17
不伪造 token，只恢复全部唯一 token 并记录剩余空间。

补回优先级：

1. selected chunk 按 query retrieval similarity 降序；
2. 在一个 chunk 内，额外预算按三个非 anchor frame 的剩余容量
   `F-base_tokens_i` 成比例分配；
3. 整数余数使用 largest-remainder，source frame ID 作为 tie-break；
4. 每帧从自己的 `omitted_indices_i` 前缀继续取 token。

不用跨帧直接比较原始 cosine 数值，因为不同帧的 score 分布未必可校准。补回过程只能增加
segment 长度，不能替换基础 novelty token。

补回必须与 packing 联动。对当前最高优先级 chunk，先请求最多可用的 extra token；如果
完整请求无法装入，在固定的成比例分配规则下二分查找最大可装 extra，再处理下一相关 chunk。
最终目标是达到 `fill_target`；若因 slot fragmentation 仍未达到，记录
`unique_backfill_unfilled_tokens`，不能转为重复 token。

需要断言同一 `(block_id, frame_offset, spatial_token_index)` 在 A17 payload 中最多出现一次。

## 7. A18：重复最高相关 chunk 的源 token

`motion_novelty_duplicate` 是刻意引入重复内容的诊断 case。它与 A16/A17 共享：

- 相同的候选排名；
- 相同的 selected chunk；
- 相同的基础 keep ratio 和 base indices；
- 与 A17 相同的最终总 token 数和每个 slot 的目标 load。

它不使用任何 `omitted_indices`。额外位置只复制最高 query-relevance 已选 chunk 的基础源
token；如果需要的重复数超过该 chunk 的基础 token 数，允许循环重复，因此一个源 token
可以出现多次。

### 7.1 重复池

最高相关 chunk 的基础 payload 构成 repeat pool。为了避免简单前缀让完整 anchor 获得不成
比例的重复，先按各 frame 的基础 token 数分配重复配额：

```text
duplicate_quota_f ~= duplicate_total * base_tokens_f / chunk_base_tokens
```

整数余数使用 largest-remainder。anchor 内按原空间顺序循环；非 anchor 内按已选 token 的
novelty 顺序循环。`base_tokens_f=0` 的 frame 不参与重复池。该规则没有新的可调权重。

### 7.2 与 A17 对齐的 slot 布局

A17 先产生 reference fill layout，包括每个 slot 的目标 token load。A18 逐 slot 填补
`target_load-base_load`，从 repeat pool 取得源 token copy，并把 copy rebase 到该目标 slot。
因此 A17/A18 具有相同：

```text
final_tokens_total
slot_token_loads
used_virtual_slots
attention tensor shape
```

复制的是同一个 bank 源 `(block, frame, spatial_index)`；copy 可以被分配到另一个目标 virtual
slot，所以 temporal RoPE 会按目标 slot 重新 rebase，V 和源内容保持复制语义。日志必须同时
记录源 frame 与目标 slot，不能把 copy 伪装成新的历史 token。

这种重复会提高对应内容在 softmax 中的总注意力质量，即使单个重复 token 的 logit 相同，
多个副本的指数项也会累加。A18 因而只回答“满长度或重加权本身是否带来变化”，不能作为
更强记忆容量的证据。

## 8. 三个 case 的共同 reference layout

为了避免 A16/A17 的差异同时包含 RoPE slot 变化，planner 每次都先构造 A17 的 reference
fill layout：

1. 冻结共同的 selected chunks 与基础 segment；
2. 生成 A17 unique backfill target；
3. 得到 reference `virtual_slot_id` 与 `slot_token_loads`；
4. A17 物化全部 unique target token；
5. A16 只物化 base token，但 base segment 沿用 reference slot；
6. A18 使用相同目标 slot load，以重复源 token 替代 A17 的新增唯一 token。

这样 A16/A17 的公共 token 具有相同 temporal slot，A17/A18 又具有相同最终 shape 和 slot
负载。若 reference layout 构造失败，三个 case 都必须对该 event 使用相同的确定性失败策略，
不能各自选择不同历史。

## 9. 计划数据结构

建议新建 `Wan21/pipeline/dykv_motion_novelty.py`，不要继续扩大已有通用 packing 文件中的
case-specific 分支。

```text
MotionFramePlan
  block_index
  frame_offset
  source_frame_id
  fov_overlap
  keep_ratio
  base_indices
  omitted_indices_in_novelty_order
  base_token_count
  backfill_token_count
  duplicate_token_count
  virtual_slot_id

MotionChunkPlan
  block_index
  retrieval_distance
  retrieval_similarity
  anchor_frame_id
  frames
  base_tokens
  raw_tokens

MotionRetrievalPlan
  selected_block_indices
  base_frames
  reference_filled_frames
  token_budget
  base_used_tokens
  unique_backfill_tokens
  duplicate_tokens
  final_used_tokens
  slot_token_loads
```

`omitted_indices_in_novelty_order` 只需保存在 planning 期间；事件日志不写完整索引列表，避免
日志体积膨胀。正式 payload 继续提供 source frame、segment 长度与 virtual slot metadata。

## 10. 计划代码改动

| 文件 | 责任 |
| --- | --- |
| `Wan21/pipeline/dykv_motion_novelty.py` | 连续比例、novelty 排名、共同选择、unique backfill、duplicate plan |
| `Wan21/pipeline/dykv_runtime.py` | 调用统一 planner；三个 case 共享候选排名；写事件诊断 |
| `Wan21/pipeline/dykv_packing.py` | 提供或抽取任意整数 segment 的确定性 packing 原语，不包含 case 名称判断 |
| `Wan21/dykv_cases.py` | 实现完成后注册 A16--A18；此前不注册 |
| `Wan21/wan/modules/dykv_rope.py` | 原则上只复用逐 segment rebase；仅在 metadata 契约不足时修改 |
| `Wan21/scripts/inference/run_dykv_cases.sh` | 三个 case 可运行后再加入默认/显式清单 |
| `Wan21/tests/test_dykv_motion_novelty.py` | 连续比例、唯一补回、重复与公平性回归 |

CPU bank 仍保存无损 K/V，不增加 store-time 压缩，也不增加公开连续超参数。FOV 探针数和
半径继续使用 dyKV 现有固定配置。

## 11. 日志契约

每个 retrieval event 至少记录：

```text
candidate_block_ids
ranked_candidate_block_ids
selected_block_ids
retrieval_similarities

motion_fov_overlaps
motion_keep_ratios
relative_rotation_degrees
relative_translation_distances
motion_geometry_invalid_block_ids

base_tokens_per_frame
base_tokens_per_chunk
base_tokens_total
unique_backfill_tokens_per_frame
unique_backfill_tokens_total
duplicate_tokens_per_frame
duplicate_tokens_total
duplicate_source_block_ids
max_source_token_multiplicity

actual_tokens_per_frame
final_tokens_total
fill_target_tokens
unused_tokens
slot_token_loads
virtual_slot_ids
```

A16 必须满足 `backfill=duplicate=0`；A17 必须满足 `duplicate=0` 且源 token 唯一；A18 必须
满足 `unique_backfill=0` 且在需要填充时 `max_source_token_multiplicity>1`。

## 12. 单元测试与验收

### 12.1 连续比例

- 相同 pose/FOV 得到 `q=0`，不被隐式抬高；
- overlap `0.1234` 得到 keep ratio `0.8766`，而不是固定档位；
- `ceil(qF)` 的边界为 0 和 F，永不越界；
- 左右旋转给出相同数量但不产生几何列 mask；
- 位移改变 overlap 并保持确定性；
- 非法几何明确跳过并记录。

### 12.2 Novelty 与 bank

- anchor 始终完整；
- base/omitted 构成 `0..F-1` 的不交并集；
- K/V 使用同一索引；
- 所有层使用同一计划长度和 layer-0 共享索引；
- materialization 不修改 CPU bank。

### 12.3 选择与 packing

- 三个 case 的 ranked/selected block ID 完全一致；
- 任意整数 segment 不超过单 slot 和总预算；
- 一个非空 source frame 在基础/unique plan 中只对应一个 slot；
- A16/A17 公共 token 的 virtual slot 一致；
- A17/A18 的最终总长度与 slot loads 一致。

### 12.4 A17 unique backfill

- 只从 omitted prefix 取 token；
- 同一源 token 最大 multiplicity 为 1；
- chunk 按 query relevance 补回；
- frame 配额按剩余容量比例分配；
- 无法完全填满时输出准确 residual，不转为重复。

### 12.5 A18 duplicate

- 不使用 omitted token；
- 所有额外 copy 都来自最高 query-relevance selected chunk；
- 重复配额与基础 frame 长度成比例；
- 需要填充时确实存在 multiplicity 大于 1；
- duplicate copy 保留 source metadata，并按目标 slot 执行 temporal rebase；
- A18 不修改 bank，也不改变 selected block 集合。

## 13. 实验设计

### 13.1 核心矩阵

| Case | 历史选择 | token 位置选择 | 最终长度 | 主要作用 |
| --- | --- | --- | --- | --- |
| `retrieval_no_compression` | 当前 query FOV | 全部 | 固定 8F | 不压缩上界 |
| `fixed_novelty` | 当前 query FOV | 固定 50% novelty | 欠填 | 固定比例内容压缩对照 |
| `yaw_intrinsics` | 当前 query FOV | 几何列裁剪 | 欠填 | 精确几何位置对照 |
| A16 `motion_novelty_unfilled` | 共同动态选择 | 连续比例 novelty | 欠填 | 新方法基础效果 |
| A17 `motion_novelty_backfill` | 与 A16 相同 | 基础 + 唯一 token | 目标 8F | 真实额外信息的作用 |
| A18 `motion_novelty_duplicate` | 与 A16 相同 | 基础 + 重复 token | 与 A17 相同 | 重加权/长度诊断 |

### 13.2 公平性要求

- A16--A18 使用相同 commit、checkpoint、prompt assignment、trajectory 和 sample seed；
- 每个 event 的 candidate ranking、selected blocks、base ratios 和 base indices 必须逐项一致；
- A17/A18 的 final token 数、slot load 和 attention shape 必须一致；
- 同时报告质量、retrieval latency、总生成时间、峰值显存和 CPU bank 字节数；
- A18 只能作为诊断，不与 A16/A17 一起宣称“记住了更多唯一历史”。

### 13.3 分阶段运行

1. 合成单元轨迹：静止、纯 yaw、横移、前进、混合旋转位移；
2. 24 latent 单 prompt checkpoint 冒烟，检查第 20 帧后事件日志；
3. `mbench_typical_4`，固定一个 seed，先检查视频和 token/slot 日志；
4. 典型样本通过后再运行八样本和完整 MBench；
5. 正式结果至少使用多个 seed，并分左右方向与 subset 报告。

### 13.4 结果解释

| 观察 | 支持的解释 |
| --- | --- |
| A17 > A16，A18 ≈ A16 | 被压掉的唯一 token 有用，单纯重复无效 |
| A17 ≈ A18 > A16 | 主要收益可能来自满长度或提高历史 attention mass |
| A18 > A17 | 模型可能更受重复重加权影响，而不是额外唯一内容 |
| A16 ≈ A17 ≈ A18 | 欠填不是主要瓶颈，基础动态压缩已足够或检索本身受限 |
| A17/A18 < A16 | 更多 token 稀释有效历史或产生 slot/attention 干扰 |

以上解释都必须结合实际 token 数、source multiplicity 和检索延迟。A18 提升不能被表述为
记忆覆盖增加，因为它没有引入新的源 token。

## 14. 已知风险

1. FOV overlap 对位移的连续比例依赖固定探针半径，不具备真实深度感知；
2. 相机静止但物体运动时 `q=0`，几何无法发现内容 novelty；
3. stored K 已包含位置相关变换，anchor-centroid cosine 不一定是纯内容相似度；
4. 连续 segment 会产生 bin fragmentation，利用率不一定自动达到 100%；
5. A18 重复 token 会系统性改变 softmax 权重，只能用于诊断；
6. sequence parallel 下相似度统计必须保证各 rank 使用一致索引；
7. 构造 A17 reference layout 会让 A16 planner 多做一次仅用于公平映射的规划，需要单独记录
   planning time，避免把它误算成 GPU attention 成本。

## 15. 实现与提交顺序

每完成一个模块后单独提交并推送：

1. `Add continuous motion novelty planning`
   - 连续 FOV ratio、layer-0 novelty 完整排序、数据结构与单元测试；
2. `Add unfilled motion novelty retrieval case`
   - relevance-first 选择、任意长度 packing、A16、runtime/RoPE 测试；
3. `Add unique-token motion novelty backfill case`
   - A17 reference fill layout、unique backfill 与公平性日志；
4. `Add repeated-token motion novelty diagnostic case`
   - A18 重复池、目标 slot 对齐、multiplicity 测试；
5. `Document motion novelty experiments`
   - runner、case 文档、实验台账与正式命令；只有前三个 case 真正注册后才能把状态改为
     “可运行”。
