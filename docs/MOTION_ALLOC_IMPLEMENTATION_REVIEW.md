# Motion Allocation 当前实现与正确性审查说明

> 目的：本文是交给独立审阅者的自包含技术说明。它描述当前代码**实际做了什么**、哪些部分已有
> 测试或实验支持、哪些部分仍是未经证明的设计假设。本文不预设当前方法正确。

## 1. 审阅结论先行

当前实现包含三个递进改动：

1. 固定选择最多 4 个历史 chunk，并固定总 retrieval token budget；
2. 用 camera motion 或 camera+content score 在 non-anchor latent 间动态分配 token 数；
3. 用 cached 3D-RoPE K 或 completely pre-RoPE K 决定每帧具体保留哪些 token。

已经确认的工程性质：

- 历史充足时选择 4 个 chunk，总 retrieval 长度严格为 `8F=12480`；
- 历史只有 3 个合法 chunk 时严格为 `6F=9360`，不会复制 token 强行填满物理 `8F`；
- 所有 anchor 完整保留；non-anchor 可以跨 virtual slot，只限制总预算和单 latent 上限 `F`；
- 最终 materialization 按原始 KV/source-frame 顺序拼接；
- 三个新 case 和 fixed baseline 在已完成的 20s×30 实验中每个 retrieval event 都选择了相同
  source set（450/450），因此主要差异确实来自 token 数量或 token index；
- 92 个 dyKV 单元测试通过。

尚不能确认正确的算法性质：

- projected overlap 是否是合适的 camera information loss；
- V cosine distance 是否能代表动态物体 novelty；
- 全局 proportional allocation 是否会让某个 chunk 的 non-anchor 覆盖不足；
- layer-0 anchor-centroid cosine 是否是正确的 token selector；
- completely pre-RoPE K 是否真的实现了“只去 temporal RoPE 污染”。实验已经表明最后一点答案为
  **否**：它同时去除了 H/W spatial RoPE，并在旋转/混合运动上显著退化。

## 2. 术语与固定尺寸

| 记号 | 当前值 | 含义 |
| --- | ---: | --- |
| `F` | 1560 | 一个 latent frame 的 spatial token 数 |
| chunk | 4 latent | 一个历史归档块：1 个 anchor + 3 个 non-anchor |
| sink | 4 latent | 永久保留的初始区域 |
| retrieval region | 8 latent slots | 最多容纳 `8F=12480` 个 retrieval token |
| local region | 8 latent | recent clean cache + current query |
| RoPE layout | 4+8+8 | sink + retrieval + local/current |

这里的 “slot” 是 RoPE/layout 的虚拟位置，不是必须独立装入一个完整 latent 的物理容器。flat
layout 允许一个 non-anchor segment 跨越 slot 边界。

## 3. 端到端数据流

```mermaid
flowchart LR
    A["Clean live KV cache"] --> B["Archive 4-frame blocks to CPU bank"]
    B --> C["FOV rank historical chunks"]
    C --> D["Take first 4 geometry-valid chunks"]
    D --> E["Camera / content scores"]
    D --> F["Layer-0 novelty token order"]
    E --> G["Exact global token allocation"]
    F --> H["Take prefix of each frame's novelty order"]
    G --> H
    H --> I["Sort segments by original source-frame order"]
    I --> J["Materialize all 30 layers' K/V"]
    J --> K["Compose sink + retrieval + local/current attention"]
```

关键点是：score 只决定“每个 latent 保留多少 token”；具体保留哪些 token 由另一套 novelty order
决定。camera geometry 不直接生成 crop mask。

## 4. 历史 KV 归档

每生成一个 clean 4-frame block，在它被 rolling cache 淘汰前，将 30 层 K/V 对应 slice 复制到
CPU bank，同时保存该 chunk 的 `viewmats`、相机内参 `K` 和 latent spatial shape。

- 普通 case 只归档模型实际 attention 使用的、已经经过 3D RoPE 的 K；
- `pre_rope_k` case 额外只在 layer 0 live cache 保存 `content_k`；它是在
  `causal_rope_apply` 之前、但在 `norm_k` 和 sequence-parallel all-to-all 之后的 K；
- `content_k` 与 K/V 使用相同 rolling slice，归档后存入 `MemoryBlock.novelty_k`；
- 额外缓存只存在于 layer 0，不复制 30 层 descriptor。

主要代码：

- `Wan21/pipeline/causal_inference.py::_initialize_kv_cache`；
- `Wan21/wan/modules/causal_model.py::CausalWanSelfAttention.forward`；
- `Wan21/pipeline/dykv_memory.py::DyKVBank.archive_clean_block`。

### 待审查点

1. sequence parallel 下 raw K 与 roped K 的 token/head 排列是否始终完全一致；
2. rolling 时 `content_k` slice 是否覆盖所有边界情况；
3. 为 layer 0 增加完整 local cache 是否值得，是否可改成更小 descriptor；
4. 当前命名 `content_k` 容易和 V-based content score 混淆，实际用途是 novelty descriptor。

## 5. Chunk retrieval 与选择

候选历史 block 必须已经完全离开 sink 和 recent local cache，防止同一帧同时出现在 local 与
retrieval region。候选通过当前 query chunk 与历史 chunk 的 FOV distance 排序。新 planner 按
ranking 顺序尝试 geometry planning，跳过缺失或非法 geometry 的 block，取得前 4 个合法 chunk。

因此“固定 4 chunk”的准确含义是：

- 候选足够时固定 4 个；
- 候选不足或 geometry 非法时少于 4 个；
- 不用更旧的无效 geometry block，也不复制已有 chunk 填满。

主要代码：

- `Wan21/pipeline/dykv_fov.py::select_fov_blocks`；
- `Wan21/pipeline/dykv_motion_novelty.py::build_motion_alloc_4chunk_plan`；
- `Wan21/pipeline/dykv_runtime.py::DyKVRuntime.retrieve`。

### 待审查点

retrieval ranking 仍使用球面采样 FOV distance，而 compression score 使用二维多深度投影。两者
的几何假设并不完全一致，尤其平移运动中可能出现“retrieval 认为相关、compression 认为大幅
出界”或相反的情况。

## 6. Camera score

对每个 4-frame chunk，frame 0 是 anchor。对 non-anchor frame `i`，使用 anchor-relative 相机
位姿和归一化内参做双向 image-plane projection。

### 6.1 多深度近似

当前没有真实 depth，使用固定 scene scale `S=8`，四个深度为：

```text
D = {S/8, S/4, S/2, S} = {1, 2, 4, 8}
```

对每个 latent token center：

1. 用 source `K^-1` 反投影到指定深度；
2. 用 `target_w2c @ inverse(source_w2c)` 变换到 target camera；
3. 用 target `K` 投影；
4. 统计正深度且落在归一化 `[0,1)×[0,1)` 图像范围内的比例。

每个深度计算 current→anchor 与 anchor→current 两个 directional overlap，并取调和平均：

```text
h_d = 2 * overlap_forward_d * overlap_backward_d
      / (overlap_forward_d + overlap_backward_d)

overlap_i = mean_d(h_d)
q_camera_i = 1 - overlap_i
```

anchor 的 `q_camera=0`，但 anchor token 数被强制设为 `F`。

主要代码：`Wan21/pipeline/dykv_projected_overlap.py::projected_motion_overlap`。

### 6.2 可能的问题

1. **无真实深度/遮挡**：四个固定平面不能表示场景深度分布、disocclusion 或动态遮挡；
2. **scene scale 固定**：`S=8` 对室内近景、远景和尺度未知的生成场景未必统一合理；
3. **平均深度权重固定**：近景/远景四个深度等权，未按可见 token 的真实深度分布加权；
4. **只比较 chunk 内 anchor**：不看 query，也不看相邻 chunk，action boundary 附近可能不连续；
5. **恒速控制导致重复 pattern**：WASD/IJKL 恒速时，不同 chunk 的三个 non-anchor camera score
   往往重复，动态分配实际上只复制同一 `[q1,q2,q3]` 模式；
6. **出界比例不等于信息量**：画面内仍重叠的区域也可能发生大幅视差，出界区域也未必重要；
7. **坐标约定需要独立复核**：`w2c` 变换方向和 normalized K 是否与生成相机实现严格一致。

## 7. Content score

为了修复“相机静止但人物/物体运动时 `q_camera=0`”的问题，camera+content case 使用 layer-0 V。
V 不应用 RoPE，因此该 score 不直接含 temporal/spatial rotary phase。

将 layer-0 V reshape 为 `[batch, 4 frames, F tokens, heads×dim]`，逐空间 token 比较 non-anchor
与 anchor 的 cosine distance：

```text
d_i,p = clamp((1 - cosine(V_i,p, V_anchor,p)) / 2, 0, 1)
q_content_i = mean_p(d_i,p)
q_alloc_i = max(q_camera_i, q_content_i)
```

### 可能的问题

1. **对应 token 未必对应同一物体**：相机一动，相同 `(h,w)` token 看向不同 world point，V
   distance 同时包含 camera motion 和 content motion；
2. **没有 camera-conditioned baseline**：直接 `max` 比较两个未校准尺度。20s×30 中
   `q_content > q_camera` 出现在 90.5% non-anchor 上；
3. **全图 mean 稀释小物体**：小范围人物手部、车轮、木屑等动态可能被大量静态背景平均掉；
4. **V 是否是合适 descriptor 未证明**：V 为 attention value，不保证 cosine 距离有语义或运动
   可比性；
5. **只用 layer 0**：浅层可能偏纹理/局部特征，未验证其他层或 latent descriptor；
6. **生成反馈**：第一次 token selection 改变后，后续 V、content score 和 allocation 都会分叉，
   所以后期 score 差异不是 frozen-state 单变量比较；
7. **平均 batch**：当前实际推理 batch=1；如果扩展 batch，多样本平均 novelty order 的语义需重审。

## 8. 全局 token allocation

假设本次成功选中 `n` 个 chunk：

- `n` 个 anchor 各保留 `F`，共 `nF`；
- 其余 `3n` 个 non-anchor 共享另一个 `nF` budget；
- 每个 non-anchor 上限为 `F`；
- 用 deterministic weighted capped allocation 按 `q_alloc` 比例分配整数 token；
- floor 后的余数按 fractional remainder 和稳定 index 顺序补齐；
- score 全为 0 时按剩余容量均分，而不是欠填。

因此总长度严格为：

```text
total = anchor nF + non-anchor nF = 2nF
```

当 `n=4` 时为 `8F`；当 `n=3` 时为 `6F`。

### 必须澄清的语义

代码保证的是**所有 chunk 的总长度为 `2nF`，平均每个 chunk `2F`**，不保证每个 chunk 单独
恰好 `2F`。camera+content 实验中曾观察到四个 chunk 分别为
`[3407, 3392, 2474, 3207]`，总和仍为 `12480`。

### 可能的问题

1. **跨 chunk 竞争**：某个 chunk 的 non-anchor 可能被其他 chunk 抢走预算，降低历史覆盖；
2. **无 per-frame floor**：score 很小时 non-anchor 可以得到 0 token；动态物体漏检时无法恢复；
3. **强制填满有效预算**：全部 score 为 0 仍均分 `nF`，可能保留大量冗余 token；
4. **anchor 固定占 50%**：不论 anchor 是否冗余，它始终占总预算一半；
5. **比例放大**：当所有 score 都很小但非零时，归一化后仍会分掉完整 `nF`，绝对 novelty 大小
   不再影响总保留量；
6. **确定性 tie-break 有顺序偏置**：integer remainder 同分时更早的 frame index 先获得 token。

主要代码：`Wan21/pipeline/dykv_motion_novelty.py::_weighted_capped_allocation`。

## 9. 具体 token 的 novelty order

token 数确定后，需要从每个 non-anchor 的 `F` 个 token 中选择指定数量。当前只使用 layer-0 K：

1. 将 anchor frame 的所有 `F` 个 K 在 spatial token 维求均值，得到一个 anchor centroid；
2. 对 non-anchor 的每个 token，计算该 token K 与 anchor centroid 的 cosine similarity；
3. 按 similarity 升序排序；
4. 取前 `keep_tokens_i` 个，即优先保留与 anchor centroid **最不相似**的 token；
5. 选中 index 最后按原空间 index 排序后 materialize，不按 novelty ranking 拼接。

### 两种 descriptor

| 模式 | 实际输入 | 含义 |
| --- | --- | --- |
| `cached_roped_k` | attention cache 的 layer-0 K | temporal + H spatial + W spatial 3D RoPE 后 K |
| `pre_rope_k` | `norm_k(k(x))` | 三种 RoPE 都未应用的 raw normalized K |

### 可能的问题

1. **centroid 丢失空间对应**：用全图 anchor centroid，而不是 anchor 对应位置或 query-aware score；
2. **最不相似不等于最有用**：异常、噪声或边界 token 也可能最不相似；
3. **不看当前 query**：selector 只看历史 chunk 内变化，不保证对本次 query attention 有用；
4. **只用 layer 0 排序所有 30 层**：默认所有层应保留相同 token index，尚未验证；
5. **cached K temporal phase 污染**：不同 frame 的 phase 会影响 cosine；
6. **completely pre-RoPE 删除过多**：`causal_rope_apply` 同时包含 temporal/H/W 三部分，raw K
   把有用 spatial phase 也删掉。20s×30 中 rotation-only 和 mixed-motion pose PSNR 显著下降；
7. **理想对照尚未实现**：真正需要审查的是保留 H/W、只移除 temporal phase 的 spatial-only
   descriptor，而不是 completely raw K。

主要代码：`Wan21/pipeline/dykv_motion_novelty.py::_novelty_order`。

## 10. Layout 与 materialization

选中 frame segment 后：

- 按 `source_frame_id` 重新排序，保持原始 KV 时间顺序；
- segment 可以跨 virtual slot，不要求一个 non-anchor 最多占单个 `F` slot；
- 根据 cumulative reference length 给 segment 分配 virtual slot id；
- 所有 30 层使用同一 block/frame/token index materialize K/V；
- 实际 attention K 仍为正常 RoPE K，pre-RoPE descriptor 只参与 token index 选择；
- 最终交给 tri-region composer，与 sink 和 local/current 组成 attention 输入。

### 可能的问题

1. flat payload 的连续 token、source frame 和 virtual slot 不是一一对应的完整 latent grid；
2. segment 跨 slot 时 RoPE position 赋值是否保持期望的空间/时间语义，需要结合
   `dykv_rope.py` 独立检查；
3. source order 保留时间顺序，但不保证高 relevance chunk 更靠近 query；
4. 每层 materialization 和 CPU→GPU copy 仍有明显开销，尚未证明真实推理加速。

## 11. 三个新 case 的唯一差异

| Case | Source retrieval | 总预算 | 数量分配 | Token selector |
| --- | --- | --- | --- | --- |
| `motion_alloc_cam_4chunk` | FOV top-4 valid | `2nF` | `q_camera` | cached 3D-RoPE layer-0 K |
| `motion_alloc_cam_content_4chunk` | 相同 | 相同 | `max(q_camera,q_content)` | cached 3D-RoPE layer-0 K |
| `motion_alloc_cam_content_prerope_4chunk` | 相同 | 相同 | 相同公式 | completely pre-RoPE layer-0 K |

fixed 对照 `retr16_compression_r033` 同样选最多 4 chunk、anchor 完整、总长度为 3 chunk 时 `6F`、
4 chunk 时 `8F`，但每个 non-anchor 固定保留约 `F/3`。

20s×30 运行日志确认：

- fixed vs camera source set：450/450 event 相同；
- camera vs camera+content source set：450/450 event 相同；
- 四个 case 的 prompt、trajectory、sample seed、initial-noise fingerprint 完全对齐。

## 12. 已有实验不能证明什么

完整结果见 [`MOTION_ALLOC_20S_ABLATION_RESULTS.md`](MOTION_ALLOC_20S_ABLATION_RESULTS.md)。审阅时
需要注意以下限制：

1. **不是 ground-truth PSNR**：outbound half 是同一生成 rollout 的 pseudo-reference，只衡量
   loop consistency；
2. **只有 seed 0**：30 个 prompt 不是 30 个独立随机种子；
3. **数据只覆盖 loop closure**：不能直接外推到开放式长程、非回环轨迹；
4. **subgroup 很小**：rotation-only 8、translation-only 8、mixed 14；
5. **多指标/多分组比较**：bootstrap CI 没有做 multiple-comparison correction；
6. **不同 case 在不同同型号 A6000 上生成**：noise fingerprint 相同，但 CUDA kernel 不保证
   bitwise deterministic；
7. **end-to-end feedback**：从第一次不同 token selection 起，后续生成状态和 score 都会分叉；
8. **没有 frozen-bank replay**：尚未在完全相同的 archived K/V 上直接衡量 token-index Jaccard、
   spatial coverage 或 query relevance；
9. **质量不等于速度**：本实验没有证明压缩后的 wall-clock 或显存收益。

已有结果只支持：

- camera allocation 对 translation-only SSIM 有局部收益，但 perceptual 指标存在代价；
- 当前 raw V `max` 补偿没有总体显著收益；
- completely pre-RoPE K 对旋转/混合运动有明确风险。

## 13. 建议审阅者逐项判断的问题

### 几何

- [ ] `target_w2c @ inverse(source_w2c)` 的方向是否与项目相机约定一致？
- [ ] normalized `K` 与 latent token center `[0,1)` 是否匹配？
- [ ] 双向调和平均是否比 intersection/union、min overlap 或单向 overlap 更合理？
- [ ] `{1,2,4,8}` 四深度等权是否有理论或数据依据？
- [ ] `1-overlap` 是否可直接解释为应该分配的 token 比例？

### Content

- [ ] layer-0 V cosine 是否是 RoPE-free content motion descriptor？
- [ ] 是否应先做 camera compensation，再计算 residual content novelty？
- [ ] mean aggregation 是否应改为 top-p/分区 robust aggregation？
- [ ] `max(camera,content)` 是否需要尺度校准？

### Allocation

- [ ] budget 应跨 chunk 全局竞争，还是每个 chunk 固定 `2F` 后只在 chunk 内分配？
- [ ] 是否需要 non-anchor minimum floor、per-chunk coverage floor 或允许欠填？
- [ ] anchor 是否必须完整，还是只保证部分 anchor coverage？

### Token selector / RoPE

- [ ] anchor spatial centroid 是否合理，还是应比较对应空间位置？
- [ ] 应按 historical novelty、current-query relevance，还是二者联合排序？
- [ ] layer-0 index 是否适用于所有层？
- [ ] 正确 descriptor 是否应为 spatial-only RoPE K？
- [ ] flat/cross-slot layout 下的 RoPE position 是否仍与训练分布相容？

### Evaluation

- [ ] primary endpoint 是否应预注册为 pose PSNR + pose LPIPS？
- [ ] 是否需要真实视频/外部 reference，而不仅是 self-loop consistency？
- [ ] 是否应增加多个 seed 和非回环轨迹？

## 14. 推荐的最小验证顺序

在继续跑完整 30-video 实验前，建议先完成：

1. **坐标单测**：手算 identity、纯 yaw、纯 pitch、左右/前后平移的 projection direction；
2. **Frozen-bank replay**：同一 archived state 比较 fixed/camera/content 和三种 K descriptor；
3. **记录 token selector 指标**：index Jaccard、2D spatial coverage、与 current query 的 layer-0
   similarity、动态区域 recall；
4. **只实现 spatial-only RoPE**：cached 3D-RoPE vs spatial-only，raw K 保留为失败对照；
5. **8-video smoke**：rotation 2、translation 2、mixed 4；
6. **通过后再跑 30-video paired confirmation**，并增加至少 3 个 seed。

这套顺序能把“planner 本身是否合理”和“end-to-end rollout 是否偶然变好”分开。

## 15. 代码与证据索引

| 内容 | 路径 |
| --- | --- |
| Case 注册 | `Wan21/dykv_cases.py` |
| 配置和 CPU bank | `Wan21/pipeline/dykv_memory.py` |
| Projected camera overlap | `Wan21/pipeline/dykv_projected_overlap.py` |
| Content score、allocation、novelty order | `Wan21/pipeline/dykv_motion_novelty.py` |
| Retrieval orchestration | `Wan21/pipeline/dykv_runtime.py` |
| Raw layer-0 K live cache | `Wan21/pipeline/causal_inference.py`、`Wan21/wan/modules/causal_model.py` |
| Tri-region RoPE/layout | `Wan21/wan/modules/dykv_rope.py` |
| 单元测试 | `Wan21/tests/test_dykv_motion_novelty.py`、`test_dykv_runtime.py` |
| 20s×30 结果 | `docs/MOTION_ALLOC_20S_ABLATION_RESULTS.md` |
| Paired analysis | `Wan21/scripts/evaluation/analyze_motion_ablation.py` |

复核测试命令：

```bash
/home/jiangyize/software/miniconda3/envs/minwm/bin/python \
  -m unittest discover -s Wan21/tests -p 'test_dykv*.py'
```

