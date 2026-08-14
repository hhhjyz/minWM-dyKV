# 基于前驱 Chunk 的增量视角压缩

## 1. 实现结论

本模块实现了两条刻意分离的决策链：

1. **检索**仍由当前正在生成的 query chunk 决定。FOV 检索器比较当前相机与所有已逐出
   历史块，把可能对当前画面有用的块排在前面。
2. **压缩**不再比较历史块与当前 query。对被检索块 `C_i`，压缩器只比较 `C_i` 与时间上
   严格相邻的前驱块 `C_(i-1)`，保留 `C_i` 相对前驱新增的世界角域。

因此，本方案回答的是两个不同问题：当前 query 决定“取谁”，前驱增量决定“取来的块保存
多少、保存哪一侧”。CPU 历史库继续保存完整、未压缩 KV；裁剪只在检索载荷实例化时发生，
所以仍然是 **retrieval-time compression**。

## 2. 代码结构

| 代码位置 | 作用 |
| --- | --- |
| `Wan21/pipeline/dykv_fov.py::select_fov_blocks` | 根据当前 query 的 FOV 对历史块排序 |
| `Wan21/pipeline/dykv_predecessor.py::_find_predecessor` | 查找 `frame_end == C_i.frame_start` 的严格前驱 |
| `dykv_predecessor.py::_predecessor_geometry` | 在世界 yaw 角域计算前驱覆盖、当前覆盖和新增角域 |
| `dykv_predecessor.py::quantize_incremental_ratio` | 将新增比例量化为四个固定档位 |
| `dykv_predecessor.py::_select_groups` | 对完整 chunk 做考虑分箱可行性的分组 0/1 选择 |
| `dykv_predecessor.py::_select_tail` | 用未选中 chunk 的单 latent 填补余量 |
| `dykv_predecessor.py::_apply_query_backfill` | 在同一 RoPE slot 的剩余原子内补回当前 query 可见列 |
| `dykv_packing.py::materialize_packed_retrieval` | 每层按同一 token 索引从无损 K/V 中取值 |
| `Wan21/wan/modules/dykv_rope.py` | 按 `source_frame_id → virtual_slot_id` 对每段 K 做 RoPE rebase |
| `dykv_runtime.py::DyKVRuntime.retrieve` | 串联当前-query 检索、前驱压缩、装箱和诊断记录 |

第 0 层 K 只在纯几何不可用时用于 novelty fallback；正常 yaw 路径的 token 列索引由相机
几何计算一次并由所有 Transformer 层共享。

## 3. 新增角域与四档压缩

对候选块 `C_i` 的第 `t` 个 latent，先由该帧相机外参和内参 `K` 得到世界水平视角区间
`H_i,t`。前驱四帧的世界视角并集为：

```text
P_i = union_t H_(i-1),t
N_i,t = H_i,t - P_i
r_i,t = angle_width(N_i,t) / angle_width(H_i,t)
r_i = max_t r_i,t
```

实现使用世界 yaw 区间而不是简单的像素位移，并在 `±π` 处先展开角度再做区间并、差，
因而可处理 360° 回绕。chunk 档位取四帧最大值，避免同一完整 chunk 内出现不可预期的
大小；latent 尾部则允许使用自己的逐帧比例。

量化严格采用以下左闭右开规则：

| 新增比例 `r` | 保留比例 `q` | 52 列 latent 的列数 | 每帧 token 数 |
| --- | ---: | ---: | ---: |
| `0 ≤ r < 1/4` | `1/4` | 13 | 390 |
| `1/4 ≤ r < 1/2` | `1/2` | 26 | 780 |
| `1/2 ≤ r < 3/4` | `3/4` | 39 | 1170 |
| `3/4 ≤ r ≤ 1` | `1` | 52 | 1560 |

当 `r=0` 时仍保留 `1/4`，作为静止或前驱已完全覆盖时的安全下限。此时没有唯一的“新增
边界”，代码改用第 0 层共享 novelty mask，而不是武断地固定裁左侧或右侧。

对于 `r>0`，代码先选择落入 `N_i,t` 的列，再从新增边界向旧区域扩展到量化档位要求的固定
列数。左右方向由世界 ray 的角度顺序决定，不把某个动作字符硬编码为“左”或“右”，所以
两种旋转的 mask 互为镜像。

## 4. 非纯 yaw 的退化策略

只有满足以下条件才采用前驱角域裁剪：

- 当前块有时间上严格相邻的前驱块；
- 两块都有合法外参、空间 latent 形状，以及所选 FOV 来源需要的内参；
- 两块相对运动可解释为同一相机中心上的纯 yaw。

在候选块自身空间形状合法的前提下，平移、pitch、roll、缺失相机几何或缺失前驱时，使用
固定 `50%`、层间共享的 novelty mask，并在载荷中记录
`predecessor_fixed_novelty_fallback`。如果连候选块的空间形状都缺失或与 frame token 数
不一致，就无法构造四分之一原子，规划器会跳过该候选而不是伪造裁剪索引。这与“前后移动
暂不做角度动态裁剪”的当前设计一致，也避免把位移误当作 yaw。

## 5. 八槽装箱与 3/4 档

retrieval region 固定为 8 个 latent slot，每个 slot 划分成 4 个 `1/4` 原子：

```text
1/4 frame = 1 atom   1/2 frame = 2 atoms
3/4 frame = 3 atoms  full frame = 4 atoms
总预算 = 8 × 4 = 32 atoms
```

`3/4` 档不能只按总 token 数做普通 knapsack。例如一个 `3/4` chunk 含四个 3-atom 段，
必须占四个不同 slot；剩下的四个 1-atom 空隙可以再放一个 `1/4` chunk。为此，
`_select_groups` 的状态是 8 个 slot 的实际载荷多重集，加入候选 chunk 前会逐段检查可分箱
性。最终 `_assign_slots` 用确定性的回溯给每个 frame segment 分配真实 slot，保证：

- 每个 segment 不被切开；
- 同一 slot 的总 token 不超过 1560；
- 总 retrieval token 不超过 `8×1560`；
- 选择效用仍由当前 query 的 FOV 排名和前驱档位共同决定。

## 6. Latent 尾部与 Query Coverage 回填

三个实现 case 逐级增加能力：

| Case | 完整 chunk | 单 latent 尾部 | 当前 query coverage 回填 |
| --- | --- | --- | --- |
| `predecessor_chunks` | 是 | 否 | 否 |
| `predecessor_chunks_latent` | 是 | 是 | 否 |
| `predecessor_query_backfill` | 是 | 是 | 是 |

完整 chunk 选择结束后，`predecessor_chunks_latent` 会从未选中的候选块中选择单 latent，
继续在剩余 slot 容量中做 0/1 选择。

完整方案还会计算“历史 frame 相对当前 query 的可见列”，但它**不改变前驱压缩档位的主
决策**。它只在该 frame 已分配的同一 virtual slot 尚有整 `1/4` 原子余量时，把尚未保留的
query 可见列按原子补回，并把 token 合并进同一个 frame segment。这样不会把同一 source
frame 拆到不同时间位置，也不会突破固定 retrieval 区域。完全空闲的 slot 由 latent 尾部
优先利用，coverage 回填不会复制 frame 来强行填满。

需要注意，“回填”不保证每次都把 8 个 slot 填满：若余量不足一个原子，或所有 query 可见
列已经包含在前驱增量裁剪中，保持欠填比加入无关 token 更合理。

## 7. RoPE Rebase

装箱后的每个 frame segment 都携带：

- `source_frame_id`：它在原始视频中的真实 latent 时间；
- `frame_token_length`：本段实际 token 数；
- `virtual_slot_id`：它在 `4..11` retrieval 区域内占用的 slot。

RoPE 层对该段 K 使用 `virtual_slot_id - source_frame_id` 的逐帧时间偏移；空间维位置不变。
多个压缩 frame 可以共享一个 slot，但同一 slot 的总 token 受 1560 上限约束。query 始终映射
到训练窗口末端 `16..19`，因此整体仍为连续的 `sink 4 | retrieval 8 | local 8`。

## 8. 诊断字段

每次 retrieval 事件写入 `dykv_summaries.jsonl`，新增关键字段：

- `predecessor_frame_starts`：每个 segment 使用的前驱起始帧；
- `incremental_yaw_degrees`：候选块相对前驱块的有符号 yaw 增量；
- `incremental_fov_ratios`：量化前新增角域比例；
- `keep_tiers`：最终 `1/4、1/2、3/4、1` 档位；
- `query_backfill_tokens`：各 segment 因 query coverage 补回的 token 数；
- `source_frame_ids` / `virtual_slot_ids`：真实历史时间与折叠后 RoPE slot；
- `packing_used_atoms` / `packing_budget_atoms`：实际原子数与固定 32 原子预算。

## 9. 运行方式

统一使用 `minwm-fa`。只比较三个新增 case：

```bash
conda activate minwm-fa
CASES=predecessor_chunks,predecessor_chunks_latent,predecessor_query_backfill \
DATA_PATH=Wan21/prompts/demos.txt \
TRAJECTORY='j*10,l*10,n*3' \
NUM_OUTPUT_FRAMES=24 \
SEED=0 \
OUTPUT_ROOT=output/predecessor_ablation \
bash Wan21/scripts/inference/run_dykv_cases.sh
```

与旧的当前-query 裁剪及不压缩检索一起做核心对比：

```bash
CASES=retrieval_no_compression,packed_chunks_latent,predecessor_chunks,predecessor_chunks_latent,predecessor_query_backfill \
OUTPUT_ROOT=output/predecessor_core \
bash Wan21/scripts/inference/run_dykv_cases.sh
```

## 10. 验证状态与限制

已在 `minwm-fa` 中通过单元测试，覆盖四档边界、左右镜像、`3/4` 分箱、回填预算、现有 dyKV
回归和 MBench 适配器。单元测试命令：

```bash
conda run -n minwm-fa python -m unittest discover -s Wan21/tests -v
```

这些测试验证算法和载荷契约，不等同于 checkpoint 视频质量实验；三个新增 case 的完整视频、
耗时和 MBench 指标仍应按 `EXPERIMENTS.md` 标记为“待运行”。当前版本也没有为平移建立
depth-aware 可见性模型，平移仍走固定 novelty fallback。

此外，真实推理入口当前将 `viewmats/Ks` 转成 BF16，而 `_pure_yaw_delta` 的旋转矩阵容差为
`1e-4`。BF16 纯 yaw 的实测矩阵误差约 `1.5e-3`，已有 predecessor 视频因而全部进入
`predecessor_fixed_novelty_fallback` 的 `1/2` 档。四档边界和左右镜像单元测试虽然通过，
但修复相机几何精度并在真实日志看到 `predecessor_incremental_yaw` 之前，不能宣称视频
已经验证四档角度裁剪。完整调用链、影响和验收字段见
[`RETRIEVAL_ROTATION_COMPRESSION_FLOW.md`](RETRIEVAL_ROTATION_COMPRESSION_FLOW.md)。
