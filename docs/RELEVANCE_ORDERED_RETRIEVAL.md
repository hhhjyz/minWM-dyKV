# 按相关性分配 Retrieval RoPE 位置

## 1. 实验目的

`retrieval_no_compression_relevance_order` 用于验证：在候选集合、FOV 检索结果、KV token
数量和计算量完全不变时，把相关性更高的历史 chunk 放到更靠近当前 query 的虚拟时间位置，
是否能改善回访主体恢复和长时一致性。

它与 `retrieval_no_compression` 构成严格配对消融。两者都使用相机内参 `K` 的 FOV 检索、
检索两个完整 4-latent chunk，并且不进行 retrieval-time compression。

## 2. 位置分配

模型仍使用连续的 `4+8+8` tri-region：

```text
0..3       4..7              8..11             12..15    16..19
sink       retrieval-left    retrieval-right    recent    current query
```

FOV 距离越小表示越相关。假设被选中的两个 chunk 为 A、B，并且 A 比 B 更相关：

```text
retrieval_no_compression:
    按源时间顺序分配 4..7、8..11

retrieval_no_compression_relevance_order:
    B（次相关） -> 4..7
    A（最相关） -> 8..11
```

每个 chunk 内部的四个 latent 保持原始时间顺序。实现不是单纯交换 K/V 张量：payload 顺序
改变后，[`dykv_rope.py`](../Wan21/wan/modules/dykv_rope.py) 会根据新的 chunk 顺序重新计算
`target_start - source_start`，因此最高相关 chunk 获得真正更靠近 query 的 RoPE 相对位置。

## 3. 代码流程

1. [`dykv_fov.py`](../Wan21/pipeline/dykv_fov.py) 仍按 FOV distance 从小到大排名，并选满
   8 个 latent；选取算法没有改变。
2. [`dykv_runtime.py`](../Wan21/pipeline/dykv_runtime.py) 从排名中取出同一 selected set，然后
   反转被选块的相关性顺序。因为 RoPE materialization 从 slot 4 向右进行，最相关块最后
   materialize，最终占据 `8..11`。
3. [`dykv_memory.py`](../Wan21/pipeline/dykv_memory.py) 在该 case 下保留 runtime 传入顺序；
   其他所有已有 case 继续按源时间排序，行为不变。
4. [`dykv_rope.py`](../Wan21/wan/modules/dykv_rope.py) 依次将两个块 rebase 到 `4..7` 和
   `8..11`。V 与 K 使用相同 payload 顺序，但 V 本身不做旋转。

该第一版只支持无压缩、非 packed retrieval，避免把动态压缩、变长 chunk 和 packing
策略混入位置顺序消融。

## 4. 日志检查

每次 retrieval event 新增或明确记录以下字段：

| 字段 | 含义 |
| --- | --- |
| `selected_frame_starts` | 被选 chunk，保持选择器原有的源时间顺序 |
| `ranked_candidate_block_ids` | FOV 从高相关到低相关的候选排名 |
| `materialized_frame_starts` | 实际进入 retrieval payload 的 chunk 顺序 |
| `materialized_virtual_chunk_starts` | 对应的 RoPE 起点，正常为 `[4, 8]` |
| `retrieval_order` | 新 case 为 `relevance_near_query` |

若最高相关块的源起点是 4、次相关块为 12，正确日志应满足：

```json
{
  "selected_frame_starts": [4, 12],
  "materialized_frame_starts": [12, 4],
  "materialized_virtual_chunk_starts": [4, 8],
  "retrieval_order": "relevance_near_query"
}
```

## 5. 运行命令

使用同一 seed 成对生成多个典型视频：

```bash
conda activate minwm-fa
CUDA_VISIBLE_DEVICES=1 \
MBENCH_ROOT=~/research/datasets/MBench-Data/MBench-A \
ASSIGNMENTS="$PWD/Wan21/prompts/mbench_typical_4.jsonl" \
CASES=retrieval_no_compression,retrieval_no_compression_relevance_order \
NUM_OUTPUT_FRAMES=100 \
SEED=0 \
MASTER_PORT=29501 \
OUTPUT_ROOT="$PWD/output/mbench_typical_4_retrieval_order_seed0" \
MODEL_PREFIX=minwm_typical4_retrieval_order \
bash Wan21/scripts/inference/run_dykv_cases.sh
```

两组必须保持 checkpoint、`ASSIGNMENTS`、`NUM_OUTPUT_FRAMES` 和 `SEED` 一致。建议先观察
相机返回旧视角后的主体恢复、背景结构漂移和转向边界连续性，再结合 retrieval 日志确认
差异是否发生在实际启用检索的尾部 chunk。
