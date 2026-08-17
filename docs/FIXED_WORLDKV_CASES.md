# minWM-back 固定压缩率与 Chunk 数对照

## 1. 目的

本模块复现 `minWM-back` 的 A--D 固定预算比较，用来回答：在相同或近似相同的物理
retrieval token 预算下，检索更多原始历史 chunk 是否优于保留较少但更完整的历史。

每个四帧 chunk 使用 WorldKV 风格的 anchor + novelty 压缩：

- 第一个 latent 是 anchor，完整保留 1560 个 token；
- 后三个 latent 分别保留与 anchor key 中心余弦相似度最低的 token；
- 固定比例 `r` 只作用于三个非 anchor latent；
- CPU KV bank 始终保存无损 KV，压缩只发生在 retrieval materialize 阶段。

## 2. 当前对照

| 对照 | 当前 Case | 原始覆盖 | Chunk 数 | 非 anchor 比例 | 单 chunk 等价容量 | 总物理容量 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | `retrieval_no_compression` | 8 latent | 2 | 1 | 4 latent | 8 latent |
| B | `retr8_compression_r050` | 8 latent | 2 | 1/2 | 2.5 latent | 5 latent |
| C | `retr12_compression_r050` | 12 latent | 3 | 1/2 | 2.5 latent | 7.5 latent |
| D | `retr16_compression_r033` | 16 latent | 4 | 1/3 | 2 latent | 8 latent |

对于每帧 1560 token，四组实际 retrieval token 分别是 12480、7800、11700、12480。
A 与 D 具有完全相同的物理预算，但 D 覆盖两倍源历史；B 与 C 使用相同压缩率，用于检查
压缩节省的容量是否应该转换成更多 chunk。

## 3. 对当前三区域设计的适配

`minWM-back` 的旧 A--D 脚本关闭 tri-region RoPE，并允许普通 attention 窗口按物理 token
长度组窗。当前项目必须保持连续 `sink 4 + retrieval 8 + local 8`，因此做如下适配：

1. 完整 anchor 各占一个 retrieval 虚拟槽；
2. 非 anchor segment 按大小做 first-fit decreasing；
3. 不同源 chunk 的压缩 segment 可以共享同一虚拟槽，但每槽最多 1560 token；
4. 每个 segment 使用自己的源帧 ID，单独执行 time-RoPE rebase；
5. 空间 RoPE 不改变，最终 retrieval 物理 token 永远不超过 12480。

当前实现按每个非 anchor latent 固定保留 `ceil(r×1560)` 个 token，并由第 0 层生成跨层
共享的新颖性索引。这与旧脚本的 pooled、逐层独立选 token 实现不完全相同；本对照严格
保持的是原始 chunk 数、anchor 语义、固定比例和总物理预算，同时避免层间 segment 边界
不一致或同一源帧跨多个时间槽。报告实验时应注明该适配，不能将结果描述为旧实现的逐
bit 复现。

## 4. 运行方式

只运行 A--D：

```bash
conda activate minwm-fa
CASES=retrieval_no_compression,retr8_compression_r050,retr12_compression_r050,retr16_compression_r033 \
OUTPUT_ROOT=output/fixed_worldkv_a_d \
bash Wan21/scripts/inference/run_dykv_cases.sh
```

与动态方法一起比较：

```bash
conda activate minwm-fa
CASES=retrieval_no_compression,retr12_compression_r050,retr16_compression_r033,yaw_intrinsics,packed_chunks_latent \
OUTPUT_ROOT=output/fixed_vs_dynamic \
bash Wan21/scripts/inference/run_dykv_cases.sh
```

建议至少记录源 latent 数、实际 retrieval token、虚拟槽占用数、检索耗时、显存峰值和
MBench/闭环质量。真实 checkpoint 与 MBench 结果目前均为待运行。
