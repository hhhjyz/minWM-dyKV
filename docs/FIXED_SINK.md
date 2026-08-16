# 所有 Case 的固定 Sink

## 1. 统一约定

所有注册实验 case 都采用相同的固定 sink：

```text
sink_mode   = fixed
sink_frames = 4 latent
```

“固定”表示在线 KV cache 滚动时始终保留生成序列最初的 4 个 latent，只有 sink 之后的
区域会被滚动淘汰。本项目不使用 periodic、random、pose-bank 或其他会替换 sink 内容的
策略。

该约定由 `Wan21/dykv_cases.py` 集中定义，不作为单独的 case 超参数。推理清单会记录
`sink_mode=fixed` 和 `sink_frames=4`，以便检查实验配置。

## 2. Baseline 与 dyKV 的公平布局

所有 case 的 attention 物理上限仍为 20 个 latent，但区域用途不同：

```text
baseline:
[ fixed sink: 4 ][ empty retrieval: 0 ][ rolling local including current: 16 ]

dyKV cases:
[ fixed sink: 4 ][ retrieval: at most 8 ][ local including current: 8 ]
```

因此 `baseline` 与 dyKV 的总 attention token 上限相同。区别是 baseline 将剩余 16 帧都
用于最近上下文，而 dyKV 将其中 8 帧容量用于历史检索。

候选记忆选择会排除源帧 0--3，避免固定 sink 又以 retrieval 历史出现一次。

## 3. 与 RoPE Rebase 的关系

固定 sink 内容和 tri-region RoPE rebase 是两个不同概念：

- 本项目所有 case 都固定保留最初 4 帧；
- 不启用 `minWM-back` 中独立的 `fixed_sink_rope_rebase`；
- baseline 从第 20 帧后的首个 chunk 起使用空 retrieval 的 `4+0+16` tri-region rebase，
  sink 映射到 0--3、历史 local 映射到 4--15、当前 query/K 映射到 16--19；
- dyKV 的三区域 RoPE 中，sink K 仍位于训练位置 0--3，retrieval 映射到 4--11，local
  与 query 映射到 12--19。

因此 baseline 与 dyKV 都不会在长视频中使用超过训练范围的绝对 RoPE；二者仍保持相同的
20-latent attention 上限，主要差异是 local 与 retrieval 的容量分配。

## 4. 当前 Case

| Case | Sink | 其余布局 |
| --- | --- | --- |
| `baseline` | 固定最初 4 帧 | retrieval 0 + rolling local 16；RoPE `4+0+16` |
| `retrieval_no_compression` | 固定最初 4 帧 | retrieval 8 + local 8 |
| `worldkv_pose_no_compression` | 固定最初 4 帧 | retrieval 8 + local 8 |
| `fixed_novelty` | 固定最初 4 帧 | retrieval 8 + local 8 |
| `yaw_intrinsics` | 固定最初 4 帧 | retrieval 8 + local 8 |
| `packed_chunks` | 固定最初 4 帧 | retrieval 8 + local 8 |
| `packed_chunks_latent` | 固定最初 4 帧 | retrieval 8 + local 8 |
| `predecessor_chunks` | 固定最初 4 帧 | retrieval 8 + local 8 |
| `predecessor_chunks_latent` | 固定最初 4 帧 | retrieval 8 + local 8 |
| `predecessor_query_backfill` | 固定最初 4 帧 | retrieval 8 + local 8 |
| `retr8_compression_r050` | 固定最初 4 帧 | retrieval 8 + local 8 |
| `retr12_compression_r050` | 固定最初 4 帧 | retrieval 8 + local 8 |
| `retr16_compression_r033` | 固定最初 4 帧 | retrieval 8 + local 8 |

新增 case 必须继承同一个固定四帧 sink。注册表测试会拒绝无意中改变该约定的 case。
