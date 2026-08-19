# 动态压缩 case 退化根因分析

## 1. 问题概述

在 `output/motion_novelty_loop_10s_honest_seed0` 的 loop closure 评测中，所有 honest
dyKV case 均超过 baseline，但**动态压缩 case 始终不如无压缩 case**：

| Case | Pose PSNR | Δ vs no\_comp | Endpoint PSNR | Δ vs no\_comp |
| --- | ---: | ---: | ---: | ---: |
| `retrieval_no_compression_honest` | **12.922** | — | **11.657** | — |
| `retr16_compression_r033_honest` | 12.621 | -0.300 | 11.303 | -0.354 |
| `motion_novelty_backfill_honest` | 12.556 | -0.365 | 11.097 | -0.560 |
| `motion_novelty_duplicate_honest` | 12.528 | -0.394 | 11.184 | -0.473 |
| `motion_novelty_unfilled_honest` | 12.481 | -0.440 | 11.196 | -0.461 |

退化在所有 30 个视频上均匀分布，不集中在个别样本。本文基于
`dykv_summaries.jsonl` 诊断日志与压缩代码路径分析根因。

## 2. 实验配置

所有 case 使用相同的基础设置：

- tri-region RoPE 布局 `[sink=4, retrieval=8, local=8]`，共 20 个虚拟位置
- `retrieval_rope_mode = "honest"`（绝对时间位置，不 rebase）
- FOV 检索，`sink_frames = 4`，`frame_tokens = 1560`
- 检索 token 预算 = `memory_frames × frame_tokens = 8 × 1560 = 12480`

各 case 的关键差异：

| Case | packing\_mode | retrieval\_frames | keep\_ratio | compression\_mode |
| --- | --- | ---: | ---: | --- |
| `retrieval_no_compression` | none | 8 | —（不压缩） | none |
| `retr16_compression_r033` | fixed\_worldkv | 16 | 0.333 | fixed\_novelty |
| `motion_novelty_backfill` | motion\_novelty\_backfill | 16 | 0.5 | motion\_novelty |
| `motion_novelty_duplicate` | motion\_novelty\_duplicate | 16 | 0.5 | motion\_novelty |
| `motion_novelty_unfilled` | motion\_novelty\_flat | 16 | 0.5 | motion\_novelty |

## 3. Token 预算实测对比

以下数据取自 video 0（`k*19,i*19,k*1`）的 `dykv_summaries.jsonl` 第一个 retrieval
event（`current_frame = 20`）：

| Case | 实际 token | 预算 | 占比 | anchor 帧 | 非 anchor 帧 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `retrieval_no_compression` | 12480 | 12480 | 100% | 1560 × 8 | 1560 × 0 |
| `retr16_compression_r033` | 9360 | 12480 | 75% | 1560 × 3 | 520 × 9 |
| `motion_novelty_backfill` | 12480 | 12480 | 100% | 1560 × 3 | 70–911 × 9 |
| `motion_novelty_duplicate` | 12480 | 12480 | 100% | 1560–3143 × 3 | 70–248 × 9 |
| `motion_novelty_unfilled` | 6195 | 12480 | **50%** | 1560 × 3 | 70–248 × 9 |

`retrieval_no_compression` 检索 8 个完整帧（每帧 1560 token），无 anchor 概念。
其余 case 检索 16 帧 = 4 个 4 帧 chunk，每 chunk 保留 1 个完整 anchor + 3 个压缩非 anchor。

### 3.1 `retr16_compression_r033` 的瞬态欠填

`r033` 在 `current_frame = 20` 时只取了 9360 token（75% 预算），原因是此时只有 3 个
可检索 chunk：

```text
chunk 0 (frames  0–3)  → sink 区，不参与检索
chunk 1 (frames  4–7)  → candidate ✓
chunk 2 (frames  8–11) → candidate ✓
chunk 3 (frames 12–15) → candidate ✓
chunk 4 (frames 16–19) → local 区（正在生成），不参与检索
```

3 chunks × (1560 + 3 × 520) = 3 × 3120 = 9360。从 `current_frame = 24` 起，chunk 4
完成并进入 candidate，始终选 4 chunks，4 × 3120 = 12480 = 100% 填满。

| current\_frame | candidates | selected | compressed | fill |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 3 | 3 | 9360 | 75% |
| 24 | 4 | 4 | 12480 | 100% |
| 28 | 5 | 4 | 12480 | 100% |
| 32 | 6 | 4 | 12480 | 100% |

这是生成前期的瞬态行为，不是系统性问题。从 frame 24 起所有压缩 case 与
`retrieval_no_compression` 拥有相同 token 预算，但压缩 case 依然更差。

### 3.2 `motion_novelty_unfilled` 的持续欠填

`unfilled` 在所有 frame 都只用了约 50–99% 预算：

| current\_frame | compressed | fill |
| ---: | ---: | ---: |
| 20 | 6195 | 50% |
| 24 | 8260 | 66% |
| 28 | 10325 | 83% |
| 32 | 12390 | 99% |

`unfilled` 模式不主动填充剩余 slot，严格劣于 `backfill` 和 `no_compression`。

## 4. 三个核心问题

### 4.1 问题 A：novelty token 选择方向与 loop closure 需求相反

**所有压缩方法都保留与 anchor 最不相似的 token（novelty），丢弃最相似的 token
（shared content）。**

`dykv_memory.py` 的 `compress_retrieved_kv` 函数（line 387）：

```python
indices = similarity.topk(keep_tokens, dim=1, largest=False).indices
#                                                ^^^^^^^^^^^^^
#                                                保留最不相似的 token
```

`dykv_packing.py` 的 `_fixed_worldkv_indices` 函数（line 743）：

```python
output.append(
    score.topk(keep_tokens, largest=False).indices.sort().values.to(torch.long)
    #                        ^^^^^^^^^^^^^
    #                        同样保留最不相似的 token
)
```

`dykv_motion_novelty.py` 的 `_novelty_order` 函数（line 179）：

```python
orders.append(torch.argsort(score, stable=True).to(torch.long))
# 后续取 novelty_order[:keep_tokens]，即最不相似的 token
```

**Loop closure 需要模型识别"回到了之前看过的位置"——这依赖共享内容（与 anchor 相似的
token），而非 novelty。** 压缩恰好丢弃了 closure 识别所需的信息。

`motion_novelty_backfill` 的 backfill 也按 novelty 顺序恢复 token
（`omitted_indices_in_novelty_order[:backfill_count]`），先恢复次 novel 的 token，
最后才恢复最 similar 的——仍然 novelty 优先。

### 4.2 问题 B：motion overlap 在 chunk 内部计算，与当前 query 无关

`motion_novelty` 的 keep\_ratio 由 `1.0 - overlap` 决定，但 overlap 是 **anchor 与
同 chunk 内其他帧** 的几何重叠，不是当前 query 与历史帧的重叠。

实测 video 0 的 motion overlap 和 keep\_ratio：

| 帧（距 anchor） | overlap | keep\_ratio | 保留 token |
| --- | ---: | ---: | ---: |
| frame 0 (anchor) | 0.0 | 1.0 | 1560 |
| frame 1 (3°) | 0.955 | **0.045** | **70** |
| frame 2 (6°) | 0.881 | 0.119 | 187 |
| frame 3 (9°) | 0.841 | 0.159 | 248 |

相机每帧转 3°，chunk 内 overlap 高达 84–96%，导致非 anchor 帧被压缩到 4.5–16%。
一个距 anchor 3° 的历史帧可能恰好处于 closure 位置——但它的 token 被压缩到 4.5%，
模型几乎看不到它。

### 4.3 问题 C：anchor 是 chunk 中时间最远的帧，最接近 turnaround 的帧被压缩最多

每个 4 帧 chunk `[t, t+1, t+2, t+3]` 中，anchor = frame t（最早），被完整保留。
frame t+3（最接近 turnaround、对 closure 最有用）被压缩最多（仅 16% = 248 tokens）。

**对 closure 最关键的帧，信息损失最大。**

## 5. 各压缩 case 的额外问题

### 5.1 `motion_novelty_unfilled`：浪费 KV 预算

`unfilled` 模式不填充剩余 slot。在 frame 20 只用了 6195/12480 = 50% 的预算，剩余
50% 完全空闲。这严格劣于 `retrieval_no_compression`（8 帧完整 = 12480 token）。

### 5.2 `motion_novelty_duplicate`：用重复内容填充预算

`duplicate` 模式通过重复高相关 chunk 的 anchor token 来填充预算。实测 video 0
frame 20 的 anchor token 达到 3143（= 1560 × 2 + 23），说明 anchor 被重复了两次。
模型看到重复内容而非新内容，有效唯一信息极低，可能误导注意力。

### 5.3 `motion_novelty_backfill`：backfill 优先级错误

`backfill` 确实填满预算（12480），但 backfill 按 novelty 顺序恢复 omitted token
（`omitted_indices_in_novelty_order`），先恢复次 novel 的 token，最后才恢复最
similar 的。最终 token 组合仍然是 novelty-heavy，shared content 不足。

实测 video 0 frame 20 的 `actual_tokens_per_frame`：

```text
chunk 1: [1560,  70, 187, 248]  ← 几乎没有 backfill（距离当前最远）
chunk 2: [1560, 823, 881, 911]  ← 大量 backfill（距离当前较近）
chunk 3: [1560, 1560, 1560, 1560]  ← 完全 backfill（距离当前最近）
```

backfill 优先分配给距离当前 query 近的 chunk，这合理；但即使填满，token 组合仍是
novelty 优先（先加 novel 的 omitted token），shared content 仍然不足。

### 5.4 `retr16_compression_r033`：固定压缩率不考虑几何相关性

所有非 anchor 帧统一保留 33%（520 token），不随相机运动调整。在相机快速旋转的场景中，
33% 可能不足以保留足够的识别信息；在相机几乎静止的场景中，33% 又浪费了预算。

## 6. 逐视频退化分析

以下对比各压缩 case 相对 `retrieval_no_compression_honest` 的 Pose PSNR 差异，
列出退化最严重的 5 个视频：

### `retr16_compression_r033_honest`（均值 -0.300）

| video | Δ pose PSNR | Δ end PSNR | 轨迹 |
| ---: | ---: | ---: | --- |
| 0 | -1.17 | -1.00 | `k*19,i*19,k*1` |
| 24 | -1.12 | -1.66 | `w*19,s*19,w*1` |
| 22 | -0.94 | -2.84 | `a*10,w*9,s*9,d*10,a*1` |
| 26 | -0.83 | +0.59 | `l*12,a*7,d*7,j*12,l*1` |
| 19 | -0.70 | -1.81 | `j*19,l*19,j*1` |

### `motion_novelty_unfilled_honest`（均值 -0.440）

| video | Δ pose PSNR | Δ end PSNR | 轨迹 |
| ---: | ---: | ---: | --- |
| 24 | -1.31 | -2.12 | `w*19,s*19,w*1` |
| 22 | -1.27 | -3.56 | `a*10,w*9,s*9,d*10,a*1` |
| 26 | -1.18 | -0.31 | `l*12,a*7,d*7,j*12,l*1` |
| 0 | -1.17 | -0.78 | `k*19,i*19,k*1` |
| 6 | -1.06 | -0.82 | `a*19,d*19,a*1` |

退化最严重的视频（0, 22, 24）是单方向长轨迹（`k*19,i*19`、`a*10,w*9,s*9`、
`w*19,s*19`），这些轨迹的 closure 帧距 anchor 最远，受问题 C 影响最大。

## 7. 建议修复方向

### 7.1 反转 token 选择：保留 shared content

将 `largest=False` 改为 `largest=True`，保留与 anchor **最相似**的 token。Loop closure
识别依赖共享内容，novelty token 对 closure 无益。

影响位置：
- `dykv_memory.py:387` — `compress_retrieved_kv`
- `dykv_packing.py:743` — `_fixed_worldkv_indices`
- `dykv_motion_novelty.py:179` — `_novelty_order`

### 7.2 overlap 应计算 query → historical

当前 motion overlap 计算 anchor 与同 chunk 内其他帧的重叠。应改为计算**当前 query
与每个历史帧**的重叠，使压缩率反映当前视角的真实信息需求。

距当前 query 视角近的历史帧应保留更多 token（高 overlap → 高 keep\_ratio），距当前
远的帧可以更激进压缩。

### 7.3 anchor 应选最接近 turnaround 的帧

将 anchor 从 chunk 第一帧改为**最后一帧**（frame t+3），使最接近 turnaround、对
closure 最有用的帧被完整保留。

或者，anchor 应选 chunk 中**距当前 query 视角最近**的帧，而非固定选第一帧。

### 7.4 `unfilled` 应改为 `backfill` 或自适应填充

`motion_novelty_unfilled` 不应浪费 50% KV 预算。至少应改为 `backfill`；理想情况下
应根据剩余预算自适应提高 keep\_ratio，使 token 总量始终接近 12480。

### 7.5 `fixed_worldkv` 应支持自适应 keep\_ratio

当可用 chunk 数 < 4 时（如 frame 20 只有 3 chunks），应自适应提高 keep\_ratio 以
填满预算。3 chunks 时 keep\_ratio ≈ 0.556 可填满 12480，而非固定 0.333 导致 75%
填充。

## 8. 验证计划

1. **反转 token 选择**（7.1）：修改 `largest=True`，重跑 10s loop closure，对比
   PSNR 变化
2. **query-aware overlap**（7.2）：修改 motion overlap 计算，重跑并对比
3. **anchor 选最后一帧**（7.3）：修改 anchor 选择逻辑，重跑并对比
4. 三个修复可独立验证，也可组合验证；建议先单独验证 7.1（影响最大、修改最小）

每个修复只需改几行代码，可用现有 `run_dykv_cases.sh` + `evaluate_loop_closure.py`
流程在 10s loop closure 上验证。

## 9. 对问题是否存在的复核结论

本节根据当前代码路径和 `output/motion_novelty_loop_10s_honest_seed0` 的 event 日志，
把原文中的判断分为三类：代码行为已经直接证实、设计与任务目标存在错配但还需要消融证实、
以及只是建议方向而不是已经证明的修复。

### 9.1 已直接证实的问题

| 问题 | 结论 | 证据 |
|---|---|---|
| `motion_novelty_unfilled` 欠填 | 确实存在 | `unfilled` 只物化 base segment；frame 20 使用 `6195/12480` token，未使用的预算不会自动补入 |
| motion overlap 不看当前 query | 确实存在 | `build_motion_chunk_plan` 使用 `poses[frame_offset]` 与 `poses[0]`，即 chunk 内 `Pi↔P0` |
| 首帧作为完整 anchor | 确实存在 | `frame_offset==0` 强制保留全部 `frame_tokens`，后三帧才执行几何比例 |
| fixed `r=1/3` 不随运动变化 | 确实存在 | `_fixed_worldkv_indices` 对所有非 anchor frame 使用同一个 `keep_ratio` |
| duplicate 引入重复 token | 确实存在 | `_duplicate_segments` 从已选 source token pool 重复生成 segment，不增加唯一历史信息 |
| backfill 仍按 novelty 顺序 | 确实存在 | backfill 拼接 `base_indices + omitted_indices_in_novelty_order[:count]` |

其中 `unfilled` 是有意保留的欠填消融，不应被描述为代码 bug；但它不适合作为 motion
novelty 的正式主方法。正式方法至少应使用 `backfill` 或其他明确的剩余预算策略。

### 9.2 真实存在但属于“任务目标错配”的问题

#### A. Novelty token 不等于 loop-closure token

当前实现确实保留与 anchor 最不相似的 token。这不是 WorldKV 语义下的实现错误：anchor
已经完整保存，novelty 选择的目标是减少历史 chunk 内的重复并增加新视野信息。

但是 loop closure 的目标不同：回到旧视角时，模型需要稳定的共享结构、主体轮廓和可重新
对应的纹理。因此，“保留 novelty”与“保留 closure 识别所需的 shared content”之间存在
明确的任务目标错配。

目前还不能据此得出“必须把 `largest=False` 改成 `largest=True`”的结论。直接全量反转会
变成 shared-only compression，可能损失真正的新视野内容。应将下面三种策略作为独立消融：

```text
novel-only   ：当前实现
shared-only  ：保留与 anchor 最相似 token
hybrid       ：固定比例分配 shared token 与 novel token
```

#### B. 当前 query 与 token 选择参考系不一致

当前几何比例使用 chunk anchor 作为参考，而 token novelty 也使用 anchor centroid；这使得
压缩计划主要回答“这个 frame 相对 chunk anchor 有多少新内容”，而不是“这个 frame 对当前
query 有多少可恢复信息”。

因此，原文建议 query-aware overlap 的方向是合理的，但不能只替换 overlap 的 pose 参数。
若只用 query 决定 token 数量，却仍用旧 anchor 决定具体 token，就会出现：

```text
query-aware token count + anchor-aware token identity
```

更一致的方案应同时评估 query 与 historical frame 的几何关系，以及 query-aware 的 token
相似度或 shared/novel 混合排序。

#### C. 首帧 anchor 可能压缩 query-nearest frame

在单向运动中，一个 chunk 的最后一帧通常比第一帧更接近当前 query。当前设计却让第一帧
完整保留，并让最后一帧承担最大的压缩比例。loop closure 中这会造成“最可能被重新访问的
frame 保存最少”的风险。

但 anchor 不能简单地无条件改成最后一帧：往返或混合轨迹中，时间上最后的 frame 不一定是
当前 query 视角最近的 frame；同时 anchor 改变会一起改变 centroid、novelty order 和 slot
分配。更稳妥的实验是比较：

```text
first-anchor
last-anchor
query-nearest-anchor
```

### 9.3 原文中尚未被证明的修复结论

下面这些说法不能直接当作已验证结论：

```text
“novelty token 一定错误”
“全部改成 shared token 就能恢复 loop closure”
“anchor 必须改成 chunk 最后一帧”
```

它们都需要在相同 checkpoint、相同 honest RoPE、相同 selected blocks 和相同 token budget
下单独验证。

特别是 `duplicate` 变好不能解释为“记忆覆盖增加”，因为 duplicate 不增加唯一 source token，
只会改变重复 token 的 attention 权重；`backfill` 变好也需要区分是新增 token 的数量作用，
还是 shared token 的内容作用。

## 10. 原分析中遗漏的实现因素

### 10.1 Novelty cosine 使用了带 temporal RoPE 的 K

在线 cache 中写入的是已经经过 temporal RoPE 的 `roped_key`。之后 `_novelty_order` 直接
读取 layer-0 K 计算 cosine。因此当前分数不是纯内容相似度，而是：

```text
content similarity + temporal RoPE phase difference
```

这个因素可能使 novelty order 随 frame position 改变，并且可能直接影响 loop closure。它应
优先于简单修改 `largest` 验证：需要保存或重建不带 temporal RoPE 的 descriptor，再比较
descriptor-based novelty 与当前 K-based novelty。

### 10.2 honest RoPE 是受控混杂，不是压缩独有问题

honest case 保留真实绝对 frame position，解决 fixed-slot 的 temporal lie，但模型训练时的
RoPE 窗口仍是 20 帧，超过 position 19 后属于 extrapolation。由于诊断中的 no-compression
和 compression 都使用 honest，这个因素不会破坏 honest 内部的压缩对比；但不能把其绝对
PSNR 直接解释为“纯压缩性能”。honest 与 fixed-slot 的结果也不能直接混合排名。

### 10.3 no-compression 不是严格的同历史覆盖对照

`retrieval_no_compression_honest` 只检索 8 个完整 latent；motion 和 fixed-WorldKV case
最多选择 16 个 source latent，再压缩到 8 latent 的 token budget。因此 no-compression 是
“8 帧完整历史上限”，不是与 motion case 完全相同 selected history 的 compression-only
对照。

判断压缩是否伤害质量时，应优先比较：

```text
retr16_compression_r033_honest
motion_novelty_unfilled_honest
motion_novelty_backfill_honest
motion_novelty_duplicate_honest
```

这些 case 都面向 16 个 source latent 和相同 retrieval budget；
`retrieval_no_compression_honest` 用于衡量 8 帧完整历史的上限。

## 11. 修复与验证优先级

建议按以下顺序做独立实验，每次只改变一个主要变量：

1. **去除 temporal RoPE 对 descriptor 的影响**：保存 pre-RoPE layer-0 descriptor，重新计算 novelty order；
2. **token 选择方向消融**：`novel-only / shared-only / hybrid`；
3. **query-aware geometry**：以当前 query 与 historical frame 的关系计算比例，并同步设计 query-aware token ranking；
4. **anchor 选择消融**：first / last / query-nearest；
5. **填充协议比较**：projected `unfilled / backfill`，并单独报告 unused token；
6. **fixed ratio 的预算自适应**：另注册新 case，不修改 `retr16_compression_r033` 的固定比例语义。

每组实验都应固定：

```text
checkpoint
prompt / trajectory
sample seed
retrieval_rope_mode
selected block IDs
token budget
```

并同时报告：Pose PSNR、endpoint PSNR、实际 token 数、unused token、source multiplicity、
retrieval latency 和峰值显存。只有当 token 数和 selected history 也被记录后，才能判断质量
差异来自压缩内容、历史覆盖范围，还是单纯的 attention budget。
