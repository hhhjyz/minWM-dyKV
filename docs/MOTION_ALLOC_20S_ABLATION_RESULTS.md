# Fixed-Budget Motion Allocation：20s × 30 消融结果

## 1. 结论摘要

本轮完成了 4 个 case、每个 30 条 20s loop-closure 视频的严格配对实验：

1. `retr16_compression_r033`；
2. `motion_alloc_cam_4chunk`；
3. `motion_alloc_cam_content_4chunk`；
4. `motion_alloc_cam_content_prerope_4chunk`。

四个 case 的 prompt index、trajectory、seed 和 initial-noise fingerprint 逐样本完全一致。所有 case
在 3 个历史 chunk 时使用 `6F=9360` token，在 4 个历史 chunk 后使用 `8F=12480` token，因此
没有总 retrieval budget 混杂。

主要结论是：

- 固定 4-chunk 公平预算后的 camera allocation **只带来局部改善，不是全面改善**。整体 closure
  SSIM 显著提高，但 pose LPIPS 显著变差；收益主要集中在 translation-only，mixed motion 的
  perceptual 指标反而下降。
- 当前 camera+content 补偿 **整体没有显著收益**。raw V content score 在 90.5% 的 non-anchor
  上高于 camera score，却平均只重分配 `568` token（`8F` 的 4.55%）；说明分数尺度没有校准，
  同时各帧 content score 过于相近。
- 完全 pre-RoPE K **不应作为默认 novelty descriptor**。它在 rotation-only 和 mixed-motion 上
  显著降低 pose PSNR。代码检查确认当前实现同时移除了 temporal 和 H/W spatial RoPE，而不是
  只去除 temporal phase；实验不能支持“raw K 一定比 cached RoPE K 更好”。

所以当前推荐保留 fixed 4-chunk/8F 基础设施；camera allocation 可继续研究，但不要将 content 或
raw pre-RoPE case 设为默认。下一步优先实现 spatial-only RoPE descriptor，并校准 content residual。

## 2. 实验协议

| 项目 | 设置 |
| --- | --- |
| 数据 | `demos_loop_closure` 20s，30 prompts |
| latent / decoded frames | 80 / 317 |
| seed | base seed 0，实际 sample seed = prompt index |
| retrieval | FOV ranking，最多 4 个完整历史 chunk |
| 总预算 | 历史 4 chunk 时固定 `8F=12480` |
| 顺序 | 原始 KV/source-frame 顺序，non-anchor 允许跨 virtual slot |
| 指标 | MAG/LPIPS matching、pose matching、exact endpoint closure |
| 图像评估 | width 256，路径两侧各最多 96 帧，LPIPS-Alex |
| 统计 | 相邻 stage paired delta，10,000 次 bootstrap 95% CI，方向归一化胜率 |

这里的 PSNR/SSIM/LPIPS 使用同一生成视频 outbound path 作为内部 pseudo-reference，衡量的是
loop closure / visual memory consistency，不是相对真实视频的 fidelity。

## 3. 四个 case 的总体均值

| Case | MAG PSNR ↑ | MAG SSIM ↑ | MAG LPIPS ↓ | Pose PSNR ↑ | Pose SSIM ↑ | Pose LPIPS ↓ | Closure PSNR ↑ | Closure SSIM ↑ | Closure LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `retr16_compression_r033` | 12.2046 | 0.2501 | 0.3367 | 10.7493 | 0.1760 | 0.4423 | 9.4598 | 0.1186 | 0.5624 |
| `motion_alloc_cam_4chunk` | 12.2718 | 0.2525 | 0.3372 | 10.7483 | 0.1775 | 0.4450 | 9.4743 | 0.1222 | 0.5666 |
| `motion_alloc_cam_content_4chunk` | 12.2620 | 0.2529 | 0.3367 | 10.7569 | 0.1776 | 0.4443 | 9.4780 | 0.1224 | 0.5670 |
| `motion_alloc_cam_content_prerope_4chunk` | 12.1884 | 0.2499 | 0.3398 | 10.7036 | 0.1761 | 0.4460 | 9.4378 | 0.1204 | 0.5665 |

均值只能描述结果，下面的相邻 paired comparison 才用于判断每一步是否有效。

## 4. 相邻阶段 paired 结果

### 4.1 Camera allocation vs fixed 1/3

| Metric | Paired Δ | Bootstrap 95% CI | Win rate | 判断 |
| --- | ---: | ---: | ---: | --- |
| Pose PSNR | -0.0010 | [-0.0402, 0.0327] | 60.0% | 无显著差异 |
| Pose SSIM | +0.0015 | [-0.0004, 0.0036] | 56.7% | 无显著差异 |
| Pose LPIPS | +0.0027 | **[+0.0002, +0.0056]** | 40.0% | 显著变差 |
| Closure PSNR | +0.0145 | [-0.0592, 0.0786] | 73.3% | 无显著差异 |
| Closure SSIM | +0.0036 | **[+0.0008, +0.0064]** | 63.3% | 显著提高 |
| Closure LPIPS | +0.0042 | [-0.0008, 0.0097] | 40.0% | 无显著差异 |

分组后，translation-only 的 pose SSIM `+0.0045 [0.0001, 0.0092]`、closure SSIM
`+0.0075 [0.0032, 0.0123]` 显著提高。mixed-motion 的 closure LPIPS
`+0.0073 [0.0021, 0.0127]` 显著变差。camera allocation 更像是对平移有帮助、对混合运动
仍不稳，而不是统一优于固定 1/3。

### 4.2 Camera+content vs camera-only

整体六个 primary metric 的 CI 都跨过 0：pose PSNR `+0.0086`、pose LPIPS `-0.0007`，其余
接近 0，没有证据证明当前 content 补偿稳定有效。唯一清晰的 subgroup 信号是 mixed-motion
pose LPIPS `-0.0025 [-0.0057, -0.0002]`，但 translation-only 多项指标方向较差，不能据此
启用全局 content 补偿。

运行时诊断进一步说明当前尺度存在问题：

- 30 条视频共 450 个 retrieval event，两个 case 的 selected source set 450/450 完全一致；
- `q_content > q_camera` 出现在 4803/5310（90.5%）个 non-anchor frame；
- 但每个 event 平均只重分配 568 token，中位数 519，约为完整 `8F` 的 4.55%。

也就是说 content 分数普遍更大，却没有形成足够有区分度的跨帧权重；它更多是在轻微扰动分配，
而不是专门救回相机静止但物体明显运动的 latent。

### 4.3 Completely pre-RoPE K vs cached 3D-RoPE K

| Group / metric | Paired Δ | Bootstrap 95% CI | 判断 |
| --- | ---: | ---: | --- |
| All pose PSNR | -0.0533 | [-0.1099, 0.0015] | 接近显著变差 |
| Rotation-only pose PSNR | -0.0731 | **[-0.1599, -0.0129]** | 显著变差 |
| Rotation-only closure PSNR | -0.1071 | **[-0.2242, -0.0001]** | 显著变差 |
| Translation-only pose PSNR | +0.0567 | [-0.0390, 0.1454] | 无显著差异 |
| Mixed-motion pose PSNR | -0.1048 | **[-0.1940, -0.0222]** | 显著变差 |

完全 pre-RoPE case 的总体 MAG PSNR 从 12.2620 降至 12.1884，MAG LPIPS 从 0.3367
升至 0.3398，也与 pose 结果一致。

直接原因不是“temporal pollution 不存在”，而是当前 descriptor 改得过多：

```text
causal_rope_apply = temporal RoPE + height spatial RoPE + width spatial RoPE
```

保存 `norm_k(k(x))` 会同时删除三部分。H/W phase 对区分空间 token 可能有用，尤其旋转时空间
对应关系显著改变。这个实验否定的是 completely raw K，不是否定“只移除 temporal phase”。

另外，camera+content 与 pre-RoPE 在前两个 retrieval event 的 score 和 token length 完全一致；
从 frame 28 开始，token selection 改变了生成状态，继而使后续 V content score 和 allocation
分叉。这是正常的 end-to-end effect，也说明不能用后期 allocation 不同来声称实验输入不公平。

## 5. 后续优化方向

### P0：保留 spatial RoPE，只移除 temporal RoPE

下一版 novelty descriptor 应为：

```text
K_descriptor = spatial_rope_hw(pre_rope_K)
```

或从 cached 3D-RoPE K 中只逆旋 temporal 分量。不要再次使用 completely raw K。实现后先比较：

- cached 3D-RoPE K；
- spatial-only RoPE K；
- completely raw K（仅作为已知较差诊断）。

三者必须在同一个 frozen bank 上输出 token-index Jaccard、spatial coverage 和 layer-0 query
similarity，再决定是否生成视频。

### P0：把 content score 改成 camera-conditioned residual

不要直接比较未校准的 `max(q_camera, q_content)`。建议先从 frozen rollout 拟合或非参数估计
正常相机运动造成的 V distance 基线：

```text
q_excess = clamp(q_content - median_content_given_camera_motion, 0, 1)
q_final  = q_camera + lambda * q_excess
```

第一轮只保留一个固定 `lambda`，避免重新引入大量超参数。小动态物体会被全图 mean 稀释，内容
聚合应同时记录 mean 与 top-p robust mean；只在离线诊断明确提升动态区域召回后选择其中一个。

### P1：保留 fixed 4-chunk/8F，但让 mixed motion 更稳定

当前 translation-only 有 SSIM 收益，而 mixed motion perceptual 指标变差。应重点检查 projected
camera score 在旋转+平移组合时是否过度集中到少数 latent。先记录每 event 的 allocation entropy、
最大 latent 占比和 action-boundary 两侧 token 比，不新增新的压缩档位。

## 6. 更简单的实验流程

后续每个想法都按三层漏斗执行，避免每次直接跑 30×20s：

| 层级 | 数据 / 状态 | 只回答的问题 | 通过标准 |
| --- | --- | --- | --- |
| A. Frozen-state planner replay | 同一批 archived KV，不回写生成 | 分数/排序是否真的改变正确 token | 同 source、同预算；动态区域 recall 或 query similarity 提高 |
| B. 8-video stratified smoke | rotation 2 + translation 2 + mixed 4 | 是否有明显方向错误或运行 bug | 无 subgroup 系统性退化；runtime contract 全通过 |
| C. 30-video confirmation | 当前 20s×30 | 效果是否可重复 | primary metric paired CI 不跨 0，且胜率方向一致 |

每个 stage 只与直接前驱比较：

1. fixed 1/3 → camera allocation；
2. camera → calibrated camera+content residual；
3. cached 3D-RoPE novelty → spatial-only RoPE novelty。

不要把第三步直接与 fixed baseline 比，否则无法判断改善/退化来自 allocation 还是 descriptor。
primary endpoint 建议预先固定为 pose PSNR + pose LPIPS；closure SSIM 作为次要指标，MAG matching
作为诊断。这样可以避免不同指标一好一坏时临时选择有利结论。

## 7. 产物

- 完整视频与 manifest：`output/motion_alloc_20s_ablation/`（默认被 Git 忽略）；
- 每 case 指标：`<case>/eval/loop_closure_metrics.{json,csv}`；
- 完整 paired 表：`output/motion_alloc_20s_ablation/PAIRED_ABLATION_REPORT.{md,json}`；
- 复现/统计脚本：`Wan21/scripts/evaluation/analyze_motion_ablation.py`。

