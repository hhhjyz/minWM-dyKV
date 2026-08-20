# Loop Closure 评测：MAG 分割匹配与轨迹匹配

> 状态：评测器与汇总脚本已实现并跑完 `motion_novelty_loop_10s_seed0` 下全部 8 个 case
> 的 30 条 10s 视频。结果见
> [`output/motion_novelty_loop_10s_seed0/loop_closure_report.md`](../output/motion_novelty_loop_10s_seed0/loop_closure_report.md)。

## 1. 目标

minWM 的 loop-closure 轨迹是"去程 + 逆序回程 + 一个尾部动作"，相机在第 0 帧和倒数
第二帧精确重访同一位姿。生成视频不一定视觉闭环，因此需要量化评测。

闭环评测把**同一条 rollout 的去程半段当作回程半段的内部伪参考**，不需要外部 ground-truth
视频。它衡量的是视觉记忆/闭环一致性，不是对真实视频的重建 fidelity。

## 2. 数据来源

| 产物 | 路径 | 作用 |
| --- | --- | --- |
| 视频 | `output/motion_novelty_loop_10s_seed0/{case}/*.mp4` | 30 条/case，157 帧，16 FPS |
| 生成清单 | `{case}/generation_manifest.jsonl` | `prompt_index`、`trajectory`、`output_path`、`status` |
| 闭环 manifest | `Wan21/prompts/demos_loop_closure/manifest.json` | `closure_pair_decoded`、`closure_pair_latent` |
| 轨迹解析 | `Wan21/wan_utils/camera_trajectory.py:parse_trajectory` | action-string → (T,4,4) w2c 位姿 |

10s 轨迹的 `closure_pair_decoded = [0, 152]`，即解码第 0 帧和第 152 帧相机位姿完全相同。
`time_mapping = "decoded_frames = 1 + 4 * (latent_frames - 1)"`，40 latent → 157 decoded。

## 3. 评测流程

### 3.1 闭环区间切分

对 `closure_pair = [closure_start, closure_end] = [0, 152]`：

```text
turnaround      = closure_start + (closure_end - closure_start) // 2 = 76
reference_full  = [0, 76)      # outbound 半段（内部伪参考）
revisit_full    = (76, 152]    # revisit 半段（待评测）
```

turnaround 帧被排除，避免边界帧的平凡匹配膨胀指标。两个半段等长（76 帧）。

若 `--max-frames-per-segment` 小于 76，则在半段内均匀子采样；默认 96 > 76，即使用全部帧。

### 3.2 匹配模式

对每个 revisit 帧，需要在 outbound 半段中找到它的"对应帧"。两种匹配方式：

#### 3.2.1 MAG 风格（LPIPS 匹配）

来自 [MAG-Bench](../../MAG/evaluate/vae_metrics.py) 与 minWM-back
`evaluate_loop_closure.py` 的口径：

```text
对每个 revisit 帧 r:
    matched(r) = argmin_{j in outbound} LPIPS(r, outbound[j])
```

PSNR/SSIM/LPIPS 在这些 LPIPS 最近匹配对上计算。容忍不同遍历速度，但可能匹配到时序
错误的帧（LPIPS 最近 ≠ 几何对应）。

#### 3.2.2 轨迹匹配（相机位姿匹配）

```text
对每个 revisit 帧 r:
    latent_r = round(decoded_r / 4)          # 解码帧 → 最近 latent 帧
    matched(r) = argmin_{j in outbound} SE3_distance(pose[latent_r], pose[latent_j])
```

其中 `SE3_distance = ||t_rel|| + angle_deg(R_rel)`（平移范数 + 旋转角度）。

对精确 string-loop 轨迹，回程是去程的逆序反向，因此轨迹匹配退化为**确定性的时间镜像**：
`matched_decoded = closure_end - decoded`。不依赖 LPIPS，不需要 GPU 做匹配。

PSNR/SSIM 在这些位姿匹配对上计算；LPIPS 也在同一匹配对上测量（不再重新取 min）。

### 3.3 指标

| 指标 | 含义 | 方向 |
| --- | --- | --- |
| `lpips_psnr/ssim/lpips` | MAG 匹配对上的 PSNR/SSIM/LPIPS | ↑/↑/↓ |
| `pose_psnr/ssim/lpips` | 位姿匹配对上的 PSNR/SSIM/LPIPS | ↑/↑/↓ |
| `closure_psnr/ssim/lpips` | 端点帧 0 vs 帧 152 的 PSNR/SSIM/LPIPS | ↑/↑/↓ |
| `match_unique_ratio` | 匹配到不同 outbound 帧的比例 | ↑ |
| `match_temporal_mae_normalized` | 匹配位置与几何镜像期望位置的归一化 MAE | ↓ |
| `match_reverse_violation_ratio` | 匹配序列出现逆序的比例 | ↓ |

`match_temporal_mae_normalized` 的期望基准是时间镜像 `expected = len(ref) - 1 - position`。
对精确 loop 轨迹，pose 匹配的 TempMAE 应接近 0；LPIPS 匹配的 TempMAE 偏高说明 LPIPS
匹配到了时序错误的帧。

## 4. 两种匹配的关系与解释

| 观察 | 解释 |
| --- | --- |
| pose PSNR >> lpips PSNR | 模型在同一位姿生成了视觉一致的内容，但 LPIPS 被中间帧干扰匹配到了错误位置 |
| pose PSNR ≈ lpips PSNR | 匹配方式不影响结论，内容在几何对应处一致 |
| lpips PSNR > pose PSNR | LPIPS 找到了比几何镜像更好的视觉匹配，说明内容在精确闭环位姿处发生了漂移 |
| pose TempMAE ≈ 0 | 轨迹匹配确认了几何镜像的精确性（对 string-loop 应总是成立） |
| lpips TempMAE >> 0 | LPIPS 匹配经常违反时序，指标可能高估或低估真实闭环质量 |

**关键区别**：MAG 匹配是"内容驱动"的——它问"回程帧看起来最像哪个去程帧"；
轨迹匹配是"几何驱动"的——它问"回程帧的相机位姿对应哪个去程帧，它们看起来像不像"。
后者直接测试模型是否在相同相机位姿下生成一致内容。

## 5. 实现

| 文件 | 责任 |
| --- | --- |
| `Wan21/scripts/evaluation/evaluate_loop_closure.py` | 读取 `generation_manifest.jsonl` + `manifest.json`，对每条视频同时计算 LPIPS 匹配和位姿匹配指标，输出 `loop_closure_metrics.{csv,json}` |
| `Wan21/scripts/evaluation/summarize_loop_closure.py` | 汇总各 case 的 JSON，生成四张 Markdown 对比表 |
| `output/motion_novelty_loop_10s_seed0/loop_closure_report.md` | 评测报告 |

### 5.1 运行

```bash
conda activate minwm-fa
cd /data/zju-151/jiangyize/research/minWM-dyKV

# 单个 case
python3 Wan21/scripts/evaluation/evaluate_loop_closure.py \
  --generation-manifest output/motion_novelty_loop_10s_seed0/baseline/generation_manifest.jsonl \
  --closure-manifest Wan21/prompts/demos_loop_closure/manifest.json \
  --output-dir output/motion_novelty_loop_10s_seed0/baseline/eval \
  --wan-root Wan21 \
  --duration-label 10s \
  --resize-width 256 \
  --max-frames-per-segment 96

# 汇总报告
python3 Wan21/scripts/evaluation/summarize_loop_closure.py \
  --root output/motion_novelty_loop_10s_seed0 \
  --output output/motion_novelty_loop_10s_seed0/loop_closure_report.md
```

### 5.2 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--resize-width` | 256 | 评测前将帧缩放到此宽度（与 minWM-back 一致） |
| `--max-frames-per-segment` | 96 | 每半段均匀子采样上限；0 = 全部帧 |
| `--lpips-net` | alex | LPIPS 网络（alex/vgg/squeeze） |
| `--lpips-batch-size` | 64 | LPIPS 批大小 |
| `--skip-lpips` | off | 跳过 LPIPS，只保留位姿匹配 + 端点 PSNR/SSIM |

## 6. 当前结果摘要

`motion_novelty_loop_10s_seed0`，8 case × 30 视频，seed 0：

### 6.1 MAG 风格（LPIPS 匹配）

| Case | PSNR ↑ | Δ | LPIPS ↓ | Δ |
| --- | ---: | ---: | ---: | ---: |
| baseline | 14.406 | +0.000 | 0.2093 | +0.0000 |
| retrieval_no_compression | 14.391 | -0.015 | 0.2057 | -0.0036 |
| motion_novelty_backfill | 14.369 | -0.037 | 0.2069 | -0.0024 |
| retr16_compression_r033 | 14.296 | -0.110 | 0.2085 | -0.0008 |
| motion_novelty_unfilled | 14.238 | -0.168 | 0.2108 | +0.0015 |

### 6.2 轨迹匹配

| Case | PSNR ↑ | Δ | LPIPS ↓ | Δ |
| --- | ---: | ---: | ---: | ---: |
| baseline | 12.077 | +0.000 | 0.3371 | +0.0000 |
| retrieval_no_compression | 12.056 | -0.022 | 0.3343 | -0.0028 |
| motion_novelty_backfill | 12.023 | -0.054 | 0.3364 | -0.0007 |
| motion_novelty_unfilled | 11.978 | -0.099 | 0.3390 | +0.0020 |

### 6.3 关键观察

1. **MAG PSNR 普遍高于 pose PSNR**（14.4 vs 12.1）：LPIPS 匹配找到了视觉更近的 outbound
   帧，但这些帧在时序上往往不是几何对应帧（LPIPS TempMAE 0.33 vs pose TempMAE 0.04）。
2. **pose TempMAE = 0.0423 对所有 case 一致**：确认了轨迹匹配的几何精确性，这是
   string-loop 轨迹的结构性质，与模型质量无关。
3. **所有 dyKV case 的闭环 PSNR 略低于 baseline**：`retrieval_no_compression` 最接近
   baseline，`motion_novelty_unfilled` 差距最大。端点 closure 各 case 差异很小（Δ PSNR
   -0.007 ~ -0.074）。
4. **两种匹配的 case 排序基本一致**：`retrieval_no_compression > backfill > retr16 >
   unfilled`，说明结论不依赖匹配方式。

完整四张表见
[`output/motion_novelty_loop_10s_seed0/loop_closure_report.md`](../output/motion_novelty_loop_10s_seed0/loop_closure_report.md)。

## 7. 与 minWM-back 的差异

| 方面 | minWM-back | minWM-dyKV |
| --- | --- | --- |
| 生成清单 | `inference_times.csv`（CSV） | `generation_manifest.jsonl`（JSONL） |
| 评测器 | `evaluate_loop_closure.py`（仅 LPIPS 匹配） | 新增位姿匹配，同时输出两种模式 |
| manifest | `Wan21/prompts/demos_loop_closure/manifest.json` | 完全相同（已迁移） |
| 轨迹 | `build_string_loop_trajectories.py` | 产物已迁移；`parse_trajectory` 在 `wan_utils` |

## 8. 局限

1. outbound 半段是同一 rollout 的伪参考，不是外部 ground truth；
2. 仅 seed 0，小差异需多 seed 验证；
3. 位姿匹配对非 string-loop 轨迹仍用 SE(3) 最近邻，但不再有精确镜像保证；
4. 解码帧到 latent 帧的映射用 `round(d/4)`，非 4 倍数帧使用最近 latent 位姿，引入
   ≤0.04 的 pose TempMAE；
5. LPIPS 匹配的 `match_reverse_violation_ratio` 和 `match_temporal_mae_normalized`
   受帧采样密度影响，不同 `--max-frames-per-segment` 设置间不可直接比较。
