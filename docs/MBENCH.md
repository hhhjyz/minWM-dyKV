# MBench-A 适配

minWM-dyKV 支持 MBench-A 的动作条件用例。MBench-T 使用随时间变化的文本片段，超出了
当前 minWM 推理接口的支持范围。

当前官方任务清单的动作分布，以及基于 yaw/FOV 的动态空间压缩适合性分析，见
[`DYNAMIC_SPATIAL_COMPRESSION.md`](DYNAMIC_SPATIAL_COMPRESSION.md)。
只做小规模视频对比时，使用项目提供的四/八样本清单，见
[`MBENCH_TYPICAL_SAMPLES.md`](MBENCH_TYPICAL_SAMPLES.md)。

## 适配内容

`mbench_adapter.py prepare` 读取基准中以模型为中心的任务分配清单，并将每个
`subset/sample_id/condition_id` 与 `samples/{subset}/{sample_id}/sample.json` 中的
caption 对齐，随后写出三个相互对应的文件：

```text
work_dir/
├── prompts.txt
├── trajectories.txt
└── cases.jsonl
```

动作映射如下：

| MBench-A 动作 | minWM 轨迹 |
| --- | --- |
| 先左转再右转 | 左偏航、右偏航，必要时补静止帧 |
| 先右转再左转 | 右偏航、左偏航，必要时补静止帧 |
| 先前进再后退 | 前进、后退，必要时补静止帧 |
| 向左/右旋转 360、720、1080 度 | 按比例缩放偏航步长以完成指定角度 |
| 静止 | 相机不运动 |

轨迹解析器支持 `n*N`，也支持 `j@2.5*40` 这类缩放步长。测试会检查适配器生成的每条
轨迹都恰好包含指定数量的 latent 相机位姿。

MBench 的长度指解码后视频帧数。Wan VAE 在时间维度上放大 4 倍，而因果 checkpoint
要求 latent 帧数是 4 的倍数。因此，官方 10 秒/161 帧条件采用最接近且较小的有效长度
40 latent 帧（解码后 157 帧），官方 25 秒/401 帧条件采用 100 latent 帧（解码后
397 帧）。默认运行器使用 100；基准报告中必须注明这一小段时长差异。

## 十四组 Case 的统一生成与打包

推荐使用统一 runner；它只准备一次 MBench 输入，然后依次生成选中的 case 并分别打包：

```bash
MBENCH_ROOT=/absolute/path/to/MBench-A-Setup \
ASSIGNMENTS=/absolute/path/to/official/samples.jsonl \
CASES=baseline,retrieval_no_compression,fixed_novelty,yaw_fixed_fov,yaw_mixed_fov,yaw_intrinsics,packed_chunks,packed_chunks_latent,predecessor_chunks,predecessor_chunks_latent,predecessor_query_backfill,retr8_compression_r050,retr12_compression_r050,retr16_compression_r033 \
OUTPUT_ROOT=output/mbench_dykv_cases \
NUM_OUTPUT_FRAMES=100 \
SEED=0 \
bash Wan21/scripts/inference/run_dykv_cases.sh
```

完整 case 定义和普通 prompt 用法见 [`CASES_AND_RUNNER.md`](CASES_AND_RUNNER.md)。

## 单组兼容默认 `yaw_intrinsics` 的入口

```bash
MBENCH_ROOT=/absolute/path/to/MBench-A-Setup \
ASSIGNMENTS=/absolute/path/to/official/samples.jsonl \
MODEL_ID=minwm_dykv_seed0 \
NUM_OUTPUT_FRAMES=100 \
SEED=0 \
bash Wan21/scripts/inference/run_mbench_dykv.sh
```

该兼容脚本不代表最新 predecessor 完整方案。运行
`predecessor_query_backfill` 应使用上一节统一 runner 并显式设置 `CASES`。

运行前应先执行 `conda activate minwm-fa`。可用筛选项包括 `SUBSETS`、`CONDITIONS` 和
`LIMIT`。若省略 `ASSIGNMENTS`，适配器使用 MBench-A 官方任务来源
`models/hy_worldplay/samples.jsonl`；运行四用例示例或自定义用例集时应显式传入清单。
当 10 秒用例的 `NUM_OUTPUT_FRAMES` 不为 40，或 25 秒用例不为 100 时，适配器会拒绝
执行，防止生成标签错误的基准包。

生成阶段写出 `generation_manifest.jsonl`，打包后得到：

```text
MBench-A-Setup/models/{MODEL_ID}/
├── samples.jsonl
└── outputs/{subset}/{sample_id}/{condition_id}/video.mp4
```

默认使用相对软链接保存视频；若数据集随后会移动到其他文件系统，请设置
`LINK_MODE=hardlink` 或 `LINK_MODE=copy`。

## 评测

在同一个 `minwm-fa` 环境中安装 MBench 依赖并执行：

```bash
mbench validate "$MBENCH_ROOT" \
  --models minwm_dykv_seed0 \
  --metrics mbencha.entity.human_identity_consistency \
  --limit 2

mbench eval "$MBENCH_ROOT" \
  --models minwm_dykv_seed0 \
  --metrics mbencha.entity.human_identity_consistency,mbencha.environment.rendering_lighting \
  --output runs/minwm_dykv_seed0
```

空间、物体几何、渲染风格和相机交互指标需要 `dataset.yaml` 所声明路径下的外部 DA3
产物。状态进展指标还可能需要已配置的 VLM 裁判模型。这些内容属于评测输入，适配器不会
伪造它们。

## 当前限制

四帧 dyKV checkpoint 路径属于 T2V，因此生成条件使用 MBench caption。基准的首帧素材
仍保留在数据集中供评测使用，但不会注入生成过程。与首帧条件世界模型比较时，必须随结果
一同报告这一差异。
