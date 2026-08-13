# 实验总记录

本文档记录 minWM-dyKV 实验的统一规范。尚未验证的结果必须明确留空，不得用估计值填充。

## 统一环境

除非某项实验另有明确记录，所有生成、适配、测试与评测均使用 Conda 环境 `minwm-fa`：

```bash
conda activate minwm-fa
```

## 可复现性约定

每次运行必须记录：

- Git commit 与工作区是否干净；
- checkpoint 路径与校验和；
- prompt/用例清单及选中的 case ID；
- seed、输出 latent 帧数、分辨率和相机轨迹；
- GPU 型号/数量、PyTorch/CUDA 版本及显存峰值；
- 生成总耗时与检索耗时；
- dyKV 记忆预算及最终采用的内部布局；
- 输出目录与评测报告目录。

## 核心对比

| 运行编号 | Case | 方法 | 目的 | 状态 |
| --- | --- | --- | --- | --- |
| B0 | `baseline` | 上游 minWM local cache | 质量/速度基线 | 待运行 |
| B1 | `retrieval_no_compression` | 不压缩的 dyKV | 单独分析检索收益 | 待运行 |
| B2 | `fixed_novelty` | 固定内容压缩 dyKV | 与相机无关的压缩对照 | 待运行 |
| B3 | `yaw_intrinsics` | 完整动态 dyKV | 评估记忆、速度与质量 | 待运行 |

以上 case 均可由 `Wan21/scripts/inference/run_dykv_cases.sh` 统一运行。FOV 来源消融另使用
`yaw_fixed_fov`、`yaw_mixed_fov` 和 `yaw_intrinsics`，定义见
[`CASES_AND_RUNNER.md`](CASES_AND_RUNNER.md)。

当前 B1/B2/B3 的固定 KV 布局为连续的 `4 + 8 + 8` latent：4 帧 sink、8 帧 retrieval、
8 帧 local（4 帧 recent + 4 帧 current），正好覆盖 20 帧 RoPE 训练窗口。
动态空间压缩及完整消融矩阵见 [`DYNAMIC_SPATIAL_COMPRESSION.md`](DYNAMIC_SPATIAL_COMPRESSION.md)
和 [`ABLATIONS.md`](ABLATIONS.md)。

## 评测分组

1. 闭环相机路径：验证模型能否恢复之前观察过的内容。
2. 长单调路径：检查检索是否会破坏新视角的稳定性。
3. MBench 用例：报告基准原生指标及逐用例产物。
4. 资源画像：记录显存峰值、CPU 记忆库字节数、检索耗时和总延迟。

## MBench 实验规范

- 使用 MBench-A 官方任务分配，并记录准确的 `samples.jsonl` 校验和。
- 10 秒/25 秒用例分别使用与 checkpoint 对齐的 40/100 个 latent 位姿；报告时同时列出
  解码后的 157/397 帧长度和官方目标 161/401 帧。
- B0/B1/B2/B3 必须使用相同的用例分配、checkpoint、latent 长度、分辨率和 seed。
- 每种方法和每个 seed 分别注册为独立的 MBench `model_id`。
- 评测前先运行接口约定校验，并记录因缺少 DA3/VLM 产物而跳过的指标。

## 结果模板

| Commit | 运行编号 | 用例数 | Seed | 帧数 | 质量报告 | 显存峰值 | 耗时 | 备注 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 待运行 | B0 | 待运行 | 0 | 待运行 | 待运行 | 待运行 | 待运行 | 基线 |
| 待运行 | B1 | 待运行 | 0 | 待运行 | 待运行 | 待运行 | 待运行 | 仅检索 |
| 待运行 | B2 | 待运行 | 0 | 待运行 | 待运行 | 待运行 | 待运行 | 固定新颖性 |
| 待运行 | B3 | 待运行 | 0 | 待运行 | 待运行 | 待运行 | 待运行 | 完整方法 |

禁止用估计值替换“待运行”，表中只记录实测结果。

## 实现冒烟测试

该测试只验证实现链路，不代表 MBench 分数。短闭环轨迹会刻意跨过第 20 帧启用边界，
使检索、压缩和三区域 RoPE 在真实 checkpoint 推理中完整执行。

| 字段 | 实测值 |
| --- | --- |
| 日期 / commit | 2026-08-13 / `d4dcdd2` |
| 环境 | `minwm-fa`；PyTorch 2.8.0+cu128；CUDA 12.8；FlashAttention 2.8.3.post1 |
| GPU | 1 × NVIDIA GeForce RTX 4090 (48 GB) |
| Checkpoint | Action2V DMD `model.pt`；SHA-256 `bdb947d45fb04513305492c2ee393d51d0621ec0e99fd312224f5d61a330aa77` |
| Prompt / seed | 1 条合成的红色木屋闭环 prompt / 0 |
| 轨迹 | `j*10,l*10,n*3` |
| 长度 / 输出 | 24 latent 帧 / 93 解码帧，832×480，16 fps |
| 生成状态 | 6/6 个因果块完成；已写出 MP4 和生成清单 |
| 结束时记忆库 | 6 个无损 CPU 块，共 6,900,940,800 字节 |
| 第 20 帧检索 | 候选 `[1,2,3]`；排序 `[1,3,2]`；选中起始帧 `[4,12]` |
| 压缩结果 | 每层 12,480 个原始选中 token → 7,800 个注意力 token |
| 检索耗时 | 2.6609 秒（选择、传输与压缩） |

运行输出位于 `/tmp/minwm_dykv_smoke`，不属于持久化基准产物。该冒烟测试未采集显存峰值，
因此不报告该数值。该记录对应 commit `d4dcdd2` 的旧三区域布局；切换到连续
`4 + 8 + 8` 布局后需要重新运行冒烟测试，不能将本行视为新布局的验证结果。正式
B0/B1/B2/B3 和 MBench 结果仍保持“待运行”，直到完成对应实验。
