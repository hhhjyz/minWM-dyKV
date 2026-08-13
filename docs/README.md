# minWM-dyKV 文档

本目录是 dyKV 实现与实验的统一事实来源。每个实现模块对应一份独立文档；
`EXPERIMENTS.md` 是共享实验台账，每当新增或运行实验时都必须同步更新。

## 统一运行环境

本项目的生成、适配器、测试与静态检查统一使用 Conda 环境 `minwm-fa`：

```bash
conda activate minwm-fa
export PYTHONPATH="$PWD/HY15:$PWD/Wan21:$PWD/shared:$PYTHONPATH"
```

文档中的 Python 与 Shell 命令均默认已进入该环境。若需要重建环境，请使用根目录
[`README.md`](../README.md) 中的安装命令，并保持环境名为 `minwm-fa`。

## 设计约束

旧原型暴露了许多彼此独立的开关。minWM-dyKV 改为使用一套完整、固定的预设：

- 所有 case 固定保留最初 4 帧 sink；baseline 使用 `4+16`，dyKV 使用连续的
  `sink 4 | retrieval 8 | local 8` latent；
- 被逐出的干净 KV 保存在 CPU 记忆库中；
- 根据相机 FOV 重叠度对候选记忆排序；
- 仅在将检索 KV 实例化为注意力输入时执行压缩；
- 将三个区域统一重映射到模型训练时的时序 RoPE 范围内。

公开推理接口保留一个启用参数和一个有限枚举：

- `--dykv`：启用完整方法；
- `--dykv-case`：从注册好的整体实验预设中选择，不暴露单个内部超参数。

区域大小等实现常量集中在一个带类型的配置对象中。这样既能在 Python 层进行消融实验，
也不会把每个内部设计都变成命令行超参数。

## 模块文档

- [`KV_MEMORY.md`](KV_MEMORY.md)：逐出存储、检索载荷与检索时压缩；
- [`FIXED_SINK.md`](FIXED_SINK.md)：所有 case 的固定四帧 sink 与公平缓存布局；
- [`TRI_REGION_ROPE.md`](TRI_REGION_ROPE.md)：有界时序位置布局；
- [`FOV_RETRIEVAL.md`](FOV_RETRIEVAL.md)：兼容 HY-WorldPlay 的 FOV 评分与选择；
- [`DYNAMIC_SPATIAL_COMPRESSION.md`](DYNAMIC_SPATIAL_COMPRESSION.md)：基于相机视场的动态空间 token 裁剪、MBench 适合性与实施计划；
- [`DYNAMIC_RETRIEVAL_PACKING.md`](DYNAMIC_RETRIEVAL_PACKING.md)：压缩后扩充历史 chunk、latent 尾部补齐与可变载荷 RoPE 方案；
- [`FIXED_WORLDKV_CASES.md`](FIXED_WORLDKV_CASES.md)：minWM-back 固定压缩率与 chunk 数 A--D 对照；
- [`ABLATIONS.md`](ABLATIONS.md)：动态压缩、检索与几何设计的消融实验矩阵；
- [`CASES_AND_RUNNER.md`](CASES_AND_RUNNER.md)：十一个可运行 case 与普通/MBench 统一 runner；
- [`MBENCH.md`](MBENCH.md)：用例转换、生成与评测流程；
- [`MBENCH_TYPICAL_SAMPLES.md`](MBENCH_TYPICAL_SAMPLES.md)：四/八个典型样本及小规模对比命令；
- [`EXPERIMENTS.md`](EXPERIMENTS.md)：实验矩阵、运行命令、环境与实测结果。

## 参考实现

- `../minWM-back`：早期原型与实验历史；
- `../WorldKV`：KV 记忆库检索及锚点/新颖性压缩；
- `../Anchor-Forcing`：有界三区域时序布局；
- `../HY-WorldPlay`：基于 FOV 重叠度的记忆选择；
- `../MBench`：基准用例与评测接口约定。

以上路径表示开发时使用的同级参考仓库；本项目运行时不会从这些仓库导入代码。
