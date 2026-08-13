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

- 注意力布局固定为连续的 `sink 4 | retrieval 8 | local 8` latent，三区域之间不留空缺；
- 被逐出的干净 KV 保存在 CPU 记忆库中；
- 根据相机 FOV 重叠度对候选记忆排序；
- 仅在将检索 KV 实例化为注意力输入时执行压缩；
- 将三个区域统一重映射到模型训练时的时序 RoPE 范围内。

公开推理接口仅保留一个 dyKV 参数：

- `--dykv`：启用完整方法；

区域大小等实现常量集中在一个带类型的配置对象中。这样既能在 Python 层进行消融实验，
也不会把每个内部设计都变成命令行超参数。

## 模块文档

- [`KV_MEMORY.md`](KV_MEMORY.md)：逐出存储、检索载荷与检索时压缩；
- [`TRI_REGION_ROPE.md`](TRI_REGION_ROPE.md)：有界时序位置布局；
- [`FOV_RETRIEVAL.md`](FOV_RETRIEVAL.md)：兼容 HY-WorldPlay 的 FOV 评分与选择；
- [`MBENCH.md`](MBENCH.md)：用例转换、生成与评测流程；
- [`EXPERIMENTS.md`](EXPERIMENTS.md)：实验矩阵、运行命令、环境与实测结果。

## 参考实现

- `../minWM-back`：早期原型与实验历史；
- `../WorldKV`：KV 记忆库检索及锚点/新颖性压缩；
- `../Anchor-Forcing`：有界三区域时序布局；
- `../HY-WorldPlay`：基于 FOV 重叠度的记忆选择；
- `../MBench`：基准用例与评测接口约定。

以上路径表示开发时使用的同级参考仓库；本项目运行时不会从这些仓库导入代码。
