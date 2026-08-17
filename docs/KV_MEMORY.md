# KV 记忆与检索时压缩

## 目标

在线因果 KV cache 仍保持固定大小和较低开销。在一个已完成去噪的干净生成块即将被
逐出在线 cache 前，dyKV 会把它在每个网络层中的 K/V 复制到 CPU 记忆库。只有同时
离开 sink 区域和近期 local 区域的块才可参与检索，从而避免同一帧被重复加入注意力。

## 生命周期

1. 模型完成一个块的去噪。
2. 最后一次干净前向将该块写入在线 KV cache。
3. `DyKVBank.archive_clean_block` 将每个 Transformer 层新增的尾部切片复制到记忆库设备。
4. 后续块通过 `evicted_candidates` 获取候选集合，再由注册 case 指定的 FOV 或 WorldKV
   平均位姿选择器从中选择记忆。
5. `materialize` 仅把选中的 K/V 移到注意力设备，并在此时执行压缩。
6. 注意力层将该载荷作为中间的 retrieval 区域使用。

扩容 case 会在第 4 步后对全部排序候选计算固定档位与实际 token 成本，再在
`8×1560` token 上限内选择更多完整 chunk；`packed_chunks_latent` 还会用未选中块中的
单个 latent 补齐剩余容量。CPU bank 的存储方式不变。

固定 WorldKV 对照则先按 case 选择 8/12/16 个源 latent，再保留每个 chunk 的完整 anchor
和非 anchor 的固定比例新颖性 token。它们同样只在 materialize 阶段压缩。

记忆库始终保存未压缩的干净 K/V。这样检索选择可逆，也不会在从未被检索的历史内容上
浪费压缩时间或损失信息。

## 压缩方法

兼容默认 case `yaw_intrinsics` 使用“历史帧相对当前 query”的 yaw/水平视场交集裁剪：

- 从历史与当前 W2C 位姿计算有向相对 yaw；
- 从相机内参 `K` 计算水平 FOV；
- 将两个视场的交集映射到 latent 网格的对应水平列；
- 对所有行使用相同列 mask，并以同一索引裁剪 K 和 V；
- 当前 4 帧 query 分别产生 mask，最终取并集。

因此，旋转方向决定裁剪左侧还是右侧，旋转角度决定保留比例。相同视角保留全部 token；
偏转半个 FOV 时约保留对应方向的一半空间列；偏转达到一个 FOV 且完全无交集时不将该
历史块加入注意力。裁剪仅发生在检索实例化阶段，CPU 记忆库保持无损。

如果已通过内参检索的块在压缩时检测到前后/横向平移或 pitch/roll，E0 算法回退到
WorldKV 的“锚点 + 新颖性”固定压缩：

- 完整保留第一帧 latent 作为锚点；
- 在空间 token 和注意力头维度上计算锚点 key 的均值；
- 按各后续帧 token 与该中心的余弦相似度排序；
- 保留相似度最低的一半，以表示尚未被锚点覆盖的内容。

对于 4 帧块和 `0.5` 的固定回退比例，存储的 4 帧在实例化后相当于
`1 + 3 * 0.5 = 2.5` 帧注意力 token。原始记忆帧预算不变；压缩只改变注意力开销，
不改变被选择的帧。

这里的“固定”表示固定 novelty 保留率，不表示固定 FOV。当前 query 缺少内参会在检索
入口报错，历史块缺少内参会被跳过，不会进入上述内容压缩回退。

## 固定档位扩容与 RoPE

扩容 case 将逐帧保留率量化为 `1`、`1/2`、`1/4` 或丢弃，以 390 token 为一个预算原子，
retrieval region 总计 32 个原子。完整四帧 chunk 优先通过 0/1 背包装入；尾部 latent
使用 4/2/1 原子大小做 first-fit decreasing。所有物理 token 始终不超过 12480。

每个裁剪后的源 latent 都显式记录源帧、token 长度和 4--11 范围内的虚拟槽位。RoPE
只按 segment 重映射时间通道；空间通道保持不变。一个虚拟槽最多容纳 1560 token，因此
可以容纳 1 个 full、2 个 half 或 4 个 quarter 源 latent，而不会侵入 local 12--19。

`retr8_compression_r050`、`retr12_compression_r050` 和 `retr16_compression_r033` 不受
四分之一档位限制，而是按实际 segment token 数装箱；完整 anchor 占一槽，固定比例的
非 anchor segment 可跨 chunk 共享槽。三者的最大物理容量仍是 8 latent。

## 公开配置

命令行只暴露 `--dykv` 和有限枚举 `--dykv-case`；区域大小、阈值和装箱参数不作为公开
连续超参数。固定方法预设如下：

| 配置项 | 值 |
| --- | ---: |
| sink 区域 | 4 latent 帧 |
| retrieval 区域 | 8 latent 帧 |
| local 区域 | 8 latent 帧（4 帧 recent + 4 帧 current） |
| 兼容默认检索压缩 | 历史相对当前 query 的 yaw/FOV 动态空间列裁剪 |
| E0 非纯 yaw 回退 | 完整 anchor + 后三帧 50% 新颖性 |
| 记忆库设备 | CPU |

三区域在 20 帧训练窗口中连续排列为 `[0,4) + [4,12) + [12,20)`，不存在保留空隙。
在线 KV cache 的物理容量为 12 latent 帧，用于保存 4 帧 sink 和完整的 8 帧 local；
retrieval KV 仍存放于 CPU 记忆库，仅在注意力计算时加入。

当预算无法按模型块大小对齐，或三区域无法放入训练时的 RoPE 范围时，
`DyKVConfig.validate` 会直接拒绝该配置。

## 验证

单元测试覆盖块尾捕获、逐出候选资格、左右方向镜像裁剪、0°/半 FOV/整 FOV/360°、
平移回退、跨层索引一致性、锚点保留、新颖性选择、检索载荷的时序顺序，以及检索压缩
不得修改无损记忆库这一约束。扩容测试另外覆盖固定档位边界、32 原子预算、完整 chunk
背包、latent 尾部补齐、共享虚拟槽逐段 rebase 和单槽容量校验。

## 运行方式

先进入统一环境。以下命令只启用兼容默认 `yaw_intrinsics`：

```bash
conda activate minwm-fa
DYKV=1 \
  bash Wan21/scripts/inference/run_infer_causal_camera.sh
```

输出目录包含 `dykv_summaries.jsonl`，每个 prompt 对应一条记录。记录包括记忆库字节数、
选中块 ID、源起始帧、压缩后 token 数以及检索耗时。

当前推理已让 dyKV 的相机位姿保持 FP32，并在 PRoPE 算子边界单独转换模型计算副本，解决
低精度几何量化干扰 yaw/FOV 判断的问题。
