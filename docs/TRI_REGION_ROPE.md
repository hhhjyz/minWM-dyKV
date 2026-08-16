# 三区域时序 RoPE

## 布局

所有注册 case 都使用 20 帧内的有界虚拟时间轴。生成尚未超过训练窗口时，原始 RoPE
位置本来就在 `0~19`；从第 20 帧之后的第一个 chunk 起，显式执行 rebase。dyKV case
使用完整三区域：

```text
0 ... 3          4 ........ 11          12 ... 15          16 ... 19
[ sink: 4 ]      [ retrieval: 8 ]       [ recent: 4 ]      [ current: 4 ]
\______________________________ 20 latent frames _______________________/
```

三个逻辑区域连续占满模型训练时的 20 帧时序范围，不再保留空缺。其中 local 区域共
8 帧，由 4 帧 recent 和 4 帧 current 组成；当前 query 在每个长时生成步骤中始终占据
最后 4 个位置。

在第 20 帧之前，minWM 使用原本单调递增的 RoPE，dyKV 检索不启用。到达边界后：

- 前 4 帧 sink K 保持在原始位置 0--3；
- 非装箱 case 的记忆块按时间顺序映射到位置 4--11；装箱 case 的 frame segment 显式映射
  到该范围内的 `virtual_slot_id`；
- 最近 4 帧映射到位置 12--15；
- 当前 4 帧 query 及其 K 映射到位置 16--19。

baseline 不创建长期记忆库，也不检索历史 KV，但复用同一套 tri-region 代码，将 retrieval
区设为空：

```text
0 ... 3          4 ........................ 15          16 ... 19
[ sink: 4 ]      [ recent: 12 ]                         [ current: 4 ]
\________________ local: 16；总计 20 latent _________________/

等价配置：sink 4 | retrieval 0 | local 16
```

因此 baseline 的物理 KV 预算仍是 `4 sink + 16 rolling local`，但长序列不再使用不断增长
的绝对时序位置：sink 固定为 `0~3`，12 帧历史 local 映射为 `4~15`，当前 query/K 固定为
`16~19`。这使 baseline 与检索 case 都使用相同的有界 RoPE 机制，消融主要剩下“16 帧
连续 local”与“8 帧 retrieval + 8 帧 local”的上下文分配差异。

## 实现入口与日志

`Wan21/wan_inference.py` 对所有 case 设置 `tri_region_rope_enabled=True`，并根据 case 写入
`memory_frames/local_frames`；`Wan21/wan_utils/wan_wrapper.py` 将这些 model kwargs 原样传给
`CausalWanModel`。`Wan21/wan/modules/causal_model.py` 再将该开关与 `dykv_enabled` 解耦：
前者决定 Q/K 是否 rebase，后者只决定是否创建长期 KV 记忆与检索运行时。因此 baseline
可以在 `dykv_enabled=False` 时独立启用有界 RoPE。

每个 `generation_manifest.jsonl` 条目都会记录：

```text
tri_region_rope_enabled
tri_region_rope_layout
tri_region_rope_train_frames
```

新 baseline 的期望值分别为 `true`、`[4,0,16]` 和 `20`；dyKV case 的 layout 为
`[4,8,8]`。正式消融前应检查这些字段，旧 baseline 视频没有该布局保证，必须重新生成。

## 重映射操作

cache 中的 K 和当前 Q 已经应用过 RoPE。因此，重映射只需在复数形式的时序通道上乘以
`target_position - source_position` 对应的相对旋转，空间高/宽通道保持不变。所有操作
都会先复制输入，避免多次去噪调用在 cache 或记忆库中累积旋转。

未装箱的完整检索块仍可按块起点重映射。动态装箱、固定 WorldKV 和 predecessor case 则
不能再按整个 chunk 假设固定长度：每个压缩 frame segment 显式携带
`source_frame_id`、`frame_token_length` 和 `virtual_slot_id`，分别乘以
`virtual_slot_id - source_frame_id` 的时序旋转。同一虚拟槽可以容纳多个 segment，但其
token 总数不得超过一个完整 latent；空间高/宽 RoPE 通道保持不变。

## 不变量

- 在线 cache 的每个切片都必须按帧对齐；
- query、recent、retrieval 和 sink 的位置范围不能重叠；
- 非扩容 case 的原始记忆不能超过源帧预算；扩容 case 可以覆盖更多源帧，但 retrieval
  物理 token 不能超过 8 个完整 latent；
- 超出可用 RoPE 频率表的位移必须显式报错；
- 预热和正式启用使用同一个 20 帧边界。

本实现参考旧版 minWM 三区域原型和 Anchor-Forcing 的有界长时 RoPE 策略。dyKV 使用固定
`4 + 8 + 8`，baseline 使用空 retrieval 的 `4 + 0 + 16`；两者都连续占满 `0~19`，不再
暴露独立的 rebase 实验开关。

常规 RoPE 注意力路径使用全部三个区域。minWM 额外的 PRoPE 分支仍保持原有的局部 cache
路径；相机几何信息只负责检索选择，存档时不会在记忆库中重复保存 PRoPE 张量。

当前测试覆盖按块 rebase、多个压缩 segment 共享槽、`1/4` 原子对齐、单槽容量上限、
query 固定映射到 16--19，以及 baseline 空 retrieval 布局。`3/4` predecessor segment
使用三个原子，沿用同一帧级协议，不需要扩大 RoPE 范围。
