# 三区域时序 RoPE

## 布局

生成长度超过训练时的 20 帧窗口后，dyKV 会把所有注意力操作数映射回固定的虚拟时间轴：

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

本实现参考旧版 minWM 三区域原型和 Anchor-Forcing 的有界长时 RoPE 策略，使用固定的
`4 + 8 + 8` 连续布局，不再分别暴露 sink、retrieval 和 rebase 开关。

常规 RoPE 注意力路径使用全部三个区域。minWM 额外的 PRoPE 分支仍保持原有的局部 cache
路径；相机几何信息只负责检索选择，存档时不会在记忆库中重复保存 PRoPE 张量。

当前测试覆盖按块 rebase、多个压缩 segment 共享槽、`1/4` 原子对齐、单槽容量上限和
query 固定映射到 16--19。`3/4` predecessor segment 使用三个原子，沿用同一帧级协议，
不需要扩大 RoPE 范围。
