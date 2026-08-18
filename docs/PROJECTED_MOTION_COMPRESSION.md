# 基于双向图像投影的 Motion-Adaptive 压缩率

> 状态：详细设计完成，尚未实现或注册 case。
>
> 本文是 [`MOTION_ADAPTIVE_NOVELTY_COMPRESSION.md`](MOTION_ADAPTIVE_NOVELTY_COMPRESSION.md)
> 中连续 FOV 比例方法的几何第二版。当前代码仍使用有限球形探针的 FOV overlap；本文提出的
> projected overlap 只有在代码、测试和真实 checkpoint 冒烟全部完成后才能成为默认方法。

## 1. 问题与目标

现有 motion novelty 方法的总体分工保持不变：

```text
相机几何：决定每个 non-anchor latent 应保留多少 token
WorldKV novelty：在该 token 数量内决定具体保留哪些 token
```

需要替换的是第一部分。当前实现把当前相机和 chunk anchor 的视锥放入半径为 8 的球形探针
空间，用三维探针的交集比例近似二维 latent token 的视野重叠。该方法对纯 yaw 基本合理，
但对平移尤其是前进运动存在明显偏差。

在当前 minWM 相机步长、`K=(fx=fy=0.5,cx=cy=0.5)`、`F=1560` 下，用 65536 个探针得到：

| Action | P1/P2/P3 keep ratio | P1/P2/P3 token |
| --- | --- | --- |
| `n` 静止 | `0, 0, 0` | `0, 0, 0` |
| `j/l` yaw | 约 `3.3%, 6.7%, 10.0%` | `52, 104/105, 156` |
| `i/k` pitch | 约 `2.7%, 5.5%, 8.5%` | `43, 86/87, 132` |
| `a/d` 横移 | 约 `1.1%, 2.2%, 3.3%` | `18/19, 35/36, 53` |
| `u/dn` 升降 | 约 `0.6%, 1.3%, 2.0%` | `10, 21, 31/32` |
| `s` 后退 | 约 `3.0%, 5.9%, 8.9%` | `47, 92, 139` |
| `w` 前进 | `0, 0, 0` | `0, 0, 0` |

`w` 与 `s` 的强烈不对称主要来自单向视锥包含关系和有限球半径，而不是生成内容的真实新颖
程度。向前移动虽然可能没有新增视线方向，仍会产生尺度变化、视差、遮挡变化和更高频细节；
不能因此删除全部 non-anchor token。

第二版方法需要满足：

1. 比例直接对应二维 latent token/image area，而不是三维采样体积；
2. yaw、pitch、roll 使用真实内参和完整相对旋转，不把 SO(3) 标量角度除以单一 FOV；
3. 横移、升降、前进、后退使用同一投影定义，并明确暴露深度不可观测性；
4. 前进/后退使用双向 overlap，避免单向包含导致一侧恒为零；
5. 混合旋转和平移不手工相加，由完整 `R,t,K` 一次投影得到；
6. 比例连续、不使用四档量化，也不设置隐式最低保留率；
7. 每个 non-anchor 仍相对 chunk anchor 计算，使匀速运动在 chunk 内得到累计的不同压缩率；
8. 不增加一组公开 CLI 超参数；除一个场景尺度外，算法常量固定在带类型配置中。

## 2. 固定语义

### 2.1 Chunk 与参考帧

每个历史 chunk 为：

```text
C = (P0, P1, P2, P3)
```

- `P0` 是完整 anchor，始终保留 `F` 个 token；
- `P1/P2/P3` 分别相对 `P0` 计算；
- 禁止改成 `P1←P0, P2←P1, P3←P2` 的相邻帧比较。

因此旋转速度为每 latent 3° 时，chunk 内计算的是 3°、6°、9°，而不是三个相同的 3°。
匀速轨迹上不同完整 chunk 重复同一组比例是预期行为：相同局部几何应得到相同压缩率。
“动态”表示比例随局部运动几何变化，不表示它必须随绝对时间变化。

### 2.2 比例和整数 token

对 frame `Pi`：

```text
overlap_ratio_i = projected_overlap(Pi, P0)
keep_ratio_i    = clamp(1 - overlap_ratio_i, 0, 1)
keep_tokens_i   = min(F, max(0, ceil(keep_ratio_i * F)))
```

这里仅有浮点比例到整数 token 的不可避免离散化，没有比例档位。若相机变换严格为 identity、
内参一致，投影必须精确返回 `keep_ratio=0`，不能依靠阈值把采样噪声压成零。

## 3. 坐标与 token 网格

### 3.1 使用二维 token 中心

使用 memory block 保存的真实 `spatial_shape=(H_t,W_t)`，要求：

```text
H_t * W_t = F
```

为每个 token 建立其图像归一化中心：

```text
u = (x + 0.5) / W_t
v = (y + 0.5) / H_t
p = [u, v, 1]^T
```

每个 token 权重相同，因此 overlap 直接表示可对应的 token 比例。不得硬编码 `30×52`，也
不需要随机或 Fibonacci 探针。

### 3.2 相对相机变换

仓库中 pose 为 world-to-camera：

```text
T_i^w2c
```

从 frame `Pi` 相机坐标变换到 anchor `P0` 相机坐标：

```text
T_(0←i) = T_0^w2c @ inverse(T_i^w2c)
```

对当前 token 射线和假设深度 `d`：

```text
X_i = d * inverse(K_i) @ p_i
X_0 = R_(0←i) @ X_i + t_(0←i)
p_0 = K_0 @ X_0
```

只有同时满足下列条件才算落入 anchor：

```text
X_0.z > 0
0 <= p_0.x / p_0.z < 1
0 <= p_0.y / p_0.z < 1
```

反方向 `P0→Pi` 使用逆变换和对应内参，定义完全相同。

## 4. 旋转：二维投影视野比例

纯旋转时 `t_(0←i)=0`，投影与深度无关。每个当前 token 反投影为射线，通过完整相对旋转
投影到 anchor，得到：

```text
o_(i→0) = valid current-token count / F
o_(0→i) = valid anchor-token count / F
```

对视野和内参相同的纯旋转，两者应接近；仍统一使用后文的双向聚合，避免为运动类型写分支。

### 4.1 Yaw

对小角度、相同内参和近似 90° 水平 FOV，结果应接近：

```text
keep_ratio ≈ abs(delta_yaw) / horizontal_FOV
```

但实现不能直接使用这个公式。二维投影会自然处理非居中主点、不同内参和大角度边界。

### 4.2 Pitch

Pitch 由垂直 FOV 决定，不能用水平 FOV 归一化。二维投影会自动使用 `fy,cy`，并处理
pitch 接近上下边界时的非线性。

### 4.3 Roll

Roll 不改变相机 forward direction，因此当前只提取 yaw/pitch 的视锥实现无法完整表达它。
二维网格投影会产生角落区域的进出，能够直接计算旋转矩形的有效 overlap。当前 minWM action
string 没有 roll，但实现不应假设未来 pose 永远没有 roll。

## 5. 平移：多深度、双向投影

### 5.1 不可辨识性

只给定 `K` 和相机 `R,t`，无法唯一计算平移造成的像素变化；必须知道场景深度。任何声称仅
凭平移距离就能得到精确 FOV 比例的方法，都隐含了深度或场景尺度假设。

第一版实现不引入深度网络，而使用一个明确、可审计的多深度先验。保留一个场景尺度：

```text
D = projection_scene_scale
```

初始值沿用当前几何的尺度 `D=8`。固定的归一化深度节点为：

```text
rho = [1/8, 1/4, 1/2, 1]
depths = D * rho
```

这四个节点是算法内部积分节点，不公开为四个超参数。正式实验只对单一 `D` 做一次敏感性
检查；若以后获得真实或估计深度，应直接替换该先验，而不是继续增加节点开关。

### 5.2 为什么必须双向

单向 `Pi→P0` overlap 对前进运动可能等于 1：当前视锥完全落在旧视锥内，于是
`keep_ratio=0`。但反方向 `P0→Pi` 会看到旧画面边缘无法映入新画面，并反映尺度变化。

对每个深度 `d` 计算：

```text
a_d = o_(i→0)(d)
b_d = o_(0→i)(d)
```

采用调和平均：

```text
o_sym(d) = 2 * a_d * b_d / (a_d + b_d)       if a_d + b_d > 0
o_sym(d) = 0                                  otherwise
```

调和平均具有以下性质：

- 任一方向 overlap 很低都会降低最终 overlap；
- 前进和后退在互换帧次序时保持对称；
- 不像 `min(a,b)` 那样完全由最坏方向控制；
- identity 时严格为 1。

多深度结果使用固定等权平均：

```text
overlap_ratio_i = mean_d(o_sym(d))
keep_ratio_i    = 1 - overlap_ratio_i
```

不根据 action 字符决定公式；action 只产生 pose，pose 是唯一几何输入。

### 5.3 横向与垂直平移

`a/d` 和 `u/dn` 会产生深度相关视差：近深度节点贡献更大的新增比例，远深度节点更小。
左右、上下的相反方向在内参对称时应得到相同数值；若主点不居中，允许出现由真实图像边界
造成的小差异。

### 5.4 前进与后退

`w/s` 主要产生尺度变化。双向 overlap 使两者都得到非零比例，并在交换 frame 次序时保持
相同的几何变化量。近深度对前后移动更敏感，远深度接近纯旋转/静止。

该比例仍只是静态多平面场景的代理，不能恢复真实遮挡和新出现细节。日志必须记录各深度节点
的两个方向 overlap，不能只记录最终平均值。

## 6. 混合运动与 action 边界

混合运动直接使用完整 `T_(0←i)`：

```text
X_0 = R_(0←i) X_i + t_(0←i)
```

禁止使用：

```text
yaw_ratio + pitch_ratio + translation_ratio
```

手工相加会重复计算同一片出界区域，并无法处理旋转与平移抵消的情况。完整投影自然覆盖：

- yaw + forward；
- pitch + lateral；
- 先旋转后沿相机局部轴移动；
- chunk 内跨越两种 action 的边界。

minWM action 是在相机局部坐标中依次应用的，最终 pose 已包含顺序信息；压缩模块不应重新解析
action string。

## 7. 各运动的预期行为

| 运动 | 几何计算 | 预期性质 | 主要风险 |
| --- | --- | --- | --- |
| 静止 `n` | identity 双向投影 | `q=0` | 动态物体不由相机几何表达 |
| Yaw `j/l` | 深度无关二维旋转投影 | 随累计 yaw 单调增加；左右近似对称 | 大角度完全不重叠后饱和为 1 |
| Pitch `i/k` | 使用垂直内参的旋转投影 | 随累计 pitch 单调增加；上下近似对称 | 极点附近投影非线性 |
| Roll | 完整旋转投影 | 根据旋转后角落出界面积增加 | 当前 action 数据不足 |
| 横移 `a/d` | 多深度双向投影 | 近景比例高于远景；左右对称 | 依赖场景尺度 `D` |
| 升降 `u/dn` | 多深度双向投影 | 近景比例高于远景；上下对称 | 依赖场景尺度 `D` |
| 前后 `w/s` | 多深度双向尺度 overlap | 两个方向均非零，交换帧次序近似对称 | 无真实深度和遮挡 |
| 混合运动 | 完整 `R,t,K` 一次投影 | 无需人工组合权重 | 深度先验误差仍存在 |
| 内参变化 | `K_i→K_0` 与 `K_0→K_i` | zoom/FOV 改变反映在比例中 | 需要合法归一化 K |

## 8. 内容运动的边界

本文解决相机运动的几何比例，不宣称解决相机静止时的人物、车辆或物体运动。静止相机下
`q_geometry=0` 是正确几何结果，但不一定是最佳视频压缩策略。

内容变化可以在后续单独建立消融：

```text
q_final = max(q_geometry, q_content)
```

但 `q_content` 不能直接从已经带 RoPE 的 cached K 做未经校准的对应位置 cosine，因为位置
变换可能污染内容距离。合理实现需要保存轻量的 pre-RoPE descriptor 或 latent descriptor，
并单独验证其 CPU 内存、相似度尺度和动态物体召回。第一版 projected motion case 不加入该
变量，避免把相机几何修正和内容变化混为一谈。

## 9. 与 retrieval 和 token 选择的关系

Projected overlap 只替换每帧 token 数计算，不修改：

- query 相对历史 chunk 的 FOV retrieval 排名；
- layer-0 WorldKV novelty 完整排序；
- anchor 完整保留；
- relevance-first chunk 选择；
- source-order materialization；
- flat 总预算和 A16/A17/A18 填充协议；
- tri-region RoPE。

具体 token 仍从原有 novelty order 中取前 `keep_tokens_i` 个。几何投影生成的是数量，不生成
精确裁剪 mask，也不强制保留图像边缘。

需要注意：当前 retrieval ranking 也使用球形 FOV overlap，因此它对平移可能存在类似偏差。
第一阶段只替换 compression ratio，保持检索算法固定以形成单变量实验；第二阶段再注册独立的
projected retrieval case，不能在同一次提交中同时更换检索和压缩几何。

## 10. 计划数据结构与接口

建议新增独立模块：

```text
Wan21/pipeline/dykv_projected_overlap.py
```

核心接口：

```python
@dataclass(frozen=True)
class ProjectedOverlapResult:
    overlap_ratio: float
    keep_ratio: float
    keep_tokens: int
    forward_overlaps: tuple[float, ...]
    backward_overlaps: tuple[float, ...]
    symmetric_overlaps: tuple[float, ...]
    depths: tuple[float, ...]
    relative_rotation_degrees: float
    relative_translation_xyz: tuple[float, float, float]


def projected_motion_overlap(
    current_w2c,
    anchor_w2c,
    current_K,
    anchor_K,
    spatial_shape,
    *,
    scene_scale: float,
) -> ProjectedOverlapResult:
    ...
```

内部缓存 `(spatial_shape,K-signature)` 对应的二维 token center 和反投影射线，避免每个 layer
重复构造。一个 retrieval event 的计划只在 layer 0 计算一次，所有 attention layer 共用 token
索引和比例。

配置只增加内部字段：

```text
motion_geometry_mode = "projected_multidepth"
projection_scene_scale = 8.0
```

它们由注册 case 固定，不增加公开 CLI flag。`projection_scene_scale` 是平移不可避免的单一
尺度假设，必须写入 summary 和生成 manifest。

## 11. 日志要求

每个 retrieval event 至少记录：

```text
motion_geometry_mode
projection_scene_scale
projection_depths
projected_forward_overlaps_per_frame_per_depth
projected_backward_overlaps_per_frame_per_depth
projected_symmetric_overlaps_per_frame_per_depth
projected_overlap_ratios
motion_keep_ratios
base_tokens_per_frame
relative_rotation_degrees
relative_translation_xyz
```

同时保留当前的 selected blocks、novelty indices、slot load、final token 和 multiplicity 日志，
以便确认几何替换没有意外改变其他实验变量。

## 12. Case 与消融设计

在实现验证前不修改现有 `motion_novelty_*` 的语义。建议先注册一个几何基础 case：

| Case | 历史选择 | 几何比例 | 剩余空间 | 目的 |
| --- | --- | --- | --- | --- |
| `motion_novelty_unfilled` | 当前 FOV retrieval | 旧 sphere overlap | 欠填 | 当前实现基线 |
| `motion_projected_unfilled` | 与上行相同 | 双向二维多深度投影 | 欠填 | 单变量验证新比例 |
| `motion_projected_backfill` | 与 projected-unfilled 相同 | 相同 | 唯一 token 回填 | 新几何下的填满主方法 |
| `motion_projected_duplicate` | 与 projected-unfilled 相同 | 相同 | 重复最高相关 chunk | 长度/重加权诊断 |

第一轮几何实验只比较前两行。只有 projected-unfilled 的动作对称性、速度单调性和真实模型
冒烟通过后，才实现 backfill/duplicate；不能一开始同时更改几何和填充策略。

### 12.1 单一尺度敏感性

平移无法避免场景尺度，因此只为 `D` 做一次开发集敏感性：

```text
D ∈ {4, 8, 16}
```

这不是三个长期保留的公开 case。选定一个值后固定到正式注册 case，完整 MBench 不重复搜索，
避免在测试集上调参。报告应分别给出旋转样本和平移样本，防止平均值掩盖尺度问题。

### 12.2 速度测试

loop-closure 10s/15s/20s/30s 只增加运动持续时间，不改变每 latent 步长，不能验证速度自适应。
必须增加合成四帧轨迹：

```text
j@0.5*3, j@1*3, j@2*3, j@4*3
i@0.5*3, i@1*3, i@2*3, i@4*3
w@0.5*3, w@1*3, w@2*3, w@4*3
a@0.5*3, a@1*3, a@2*3, a@4*3
```

同一运动方向必须随 multiplier 单调增加。匀速轨迹不同 chunk 重复同一比例不是失败；不同
速度仍给出相同模式才是失败。

## 13. 单元测试与验收条件

### 13.1 几何单元测试

1. Identity：所有深度、双向 overlap 均为 1，keep ratio 和 token 为 0；
2. 纯 yaw：3°、6°、9° 单调，`j/l` 在对称 K 下数值一致；
3. 纯 pitch：3°、6°、9° 单调，`i/k` 在对称 K 下数值一致；
4. Roll：非零 roll 产生非零比例，正负 roll 对称；
5. 横移：`a/d` 对称，近深度变化大于远深度；
6. 升降：`u/dn` 对称，近深度变化大于远深度；
7. 前后：`w/s` 均为非零，交换 anchor/current 后 symmetric overlap 不变；
8. 速度：`@0.5 < @1 < @2 < @4`；
9. 混合变换：完整投影结果有界且确定，不等于简单比例相加；
10. 内参变化：改变 fx/fy/cx/cy 后结果按真实投影变化；
11. 非法 K、非有限 pose、`z<=0` 和 shape 不一致必须显式报错；
12. CPU 重复运行和不同 attention layer 使用完全相同的整数索引。

### 13.2 Planner 公平性

旧 sphere case 与 projected case 的下列字段必须相同：

```text
candidate_block_ids
ranked_candidate_block_ids
retrieval similarities
token budget
fill mode
retrieval layout
```

允许且预期变化的字段只有：

```text
motion keep ratios
base token counts
由 token 预算导致的 selected block 数
slot loads
```

如果 retrieval ranking 也发生变化，说明实现不再是单变量实验。

### 13.3 真实 checkpoint 冒烟

按以下顺序：

1. 每类 action 一个 24-latent prompt，检查至少一次 retrieval event；
2. MBench typical-8，比较旧 sphere 与 projected-unfilled；
3. loop-closure 10s 的 30 个 prompt，按旋转、平移、混合运动分组；
4. 通过后再运行 15s/20s/30s；
5. 最终质量报告必须同时给出 token 数、峰值显存和 retrieval latency。

## 14. 已知边界

1. 多深度投影仍不是真实深度，不能精确描述遮挡与 disocclusion；
2. 双向 overlap 把尺度变化视为需要保留的信息，这是面向视频生成的工程定义，不是严格的
   “新视野面积”；
3. 静止相机的动态物体不属于相机几何，需要独立 content-motion 消融；
4. 相机训练轨迹单位没有绝对米制含义，`D=8` 必须通过开发集敏感性验证；
5. projected compression 修正后，旧 sphere FOV retrieval 仍可能限制平移场景的历史选择；
6. `ceil(qF)` 会让任何严格正比例至少保留一个 token，但这只是整数载荷要求，不是比例量化；
7. 匀速纯 action 在不同 chunk 重复同一压缩模板是正确结果，不能误判为 planner 没有动态运行。

## 15. 实现与提交顺序

每个模块单独提交并推送：

1. `Add bidirectional projected motion overlap`
   - 二维网格、完整旋转、单深度双向投影和几何测试；
2. `Add multidepth translation compression ratios`
   - 单一 scene scale、多深度聚合、所有 action 与速度测试；
3. `Add projected motion novelty retrieval case`
   - `motion_projected_unfilled`、planner/runtime 日志和公平性测试；
4. `Validate projected motion compression on checkpoint`
   - 24-latent action 冒烟、typical-8 和 loop-closure 记录；
5. `Add projected fill strategy ablations`
   - 只有基础 case 通过后再增加 backfill/duplicate。
