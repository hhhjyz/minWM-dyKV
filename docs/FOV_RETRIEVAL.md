# FOV 检索

本文描述 `retrieval_mode=fov`。当前另有一个仅用于公平消融的
`worldkv_pose_no_compression` case，它使用 WorldKV 平均 C2W 位姿得分，不经过本模块；
两者的固定变量和原仓库非检索差异见
[`WORLDKV_RETRIEVAL_ABLATION.md`](WORLDKV_RETRIEVAL_ABLATION.md)。

## 度量方法

dyKV 采用 HY-WorldPlay 的检索策略。对于当前相机位姿 `C` 和历史位姿 `H`，算法在以
`C` 为中心、半径为 8 的球体内采样，并计算：

```text
overlap(C, H) = count(points in FOV(C) and FOV(H)) / count(points in FOV(C))
distance(C, H) = 1 - overlap(C, H)
```

视锥只从当前帧和历史帧的归一化相机内参 `K` 推导边界：

```text
left   = atan(-cx / fx)          right  = atan((1 - cx) / fx)
top    = atan(-cy / fy)          bottom = atan((1 - cy) / fy)
```

当前 `Wan21/wan_inference.py` 的轨迹推理入口实际构造
`fx=fy=cx=cy=0.5`，对应 `90°×90°`；该矩阵对同一视频的所有帧相同。算法只从 `K`
推导 FOV，不再读取独立的固定角度参数，但当前也尚未从 MBench 样本读取逐样本真实标定。
相对于历史相机距离超过半径 8 的历史点仍不计入重叠。

固定 FOV 和混合 FOV 消融已经移除，运行配置中也不再保存水平/垂直固定角度。当前 query
缺少或包含非法 `K` 时检索直接报错；历史块缺少 `K` 时跳过该块。这样不会在不同样本间
悄然混用真实内参与固定角度。

检索块时，将每个当前帧分别与历史块的首帧位姿和中间帧位姿比较。先对这两个距离取平均，
再对所有当前帧取平均。选择器同时返回两种结果：按源帧预算截断且已恢复时间顺序的列表，
供非扩容和固定 WorldKV case 使用；以及完整的距离排序，供 `packed_chunks*` 在
8-latent **物理 token 预算**内重新装入更多压缩历史。因而扩容 case
不是只对最前面的 8 个源 latent 做裁剪。

## 确定性探针

HY-WorldPlay 使用随机蒙特卡洛采样点。dyKV 将其替换为确定性的黄金角球面序列，并对半径
进行体积分布校正。该修改保留原几何估计方式，同时保证：

- 同一次运行中的候选得分可比较；
- 视频生成使用的 seed 不会改变检索决策；
- 单元测试与重复实验可复现。

探针张量在每次推理开始时只生成一次，后续块重复使用。

## 相机几何精度

推理入口当前使用 FP32 保存权威 `viewmats/Ks`，FOV 排序、CPU bank 归档和 yaw 空间裁剪
共享这份无 BF16 预量化的几何数据。PRoPE 只在自身算子入口把局部副本转换成 Q/K/V dtype，
保持原有低精度模型计算路径。该修复解决了纯 `j/l` 旋转因 BF16 矩阵误差进入 novelty
fallback 的问题；旧视频不会自动改变，仍需重新生成并核验 `compression_modes`。完整修复
和日志证据见
[`RETRIEVAL_ROTATION_COMPRESSION_FLOW.md`](RETRIEVAL_ROTATION_COMPRESSION_FLOW.md)。

## 候选边界

`DyKVBank.evicted_candidates` 会先排除仍存在于在线 sink 或 recent cache 中的块，之后才
进行 FOV 评分。缺少相机位姿矩阵或内参的历史块都会被跳过，不会静默回退到第二种检索
度量或固定视场。

## 验证

测试覆盖确定且有界的探针、相同相机下接近完整的重叠、相反朝向相机下接近零的重叠、
内参对应的实际视场角、缺失内参拒绝策略、闭环路径偏好、
检索载荷的时序顺序、非扩容 case 的源帧预算，以及扩容 case 的物理 token 预算。
