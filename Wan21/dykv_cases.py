"""Registered dyKV experiment cases.

The case name is the only public experiment selector.  It expands into a
coherent retrieval/compression preset so individual geometry switches do not
become an unmanageable collection of CLI hyperparameters.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


FIXED_SINK_MODE = "fixed"
FIXED_SINK_FRAMES = 4


@dataclass(frozen=True)
class DyKVCase:
    name: str
    enabled: bool
    compression_mode: str
    description: str
    packing_mode: str = "none"
    retrieval_mode: str = "fov"
    retrieval_frames: int = 8
    compression_keep_ratio: float = 0.5
    retrieval_layout: str = "source_ordered"
    retrieval_order: str = "source_ordered"
    motion_geometry_mode: str = "projected_multidepth"
    sink_mode: str = FIXED_SINK_MODE
    sink_frames: int = FIXED_SINK_FRAMES
    retrieval_rope_mode: str = "fixed_slot"
    motion_allocation_mode: str = "legacy"
    novelty_feature_mode: str = "cached_roped_k"

    @property
    def local_frames(self) -> int:
        """Live rolling context, including the current four-frame query."""

        return 8 if self.enabled else 16

    @property
    def memory_frames(self) -> int:
        """Virtual retrieval region; baseline intentionally leaves it empty."""

        return 8 if self.enabled else 0


DYKV_CASES = {
    case.name: case
    for case in (
        DyKVCase(
            "baseline",
            False,
            "none",
            "tri-region RoPE（4+0+16）+ rolling local，不启用长期检索",
            retrieval_mode="none",
            retrieval_frames=0,
            compression_keep_ratio=1.0,
        ),
        DyKVCase(
            "baseline_honest",
            False,
            "none",
            "baseline + honest 绝对位置 RoPE（local 和 query 不 rebase，sink 相对距离随长度增长）",
            retrieval_mode="none",
            retrieval_frames=0,
            compression_keep_ratio=1.0,
            retrieval_rope_mode="honest",
        ),
        DyKVCase(
            "retrieval_no_compression",
            True,
            "none",
            "按相机内参进行 FOV 检索，但不压缩检索 KV",
        ),
        DyKVCase(
            "retrieval_no_compression_honest",
            True,
            "none",
            "无压缩 FOV 检索 + honest 绝对位置 RoPE（不 rebase，相对距离精确）",
            retrieval_rope_mode="honest",
        ),
        DyKVCase(
            "retrieval_no_compression_age",
            True,
            "none",
            "无压缩 FOV 检索 + age-ordered 检索 RoPE（按真实 age 排序映射到 4-11）",
            retrieval_rope_mode="age_ordered",
        ),
        DyKVCase(
            "retrieval_no_compression_relevance_order",
            True,
            "none",
            "与无压缩 FOV 检索相同，但高相关 chunk 的 RoPE 位置更靠近当前 query",
            retrieval_order="relevance_near_query",
        ),
        DyKVCase(
            "worldkv_pose_no_compression",
            True,
            "none",
            "WorldKV 原始平均位姿检索，但不压缩检索 KV",
            retrieval_mode="worldkv_pose",
        ),
        DyKVCase(
            "yaw_intrinsics",
            True,
            "yaw_fov",
            "检索和动态裁剪都使用相机内参（E0，兼容默认）",
        ),
        DyKVCase(
            "packed_chunks",
            True,
            "yaw_fov",
            "固定档位动态压缩，并用完整 chunk 扩充 retrieval（E1）",
            packing_mode="whole_chunks",
        ),
        DyKVCase(
            "packed_chunks_latent",
            True,
            "yaw_fov",
            "固定档位完整 chunk 装箱，并用单 latent 补齐尾部（E2）",
            packing_mode="whole_chunks_and_latents",
        ),
        DyKVCase(
            "retr8_compression_r050",
            True,
            "fixed_novelty",
            "minWM-back B：检索 8 帧，每 chunk 保留完整 anchor + 3×50%",
            packing_mode="fixed_worldkv",
            retrieval_frames=8,
            compression_keep_ratio=0.5,
        ),
        DyKVCase(
            "retr12_compression_r050",
            True,
            "fixed_novelty",
            "minWM-back C：检索 12 帧，每 chunk 保留完整 anchor + 3×50%",
            packing_mode="fixed_worldkv",
            retrieval_frames=12,
            compression_keep_ratio=0.5,
        ),
        DyKVCase(
            "retr16_r033_slot_packed",
            True,
            "fixed_novelty",
            "retr16 固定 1/3 novelty 的旧 slot-order 布局诊断",
            packing_mode="fixed_worldkv",
            retrieval_frames=16,
            compression_keep_ratio=1.0 / 3.0,
            retrieval_layout="slot_packed",
        ),
        DyKVCase(
            "retr16_compression_r033",
            True,
            "fixed_novelty",
            "minWM-back D：检索 16 帧，每 chunk 保留完整 anchor + 3×1/3",
            packing_mode="fixed_worldkv",
            retrieval_frames=16,
            compression_keep_ratio=1.0 / 3.0,
        ),
        DyKVCase(
            "retr16_compression_r033_honest",
            True,
            "fixed_novelty",
            "retr16_compression_r033 + honest 绝对位置 RoPE",
            packing_mode="fixed_worldkv",
            retrieval_frames=16,
            compression_keep_ratio=1.0 / 3.0,
            retrieval_rope_mode="honest",
        ),
        DyKVCase(
            "motion_novelty_sphere_unfilled",
            True,
            "motion_novelty",
            "旧 sphere-FOV 比例 + novelty，flat 总预算且允许欠填（几何消融）",
            packing_mode="motion_novelty_flat",
            retrieval_frames=16,
            retrieval_layout="flat_source_ordered",
            motion_geometry_mode="sphere_fov",
        ),
        DyKVCase(
            "motion_novelty_sphere_unfilled_honest",
            True,
            "motion_novelty",
            "motion_novelty_sphere_unfilled + honest 绝对位置 RoPE",
            packing_mode="motion_novelty_flat",
            retrieval_frames=16,
            retrieval_layout="flat_source_ordered",
            motion_geometry_mode="sphere_fov",
            retrieval_rope_mode="honest",
        ),
        DyKVCase(
            "motion_novelty_slot_capped",
            True,
            "motion_novelty",
            "双向二维多深度投影比例 + novelty，按源顺序并保留单槽容量限制",
            packing_mode="motion_novelty_slot_capped",
            retrieval_frames=16,
            retrieval_layout="source_ordered",
        ),
        DyKVCase(
            "motion_novelty_slot_capped_honest",
            True,
            "motion_novelty",
            "motion_novelty_slot_capped + honest 绝对位置 RoPE",
            packing_mode="motion_novelty_slot_capped",
            retrieval_frames=16,
            retrieval_layout="source_ordered",
            retrieval_rope_mode="honest",
        ),
        DyKVCase(
            "motion_novelty_unfilled",
            True,
            "motion_novelty",
            "双向二维多深度投影比例 + novelty，flat 总预算且允许欠填",
            packing_mode="motion_novelty_flat",
            retrieval_frames=16,
            retrieval_layout="flat_source_ordered",
        ),
        DyKVCase(
            "motion_novelty_unfilled_honest",
            True,
            "motion_novelty",
            "motion_novelty_unfilled + honest 绝对位置 RoPE",
            packing_mode="motion_novelty_flat",
            retrieval_frames=16,
            retrieval_layout="flat_source_ordered",
            retrieval_rope_mode="honest",
        ),
        DyKVCase(
            "motion_novelty_backfill",
            True,
            "motion_novelty",
            "投影比例 novelty 后补回唯一 token 至 reference 目标",
            packing_mode="motion_novelty_backfill",
            retrieval_frames=16,
            retrieval_layout="flat_source_ordered",
        ),
        DyKVCase(
            "motion_novelty_backfill_honest",
            True,
            "motion_novelty",
            "motion_novelty_backfill + honest 绝对位置 RoPE",
            packing_mode="motion_novelty_backfill",
            retrieval_frames=16,
            retrieval_layout="flat_source_ordered",
            retrieval_rope_mode="honest",
        ),
        DyKVCase(
            "motion_novelty_duplicate",
            True,
            "motion_novelty",
            "投影比例后重复最高相关 chunk 的基础 token，对齐 backfill 长度与 slot load",
            packing_mode="motion_novelty_duplicate",
            retrieval_frames=16,
            retrieval_layout="flat_source_ordered",
        ),
        DyKVCase(
            "motion_novelty_duplicate_honest",
            True,
            "motion_novelty",
            "motion_novelty_duplicate + honest 绝对位置 RoPE",
            packing_mode="motion_novelty_duplicate",
            retrieval_frames=16,
            retrieval_layout="flat_source_ordered",
            retrieval_rope_mode="honest",
        ),
        DyKVCase(
            "motion_alloc_cam_4chunk",
            True,
            "motion_novelty",
            "固定检索 4 chunk/8F；anchor 完整，剩余 4F 按相机运动比例连续分配",
            packing_mode="motion_alloc_4chunk",
            retrieval_frames=16,
            retrieval_layout="flat_source_ordered",
            motion_allocation_mode="camera_budgeted",
        ),
        DyKVCase(
            "motion_alloc_cam_content_4chunk",
            True,
            "motion_novelty",
            "固定 4 chunk/8F；按 max(相机运动, RoPE-free V 内容变化) 分配",
            packing_mode="motion_alloc_4chunk",
            retrieval_frames=16,
            retrieval_layout="flat_source_ordered",
            motion_allocation_mode="camera_content_budgeted",
        ),
        DyKVCase(
            "motion_alloc_cam_content_prerope_4chunk",
            True,
            "motion_novelty",
            "camera+content 固定预算，并用 layer-0 pre-RoPE K 决定保留 token",
            packing_mode="motion_alloc_4chunk",
            retrieval_frames=16,
            retrieval_layout="flat_source_ordered",
            motion_allocation_mode="camera_content_budgeted",
            novelty_feature_mode="pre_rope_k",
        ),
    )
}

DEFAULT_DYKV_CASE = "yaw_intrinsics"


def get_dykv_case(name: str) -> DyKVCase:
    try:
        return DYKV_CASES[str(name)]
    except KeyError as error:
        choices = ", ".join(DYKV_CASES)
        raise ValueError(f"unknown dyKV case {name!r}; choose one of: {choices}") from error


def resolve_dykv_case(name: str | None, *, enabled: bool) -> DyKVCase:
    requested = name or (DEFAULT_DYKV_CASE if enabled else "baseline")
    case = get_dykv_case(requested)
    if bool(case.enabled) != bool(enabled):
        expected = "with --dykv" if case.enabled else "without --dykv"
        raise ValueError(f"case {case.name!r} must be run {expected}")
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description="列出或校验 dyKV 实验 case")
    parser.add_argument("--validate", metavar="CASE", help="只校验一个 case 名称")
    args = parser.parse_args()
    if args.validate:
        get_dykv_case(args.validate)
        return
    for case in DYKV_CASES.values():
        marker = " (default)" if case.name == DEFAULT_DYKV_CASE else ""
        print(
            f"{case.name}{marker}\t"
            f"sink={case.sink_mode}:{case.sink_frames}\t"
            f"retrieval={case.retrieval_mode}\t{case.description}"
        )


if __name__ == "__main__":
    main()
