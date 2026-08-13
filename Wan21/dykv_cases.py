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
    retrieval_fov_source: str
    compression_fov_source: str
    description: str
    packing_mode: str = "none"
    retrieval_frames: int = 8
    compression_keep_ratio: float = 0.5
    sink_mode: str = FIXED_SINK_MODE
    sink_frames: int = FIXED_SINK_FRAMES

    @property
    def local_frames(self) -> int:
        """Live rolling context, including the current four-frame query."""

        return 8 if self.enabled else 16


DYKV_CASES = {
    case.name: case
    for case in (
        DyKVCase(
            "baseline",
            False,
            "none",
            "intrinsics",
            "intrinsics",
            "固定前 4 帧 sink + rolling local，不启用长期检索",
        ),
        DyKVCase(
            "retrieval_no_compression",
            True,
            "none",
            "intrinsics",
            "intrinsics",
            "按相机内参进行 FOV 检索，但不压缩检索 KV",
        ),
        DyKVCase(
            "fixed_novelty",
            True,
            "fixed_novelty",
            "intrinsics",
            "intrinsics",
            "内参 FOV 检索与固定锚点/新颖性压缩",
        ),
        DyKVCase(
            "yaw_fixed_fov",
            True,
            "yaw_fov",
            "fixed",
            "fixed",
            "固定 60°×35° 检索与固定 60° 横向裁剪（F0）",
        ),
        DyKVCase(
            "yaw_mixed_fov",
            True,
            "yaw_fov",
            "fixed",
            "intrinsics",
            "固定角度检索、内参裁剪，用于隔离两处 FOV 的影响（F1）",
        ),
        DyKVCase(
            "yaw_intrinsics",
            True,
            "yaw_fov",
            "intrinsics",
            "intrinsics",
            "检索和动态裁剪都使用相机内参（F2，默认完整方法）",
        ),
        DyKVCase(
            "packed_chunks",
            True,
            "yaw_fov",
            "intrinsics",
            "intrinsics",
            "固定档位动态压缩，并用完整 chunk 扩充 retrieval（E1）",
            packing_mode="whole_chunks",
        ),
        DyKVCase(
            "packed_chunks_latent",
            True,
            "yaw_fov",
            "intrinsics",
            "intrinsics",
            "固定档位完整 chunk 装箱，并用单 latent 补齐尾部（E2）",
            packing_mode="whole_chunks_and_latents",
        ),
        DyKVCase(
            "retr8_compression_r050",
            True,
            "fixed_novelty",
            "intrinsics",
            "intrinsics",
            "minWM-back B：检索 8 帧，每 chunk 保留完整 anchor + 3×50%",
            packing_mode="fixed_worldkv",
            retrieval_frames=8,
            compression_keep_ratio=0.5,
        ),
        DyKVCase(
            "retr12_compression_r050",
            True,
            "fixed_novelty",
            "intrinsics",
            "intrinsics",
            "minWM-back C：检索 12 帧，每 chunk 保留完整 anchor + 3×50%",
            packing_mode="fixed_worldkv",
            retrieval_frames=12,
            compression_keep_ratio=0.5,
        ),
        DyKVCase(
            "retr16_compression_r033",
            True,
            "fixed_novelty",
            "intrinsics",
            "intrinsics",
            "minWM-back D：检索 16 帧，每 chunk 保留完整 anchor + 3×1/3",
            packing_mode="fixed_worldkv",
            retrieval_frames=16,
            compression_keep_ratio=1.0 / 3.0,
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
            f"sink={case.sink_mode}:{case.sink_frames}\t{case.description}"
        )


if __name__ == "__main__":
    main()
