"""Bidirectional image-plane overlap for motion-adaptive KV compression."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import torch


PROJECTED_MULTIDEPTH_MODE = "projected_multidepth"
DEFAULT_SCENE_SCALE = 8.0
DEPTH_FRACTIONS = (1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0)


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

    @property
    def relative_translation_distance(self) -> float:
        return math.sqrt(sum(value * value for value in self.relative_translation_xyz))


def _matrix(value: torch.Tensor, *, size: int, label: str) -> torch.Tensor:
    output = value.detach().to(device="cpu", dtype=torch.float64)
    if output.shape != (size, size):
        raise ValueError(f"projected motion {label} must have shape {size}x{size}")
    if not bool(torch.isfinite(output).all()):
        raise ValueError(f"projected motion {label} must be finite")
    return output


def _intrinsics(value: torch.Tensor, *, label: str) -> torch.Tensor:
    output = _matrix(value, size=3, label=label)
    if float(output[0, 0]) <= 0.0 or float(output[1, 1]) <= 0.0:
        raise ValueError(f"projected motion {label} focal lengths must be positive")
    try:
        torch.linalg.inv(output)
    except RuntimeError as error:
        raise ValueError(f"projected motion {label} must be invertible") from error
    return output


def _pose(value: torch.Tensor, *, label: str) -> torch.Tensor:
    output = _matrix(value, size=4, label=label)
    try:
        torch.linalg.inv(output)
    except RuntimeError as error:
        raise ValueError(f"projected motion {label} must be invertible") from error
    return output


@lru_cache(maxsize=16)
def _token_centers(height: int, width: int) -> torch.Tensor:
    if height <= 0 or width <= 0:
        raise ValueError("projected motion spatial shape must be positive")
    ys = (torch.arange(height, dtype=torch.float64) + 0.5) / float(height)
    xs = (torch.arange(width, dtype=torch.float64) + 0.5) / float(width)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack(
        (grid_x.reshape(-1), grid_y.reshape(-1), torch.ones(height * width)),
        dim=0,
    ).contiguous()


def _directional_overlap(
    source_w2c: torch.Tensor,
    target_w2c: torch.Tensor,
    source_K: torch.Tensor,
    target_K: torch.Tensor,
    pixels: torch.Tensor,
    *,
    depth: float,
) -> float:
    relative = target_w2c @ torch.linalg.inv(source_w2c)
    rays = torch.linalg.solve(source_K, pixels)
    source_points = rays * float(depth)
    target_points = (
        relative[:3, :3] @ source_points + relative[:3, 3].unsqueeze(1)
    )
    z = target_points[2]
    projected = target_K @ target_points
    safe_z = torch.where(z.abs() > 1e-12, z, torch.ones_like(z))
    u = projected[0] / safe_z
    v = projected[1] / safe_z
    valid = (
        (z > 1e-12)
        & (u >= 0.0)
        & (u < 1.0)
        & (v >= 0.0)
        & (v < 1.0)
        & torch.isfinite(u)
        & torch.isfinite(v)
    )
    return float(valid.sum().item()) / float(pixels.shape[1])


def _rotation_degrees(rotation: torch.Tensor) -> float:
    cosine = ((torch.trace(rotation) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return math.degrees(float(torch.acos(cosine)))


def projected_motion_overlap(
    current_w2c: torch.Tensor,
    anchor_w2c: torch.Tensor,
    current_K: torch.Tensor,
    anchor_K: torch.Tensor,
    spatial_shape: Sequence[int],
    *,
    scene_scale: float = DEFAULT_SCENE_SCALE,
) -> ProjectedOverlapResult:
    """Measure symmetric current/anchor image-grid overlap at fixed depths.

    Poses are world-to-camera matrices and intrinsics use normalized image
    coordinates. Translation is not identifiable without scene depth, so the
    single scene scale expands into fixed log-spaced quadrature depths.
    """

    if len(spatial_shape) != 2:
        raise ValueError("projected motion spatial shape must contain height and width")
    height, width = int(spatial_shape[0]), int(spatial_shape[1])
    pixels = _token_centers(height, width)
    current_pose = _pose(current_w2c, label="current pose")
    anchor_pose = _pose(anchor_w2c, label="anchor pose")
    current_intrinsics = _intrinsics(current_K, label="current intrinsics")
    anchor_intrinsics = _intrinsics(anchor_K, label="anchor intrinsics")
    scale = float(scene_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("projected motion scene scale must be positive and finite")
    depths = tuple(scale * fraction for fraction in DEPTH_FRACTIONS)

    forward = []
    backward = []
    symmetric = []
    for depth in depths:
        current_to_anchor = _directional_overlap(
            current_pose,
            anchor_pose,
            current_intrinsics,
            anchor_intrinsics,
            pixels,
            depth=depth,
        )
        anchor_to_current = _directional_overlap(
            anchor_pose,
            current_pose,
            anchor_intrinsics,
            current_intrinsics,
            pixels,
            depth=depth,
        )
        denominator = current_to_anchor + anchor_to_current
        harmonic = (
            2.0 * current_to_anchor * anchor_to_current / denominator
            if denominator > 0.0
            else 0.0
        )
        forward.append(current_to_anchor)
        backward.append(anchor_to_current)
        symmetric.append(harmonic)

    overlap = min(1.0, max(0.0, sum(symmetric) / len(symmetric)))
    keep_ratio = min(1.0, max(0.0, 1.0 - overlap))
    frame_tokens = height * width
    keep_tokens = min(
        frame_tokens,
        max(0, int(math.ceil(keep_ratio * frame_tokens))),
    )
    relative = anchor_pose @ torch.linalg.inv(current_pose)
    translation = tuple(float(value) for value in relative[:3, 3])
    return ProjectedOverlapResult(
        overlap_ratio=overlap,
        keep_ratio=keep_ratio,
        keep_tokens=keep_tokens,
        forward_overlaps=tuple(forward),
        backward_overlaps=tuple(backward),
        symmetric_overlaps=tuple(symmetric),
        depths=depths,
        relative_rotation_degrees=_rotation_degrees(relative[:3, :3]),
        relative_translation_xyz=translation,
    )
