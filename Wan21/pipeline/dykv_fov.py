"""HY-WorldPlay-style camera-FOV retrieval for dyKV memory blocks."""

from __future__ import annotations

import math
from typing import Sequence

import torch


def deterministic_sphere_points(
    count: int,
    radius: float,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Deterministic, approximately uniform probe points inside a sphere."""

    if count <= 0 or radius <= 0:
        raise ValueError("FOV probe count and radius must be positive")
    index = torch.arange(count, device=device, dtype=dtype) + 0.5
    fraction = index / float(count)
    z = 1.0 - 2.0 * fraction
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    theta = index * golden_angle
    xy = torch.sqrt(torch.clamp(1.0 - z.square(), min=0.0))
    direction = torch.stack(
        (xy * torch.cos(theta), xy * torch.sin(theta), z), dim=-1
    )
    return direction * (float(radius) * fraction.pow(1.0 / 3.0))[:, None]


def _camera_center_and_angles(w2c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]
    center = -rotation.T @ translation
    forward = rotation.T[:, 2]
    yaw = torch.atan2(forward[0], forward[2]) * (180.0 / math.pi)
    pitch = torch.atan2(
        forward[1], torch.sqrt(forward[0].square() + forward[2].square())
    ) * (180.0 / math.pi)
    return center, pitch, yaw


def _inside_angular_fov(
    points: torch.Tensor,
    center: torch.Tensor,
    pitch: torch.Tensor,
    yaw: torch.Tensor,
    *,
    horizontal_bounds: tuple[float, float],
    vertical_bounds: tuple[float, float],
) -> torch.Tensor:
    vectors = points - center[None, :]
    x, y, z = vectors.unbind(dim=-1)
    azimuth = torch.atan2(x, z) * (180.0 / math.pi)
    elevation = torch.atan2(y, torch.sqrt(x.square() + z.square())) * (180.0 / math.pi)
    yaw_delta = torch.remainder(azimuth - yaw + 180.0, 360.0) - 180.0
    pitch_delta = torch.remainder(elevation - pitch + 180.0, 360.0) - 180.0
    return (
        (yaw_delta >= float(horizontal_bounds[0]))
        & (yaw_delta <= float(horizontal_bounds[1]))
        & (pitch_delta >= float(vertical_bounds[0]))
        & (pitch_delta <= float(vertical_bounds[1]))
    )


def angular_fov_bounds(
    K: torch.Tensor | None,
    *,
    source: str = "intrinsics",
    horizontal_degrees: float = 60.0,
    vertical_degrees: float = 35.0,
) -> tuple[tuple[float, float], tuple[float, float], str]:
    """Resolve angular image bounds in degrees.

    Intrinsics are normalized to image width/height.  Missing or invalid
    intrinsics deliberately fall back to HY-WorldPlay's fixed FOV so older
    archives remain retrievable.
    """

    if source not in {"fixed", "intrinsics"}:
        raise ValueError("FOV source must be fixed or intrinsics")
    if source == "intrinsics" and K is not None:
        matrix = K.detach().to(device="cpu", dtype=torch.float64)
        if matrix.shape == (3, 3):
            fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
            cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
            if all(math.isfinite(value) for value in (fx, fy, cx, cy)) and fx > 0 and fy > 0:
                scale = 180.0 / math.pi
                return (
                    (math.atan(-cx / fx) * scale, math.atan((1.0 - cx) / fx) * scale),
                    (math.atan(-cy / fy) * scale, math.atan((1.0 - cy) / fy) * scale),
                    "intrinsics",
                )
    if not 0.0 < float(horizontal_degrees) < 180.0:
        raise ValueError("horizontal FOV must be in (0, 180) degrees")
    if not 0.0 < float(vertical_degrees) < 180.0:
        raise ValueError("vertical FOV must be in (0, 180) degrees")
    return (
        (-float(horizontal_degrees) / 2.0, float(horizontal_degrees) / 2.0),
        (-float(vertical_degrees) / 2.0, float(vertical_degrees) / 2.0),
        "fixed" if source == "fixed" else "fixed_fallback",
    )


def fov_overlap(
    current_w2c: torch.Tensor,
    historical_w2c: torch.Tensor,
    probe_points: torch.Tensor,
    *,
    current_K: torch.Tensor | None = None,
    historical_K: torch.Tensor | None = None,
    fov_source: str = "intrinsics",
    horizontal_degrees: float = 60.0,
    vertical_degrees: float = 35.0,
    radius: float = 8.0,
) -> torch.Tensor:
    """Return ``|current FOV intersect historical FOV| / |current FOV|``.

    As in HY-WorldPlay, poses are first expressed relative to the current camera,
    angular inclusion is evaluated in world coordinates, and historical probes
    outside the finite memory radius are discarded.
    """

    current_w2c = current_w2c.to(device=probe_points.device, dtype=torch.float32)
    historical_w2c = historical_w2c.to(device=probe_points.device, dtype=torch.float32)
    current_c2w = torch.linalg.inv(current_w2c)
    historical_c2w = torch.linalg.inv(historical_w2c)
    reference = current_w2c
    current_relative = torch.linalg.inv(reference @ current_c2w)
    historical_relative = torch.linalg.inv(reference @ historical_c2w)

    current_center, current_pitch, current_yaw = _camera_center_and_angles(current_relative)
    historical_center, historical_pitch, historical_yaw = _camera_center_and_angles(
        historical_relative
    )
    current_horizontal, current_vertical, _ = angular_fov_bounds(
        current_K,
        source=fov_source,
        horizontal_degrees=horizontal_degrees,
        vertical_degrees=vertical_degrees,
    )
    historical_horizontal, historical_vertical, _ = angular_fov_bounds(
        historical_K,
        source=fov_source,
        horizontal_degrees=horizontal_degrees,
        vertical_degrees=vertical_degrees,
    )
    points_world = probe_points + current_center[None, :]
    current_mask = _inside_angular_fov(
        points_world,
        current_center,
        current_pitch,
        current_yaw,
        horizontal_bounds=current_horizontal,
        vertical_bounds=current_vertical,
    )
    historical_mask = _inside_angular_fov(
        points_world,
        historical_center,
        historical_pitch,
        historical_yaw,
        horizontal_bounds=historical_horizontal,
        vertical_bounds=historical_vertical,
    )
    historical_mask &= (
        torch.linalg.vector_norm(points_world - historical_center[None, :], dim=-1)
        < float(radius)
    )
    denominator = current_mask.sum()
    if int(denominator.item()) == 0:
        return torch.zeros((), device=probe_points.device, dtype=torch.float32)
    return (current_mask & historical_mask).sum().float() / denominator.float()


def chunk_fov_distance(
    current_viewmats: torch.Tensor,
    historical_viewmats: torch.Tensor,
    probe_points: torch.Tensor,
    *,
    current_Ks: torch.Tensor | None = None,
    historical_Ks: torch.Tensor | None = None,
    fov_source: str = "intrinsics",
    horizontal_degrees: float = 60.0,
    vertical_degrees: float = 35.0,
    radius: float = 8.0,
) -> torch.Tensor:
    """HY-WorldPlay's query-chunk to history-chunk distance.

    Every current frame is compared with two historical representatives: the
    first frame and the midpoint frame. Distances are averaged across query
    frames, matching the reference four-frame memory selector.
    """

    if current_viewmats.ndim == 4:
        current_viewmats = current_viewmats[0]
    if historical_viewmats.ndim == 4:
        historical_viewmats = historical_viewmats[0]
    if current_Ks is not None and current_Ks.ndim == 4:
        current_Ks = current_Ks[0]
    if historical_Ks is not None and historical_Ks.ndim == 4:
        historical_Ks = historical_Ks[0]
    if current_viewmats.shape[0] == 0 or historical_viewmats.shape[0] == 0:
        raise ValueError("FOV retrieval chunks cannot be empty")
    representatives = [0, min(historical_viewmats.shape[0] // 2, historical_viewmats.shape[0] - 1)]
    per_query = []
    for query_index, current_pose in enumerate(current_viewmats):
        current_K = None
        if current_Ks is not None and query_index < current_Ks.shape[0]:
            current_K = current_Ks[query_index]
        similarities = [
            fov_overlap(
                current_pose,
                historical_viewmats[index],
                probe_points,
                current_K=current_K,
                historical_K=(
                    historical_Ks[index]
                    if historical_Ks is not None and index < historical_Ks.shape[0]
                    else None
                ),
                fov_source=fov_source,
                horizontal_degrees=horizontal_degrees,
                vertical_degrees=vertical_degrees,
                radius=radius,
            )
            for index in representatives
        ]
        per_query.append(1.0 - torch.stack(similarities).mean())
    return torch.stack(per_query).mean()


def select_fov_blocks(
    bank,
    candidate_indices: Sequence[int],
    *,
    current_viewmats: torch.Tensor,
    current_Ks: torch.Tensor | None = None,
    memory_frames: int,
    probe_points: torch.Tensor,
    horizontal_degrees: float = 60.0,
    vertical_degrees: float = 35.0,
    radius: float = 8.0,
    fov_source: str = "intrinsics",
) -> tuple[list[int], list[int], list[float]]:
    """Select closest blocks and return their complete FOV ranking.

    The selected indices are chronological for attention composition. Ranked
    indices and distances remain score-aligned for diagnostics.
    """

    scored: list[tuple[int, float]] = []
    for index in candidate_indices:
        block = bank.blocks[int(index)]
        if block.viewmats is None:
            continue
        distance = chunk_fov_distance(
            current_viewmats,
            block.viewmats,
            probe_points,
            current_Ks=current_Ks,
            historical_Ks=getattr(block, "Ks", None),
            fov_source=fov_source,
            horizontal_degrees=horizontal_degrees,
            vertical_degrees=vertical_degrees,
            radius=radius,
        )
        scored.append((int(index), float(distance.item())))
    scored.sort(key=lambda item: (item[1], bank.blocks[item[0]].frame_start))

    selected: list[int] = []
    used_frames = 0
    for index, _ in scored:
        block_frames = int(bank.blocks[index].frame_count)
        if used_frames + block_frames > int(memory_frames):
            continue
        selected.append(index)
        used_frames += block_frames
        if used_frames == int(memory_frames):
            break
    selected.sort(key=lambda index: bank.blocks[index].frame_start)
    return (
        selected,
        [index for index, _ in scored],
        [distance for _, distance in scored],
    )
