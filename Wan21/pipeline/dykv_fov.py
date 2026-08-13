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
    half_horizontal: float,
    half_vertical: float,
) -> torch.Tensor:
    vectors = points - center[None, :]
    x, y, z = vectors.unbind(dim=-1)
    azimuth = torch.atan2(x, z) * (180.0 / math.pi)
    elevation = torch.atan2(y, torch.sqrt(x.square() + z.square())) * (180.0 / math.pi)
    yaw_delta = torch.remainder(azimuth - yaw + 180.0, 360.0) - 180.0
    pitch_delta = torch.remainder(elevation - pitch + 180.0, 360.0) - 180.0
    return (yaw_delta.abs() < half_horizontal) & (pitch_delta.abs() < half_vertical)


def fov_overlap(
    current_w2c: torch.Tensor,
    historical_w2c: torch.Tensor,
    probe_points: torch.Tensor,
    *,
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
    points_world = probe_points + current_center[None, :]
    current_mask = _inside_angular_fov(
        points_world,
        current_center,
        current_pitch,
        current_yaw,
        half_horizontal=float(horizontal_degrees) / 2.0,
        half_vertical=float(vertical_degrees) / 2.0,
    )
    historical_mask = _inside_angular_fov(
        points_world,
        historical_center,
        historical_pitch,
        historical_yaw,
        half_horizontal=float(horizontal_degrees) / 2.0,
        half_vertical=float(vertical_degrees) / 2.0,
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
    if current_viewmats.shape[0] == 0 or historical_viewmats.shape[0] == 0:
        raise ValueError("FOV retrieval chunks cannot be empty")
    representatives = [0, min(historical_viewmats.shape[0] // 2, historical_viewmats.shape[0] - 1)]
    per_query = []
    for current_pose in current_viewmats:
        similarities = [
            fov_overlap(
                current_pose,
                historical_viewmats[index],
                probe_points,
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
    memory_frames: int,
    probe_points: torch.Tensor,
    horizontal_degrees: float = 60.0,
    vertical_degrees: float = 35.0,
    radius: float = 8.0,
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
