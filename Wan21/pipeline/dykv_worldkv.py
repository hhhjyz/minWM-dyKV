"""WorldKV camera-pose retrieval adapted to the dyKV memory bank.

Only the ranking score is ported from WorldKV. Candidate eligibility, memory
budget, chronological materialization, and RoPE rebasing remain owned by dyKV
so retrieval-algorithm ablations differ by one controlled variable.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch


def _mean_c2w_pose(viewmats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return WorldKV's mean translation and mean rotation for one chunk.

    minWM stores W2C matrices, whereas WorldKV scores absolute C2W poses. The
    reference implementation averages the matrices directly; it does not
    project the mean rotation back to SO(3), so this port intentionally does
    the same.
    """

    poses = viewmats.detach().to(device="cpu", dtype=torch.float32)
    if poses.ndim == 4:
        if poses.shape[0] != 1:
            raise ValueError("WorldKV pose retrieval requires batch size one")
        poses = poses[0]
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4) or poses.shape[0] == 0:
        raise ValueError(
            "WorldKV pose retrieval requires non-empty [frames,4,4] viewmats"
        )
    c2ws = torch.linalg.inv(poses)
    return c2ws[:, :3, 3].mean(dim=0), c2ws[:, :3, :3].mean(dim=0)


def _normalize_by_candidate_max(values: torch.Tensor) -> torch.Tensor:
    """Match WorldKV's per-query candidate-set normalization."""

    maximum = values.max()
    return values / maximum if bool(maximum > 0) else values


def select_worldkv_pose_blocks(
    bank,
    candidate_indices: Sequence[int],
    *,
    current_viewmats: torch.Tensor,
    memory_frames: int,
) -> tuple[list[int], list[int], list[float], dict[str, list[float]]]:
    """Rank candidates with WorldKV's translation/rotation pose distance.

    The reference score is

    ``0.5 * normalize(||t_h-t_q||_2^2) + 0.5 * normalize(geodesic(R_h,R_q))``.

    Normalization is performed independently over the current candidate set.
    Score ties use the older frame first for deterministic compatibility with
    the existing dyKV selector; selected blocks are then made chronological
    before attention composition.
    """

    current_translation, current_rotation = _mean_c2w_pose(current_viewmats)
    valid_indices: list[int] = []
    translations: list[torch.Tensor] = []
    rotations: list[torch.Tensor] = []
    for index in candidate_indices:
        block = bank.blocks[int(index)]
        if block.viewmats is None:
            continue
        translation, rotation = _mean_c2w_pose(block.viewmats)
        valid_indices.append(int(index))
        translations.append(translation)
        rotations.append(rotation)

    empty_components = {
        "translation_squared": [],
        "rotation_degrees": [],
        "translation_normalized": [],
        "rotation_normalized": [],
    }
    if not valid_indices:
        return [], [], [], empty_components

    candidate_translations = torch.stack(translations)
    translation_squared = (
        (candidate_translations - current_translation) ** 2
    ).sum(dim=-1)

    candidate_rotations = torch.stack(rotations)
    relative_rotation = (
        candidate_rotations.transpose(-1, -2) @ current_rotation.unsqueeze(0)
    )
    trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    rotation_radians = torch.acos(cosine)

    translation_normalized = _normalize_by_candidate_max(translation_squared)
    rotation_normalized = _normalize_by_candidate_max(rotation_radians)
    combined = 0.5 * translation_normalized + 0.5 * rotation_normalized

    score_rows = [
        (
            index,
            float(combined[offset]),
            float(translation_squared[offset]),
            math.degrees(float(rotation_radians[offset])),
            float(translation_normalized[offset]),
            float(rotation_normalized[offset]),
        )
        for offset, index in enumerate(valid_indices)
    ]
    score_rows.sort(key=lambda row: (row[1], bank.blocks[row[0]].frame_start))

    selected: list[int] = []
    used_frames = 0
    for index, *_ in score_rows:
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
        [row[0] for row in score_rows],
        [row[1] for row in score_rows],
        {
            "translation_squared": [row[2] for row in score_rows],
            "rotation_degrees": [row[3] for row in score_rows],
            "translation_normalized": [row[4] for row in score_rows],
            "rotation_normalized": [row[5] for row in score_rows],
        },
    )
