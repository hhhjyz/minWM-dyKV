"""Predecessor-relative yaw compression and retrieval-region packing.

Retrieval relevance is intentionally computed elsewhere from the current query.
This module only decides how much of each retrieved historical chunk is novel
relative to that chunk's immediately preceding chunk, then packs those tokens
into the fixed eight-frame retrieval region.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import torch

from .dykv_memory import (
    DyKVBank,
    MemoryBlock,
    _horizontal_fov_bounds,
    _horizontal_ray_angles,
    _pure_yaw_delta,
    _single_batch_frames,
    build_yaw_crop_plan,
)
from .dykv_packing import (
    PACKING_ATOMS_PER_SLOT,
    FrameCrop,
    PackedFramePlan,
    PackedRetrievalPlan,
    _novelty_frame_indices,
    _spatial_token_indices,
    materialize_packed_retrieval,
    tier_atoms,
)


PREDECESSOR_KEEP_TIERS = (0.25, 0.5, 0.75, 1.0)


def quantize_incremental_ratio(ratio: float) -> float:
    """Quantize new-FOV ratio using the four user-defined left-closed bins."""

    value = float(ratio)
    if not math.isfinite(value):
        raise ValueError("incremental FOV ratio must be finite")
    value = min(1.0, max(0.0, value))
    if value < 0.25:
        return 0.25
    if value < 0.5:
        return 0.5
    if value < 0.75:
        return 0.75
    return 1.0


@dataclass(frozen=True)
class PredecessorCandidate:
    block_index: int
    predecessor_index: int | None
    predecessor_frame_start: int | None
    distance: float
    similarity: float
    keep_tier: float
    utility: float
    chunk_frames: tuple[FrameCrop, ...]
    tail_frames: tuple[FrameCrop, ...]
    incremental_yaw_degrees: float | None
    fallback_reason: str | None

    @property
    def cost_atoms(self) -> int:
        return sum(frame.atom_count for frame in self.chunk_frames)


def _intrinsics_for_block(
    block: MemoryBlock,
) -> torch.Tensor | None:
    poses = _single_batch_frames(block.viewmats, matrix_size=4)
    if poses is None:
        return None
    return _single_batch_frames(block.Ks, matrix_size=3)


def _merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for left, right in sorted(intervals):
        if not merged or left > merged[-1][1] + 1e-12:
            merged.append([float(left), float(right)])
        else:
            merged[-1][1] = max(merged[-1][1], float(right))
    return [(left, right) for left, right in merged]


def _subtract_interval(
    interval: tuple[float, float], covered: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    pieces = [interval]
    for cover_left, cover_right in covered:
        next_pieces = []
        for left, right in pieces:
            if cover_right <= left or cover_left >= right:
                next_pieces.append((left, right))
                continue
            if cover_left > left:
                next_pieces.append((left, min(right, cover_left)))
            if cover_right < right:
                next_pieces.append((max(left, cover_right), right))
        pieces = next_pieces
    return [(left, right) for left, right in pieces if right - left > 1e-12]


def _nearest_unwrapped(angle: float, reference: float) -> float:
    return float(angle) + round((float(reference) - float(angle)) / (2.0 * math.pi)) * (
        2.0 * math.pi
    )


def _rank_directional_columns(
    rays: torch.Tensor,
    *,
    yaw: float,
    novel_intervals: Sequence[tuple[float, float]],
    signed_increment: float,
    target_columns: int,
) -> tuple[int, ...]:
    world_rays = [yaw + float(value) for value in rays.tolist()]

    def interval_distance(value: float) -> float:
        if not novel_intervals:
            return 0.0
        return min(
            0.0 if left <= value <= right else min(abs(value - left), abs(value - right))
            for left, right in novel_intervals
        )

    if novel_intervals:
        ranked = sorted(
            range(len(world_rays)),
            key=lambda column: (interval_distance(world_rays[column]), column),
        )
    elif signed_increment > 0.0:
        ranked = sorted(range(len(world_rays)), key=lambda column: (-world_rays[column], column))
    else:
        ranked = sorted(range(len(world_rays)), key=lambda column: (world_rays[column], column))
    return tuple(sorted(ranked[: int(target_columns)]))


def _predecessor_geometry(
    predecessor: MemoryBlock,
    block: MemoryBlock,
    *,
    frame_tokens: int,
) -> tuple[tuple[FrameCrop, ...], tuple[FrameCrop, ...], float] | None:
    if block.spatial_shape is None:
        return None
    height, width = map(int, block.spatial_shape)
    if height * width != int(frame_tokens) or width % PACKING_ATOMS_PER_SLOT:
        return None
    previous_poses = _single_batch_frames(predecessor.viewmats, matrix_size=4)
    poses = _single_batch_frames(block.viewmats, matrix_size=4)
    previous_Ks = _intrinsics_for_block(predecessor)
    Ks = _intrinsics_for_block(block)
    if previous_poses is None or poses is None or previous_Ks is None or Ks is None:
        return None
    if previous_poses.shape[0] != predecessor.frame_count or poses.shape[0] != block.frame_count:
        return None
    if previous_Ks.shape[0] != predecessor.frame_count or Ks.shape[0] != block.frame_count:
        return None

    base = previous_poses[0]
    previous_yaws = [_pure_yaw_delta(base, pose) for pose in previous_poses]
    yaws = [_pure_yaw_delta(base, pose) for pose in poses]
    if any(value is None for value in previous_yaws + yaws):
        return None
    previous_yaws = [float(value) for value in previous_yaws]
    yaws = [float(value) for value in yaws]
    signed_increment = sum(yaws) / len(yaws) - sum(previous_yaws) / len(previous_yaws)
    signed_increment = math.atan2(math.sin(signed_increment), math.cos(signed_increment))

    raw_ratios: list[float] = []
    per_frame_geometry = []
    for frame_index, yaw in enumerate(yaws):
        current_bounds = _horizontal_fov_bounds(Ks[frame_index])
        rays = _horizontal_ray_angles(Ks[frame_index], width)
        if current_bounds is None or rays is None:
            return None
        current_interval = (yaw + current_bounds[0], yaw + current_bounds[1])
        previous_intervals = []
        for previous_index, previous_yaw in enumerate(previous_yaws):
            bounds = _horizontal_fov_bounds(previous_Ks[previous_index])
            if bounds is None:
                return None
            unwrapped = _nearest_unwrapped(previous_yaw, yaw)
            previous_intervals.append((unwrapped + bounds[0], unwrapped + bounds[1]))
        novel = _subtract_interval(current_interval, _merge_intervals(previous_intervals))
        new_angle = sum(right - left for left, right in novel)
        horizontal_fov = current_interval[1] - current_interval[0]
        raw_ratio = min(1.0, max(0.0, new_angle / horizontal_fov))
        raw_ratios.append(raw_ratio)
        per_frame_geometry.append((rays, yaw, novel, horizontal_fov))

    chunk_tier = quantize_incremental_ratio(max(raw_ratios, default=0.0))
    if max(raw_ratios, default=0.0) <= 1e-12:
        novelty = _novelty_frame_indices(
            block, frame_tokens=frame_tokens, keep_tier=chunk_tier
        )
        chunk_frames = tuple(
            FrameCrop(
                frame_offset=index,
                token_indices=indices,
                kept_columns=(),
                raw_overlap_ratio=raw_ratios[index],
                keep_tier=chunk_tier,
                compression_mode="predecessor_static_novelty_safety",
                delta_yaw_degrees=math.degrees(signed_increment),
                horizontal_fov_degrees=math.degrees(per_frame_geometry[index][3]),
            )
            for index, indices in enumerate(novelty)
        )
        return chunk_frames, chunk_frames, math.degrees(signed_increment)

    def make_crop(frame_index: int, keep_tier: float) -> FrameCrop:
        rays, yaw, novel, horizontal_fov = per_frame_geometry[frame_index]
        columns = _rank_directional_columns(
            rays,
            yaw=yaw,
            novel_intervals=novel,
            signed_increment=signed_increment,
            target_columns=round(width * keep_tier),
        )
        return FrameCrop(
            frame_offset=frame_index,
            token_indices=_spatial_token_indices(columns, height=height, width=width),
            kept_columns=columns,
            raw_overlap_ratio=raw_ratios[frame_index],
            keep_tier=keep_tier,
            compression_mode="predecessor_incremental_yaw",
            delta_yaw_degrees=math.degrees(signed_increment),
            horizontal_fov_degrees=math.degrees(horizontal_fov),
        )

    chunk_frames = tuple(make_crop(index, chunk_tier) for index in range(block.frame_count))
    tail_frames = tuple(
        make_crop(index, quantize_incremental_ratio(raw_ratios[index]))
        for index in range(block.frame_count)
    )
    return chunk_frames, tail_frames, math.degrees(signed_increment)


def _find_predecessor(bank: DyKVBank, block_index: int) -> int | None:
    start = bank.blocks[int(block_index)].frame_start
    matches = [
        index for index, candidate in enumerate(bank.blocks) if candidate.frame_end == start
    ]
    return max(matches, key=lambda index: bank.blocks[index].frame_start) if matches else None


def build_predecessor_candidate(
    bank: DyKVBank,
    *,
    block_index: int,
    distance: float,
    frame_tokens: int,
) -> PredecessorCandidate | None:
    block = bank.blocks[int(block_index)]
    predecessor_index = _find_predecessor(bank, int(block_index))
    geometry = None
    if predecessor_index is not None:
        geometry = _predecessor_geometry(
            bank.blocks[predecessor_index],
            block,
            frame_tokens=frame_tokens,
        )
    fallback_reason = None
    if geometry is None:
        if block.spatial_shape is None or math.prod(block.spatial_shape) != int(frame_tokens):
            return None
        keep_tier = 0.5
        novelty = _novelty_frame_indices(block, frame_tokens=frame_tokens, keep_tier=keep_tier)
        chunk_frames = tuple(
            FrameCrop(
                frame_offset=index,
                token_indices=indices,
                kept_columns=(),
                raw_overlap_ratio=None,
                keep_tier=keep_tier,
                compression_mode="predecessor_fixed_novelty_fallback",
            )
            for index, indices in enumerate(novelty)
        )
        tail_frames = chunk_frames
        incremental_yaw = None
        fallback_reason = "missing_predecessor_or_non_yaw_geometry"
    else:
        chunk_frames, tail_frames, incremental_yaw = geometry
        keep_tier = chunk_frames[0].keep_tier
    similarity = min(1.0, max(0.0, 1.0 - float(distance)))
    return PredecessorCandidate(
        block_index=int(block_index),
        predecessor_index=predecessor_index,
        predecessor_frame_start=(
            None
            if predecessor_index is None
            else bank.blocks[predecessor_index].frame_start
        ),
        distance=float(distance),
        similarity=similarity,
        keep_tier=keep_tier,
        utility=similarity * math.sqrt(keep_tier),
        chunk_frames=chunk_frames,
        tail_frames=tail_frames,
        incremental_yaw_degrees=incremental_yaw,
        fallback_reason=fallback_reason,
    )


def _place_sizes(state: tuple[int, ...], sizes: Sequence[int]) -> tuple[int, ...] | None:
    states = {tuple(sorted(state))}
    for size in sorted((int(value) for value in sizes), reverse=True):
        next_states = set()
        for loads in states:
            for slot, load in enumerate(loads):
                if load + size > PACKING_ATOMS_PER_SLOT:
                    continue
                if slot and loads[slot - 1] == load:
                    continue
                changed = list(loads)
                changed[slot] += size
                next_states.add(tuple(sorted(changed)))
        states = next_states
        if not states:
            return None
    return min(states)


def _select_groups(
    candidates: Sequence[PredecessorCandidate], *, slot_count: int
) -> tuple[tuple[PredecessorCandidate, ...], tuple[int, ...]]:
    initial = (0,) * int(slot_count)
    states: dict[tuple[int, ...], tuple[float, tuple[int, ...]]] = {initial: (0.0, ())}
    for position, candidate in enumerate(candidates):
        updates = dict(states)
        sizes = [frame.atom_count for frame in candidate.chunk_frames]
        for state, (utility, chosen) in states.items():
            target = _place_sizes(state, sizes)
            if target is None:
                continue
            proposal = (utility + candidate.utility, chosen + (position,))
            previous = updates.get(target)
            if previous is None or proposal[0] > previous[0] + 1e-12 or (
                abs(proposal[0] - previous[0]) <= 1e-12 and proposal[1] < previous[1]
            ):
                updates[target] = proposal
        states = updates
    best_state, best = max(
        states.items(),
        key=lambda item: (item[1][0], sum(item[0]), tuple(-value for value in item[1][1])),
    )
    return tuple(candidates[index] for index in best[1]), best_state


def _select_tail(
    frames: Sequence[PackedFramePlan], initial_state: tuple[int, ...]
) -> tuple[tuple[PackedFramePlan, ...], tuple[int, ...]]:
    states: dict[tuple[int, ...], tuple[float, tuple[int, ...]]] = {
        tuple(initial_state): (0.0, ())
    }
    for position, frame in enumerate(frames):
        updates = dict(states)
        for state, (utility, chosen) in states.items():
            target = _place_sizes(state, [frame.atom_count])
            if target is None:
                continue
            proposal = (utility + frame.utility, chosen + (position,))
            previous = updates.get(target)
            if previous is None or proposal[0] > previous[0] + 1e-12 or (
                abs(proposal[0] - previous[0]) <= 1e-12 and proposal[1] < previous[1]
            ):
                updates[target] = proposal
        states = updates
    best_state, best = max(states.items(), key=lambda item: (item[1][0], sum(item[0])))
    return tuple(frames[index] for index in best[1]), best_state


def _assign_slots(frames: Sequence[PackedFramePlan], *, slot_count: int, sink_frames: int):
    order = sorted(
        range(len(frames)),
        key=lambda index: (-frames[index].atom_count, -frames[index].utility, frames[index].source_frame_id),
    )
    loads = [0] * int(slot_count)
    assignments = [-1] * len(frames)

    def search(position: int) -> bool:
        if position == len(order):
            return True
        frame_index = order[position]
        size = frames[frame_index].atom_count
        seen = set()
        for slot, load in enumerate(loads):
            if load in seen or load + size > PACKING_ATOMS_PER_SLOT:
                continue
            seen.add(load)
            loads[slot] += size
            assignments[frame_index] = slot
            if search(position + 1):
                return True
            assignments[frame_index] = -1
            loads[slot] -= size
        return False

    if not search(0):
        raise RuntimeError("predecessor packing plan cannot be assigned to retrieval slots")
    return (
        [replace(frame, virtual_slot_id=int(sink_frames) + assignments[index]) for index, frame in enumerate(frames)],
        loads,
    )


def _as_frame(
    candidate: PredecessorCandidate,
    crop: FrameCrop,
    block: MemoryBlock,
    kind: str,
) -> PackedFramePlan:
    ratio = crop.raw_overlap_ratio
    return PackedFramePlan(
        block_index=candidate.block_index,
        frame_offset=crop.frame_offset,
        source_frame_id=block.frame_start + crop.frame_offset,
        token_indices=crop.token_indices,
        kept_columns=crop.kept_columns,
        raw_overlap_ratio=ratio,
        keep_tier=crop.keep_tier,
        compression_mode=crop.compression_mode,
        selection_kind=kind,
        utility=candidate.similarity * math.sqrt(crop.keep_tier),
        delta_yaw_degrees=crop.delta_yaw_degrees,
        horizontal_fov_degrees=crop.horizontal_fov_degrees,
        predecessor_frame_start=candidate.predecessor_frame_start,
        incremental_yaw_degrees=candidate.incremental_yaw_degrees,
        incremental_fov_ratio=ratio,
    )
def _apply_query_backfill(
    bank: DyKVBank,
    frames: list[PackedFramePlan],
    slot_loads: list[int],
    *,
    sink_frames: int,
    current_viewmats: torch.Tensor,
    current_Ks: torch.Tensor | None,
    frame_tokens: int,
) -> tuple[list[PackedFramePlan], list[int]]:
    output = list(frames)
    for index in sorted(range(len(output)), key=lambda value: -output[value].utility):
        frame = output[index]
        if not frame.kept_columns:
            continue
        slot = frame.virtual_slot_id - int(sink_frames)
        remaining_atoms = PACKING_ATOMS_PER_SLOT - slot_loads[slot]
        if remaining_atoms <= 0:
            continue
        block = bank.blocks[frame.block_index]
        exact = build_yaw_crop_plan(
            historical_viewmats=block.viewmats,
            historical_Ks=block.Ks,
            current_viewmats=current_viewmats,
            current_Ks=current_Ks,
            frame_count=block.frame_count,
            frame_tokens=frame_tokens,
            spatial_shape=block.spatial_shape,
        )
        if exact is None:
            continue
        visible = set(exact.kept_columns_per_frame[frame.frame_offset])
        missing = sorted(visible.difference(frame.kept_columns))
        width = int(block.spatial_shape[1])
        height = int(block.spatial_shape[0])
        atom_columns = width // PACKING_ATOMS_PER_SLOT
        add_atoms = min(remaining_atoms, len(missing) // atom_columns)
        if add_atoms <= 0:
            continue
        additions = missing[: add_atoms * atom_columns]
        columns = tuple(sorted(set(frame.kept_columns).union(additions)))
        added_tokens = len(additions) * height
        output[index] = replace(
            frame,
            token_indices=_spatial_token_indices(columns, height=height, width=width),
            kept_columns=columns,
            keep_tier=len(columns) / float(width),
            compression_mode=frame.compression_mode + "+query_backfill",
            query_backfill_tokens=added_tokens,
        )
        slot_loads[slot] += add_atoms
    return output, slot_loads


def build_predecessor_retrieval_plan(
    bank: DyKVBank,
    ranked_block_indices: Sequence[int],
    ranked_distances: Sequence[float],
    *,
    current_viewmats: torch.Tensor,
    current_Ks: torch.Tensor | None,
    frame_tokens: int,
    memory_frames: int,
    sink_frames: int,
    include_tail_latents: bool,
    query_backfill: bool,
) -> PackedRetrievalPlan:
    if len(ranked_block_indices) != len(ranked_distances):
        raise ValueError("predecessor retrieval ranking and distances must align")
    if int(frame_tokens) % PACKING_ATOMS_PER_SLOT:
        raise ValueError("predecessor packing requires frame tokens divisible by four")
    candidates = []
    for block_index, distance in zip(ranked_block_indices, ranked_distances):
        candidate = build_predecessor_candidate(
            bank,
            block_index=int(block_index),
            distance=float(distance),
            frame_tokens=frame_tokens,
        )
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(
        key=lambda item: (item.distance, -bank.blocks[item.block_index].frame_start, item.block_index)
    )
    selected_full, state = _select_groups(candidates, slot_count=memory_frames)
    full_indices = {candidate.block_index for candidate in selected_full}
    frames = []
    for candidate in selected_full:
        block = bank.blocks[candidate.block_index]
        frames.extend(_as_frame(candidate, crop, block, "predecessor_full_chunk") for crop in candidate.chunk_frames)

    selected_tail: tuple[PackedFramePlan, ...] = ()
    if include_tail_latents:
        tail_candidates = []
        for candidate in candidates:
            if candidate.block_index in full_indices:
                continue
            block = bank.blocks[candidate.block_index]
            tail_candidates.extend(
                _as_frame(candidate, crop, block, "predecessor_tail_latent")
                for crop in candidate.tail_frames
            )
        selected_tail, state = _select_tail(tail_candidates, state)
        frames.extend(selected_tail)

    frames, slot_loads = _assign_slots(frames, slot_count=memory_frames, sink_frames=sink_frames)
    if query_backfill and frames:
        frames, slot_loads = _apply_query_backfill(
            bank,
            frames,
            slot_loads,
            sink_frames=sink_frames,
            current_viewmats=current_viewmats,
            current_Ks=current_Ks,
            frame_tokens=frame_tokens,
        )
    frames.sort(key=lambda frame: (frame.virtual_slot_id, frame.source_frame_id))
    used_atoms = sum(frame.atom_count for frame in frames)
    used_tokens = sum(frame.token_count for frame in frames)
    atom_tokens = int(frame_tokens) // PACKING_ATOMS_PER_SLOT
    if used_tokens != used_atoms * atom_tokens or used_atoms > memory_frames * PACKING_ATOMS_PER_SLOT:
        raise RuntimeError("predecessor plan violates the retrieval token budget")
    return PackedRetrievalPlan(
        frames=tuple(frames),
        selected_full_blocks=tuple(
            sorted(full_indices, key=lambda index: bank.blocks[index].frame_start)
        ),
        selected_tail_frames=tuple((frame.block_index, frame.frame_offset) for frame in selected_tail),
        token_budget=int(memory_frames) * int(frame_tokens),
        used_tokens=used_tokens,
        used_virtual_slots=sum(load > 0 for load in slot_loads),
        budget_atoms=int(memory_frames) * PACKING_ATOMS_PER_SLOT,
        used_atoms=used_atoms,
        candidate_block_indices=tuple(candidate.block_index for candidate in candidates),
    )


__all__ = [
    "PREDECESSOR_KEEP_TIERS",
    "build_predecessor_candidate",
    "build_predecessor_retrieval_plan",
    "materialize_packed_retrieval",
    "quantize_incremental_ratio",
]
