"""Continuous FOV-ratio WorldKV novelty planning for dyKV retrieval."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import torch

from .dykv_fov import fov_overlap
from .dykv_memory import DyKVBank, MemoryBlock


SOURCE_ORDERED_LAYOUT = "source_ordered"
FLAT_SOURCE_ORDERED_LAYOUT = "flat_source_ordered"


def motion_keep_ratio_and_token_count(
    overlap_ratio: float,
    frame_tokens: int,
) -> tuple[float, int]:
    overlap = float(overlap_ratio)
    if not math.isfinite(overlap):
        raise ValueError("motion novelty FOV overlap is not finite")
    overlap = min(1.0, max(0.0, overlap))
    keep_ratio = min(1.0, max(0.0, 1.0 - overlap))
    keep_tokens = min(
        int(frame_tokens),
        max(0, int(math.ceil(keep_ratio * int(frame_tokens)))),
    )
    return keep_ratio, keep_tokens


@dataclass(frozen=True)
class MotionFramePlan:
    block_index: int
    frame_offset: int
    source_frame_id: int
    fov_overlap: float
    keep_ratio: float
    base_indices: torch.Tensor
    base_indices_in_selection_order: torch.Tensor
    omitted_indices_in_novelty_order: torch.Tensor
    relative_rotation_degrees: float
    relative_translation_distance: float
    virtual_slot_id: int = -1

    @property
    def base_token_count(self) -> int:
        return int(self.base_indices.numel())

    @property
    def omitted_token_count(self) -> int:
        return int(self.omitted_indices_in_novelty_order.numel())

    @property
    def selection_kind(self) -> str:
        return "anchor" if self.frame_offset == 0 else "motion_novelty"


@dataclass(frozen=True)
class MotionChunkPlan:
    block_index: int
    retrieval_distance: float
    retrieval_similarity: float
    frames: tuple[MotionFramePlan, ...]

    @property
    def base_tokens(self) -> int:
        return sum(frame.base_token_count for frame in self.frames)


@dataclass(frozen=True)
class MotionSegmentPlan:
    block_index: int
    frame_offset: int
    source_frame_id: int
    token_indices: torch.Tensor
    virtual_slot_id: int
    selection_kind: str
    duplicate_ordinal: int = 0

    @property
    def token_count(self) -> int:
        return int(self.token_indices.numel())


@dataclass(frozen=True)
class MotionRetrievalPlan:
    chunks: tuple[MotionChunkPlan, ...]
    selected_block_indices: tuple[int, ...]
    candidate_block_indices: tuple[int, ...]
    geometry_invalid_block_indices: tuple[int, ...]
    token_budget: int
    base_used_tokens: int
    fill_target_tokens: int
    reference_frame_lengths: tuple[int, ...]
    slot_token_loads: tuple[int, ...]
    retrieval_layout: str
    fill_mode: str
    segments: tuple[MotionSegmentPlan, ...]
    unique_backfill_tokens_per_frame: tuple[int, ...]
    duplicate_tokens_per_frame: tuple[int, ...]
    duplicate_source_block_indices: tuple[int, ...]
    max_source_token_multiplicity: int

    @property
    def frames(self) -> tuple[MotionFramePlan, ...]:
        return tuple(frame for chunk in self.chunks for frame in chunk.frames)


def _frame_matrices(
    tensor: torch.Tensor | None,
    *,
    frame_count: int,
    matrix_size: int,
) -> torch.Tensor:
    if tensor is None:
        raise ValueError("motion novelty geometry is missing")
    output = tensor.detach().to(device="cpu", dtype=torch.float64)
    if output.ndim == 4:
        if output.shape[0] != 1:
            raise ValueError("motion novelty geometry requires batch size one")
        output = output[0]
    if output.shape != (int(frame_count), matrix_size, matrix_size):
        raise ValueError("motion novelty geometry has an invalid shape")
    if not bool(torch.isfinite(output).all()):
        raise ValueError("motion novelty geometry must be finite")
    return output


def _relative_motion(
    anchor_w2c: torch.Tensor,
    frame_w2c: torch.Tensor,
) -> tuple[float, float]:
    anchor_c2w = torch.linalg.inv(anchor_w2c)
    frame_c2w = torch.linalg.inv(frame_w2c)
    relative_rotation = anchor_c2w[:3, :3].T @ frame_c2w[:3, :3]
    cosine = ((torch.trace(relative_rotation) - 1.0) / 2.0).clamp(-1.0, 1.0)
    rotation = math.degrees(float(torch.acos(cosine)))
    translation = float(
        torch.linalg.vector_norm(frame_c2w[:3, 3] - anchor_c2w[:3, 3])
    )
    return rotation, translation


def _novelty_order(block: MemoryBlock, *, frame_tokens: int) -> tuple[torch.Tensor, ...]:
    if not block.layers:
        raise ValueError("motion novelty requires stored KV layers")
    layer0 = block.layers[0].k.detach().to(device="cpu")
    expected = int(block.frame_count) * int(frame_tokens)
    if layer0.ndim != 4 or int(layer0.shape[1]) != expected:
        raise ValueError("motion novelty requires complete frame-aligned KV")
    batch, _, heads, dim = layer0.shape
    frames = layer0.reshape(batch, block.frame_count, frame_tokens, heads, dim)
    centroid = frames[:, 0].float().mean(dim=1)
    centroid_norm = torch.linalg.vector_norm(centroid, dim=(-2, -1))
    eps = torch.finfo(torch.float32).eps
    orders = [torch.arange(int(frame_tokens), dtype=torch.long)]
    for frame_index in range(1, int(block.frame_count)):
        values = frames[:, frame_index].float()
        similarity = (values * centroid.unsqueeze(1)).sum(dim=(-2, -1))
        similarity = similarity / (
            torch.linalg.vector_norm(values, dim=(-2, -1))
            * centroid_norm.unsqueeze(1)
            + eps
        )
        score = similarity.mean(dim=0)
        orders.append(torch.argsort(score, stable=True).to(torch.long))
    return tuple(orders)


def build_motion_chunk_plan(
    block: MemoryBlock,
    *,
    block_index: int,
    retrieval_distance: float,
    probe_points: torch.Tensor,
    radius: float,
    frame_tokens: int,
) -> MotionChunkPlan:
    if int(block.frame_count) != 4:
        raise ValueError("motion novelty requires four-frame chunks")
    poses = _frame_matrices(block.viewmats, frame_count=4, matrix_size=4)
    intrinsics = _frame_matrices(block.Ks, frame_count=4, matrix_size=3)
    orders = _novelty_order(block, frame_tokens=frame_tokens)
    frames = []
    for frame_offset in range(4):
        rotation, translation = _relative_motion(poses[0], poses[frame_offset])
        if frame_offset == 0:
            overlap = 0.0
            keep_ratio = 1.0
            keep_tokens = int(frame_tokens)
        else:
            overlap_tensor = fov_overlap(
                poses[frame_offset],
                poses[0],
                probe_points,
                current_K=intrinsics[frame_offset],
                historical_K=intrinsics[0],
                radius=radius,
            )
            overlap = float(overlap_tensor.item())
            if not math.isfinite(overlap):
                raise ValueError("motion novelty FOV overlap is not finite")
            overlap = min(1.0, max(0.0, overlap))
            keep_ratio, keep_tokens = motion_keep_ratio_and_token_count(
                overlap,
                frame_tokens,
            )
        novelty_order = orders[frame_offset]
        base_selection_order = novelty_order[:keep_tokens].clone()
        base = base_selection_order.sort().values
        omitted = novelty_order[keep_tokens:].clone()
        frames.append(
            MotionFramePlan(
                block_index=int(block_index),
                frame_offset=frame_offset,
                source_frame_id=int(block.frame_start) + frame_offset,
                fov_overlap=overlap,
                keep_ratio=keep_ratio,
                base_indices=base,
                base_indices_in_selection_order=base_selection_order,
                omitted_indices_in_novelty_order=omitted,
                relative_rotation_degrees=rotation,
                relative_translation_distance=translation,
            )
        )
    distance = float(retrieval_distance)
    return MotionChunkPlan(
        block_index=int(block_index),
        retrieval_distance=distance,
        retrieval_similarity=1.0 - distance,
        frames=tuple(frames),
    )


def _largest_remainder_allocation(capacities: Sequence[int], total: int) -> list[int]:
    capacities = [max(0, int(value)) for value in capacities]
    total = min(max(0, int(total)), sum(capacities))
    if total == 0 or not capacities:
        return [0] * len(capacities)
    capacity_sum = sum(capacities)
    quotas = [total * capacity / capacity_sum for capacity in capacities]
    output = [
        min(capacity, int(math.floor(quota)))
        for capacity, quota in zip(capacities, quotas)
    ]
    remaining = total - sum(output)
    order = sorted(
        range(len(capacities)),
        key=lambda index: (-(quotas[index] - output[index]), index),
    )
    while remaining:
        progressed = False
        for index in order:
            if output[index] < capacities[index]:
                output[index] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("largest-remainder allocation could not reach target")
    return output


def _proportional_allocation(weights: Sequence[int], total: int) -> list[int]:
    weights = [max(0, int(value)) for value in weights]
    total = max(0, int(total))
    if total == 0:
        return [0] * len(weights)
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("duplicate allocation requires a non-empty repeat pool")
    quotas = [total * weight / weight_sum for weight in weights]
    output = [int(math.floor(quota)) for quota in quotas]
    remaining = total - sum(output)
    order = sorted(
        range(len(weights)),
        key=lambda index: (-(quotas[index] - output[index]), index),
    )
    for index in order[:remaining]:
        output[index] += 1
    return output


def _base_segments(frames: Sequence[MotionFramePlan]) -> list[MotionSegmentPlan]:
    return [
        MotionSegmentPlan(
            block_index=frame.block_index,
            frame_offset=frame.frame_offset,
            source_frame_id=frame.source_frame_id,
            token_indices=frame.base_indices,
            virtual_slot_id=frame.virtual_slot_id,
            selection_kind=frame.selection_kind,
        )
        for frame in frames
        if frame.base_token_count > 0
    ]


def _backfilled_segments(
    frames: Sequence[MotionFramePlan],
    reference_lengths: Sequence[int],
) -> tuple[list[MotionSegmentPlan], tuple[int, ...]]:
    segments = []
    counts = []
    for frame, reference_length in zip(frames, reference_lengths):
        backfill_count = int(reference_length) - frame.base_token_count
        if backfill_count < 0 or backfill_count > frame.omitted_token_count:
            raise RuntimeError("motion novelty backfill count is invalid")
        counts.append(backfill_count)
        indices = torch.cat(
            (
                frame.base_indices,
                frame.omitted_indices_in_novelty_order[:backfill_count],
            )
        ).sort().values
        if int(indices.unique().numel()) != int(indices.numel()):
            raise RuntimeError("motion novelty backfill repeated a source token")
        if indices.numel():
            segments.append(
                MotionSegmentPlan(
                    block_index=frame.block_index,
                    frame_offset=frame.frame_offset,
                    source_frame_id=frame.source_frame_id,
                    token_indices=indices,
                    virtual_slot_id=frame.virtual_slot_id,
                    selection_kind=(
                        "anchor"
                        if frame.frame_offset == 0
                        else "motion_novelty_backfill"
                    ),
                )
            )
    return segments, tuple(counts)


def _duplicate_segments(
    chunks: Sequence[MotionChunkPlan],
    source_ordered_frames: Sequence[MotionFramePlan],
    *,
    target_slot_loads: Sequence[int],
    sink_frames: int,
    memory_frames: int,
) -> tuple[list[MotionSegmentPlan], tuple[int, ...], tuple[int, ...], int]:
    segments = _base_segments(source_ordered_frames)
    if not chunks:
        return segments, (), (), 0
    base_slot_loads = [0] * int(memory_frames)
    for frame in source_ordered_frames:
        if frame.base_token_count:
            base_slot_loads[frame.virtual_slot_id - int(sink_frames)] += (
                frame.base_token_count
            )
    slot_needs = [
        int(target) - int(base)
        for target, base in zip(target_slot_loads, base_slot_loads)
    ]
    if any(value < 0 for value in slot_needs):
        raise RuntimeError("duplicate target slot load is below its base load")
    duplicate_total = sum(slot_needs)
    repeat_frames = list(chunks[0].frames)
    weights = [frame.base_token_count for frame in repeat_frames]
    quotas = _proportional_allocation(weights, duplicate_total)
    remaining_by_frame = list(quotas)
    cursors = [0] * len(repeat_frames)
    ordinals = [0] * len(repeat_frames)
    duplicate_per_source = {
        frame.source_frame_id: 0 for frame in source_ordered_frames
    }
    multiplicities: dict[tuple[int, int, int], int] = {}
    for frame in source_ordered_frames:
        for token_index in frame.base_indices.tolist():
            key = (frame.block_index, frame.frame_offset, int(token_index))
            multiplicities[key] = multiplicities.get(key, 0) + 1

    for slot_offset, slot_need in enumerate(slot_needs):
        if slot_need <= 0:
            continue
        allocation = _largest_remainder_allocation(remaining_by_frame, slot_need)
        for frame_index, count in enumerate(allocation):
            if count <= 0:
                continue
            frame = repeat_frames[frame_index]
            pool = frame.base_indices_in_selection_order
            if pool.numel() == 0:
                raise RuntimeError("duplicate allocation selected an empty source frame")
            positions = (
                torch.arange(count, dtype=torch.long) + cursors[frame_index]
            ) % int(pool.numel())
            indices = pool.index_select(0, positions)
            cursors[frame_index] += count
            ordinals[frame_index] += 1
            remaining_by_frame[frame_index] -= count
            duplicate_per_source[frame.source_frame_id] += count
            for token_index in indices.tolist():
                key = (frame.block_index, frame.frame_offset, int(token_index))
                multiplicities[key] = multiplicities.get(key, 0) + 1
            segments.append(
                MotionSegmentPlan(
                    block_index=frame.block_index,
                    frame_offset=frame.frame_offset,
                    source_frame_id=frame.source_frame_id,
                    token_indices=indices,
                    virtual_slot_id=int(sink_frames) + slot_offset,
                    selection_kind="duplicate",
                    duplicate_ordinal=ordinals[frame_index],
                )
            )
    if any(remaining_by_frame):
        raise RuntimeError("duplicate allocation did not consume all frame quotas")
    segments.sort(
        key=lambda segment: (
            segment.source_frame_id,
            0 if segment.selection_kind != "duplicate" else 1,
            segment.duplicate_ordinal,
            segment.virtual_slot_id,
        )
    )
    duplicate_counts = tuple(
        duplicate_per_source[frame.source_frame_id]
        for frame in source_ordered_frames
    )
    duplicate_sources = (
        (chunks[0].block_index,) if duplicate_total > 0 else ()
    )
    max_multiplicity = max(multiplicities.values(), default=0)
    return segments, duplicate_counts, duplicate_sources, max_multiplicity


def _reference_lengths(
    chunks: Sequence[MotionChunkPlan],
    *,
    frame_tokens: int,
    token_budget: int,
) -> tuple[tuple[int, ...], int]:
    frames = [frame for chunk in chunks for frame in chunk.frames]
    lengths = [frame.base_token_count for frame in frames]
    fill_target = min(int(token_budget), len(chunks) * 4 * int(frame_tokens))
    remaining = fill_target - sum(lengths)
    frame_position = {id(frame): index for index, frame in enumerate(frames)}
    for chunk in chunks:
        if remaining <= 0:
            break
        non_anchor = list(chunk.frames[1:])
        capacities = [frame.omitted_token_count for frame in non_anchor]
        chunk_add = min(remaining, sum(capacities))
        allocation = _largest_remainder_allocation(capacities, chunk_add)
        for frame, add in zip(non_anchor, allocation):
            lengths[frame_position[id(frame)]] += int(add)
        remaining -= chunk_add
    if remaining:
        raise RuntimeError("motion novelty reference layout did not reach fill target")
    return tuple(lengths), fill_target


def _reference_slots(
    frames: Sequence[MotionFramePlan],
    reference_lengths: Sequence[int],
    *,
    frame_tokens: int,
    sink_frames: int,
    memory_frames: int,
) -> tuple[tuple[MotionFramePlan, ...], tuple[int, ...]]:
    prefix = 0
    output = []
    loads = [0] * int(memory_frames)
    for frame, reference_length in zip(frames, reference_lengths):
        reference_length = int(reference_length)
        center = prefix + reference_length / 2.0
        slot_offset = min(int(memory_frames) - 1, int(math.floor(center / frame_tokens)))
        slot = int(sink_frames) + slot_offset
        output.append(replace(frame, virtual_slot_id=slot))
        loads[slot_offset] += frame.base_token_count
        prefix += reference_length
    return tuple(output), tuple(loads)


def _capped_slots(
    frames: Sequence[MotionFramePlan],
    *,
    frame_tokens: int,
    sink_frames: int,
    memory_frames: int,
) -> tuple[tuple[MotionFramePlan, ...], tuple[int, ...]] | None:
    nonempty = [frame for frame in frames if frame.base_token_count > 0]
    anchors = [frame for frame in nonempty if frame.frame_offset == 0]
    novelty = [frame for frame in nonempty if frame.frame_offset != 0]
    if len(anchors) > int(memory_frames):
        return None
    bins: list[list[MotionFramePlan]] = [[frame] for frame in anchors]
    remaining = [0] * len(anchors)
    for frame in sorted(
        novelty,
        key=lambda item: (-item.base_token_count, item.source_frame_id),
    ):
        for index, capacity in enumerate(remaining):
            if capacity >= frame.base_token_count:
                bins[index].append(frame)
                remaining[index] -= frame.base_token_count
                break
        else:
            if len(bins) >= int(memory_frames):
                return None
            bins.append([frame])
            remaining.append(int(frame_tokens) - frame.base_token_count)
    assigned = []
    loads = [0] * int(memory_frames)
    for bin_index, members in enumerate(bins):
        slot = int(sink_frames) + bin_index
        for frame in members:
            assigned.append(replace(frame, virtual_slot_id=slot))
            loads[bin_index] += frame.base_token_count
    assigned.sort(key=lambda frame: frame.source_frame_id)
    return tuple(assigned), tuple(loads)


def build_motion_retrieval_plan(
    bank: DyKVBank,
    ranked_block_indices: Sequence[int],
    ranked_distances: Sequence[float],
    *,
    probe_points: torch.Tensor,
    radius: float,
    frame_tokens: int,
    memory_frames: int,
    sink_frames: int,
    slot_capped: bool,
    candidate_block_indices: Sequence[int] | None = None,
    fill_mode: str = "unfilled",
) -> MotionRetrievalPlan:
    if len(ranked_block_indices) != len(ranked_distances):
        raise ValueError("motion novelty ranking and distances must align")
    fill_mode = str(fill_mode)
    if fill_mode not in {"unfilled", "backfill", "duplicate"}:
        raise ValueError(f"unsupported motion novelty fill mode: {fill_mode}")
    if slot_capped and fill_mode != "unfilled":
        raise ValueError("slot-capped motion novelty only supports unfilled mode")
    token_budget = int(memory_frames) * int(frame_tokens)
    selected: list[MotionChunkPlan] = []
    all_candidates = tuple(
        int(index)
        for index in (
            ranked_block_indices
            if candidate_block_indices is None
            else candidate_block_indices
        )
    )
    ranked_set = {int(index) for index in ranked_block_indices}
    invalid: list[int] = [
        index for index in all_candidates if index not in ranked_set
    ]
    for block_index, distance in zip(ranked_block_indices, ranked_distances):
        try:
            candidate = build_motion_chunk_plan(
                bank.blocks[int(block_index)],
                block_index=int(block_index),
                retrieval_distance=float(distance),
                probe_points=probe_points,
                radius=radius,
                frame_tokens=frame_tokens,
            )
        except (RuntimeError, ValueError):
            invalid.append(int(block_index))
            continue
        if sum(chunk.base_tokens for chunk in selected) + candidate.base_tokens > token_budget:
            continue
        if slot_capped:
            tentative_frames = [
                frame for chunk in (*selected, candidate) for frame in chunk.frames
            ]
            if _capped_slots(
                tentative_frames,
                frame_tokens=frame_tokens,
                sink_frames=sink_frames,
                memory_frames=memory_frames,
            ) is None:
                continue
        selected.append(candidate)

    frames = [frame for chunk in selected for frame in chunk.frames]
    reference_lengths_ranked, fill_target = _reference_lengths(
        selected,
        frame_tokens=frame_tokens,
        token_budget=token_budget,
    )
    reference_by_source = {
        frame.source_frame_id: length
        for frame, length in zip(frames, reference_lengths_ranked)
    }
    source_ordered_frames = sorted(frames, key=lambda frame: frame.source_frame_id)
    reference_lengths = tuple(
        reference_by_source[frame.source_frame_id] for frame in source_ordered_frames
    )
    if slot_capped:
        assigned = _capped_slots(
            frames,
            frame_tokens=frame_tokens,
            sink_frames=sink_frames,
            memory_frames=memory_frames,
        )
        if assigned is None:
            raise RuntimeError("selected capped motion plan no longer fits")
        assigned_frames, slot_loads = assigned
        layout = SOURCE_ORDERED_LAYOUT
    else:
        assigned_frames, slot_loads = _reference_slots(
            source_ordered_frames,
            reference_lengths,
            frame_tokens=frame_tokens,
            sink_frames=sink_frames,
            memory_frames=memory_frames,
        )
        layout = FLAT_SOURCE_ORDERED_LAYOUT

    assigned_by_source = {frame.source_frame_id: frame for frame in assigned_frames}
    assigned_chunks = []
    for chunk in selected:
        anchor = assigned_by_source[chunk.frames[0].source_frame_id]
        chunk_frames = tuple(
            assigned_by_source.get(
                frame.source_frame_id,
                replace(frame, virtual_slot_id=anchor.virtual_slot_id),
            )
            for frame in chunk.frames
        )
        assigned_chunks.append(replace(chunk, frames=chunk_frames))
    source_ordered_assigned = tuple(
        sorted(
            (frame for chunk in assigned_chunks for frame in chunk.frames),
            key=lambda frame: frame.source_frame_id,
        )
    )
    target_slot_loads = [0] * int(memory_frames)
    for frame, reference_length in zip(
        source_ordered_assigned, reference_lengths
    ):
        target_slot_loads[frame.virtual_slot_id - int(sink_frames)] += int(
            reference_length
        )

    if fill_mode == "backfill":
        segments, backfill_counts = _backfilled_segments(
            source_ordered_assigned,
            reference_lengths,
        )
        duplicate_counts = (0,) * len(source_ordered_assigned)
        duplicate_sources: tuple[int, ...] = ()
        max_multiplicity = 1 if segments else 0
        final_slot_loads = tuple(target_slot_loads)
    elif fill_mode == "duplicate":
        (
            segments,
            duplicate_counts,
            duplicate_sources,
            max_multiplicity,
        ) = _duplicate_segments(
            tuple(assigned_chunks),
            source_ordered_assigned,
            target_slot_loads=target_slot_loads,
            sink_frames=sink_frames,
            memory_frames=memory_frames,
        )
        backfill_counts = (0,) * len(source_ordered_assigned)
        final_slot_loads = tuple(target_slot_loads)
    else:
        segments = _base_segments(source_ordered_assigned)
        backfill_counts = (0,) * len(source_ordered_assigned)
        duplicate_counts = (0,) * len(source_ordered_assigned)
        duplicate_sources = ()
        max_multiplicity = 1 if segments else 0
        final_slot_loads = slot_loads
    return MotionRetrievalPlan(
        chunks=tuple(assigned_chunks),
        selected_block_indices=tuple(chunk.block_index for chunk in selected),
        candidate_block_indices=all_candidates,
        geometry_invalid_block_indices=tuple(invalid),
        token_budget=token_budget,
        base_used_tokens=sum(chunk.base_tokens for chunk in selected),
        fill_target_tokens=fill_target,
        reference_frame_lengths=reference_lengths,
        slot_token_loads=final_slot_loads,
        retrieval_layout=layout,
        fill_mode=fill_mode,
        segments=tuple(segments),
        unique_backfill_tokens_per_frame=tuple(backfill_counts),
        duplicate_tokens_per_frame=tuple(duplicate_counts),
        duplicate_source_block_indices=duplicate_sources,
        max_source_token_multiplicity=max_multiplicity,
    )


def materialize_motion_retrieval(
    bank: DyKVBank,
    plan: MotionRetrievalPlan,
    *,
    target_device: torch.device | str,
    frame_tokens: int,
) -> list[dict]:
    segments = plan.segments
    if not segments:
        return []
    layer_count = len(bank.blocks[segments[0].block_index].layers)
    if any(
        len(bank.blocks[segment.block_index].layers) != layer_count
        for segment in segments
    ):
        raise RuntimeError("motion novelty bank blocks have inconsistent layer counts")
    device = torch.device(target_device)
    payloads = []
    source_frames = [segment.source_frame_id for segment in segments]
    frame_lengths = [segment.token_count for segment in segments]
    virtual_slots = [segment.virtual_slot_id for segment in segments]
    selected_starts = sorted(
        bank.blocks[index].frame_start for index in plan.selected_block_indices
    )
    diagnostic_frames = sorted(plan.frames, key=lambda frame: frame.source_frame_id)
    base_per_frame = [frame.base_token_count for frame in diagnostic_frames]
    backfill_per_frame = list(plan.unique_backfill_tokens_per_frame)
    duplicate_per_frame = list(plan.duplicate_tokens_per_frame)
    actual_per_frame = [
        base + backfill + duplicate
        for base, backfill, duplicate in zip(
            base_per_frame, backfill_per_frame, duplicate_per_frame
        )
    ]
    actual_by_source = {
        frame.source_frame_id: actual
        for frame, actual in zip(diagnostic_frames, actual_per_frame)
    }
    base_per_chunk = [chunk.base_tokens for chunk in plan.chunks]
    actual_per_chunk = [
        sum(actual_by_source[frame.source_frame_id] for frame in chunk.frames)
        for chunk in plan.chunks
    ]
    final_tokens = sum(frame_lengths)
    if final_tokens != sum(actual_per_frame):
        raise RuntimeError("motion novelty frame and segment totals differ")
    if final_tokens != sum(plan.slot_token_loads):
        raise RuntimeError("motion novelty segment and slot totals differ")
    for layer_index in range(layer_count):
        key_parts = []
        value_parts = []
        for segment in segments:
            block = bank.blocks[segment.block_index]
            indices = segment.token_indices.to(device=device) + (
                segment.frame_offset * int(frame_tokens)
            )
            raw_k = block.layers[layer_index].k.to(device=device)
            raw_v = block.layers[layer_index].v.to(device=device)
            key_parts.append(raw_k.index_select(1, indices))
            value_parts.append(raw_v.index_select(1, indices))
        packed_k = torch.cat(key_parts, dim=1)
        packed_v = torch.cat(value_parts, dim=1)
        if int(packed_k.shape[1]) != final_tokens:
            raise RuntimeError("motion novelty materialization differs from its plan")
        payloads.append(
            {
                "k": packed_k,
                "v": packed_v,
                "source_frame_ids": source_frames,
                "frame_token_lengths": frame_lengths,
                "virtual_slot_ids": virtual_slots,
                "retrieval_layout": plan.retrieval_layout,
                "src_frame_ids": selected_starts,
                "chunk_frame_counts": [4] * len(plan.chunks),
                "chunk_token_lengths": actual_per_chunk,
                "parent_block_ids": [
                    bank.blocks[segment.block_index].block_id for segment in segments
                ],
                "selection_kinds": [
                    segment.selection_kind for segment in segments
                ],
                "duplicate_ordinals": [
                    segment.duplicate_ordinal for segment in segments
                ],
                "compression_modes": [
                    f"motion_novelty_{plan.fill_mode}"
                ] * len(segments),
                "motion_fov_overlaps": [
                    frame.fov_overlap for frame in diagnostic_frames
                ],
                "motion_keep_ratios": [
                    frame.keep_ratio for frame in diagnostic_frames
                ],
                "relative_rotation_degrees": [
                    frame.relative_rotation_degrees for frame in diagnostic_frames
                ],
                "relative_translation_distances": [
                    frame.relative_translation_distance for frame in diagnostic_frames
                ],
                "motion_geometry_invalid_block_ids": [
                    bank.blocks[index].block_id
                    for index in plan.geometry_invalid_block_indices
                ],
                "retrieval_similarities": [
                    chunk.retrieval_similarity for chunk in plan.chunks
                ],
                "base_tokens_per_frame": base_per_frame,
                "base_tokens_per_chunk": base_per_chunk,
                "base_tokens_total": plan.base_used_tokens,
                "unique_backfill_tokens_per_frame": backfill_per_frame,
                "unique_backfill_tokens_total": sum(backfill_per_frame),
                "duplicate_tokens_per_frame": duplicate_per_frame,
                "duplicate_tokens_total": sum(duplicate_per_frame),
                "duplicate_source_block_ids": [
                    bank.blocks[index].block_id
                    for index in plan.duplicate_source_block_indices
                ],
                "max_source_token_multiplicity": (
                    plan.max_source_token_multiplicity
                ),
                "actual_tokens_per_frame": actual_per_frame,
                "final_tokens_total": final_tokens,
                "fill_target_tokens": plan.fill_target_tokens,
                "unused_tokens": plan.token_budget - final_tokens,
                "slot_token_loads": list(plan.slot_token_loads),
                "segments_source_ordered": True,
                "kept_tokens_per_frame": actual_per_frame,
                "packing_used_virtual_slots": sum(
                    1 for load in plan.slot_token_loads if load > 0
                ),
                "raw_tokens": len(plan.chunks) * 4 * int(frame_tokens),
                "kept_tokens": final_tokens,
                "token_budget": plan.token_budget,
            }
        )
    return payloads
