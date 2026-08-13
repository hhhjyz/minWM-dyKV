"""Token-budgeted retrieval packing for quantized dynamic KV crops."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import torch

from .dykv_memory import (
    DyKVBank,
    MemoryBlock,
    _horizontal_ray_angles,
    _single_batch_frames,
    _wrap_radians,
    build_yaw_crop_plan,
)


PACKING_ATOM_RATIO = 0.25
PACKING_ATOMS_PER_SLOT = 4
KEEP_TIERS = (1.0, 0.5, 0.25)


def quantize_keep_tier(overlap_ratio: float) -> float:
    """Map exact horizontal overlap to the registered fixed-size tiers."""

    ratio = float(overlap_ratio)
    if not math.isfinite(ratio) or ratio <= 0.0:
        return 0.0
    if ratio >= 0.75:
        return 1.0
    if ratio >= 0.375:
        return 0.5
    return 0.25


def tier_atoms(keep_tier: float) -> int:
    if keep_tier not in KEEP_TIERS:
        raise ValueError(f"unsupported packed retrieval tier: {keep_tier}")
    return int(round(float(keep_tier) / PACKING_ATOM_RATIO))


@dataclass(frozen=True)
class FrameCrop:
    frame_offset: int
    token_indices: torch.Tensor
    kept_columns: tuple[int, ...]
    raw_overlap_ratio: float | None
    keep_tier: float
    compression_mode: str
    delta_yaw_degrees: float | None = None
    horizontal_fov_degrees: float | None = None

    @property
    def token_count(self) -> int:
        return int(self.token_indices.numel())

    @property
    def atom_count(self) -> int:
        return tier_atoms(self.keep_tier)


@dataclass(frozen=True)
class BlockPackingCandidate:
    block_index: int
    distance: float
    similarity: float
    keep_tier: float
    cost_atoms: int
    utility: float
    chunk_frames: tuple[FrameCrop, ...]
    tail_frames: tuple[FrameCrop, ...]


@dataclass(frozen=True)
class PackedFramePlan:
    block_index: int
    frame_offset: int
    source_frame_id: int
    token_indices: torch.Tensor
    kept_columns: tuple[int, ...]
    raw_overlap_ratio: float | None
    keep_tier: float
    compression_mode: str
    selection_kind: str
    utility: float
    virtual_slot_id: int = -1
    delta_yaw_degrees: float | None = None
    horizontal_fov_degrees: float | None = None

    @property
    def token_count(self) -> int:
        return int(self.token_indices.numel())

    @property
    def atom_count(self) -> int:
        return tier_atoms(self.keep_tier)


@dataclass(frozen=True)
class PackedRetrievalPlan:
    frames: tuple[PackedFramePlan, ...]
    selected_full_blocks: tuple[int, ...]
    selected_tail_frames: tuple[tuple[int, int], ...]
    token_budget: int
    used_tokens: int
    used_virtual_slots: int
    budget_atoms: int
    used_atoms: int
    candidate_block_indices: tuple[int, ...]


@dataclass(frozen=True)
class FixedWorldKVFramePlan:
    block_index: int
    frame_offset: int
    source_frame_id: int
    token_indices: torch.Tensor
    virtual_slot_id: int
    selection_kind: str

    @property
    def token_count(self) -> int:
        return int(self.token_indices.numel())


@dataclass(frozen=True)
class FixedWorldKVRetrievalPlan:
    frames: tuple[FixedWorldKVFramePlan, ...]
    selected_blocks: tuple[int, ...]
    retrieval_frames: int
    keep_ratio: float
    keep_tokens_per_non_anchor: int
    token_budget: int
    used_tokens: int
    used_virtual_slots: int


def _spatial_token_indices(
    columns: Sequence[int],
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    column_tensor = torch.tensor(tuple(columns), dtype=torch.long)
    if column_tensor.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    rows = torch.arange(int(height), dtype=torch.long)[:, None]
    return (rows * int(width) + column_tensor[None, :]).flatten()


def _quantized_columns(
    *,
    visible_columns: Sequence[int],
    K: torch.Tensor,
    delta_yaw_degrees: float,
    width: int,
    keep_tier: float,
) -> tuple[int, ...]:
    target = int(round(float(keep_tier) * int(width)))
    if target <= 0:
        return ()
    if target >= int(width):
        return tuple(range(int(width)))
    rays = _horizontal_ray_angles(K, int(width))
    if rays is None:
        raise ValueError("packed geometry crop requires valid historical intrinsics")
    delta = math.radians(float(delta_yaw_degrees))
    distance = _wrap_radians(rays - delta).abs().tolist()
    visible = {int(value) for value in visible_columns}
    ranked = sorted(
        range(int(width)),
        key=lambda column: (0 if column in visible else 1, distance[column], column),
    )
    return tuple(sorted(ranked[:target]))


def _novelty_frame_indices(
    block: MemoryBlock,
    *,
    frame_tokens: int,
    keep_tier: float,
) -> tuple[torch.Tensor, ...]:
    """Choose one layer-shared fixed-size novelty mask for every source frame."""

    layer0 = block.layers[0].k.detach().to(device="cpu")
    expected = int(block.frame_count) * int(frame_tokens)
    if layer0.ndim != 4 or layer0.shape[1] != expected:
        raise ValueError("packed fallback requires complete frame-aligned layer-0 KV")
    batch, _, heads, dim = layer0.shape
    frames = layer0.reshape(batch, block.frame_count, frame_tokens, heads, dim)
    centroid = frames[:, 0].float().mean(dim=1)
    centroid_norm = torch.linalg.vector_norm(centroid, dim=(-2, -1))
    keep_tokens = int(round(float(keep_tier) * int(frame_tokens)))
    eps = torch.finfo(torch.float32).eps
    output = []
    for frame_index in range(block.frame_count):
        values = frames[:, frame_index].float()
        similarity = (values * centroid.unsqueeze(1)).sum(dim=(-2, -1))
        similarity = similarity / (
            torch.linalg.vector_norm(values, dim=(-2, -1))
            * centroid_norm.unsqueeze(1)
            + eps
        )
        # Inference banks use batch one. Averaging keeps the plan deterministic
        # and layer-shared if a diagnostic batch is ever used.
        score = similarity.mean(dim=0)
        indices = score.topk(keep_tokens, largest=False).indices.sort().values
        output.append(indices.to(dtype=torch.long))
    return tuple(output)


def _geometry_crop_for_tier(
    *,
    exact_plan,
    historical_Ks: torch.Tensor,
    frame_index: int,
    height: int,
    width: int,
    keep_tier: float,
) -> FrameCrop:
    visible = exact_plan.kept_columns_per_frame[frame_index]
    raw_ratio = len(visible) / float(width)
    delta = float(exact_plan.delta_yaw_degrees[frame_index])
    columns = _quantized_columns(
        visible_columns=visible,
        K=historical_Ks[frame_index],
        delta_yaw_degrees=delta,
        width=width,
        keep_tier=keep_tier,
    )
    return FrameCrop(
        frame_offset=int(frame_index),
        token_indices=_spatial_token_indices(columns, height=height, width=width),
        kept_columns=columns,
        raw_overlap_ratio=raw_ratio,
        keep_tier=float(keep_tier),
        compression_mode="yaw_fov_quantized",
        delta_yaw_degrees=delta,
        horizontal_fov_degrees=float(
            exact_plan.horizontal_fov_degrees[frame_index]
        ),
    )


def build_block_packing_candidate(
    block: MemoryBlock,
    *,
    block_index: int,
    distance: float,
    current_viewmats: torch.Tensor,
    current_Ks: torch.Tensor | None,
    frame_tokens: int,
    compression_fov_source: str,
    fixed_horizontal_degrees: float,
) -> BlockPackingCandidate | None:
    """Create fixed-size chunk and per-frame alternatives for one bank block."""

    if block.spatial_shape is None:
        return None
    height, width = (int(block.spatial_shape[0]), int(block.spatial_shape[1]))
    if height * width != int(frame_tokens):
        raise ValueError("packed retrieval spatial shape must match frame tokens")
    if width % PACKING_ATOMS_PER_SLOT:
        raise ValueError("packed retrieval latent width must be divisible by four")

    exact = build_yaw_crop_plan(
        historical_viewmats=block.viewmats,
        historical_Ks=block.Ks,
        current_viewmats=current_viewmats,
        current_Ks=current_Ks,
        frame_count=block.frame_count,
        frame_tokens=frame_tokens,
        spatial_shape=block.spatial_shape,
        fov_source=compression_fov_source,
        fixed_horizontal_degrees=fixed_horizontal_degrees,
    )
    if exact is not None:
        historical_Ks = _single_batch_frames(block.Ks, matrix_size=3)
        if compression_fov_source == "fixed":
            focal = 0.5 / math.tan(
                math.radians(float(fixed_horizontal_degrees)) / 2.0
            )
            fixed_K = torch.tensor(
                [[focal, 0.0, 0.5], [0.0, focal, 0.5], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            )
            historical_Ks = fixed_K.repeat(block.frame_count, 1, 1)
        if historical_Ks is None:
            return None
        raw_ratios = [
            len(columns) / float(width) for columns in exact.kept_columns_per_frame
        ]
        chunk_tier = quantize_keep_tier(max(raw_ratios, default=0.0))
        if chunk_tier == 0.0:
            return None
        chunk_frames = tuple(
            _geometry_crop_for_tier(
                exact_plan=exact,
                historical_Ks=historical_Ks,
                frame_index=index,
                height=height,
                width=width,
                keep_tier=chunk_tier,
            )
            for index in range(block.frame_count)
        )
        tail_frames = tuple(
            _geometry_crop_for_tier(
                exact_plan=exact,
                historical_Ks=historical_Ks,
                frame_index=index,
                height=height,
                width=width,
                keep_tier=frame_tier,
            )
            for index, ratio in enumerate(raw_ratios)
            if (frame_tier := quantize_keep_tier(ratio)) > 0.0
        )
    else:
        # Translation, pitch, roll, or missing geometry uses the registered
        # 50% layer-shared novelty fallback.
        chunk_tier = 0.5
        novelty = _novelty_frame_indices(
            block, frame_tokens=frame_tokens, keep_tier=chunk_tier
        )
        chunk_frames = tuple(
            FrameCrop(
                frame_offset=index,
                token_indices=indices,
                kept_columns=(),
                raw_overlap_ratio=None,
                keep_tier=chunk_tier,
                compression_mode="fixed_novelty_quantized_fallback",
            )
            for index, indices in enumerate(novelty)
        )
        tail_frames = chunk_frames

    similarity = min(1.0, max(0.0, 1.0 - float(distance)))
    cost_atoms = int(block.frame_count) * tier_atoms(chunk_tier)
    return BlockPackingCandidate(
        block_index=int(block_index),
        distance=float(distance),
        similarity=similarity,
        keep_tier=chunk_tier,
        cost_atoms=cost_atoms,
        utility=similarity * math.sqrt(chunk_tier),
        chunk_frames=chunk_frames,
        tail_frames=tail_frames,
    )


def _knapsack(items, *, capacity: int, cost, utility) -> tuple:
    """Deterministic 0/1 knapsack over the tiny registered atom budget."""

    states: list[tuple[float, tuple[int, ...]] | None] = [None] * (capacity + 1)
    states[0] = (0.0, ())
    for item_index, item in enumerate(items):
        item_cost = int(cost(item))
        item_utility = float(utility(item))
        if item_cost <= 0 or item_cost > capacity or item_utility <= 0.0:
            continue
        for used in range(capacity - item_cost, -1, -1):
            state = states[used]
            if state is None:
                continue
            target = used + item_cost
            candidate = (state[0] + item_utility, state[1] + (item_index,))
            previous = states[target]
            if previous is None or candidate[0] > previous[0] + 1e-12:
                states[target] = candidate
            elif abs(candidate[0] - previous[0]) <= 1e-12 and candidate[1] < previous[1]:
                states[target] = candidate

    best_used = 0
    best = states[0]
    for used, state in enumerate(states):
        if state is None:
            continue
        if state[0] > best[0] + 1e-12:
            best_used, best = used, state
        elif abs(state[0] - best[0]) <= 1e-12 and used > best_used:
            best_used, best = used, state
    return tuple(items[index] for index in best[1])


def select_full_chunk_candidates(
    candidates: Sequence[BlockPackingCandidate],
    *,
    budget_atoms: int,
) -> tuple[BlockPackingCandidate, ...]:
    return _knapsack(
        tuple(candidates),
        capacity=int(budget_atoms),
        cost=lambda item: item.cost_atoms,
        utility=lambda item: item.utility,
    )


def _as_packed_frame(
    candidate: BlockPackingCandidate,
    crop: FrameCrop,
    block: MemoryBlock,
    *,
    selection_kind: str,
) -> PackedFramePlan:
    similarity = (
        candidate.similarity
        if crop.raw_overlap_ratio is None
        else float(crop.raw_overlap_ratio)
    )
    return PackedFramePlan(
        block_index=candidate.block_index,
        frame_offset=crop.frame_offset,
        source_frame_id=block.frame_start + crop.frame_offset,
        token_indices=crop.token_indices,
        kept_columns=crop.kept_columns,
        raw_overlap_ratio=crop.raw_overlap_ratio,
        keep_tier=crop.keep_tier,
        compression_mode=crop.compression_mode,
        selection_kind=selection_kind,
        utility=similarity * math.sqrt(crop.keep_tier),
        delta_yaw_degrees=crop.delta_yaw_degrees,
        horizontal_fov_degrees=crop.horizontal_fov_degrees,
    )


def _assign_full_chunk_slots(
    frames: list[PackedFramePlan],
    *,
    first_slot: int,
) -> tuple[list[PackedFramePlan], int]:
    output = []
    slot = int(first_slot)
    used_in_slot = 0
    for frame in frames:
        atoms = frame.atom_count
        if used_in_slot + atoms > PACKING_ATOMS_PER_SLOT:
            slot += 1
            used_in_slot = 0
        output.append(replace(frame, virtual_slot_id=slot))
        used_in_slot += atoms
        if used_in_slot == PACKING_ATOMS_PER_SLOT:
            slot += 1
            used_in_slot = 0
    if used_in_slot:
        raise RuntimeError("a complete quantized chunk must end on a slot boundary")
    return output, slot


def _assign_tail_slots(
    frames: Sequence[PackedFramePlan],
    *,
    first_slot: int,
    slot_count: int,
) -> list[PackedFramePlan]:
    capacities = [PACKING_ATOMS_PER_SLOT] * int(slot_count)
    output = []
    ordered = sorted(
        frames,
        key=lambda frame: (
            -frame.atom_count,
            -frame.utility,
            frame.source_frame_id,
            frame.block_index,
        ),
    )
    for frame in ordered:
        for bin_index, remaining in enumerate(capacities):
            if remaining >= frame.atom_count:
                capacities[bin_index] -= frame.atom_count
                output.append(
                    replace(frame, virtual_slot_id=int(first_slot) + bin_index)
                )
                break
        else:
            raise RuntimeError("selected tail frame does not fit retrieval slots")
    return output


def build_packed_retrieval_plan(
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
    compression_fov_source: str,
    fixed_horizontal_degrees: float,
) -> PackedRetrievalPlan:
    if len(ranked_block_indices) != len(ranked_distances):
        raise ValueError("packed retrieval ranking and distances must align")
    atom_tokens = int(round(PACKING_ATOM_RATIO * int(frame_tokens)))
    if atom_tokens * PACKING_ATOMS_PER_SLOT != int(frame_tokens):
        raise ValueError("frame token count must be divisible into four packing atoms")
    budget_atoms = int(memory_frames) * PACKING_ATOMS_PER_SLOT
    candidates = []
    for block_index, distance in zip(ranked_block_indices, ranked_distances):
        candidate = build_block_packing_candidate(
            bank.blocks[int(block_index)],
            block_index=int(block_index),
            distance=float(distance),
            current_viewmats=current_viewmats,
            current_Ks=current_Ks,
            frame_tokens=frame_tokens,
            compression_fov_source=compression_fov_source,
            fixed_horizontal_degrees=fixed_horizontal_degrees,
        )
        if candidate is not None:
            candidates.append(candidate)

    # FOV score first, then newer history, then stable block identity.  The
    # knapsack uses this order to resolve equal-utility plans deterministically.
    candidates.sort(
        key=lambda item: (
            item.distance,
            -bank.blocks[item.block_index].frame_start,
            bank.blocks[item.block_index].block_id,
        )
    )
    selected_full = select_full_chunk_candidates(candidates, budget_atoms=budget_atoms)
    selected_full_indices = {item.block_index for item in selected_full}
    used_atoms = sum(item.cost_atoms for item in selected_full)
    packed_frames: list[PackedFramePlan] = []
    next_slot = int(sink_frames)
    for candidate in sorted(
        selected_full, key=lambda item: bank.blocks[item.block_index].frame_start
    ):
        block = bank.blocks[candidate.block_index]
        chunk_frames = [
            _as_packed_frame(candidate, crop, block, selection_kind="full_chunk")
            for crop in candidate.chunk_frames
        ]
        assigned, next_slot = _assign_full_chunk_slots(
            chunk_frames, first_slot=next_slot
        )
        packed_frames.extend(assigned)

    selected_tail: tuple[PackedFramePlan, ...] = ()
    if include_tail_latents and used_atoms < budget_atoms:
        tail_candidates = []
        for candidate in candidates:
            if candidate.block_index in selected_full_indices:
                continue
            block = bank.blocks[candidate.block_index]
            tail_candidates.extend(
                _as_packed_frame(candidate, crop, block, selection_kind="tail_latent")
                for crop in candidate.tail_frames
            )
        selected_tail = _knapsack(
            tuple(tail_candidates),
            capacity=budget_atoms - used_atoms,
            cost=lambda frame: frame.atom_count,
            utility=lambda frame: frame.utility,
        )
        remaining_slots = int(sink_frames) + int(memory_frames) - next_slot
        packed_frames.extend(
            _assign_tail_slots(
                selected_tail,
                first_slot=next_slot,
                slot_count=remaining_slots,
            )
        )
        used_atoms += sum(frame.atom_count for frame in selected_tail)

    packed_frames.sort(key=lambda frame: (frame.virtual_slot_id, frame.source_frame_id))
    used_tokens = used_atoms * atom_tokens
    actual_tokens = sum(frame.token_count for frame in packed_frames)
    if actual_tokens != used_tokens or used_tokens > int(memory_frames) * int(frame_tokens):
        raise RuntimeError("packed retrieval plan violates its physical token budget")
    used_slots = len({frame.virtual_slot_id for frame in packed_frames})
    return PackedRetrievalPlan(
        frames=tuple(packed_frames),
        selected_full_blocks=tuple(
            item.block_index
            for item in sorted(
                selected_full, key=lambda item: bank.blocks[item.block_index].frame_start
            )
        ),
        selected_tail_frames=tuple(
            (frame.block_index, frame.frame_offset) for frame in selected_tail
        ),
        token_budget=int(memory_frames) * int(frame_tokens),
        used_tokens=used_tokens,
        used_virtual_slots=used_slots,
        budget_atoms=budget_atoms,
        used_atoms=used_atoms,
        candidate_block_indices=tuple(item.block_index for item in candidates),
    )


def materialize_packed_retrieval(
    bank: DyKVBank,
    plan: PackedRetrievalPlan,
    *,
    target_device: torch.device | str,
    frame_tokens: int,
) -> list[dict]:
    if not plan.frames:
        return []
    layer_count = len(bank.blocks[plan.frames[0].block_index].layers)
    if any(len(bank.blocks[frame.block_index].layers) != layer_count for frame in plan.frames):
        raise RuntimeError("packed retrieval bank blocks have inconsistent layer counts")
    device = torch.device(target_device)
    payloads = []
    for layer_index in range(layer_count):
        key_parts = []
        value_parts = []
        for frame in plan.frames:
            block = bank.blocks[frame.block_index]
            source_start = int(frame.frame_offset) * int(frame_tokens)
            indices = frame.token_indices.to(device=device) + source_start
            raw_k = block.layers[layer_index].k.to(device=device)
            raw_v = block.layers[layer_index].v.to(device=device)
            key_parts.append(raw_k.index_select(1, indices))
            value_parts.append(raw_v.index_select(1, indices))
        packed_k = torch.cat(key_parts, dim=1)
        packed_v = torch.cat(value_parts, dim=1)
        frame_lengths = [frame.token_count for frame in plan.frames]
        if int(packed_k.shape[1]) != plan.used_tokens:
            raise RuntimeError("materialized retrieval length differs from packing plan")
        payloads.append(
            {
                "k": packed_k,
                "v": packed_v,
                "src_frame_ids": [frame.source_frame_id for frame in plan.frames],
                "chunk_frame_counts": [1] * len(plan.frames),
                "chunk_token_lengths": frame_lengths,
                "source_frame_ids": [frame.source_frame_id for frame in plan.frames],
                "frame_token_lengths": frame_lengths,
                "virtual_slot_ids": [frame.virtual_slot_id for frame in plan.frames],
                "parent_block_ids": [
                    bank.blocks[frame.block_index].block_id for frame in plan.frames
                ],
                "selection_kinds": [frame.selection_kind for frame in plan.frames],
                "keep_tiers": [frame.keep_tier for frame in plan.frames],
                "raw_overlap_ratios": [
                    frame.raw_overlap_ratio for frame in plan.frames
                ],
                "compression_modes": [
                    frame.compression_mode for frame in plan.frames
                ],
                "kept_tokens_per_frame": frame_lengths,
                "kept_columns_per_frame": [
                    list(frame.kept_columns) for frame in plan.frames
                ],
                "delta_yaw_degrees": [
                    frame.delta_yaw_degrees for frame in plan.frames
                ],
                "horizontal_fov_degrees": [
                    frame.horizontal_fov_degrees for frame in plan.frames
                ],
                "selected_full_block_ids": [
                    bank.blocks[index].block_id for index in plan.selected_full_blocks
                ],
                "selected_tail_frame_ids": [
                    bank.blocks[index].frame_start + offset
                    for index, offset in plan.selected_tail_frames
                ],
                "packing_budget_atoms": plan.budget_atoms,
                "packing_used_atoms": plan.used_atoms,
                "packing_atom_tokens": int(frame_tokens) // PACKING_ATOMS_PER_SLOT,
                "packing_used_virtual_slots": plan.used_virtual_slots,
                "raw_tokens": len(plan.frames) * int(frame_tokens),
                "kept_tokens": int(packed_k.shape[1]),
                "token_budget": plan.token_budget,
            }
        )
    return payloads


def _fixed_worldkv_indices(
    block: MemoryBlock,
    *,
    frame_tokens: int,
    keep_ratio: float,
) -> tuple[torch.Tensor, ...]:
    """Build a layer-shared anchor-plus-novelty mask for one whole chunk."""

    layer0 = block.layers[0].k.detach().to(device="cpu")
    expected = int(block.frame_count) * int(frame_tokens)
    if layer0.ndim != 4 or layer0.shape[1] != expected:
        raise ValueError("fixed WorldKV packing requires complete frame-aligned KV")
    batch, _, heads, dim = layer0.shape
    frames = layer0.reshape(batch, block.frame_count, frame_tokens, heads, dim)
    anchor = frames[:, 0]
    centroid = anchor.float().mean(dim=1)
    centroid_norm = torch.linalg.vector_norm(centroid, dim=(-2, -1))
    keep_tokens = max(1, int(math.ceil(float(keep_ratio) * int(frame_tokens))))
    keep_tokens = min(int(frame_tokens), keep_tokens)
    eps = torch.finfo(torch.float32).eps
    output = [torch.arange(int(frame_tokens), dtype=torch.long)]
    for frame_index in range(1, int(block.frame_count)):
        values = frames[:, frame_index].float()
        similarity = (values * centroid.unsqueeze(1)).sum(dim=(-2, -1))
        similarity = similarity / (
            torch.linalg.vector_norm(values, dim=(-2, -1))
            * centroid_norm.unsqueeze(1)
            + eps
        )
        score = similarity.mean(dim=0)
        output.append(
            score.topk(keep_tokens, largest=False).indices.sort().values.to(torch.long)
        )
    return tuple(output)


def build_fixed_worldkv_retrieval_plan(
    bank: DyKVBank,
    selected_block_indices: Sequence[int],
    *,
    frame_tokens: int,
    memory_frames: int,
    sink_frames: int,
    retrieval_frames: int,
    keep_ratio: float,
) -> FixedWorldKVRetrievalPlan:
    """Pack minWM-back's fixed anchor+novelty cases into eight token slots.

    Full anchor frames occupy dedicated slots.  Fixed-size non-anchor segments
    are then packed first-fit into the remaining slots, allowing segments from
    different source chunks to share a virtual time position.
    """

    if not 0.0 < float(keep_ratio) <= 1.0:
        raise ValueError("fixed WorldKV keep_ratio must be in (0, 1]")
    selected = tuple(int(index) for index in selected_block_indices)
    selected = tuple(
        sorted(selected, key=lambda index: bank.blocks[index].frame_start)
    )
    raw_frames = sum(int(bank.blocks[index].frame_count) for index in selected)
    if raw_frames > int(retrieval_frames):
        raise ValueError("fixed WorldKV selection exceeds its source-frame budget")
    token_budget = int(memory_frames) * int(frame_tokens)
    segment_specs = []
    for block_index in selected:
        block = bank.blocks[block_index]
        indices_per_frame = _fixed_worldkv_indices(
            block,
            frame_tokens=frame_tokens,
            keep_ratio=keep_ratio,
        )
        for frame_offset, indices in enumerate(indices_per_frame):
            segment_specs.append(
                (
                    block_index,
                    frame_offset,
                    block.frame_start + frame_offset,
                    indices,
                    "anchor" if frame_offset == 0 else "novelty",
                )
            )

    # Anchors must remain full, so reserve their bins first.  Novelty segments
    # from different chunks may share the remaining virtual slots.
    bins: list[dict] = []
    anchors = [segment for segment in segment_specs if segment[4] == "anchor"]
    novelty = [segment for segment in segment_specs if segment[4] == "novelty"]
    for segment in anchors:
        bins.append({"remaining": 0, "segments": [segment]})
    for segment in sorted(novelty, key=lambda item: (-item[3].numel(), item[2])):
        length = int(segment[3].numel())
        for bin_item in bins:
            if int(bin_item["remaining"]) >= length:
                bin_item["remaining"] = int(bin_item["remaining"]) - length
                bin_item["segments"].append(segment)
                break
        else:
            if len(bins) >= int(memory_frames):
                raise ValueError("fixed WorldKV payload exceeds retrieval token slots")
            bins.append(
                {
                    "remaining": int(frame_tokens) - length,
                    "segments": [segment],
                }
            )

    frames = []
    for bin_index, bin_item in enumerate(bins):
        slot = int(sink_frames) + bin_index
        for block_index, frame_offset, source_frame, indices, kind in sorted(
            bin_item["segments"], key=lambda item: item[2]
        ):
            frames.append(
                FixedWorldKVFramePlan(
                    block_index=block_index,
                    frame_offset=frame_offset,
                    source_frame_id=source_frame,
                    token_indices=indices,
                    virtual_slot_id=slot,
                    selection_kind=kind,
                )
            )
    used_tokens = sum(frame.token_count for frame in frames)
    if used_tokens > token_budget:
        raise ValueError("fixed WorldKV payload exceeds retrieval token budget")
    return FixedWorldKVRetrievalPlan(
        frames=tuple(frames),
        selected_blocks=selected,
        retrieval_frames=int(retrieval_frames),
        keep_ratio=float(keep_ratio),
        keep_tokens_per_non_anchor=max(
            1, int(math.ceil(float(keep_ratio) * int(frame_tokens)))
        ),
        token_budget=token_budget,
        used_tokens=used_tokens,
        used_virtual_slots=len(bins),
    )


def materialize_fixed_worldkv_retrieval(
    bank: DyKVBank,
    plan: FixedWorldKVRetrievalPlan,
    *,
    target_device: torch.device | str,
    frame_tokens: int,
) -> list[dict]:
    if not plan.frames:
        return []
    layer_count = len(bank.blocks[plan.frames[0].block_index].layers)
    if any(
        len(bank.blocks[frame.block_index].layers) != layer_count
        for frame in plan.frames
    ):
        raise RuntimeError("fixed WorldKV bank blocks have inconsistent layer counts")
    device = torch.device(target_device)
    payloads = []
    frame_lengths = [frame.token_count for frame in plan.frames]
    source_frames = [frame.source_frame_id for frame in plan.frames]
    virtual_slots = [frame.virtual_slot_id for frame in plan.frames]
    for layer_index in range(layer_count):
        key_parts = []
        value_parts = []
        for frame in plan.frames:
            block = bank.blocks[frame.block_index]
            source_start = frame.frame_offset * int(frame_tokens)
            indices = frame.token_indices.to(device=device) + source_start
            key_parts.append(
                block.layers[layer_index].k.to(device=device).index_select(1, indices)
            )
            value_parts.append(
                block.layers[layer_index].v.to(device=device).index_select(1, indices)
            )
        packed_k = torch.cat(key_parts, dim=1)
        packed_v = torch.cat(value_parts, dim=1)
        if int(packed_k.shape[1]) != plan.used_tokens:
            raise RuntimeError("fixed WorldKV materialization differs from its plan")
        payloads.append(
            {
                "k": packed_k,
                "v": packed_v,
                "source_frame_ids": source_frames,
                "frame_token_lengths": frame_lengths,
                "virtual_slot_ids": virtual_slots,
                "src_frame_ids": [
                    bank.blocks[index].frame_start for index in plan.selected_blocks
                ],
                "chunk_frame_counts": [
                    bank.blocks[index].frame_count for index in plan.selected_blocks
                ],
                "chunk_token_lengths": [
                    int(frame_tokens)
                    + (bank.blocks[index].frame_count - 1)
                    * plan.keep_tokens_per_non_anchor
                    for index in plan.selected_blocks
                ],
                "parent_block_ids": [
                    bank.blocks[frame.block_index].block_id for frame in plan.frames
                ],
                "selection_kinds": [frame.selection_kind for frame in plan.frames],
                "compression_modes": ["fixed_worldkv_anchor_novelty"] * len(plan.frames),
                "kept_tokens_per_frame": frame_lengths,
                "fixed_keep_ratio": plan.keep_ratio,
                "fixed_retrieval_frames": plan.retrieval_frames,
                "packing_used_virtual_slots": plan.used_virtual_slots,
                "raw_tokens": sum(
                    bank.blocks[index].frame_count for index in plan.selected_blocks
                )
                * int(frame_tokens),
                "kept_tokens": plan.used_tokens,
                "token_budget": plan.token_budget,
            }
        )
    return payloads
