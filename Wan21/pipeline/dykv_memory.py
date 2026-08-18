"""Minimal KV memory primitives used by minWM-dyKV inference.

The live attention cache remains owned by ``CausalWanSelfAttention``.  This
module archives each clean block before the rolling cache can evict it, keeps
the archive off GPU, and materializes selected blocks only when they are about
to be attended to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class DyKVConfig:
    """Single coherent dyKV preset.

    Region sizes define the method and are kept together so the contiguous
    4 + 8 + 8 layout can be validated in one place. They are not public CLI
    hyperparameters.
    """

    enabled: bool = False
    memory_frames: int = 8
    # Number of raw source frames considered before retrieval-time packing.
    retrieval_frames: int = 8
    sink_frames: int = 4
    # The local region includes both recent cached frames and the current query.
    local_frames: int = 8
    rope_train_frames: int = 20
    compression_keep_ratio: float = 0.5
    compression_mode: str = "yaw_fov"
    packing_mode: str = "none"
    retrieval_layout: str = "source_ordered"
    # Unpacked chunk-to-RoPE-slot assignment. Relevance ordering keeps the
    # selected set unchanged and only puts the best match nearest the query.
    retrieval_order: str = "source_ordered"
    retrieval_mode: str = "fov"
    bank_device: str = "cpu"
    fov_samples: int = 8192
    fov_radius: float = 8.0
    motion_geometry_mode: str = "projected_multidepth"
    projection_scene_scale: float = 8.0

    def validate(self, *, chunk_frames: int) -> "DyKVConfig":
        if self.memory_frames <= 0:
            raise ValueError("dyKV memory_frames must be positive")
        if chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        if self.memory_frames % chunk_frames:
            raise ValueError(
                "dyKV memory_frames must be divisible by the model chunk size"
            )
        if self.sink_frames < 0 or self.local_frames < chunk_frames:
            raise ValueError(
                "dyKV sink frames cannot be negative and the local region must "
                "contain the current chunk"
            )
        occupied = self.sink_frames + self.memory_frames + self.local_frames
        if occupied != self.rope_train_frames:
            raise ValueError(
                "dyKV regions must exactly fill the trained RoPE range: "
                f"sink({self.sink_frames}) + memory({self.memory_frames}) + "
                f"local_including_current({self.local_frames}) != "
                f"rope_train_frames({self.rope_train_frames})"
            )
        if not 0.0 < self.compression_keep_ratio <= 1.0:
            raise ValueError("dyKV compression_keep_ratio must be in (0, 1]")
        if self.compression_mode not in {
            "none",
            "fixed_novelty",
            "yaw_fov",
            "motion_novelty",
        }:
            raise ValueError(
                "unsupported dyKV compression_mode"
            )
        if self.retrieval_mode not in {"fov", "worldkv_pose"}:
            raise ValueError("dyKV retrieval_mode must be fov or worldkv_pose")
        if self.motion_geometry_mode not in {"projected_multidepth", "sphere_fov"}:
            raise ValueError("unsupported dyKV motion_geometry_mode")
        if (
            not math.isfinite(self.projection_scene_scale)
            or self.projection_scene_scale <= 0.0
        ):
            raise ValueError("dyKV projection_scene_scale must be positive and finite")
        if self.packing_mode not in {
            "none",
            "whole_chunks",
            "whole_chunks_and_latents",
            "fixed_worldkv",
            "motion_novelty_slot_capped",
            "motion_novelty_flat",
            "motion_novelty_backfill",
            "motion_novelty_duplicate",
        }:
            raise ValueError(
                "unsupported dyKV packing_mode"
            )
        if self.retrieval_layout not in {
            "slot_packed",
            "source_ordered",
            "flat_source_ordered",
        }:
            raise ValueError("unsupported dyKV retrieval_layout")
        if self.retrieval_order not in {
            "source_ordered",
            "relevance_near_query",
        }:
            raise ValueError("unsupported dyKV retrieval_order")
        if (
            self.retrieval_order == "relevance_near_query"
            and self.packing_mode != "none"
        ):
            raise ValueError(
                "relevance_near_query currently requires unpacked retrieval"
            )
        if self.retrieval_frames <= 0 or self.retrieval_frames % chunk_frames:
            raise ValueError(
                "dyKV retrieval_frames must be positive and chunk-aligned"
            )
        if self.packing_mode == "none" and self.retrieval_frames > self.memory_frames:
            raise ValueError(
                "unpacked dyKV retrieval_frames cannot exceed memory_frames"
            )
        return self


@dataclass(frozen=True)
class LayerKV:
    """One transformer's clean key/value tensors for a historical block."""

    k: torch.Tensor
    v: torch.Tensor


@dataclass(frozen=True)
class MemoryBlock:
    """Frame-aligned KV archive shared by every transformer layer."""

    block_id: int
    frame_start: int
    frame_count: int
    layers: tuple[LayerKV, ...]
    viewmats: torch.Tensor | None = None
    Ks: torch.Tensor | None = None
    spatial_shape: tuple[int, int] | None = None

    @property
    def frame_end(self) -> int:
        return self.frame_start + self.frame_count


def _copy_off_cache(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.detach().to(device=device).contiguous().clone()


@dataclass(frozen=True)
class YawCropPlan:
    """Shared token indices and diagnostics for one historical block."""

    token_indices: torch.Tensor
    kept_tokens_per_frame: tuple[int, ...]
    kept_columns_per_frame: tuple[tuple[int, ...], ...]
    delta_yaw_degrees: tuple[float, ...]
    horizontal_fov_degrees: tuple[float, ...]


def _single_batch_frames(
    tensor: torch.Tensor | None,
    *,
    matrix_size: int,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    tensor = tensor.detach().to(device="cpu", dtype=torch.float64)
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            return None
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[-2:] != (matrix_size, matrix_size):
        return None
    return tensor


def _camera_center_and_rotation(w2c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c2w = torch.linalg.inv(w2c)
    return c2w[:3, 3], c2w[:3, :3]


def _wrap_radians(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def _pure_yaw_delta(
    historical_w2c: torch.Tensor,
    current_w2c: torch.Tensor,
    *,
    tolerance: float = 1e-4,
) -> float | None:
    """Return current yaw relative to history, or ``None`` for non-yaw motion."""

    historical_center, historical_rotation = _camera_center_and_rotation(historical_w2c)
    current_center, current_rotation = _camera_center_and_rotation(current_w2c)
    if float(torch.linalg.vector_norm(current_center - historical_center)) > tolerance:
        return None

    relative = historical_rotation.T @ current_rotation
    yaw = torch.atan2(relative[0, 2], relative[2, 2])
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    expected = torch.stack(
        (
            torch.stack((cosine, cosine.new_zeros(()), sine)),
            torch.stack((cosine.new_zeros(()), cosine.new_ones(()), cosine.new_zeros(()))),
            torch.stack((-sine, cosine.new_zeros(()), cosine)),
        )
    )
    if float((relative - expected).abs().max()) > tolerance:
        return None
    return float(_wrap_radians(yaw).item())


def _horizontal_ray_angles(K: torch.Tensor, width: int) -> torch.Tensor | None:
    fx = float(K[0, 0])
    cx = float(K[0, 2])
    if not math.isfinite(fx) or not math.isfinite(cx) or fx <= 0.0:
        return None
    centers = (torch.arange(width, dtype=torch.float64) + 0.5) / float(width)
    return torch.atan((centers - cx) / fx)


def _horizontal_fov_bounds(K: torch.Tensor) -> tuple[float, float] | None:
    fx = float(K[0, 0])
    cx = float(K[0, 2])
    if not math.isfinite(fx) or not math.isfinite(cx) or fx <= 0.0:
        return None
    return math.atan((0.0 - cx) / fx), math.atan((1.0 - cx) / fx)


def build_yaw_crop_plan(
    *,
    historical_viewmats: torch.Tensor | None,
    historical_Ks: torch.Tensor | None,
    current_viewmats: torch.Tensor | None,
    current_Ks: torch.Tensor | None,
    frame_count: int,
    frame_tokens: int,
    spatial_shape: tuple[int, int] | None,
) -> YawCropPlan | None:
    """Build a direction-aware latent-column crop for pure yaw motion.

    ``None`` means geometry is missing or the motion includes translation,
    pitch, or roll, so callers must use the fixed novelty fallback. An empty
    ``token_indices`` tensor is a valid plan and means there is no horizontal
    FOV overlap with the current query chunk.
    """

    historical_poses = _single_batch_frames(historical_viewmats, matrix_size=4)
    current_poses = _single_batch_frames(current_viewmats, matrix_size=4)
    if historical_poses is None or current_poses is None:
        return None
    historical_intrinsics = _single_batch_frames(historical_Ks, matrix_size=3)
    current_intrinsics = _single_batch_frames(current_Ks, matrix_size=3)
    if historical_intrinsics is None or current_intrinsics is None:
        return None
    if spatial_shape is None or len(spatial_shape) != 2:
        return None
    height, width = (int(spatial_shape[0]), int(spatial_shape[1]))
    if height <= 0 or width <= 0 or height * width != int(frame_tokens):
        return None
    if historical_poses.shape[0] != int(frame_count):
        return None
    if historical_intrinsics.shape[0] != int(frame_count):
        return None
    if current_poses.shape[0] == 0 or current_intrinsics.shape[0] != current_poses.shape[0]:
        return None

    all_indices: list[torch.Tensor] = []
    kept_tokens: list[int] = []
    kept_columns: list[tuple[int, ...]] = []
    delta_yaws: list[float] = []
    horizontal_fovs: list[float] = []

    for frame_index in range(int(frame_count)):
        ray_angles = _horizontal_ray_angles(historical_intrinsics[frame_index], width)
        if ray_angles is None:
            return None
        column_mask = torch.zeros(width, dtype=torch.bool)
        frame_deltas: list[float] = []
        frame_fovs: list[float] = []
        for query_index in range(current_poses.shape[0]):
            delta = _pure_yaw_delta(
                historical_poses[frame_index], current_poses[query_index]
            )
            bounds = _horizontal_fov_bounds(current_intrinsics[query_index])
            if delta is None or bounds is None:
                return None
            left, right = bounds
            relative_rays = _wrap_radians(ray_angles - delta)
            column_mask |= (relative_rays >= left) & (relative_rays <= right)
            frame_deltas.append(delta)
            frame_fovs.append(right - left)

        columns = torch.nonzero(column_mask, as_tuple=False).flatten()
        if columns.numel():
            rows = torch.arange(height, dtype=torch.long)[:, None]
            spatial_indices = (rows * width + columns[None, :]).flatten()
            all_indices.append(spatial_indices + frame_index * int(frame_tokens))
        kept_tokens.append(int(columns.numel()) * height)
        kept_columns.append(tuple(int(value) for value in columns.tolist()))
        nearest = min(frame_deltas, key=lambda value: abs(value))
        delta_yaws.append(math.degrees(nearest))
        horizontal_fovs.append(math.degrees(max(frame_fovs)))

    token_indices = (
        torch.cat(all_indices)
        if all_indices
        else torch.empty(0, dtype=torch.long)
    )
    return YawCropPlan(
        token_indices=token_indices,
        kept_tokens_per_frame=tuple(kept_tokens),
        kept_columns_per_frame=tuple(kept_columns),
        delta_yaw_degrees=tuple(delta_yaws),
        horizontal_fov_degrees=tuple(horizontal_fovs),
    )


def compress_retrieved_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    chunk_frames: int,
    frame_tokens: int,
    keep_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """WorldKV anchor-plus-novelty compression at retrieval time.

    The first frame of every chunk is retained in full.  In each later frame,
    tokens least similar to the anchor-key centroid are retained as novel
    content.  Storage remains lossless and compression work is paid only for
    blocks that retrieval actually selects.
    """

    if k.shape != v.shape or k.ndim != 4:
        raise ValueError("retrieved K/V must share [batch, tokens, heads, dim]")
    chunk_tokens = int(chunk_frames) * int(frame_tokens)
    if chunk_tokens <= 0 or k.shape[1] % chunk_tokens:
        raise ValueError("retrieved K/V must contain complete frame-aligned chunks")
    keep_tokens = max(1, int(math.ceil(float(keep_ratio) * frame_tokens)))
    if keep_tokens >= frame_tokens or k.shape[1] == 0:
        return k, v

    batch, _, heads, dim = k.shape
    output_k: list[torch.Tensor] = []
    output_v: list[torch.Tensor] = []
    eps = torch.finfo(torch.float32).eps

    for start in range(0, k.shape[1], chunk_tokens):
        chunk_k = k[:, start:start + chunk_tokens].reshape(
            batch, chunk_frames, frame_tokens, heads, dim
        )
        chunk_v = v[:, start:start + chunk_tokens].reshape(
            batch, chunk_frames, frame_tokens, heads, dim
        )
        anchor_k = chunk_k[:, 0]
        anchor_v = chunk_v[:, 0]
        centroid = anchor_k.float().mean(dim=1)
        centroid_norm = torch.linalg.vector_norm(centroid, dim=(-2, -1))
        kept_k = [anchor_k]
        kept_v = [anchor_v]

        for frame_index in range(1, chunk_frames):
            frame_k = chunk_k[:, frame_index]
            frame_float = frame_k.float()
            similarity = (frame_float * centroid.unsqueeze(1)).sum(dim=(-2, -1))
            similarity = similarity / (
                torch.linalg.vector_norm(frame_float, dim=(-2, -1))
                * centroid_norm.unsqueeze(1)
                + eps
            )
            indices = similarity.topk(keep_tokens, dim=1, largest=False).indices
            indices = indices.sort(dim=1).values
            gather = indices[:, :, None, None].expand(-1, -1, heads, dim)
            kept_k.append(torch.gather(frame_k, 1, gather))
            kept_v.append(torch.gather(chunk_v[:, frame_index], 1, gather))

        output_k.append(torch.cat(kept_k, dim=1))
        output_v.append(torch.cat(kept_v, dim=1))

    return torch.cat(output_k, dim=1), torch.cat(output_v, dim=1)


class DyKVBank:
    """CPU archive for clean blocks that will leave the rolling live cache."""

    def __init__(self, *, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.blocks: list[MemoryBlock] = []

    def clear(self) -> None:
        self.blocks.clear()

    def archive_clean_block(
        self,
        caches: Sequence[dict],
        *,
        frame_start: int,
        frame_count: int,
        frame_tokens: int,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
        spatial_shape: tuple[int, int] | None = None,
    ) -> MemoryBlock:
        """Copy the newest clean block out of every live layer cache.

        Archiving at clean-cache commit time guarantees the tensors are captured
        before a later attention call rolls them out of the fixed-size cache.
        Eligibility is handled separately, so blocks still present in the local
        region cannot be retrieved twice.
        """

        token_count = int(frame_count) * int(frame_tokens)
        if token_count <= 0:
            raise ValueError("archive block must contain at least one token")
        layers: list[LayerKV] = []
        for layer_index, cache in enumerate(caches):
            local_end = int(cache["local_end_index"].item())
            local_start = local_end - token_count
            if local_start < 0:
                raise RuntimeError(
                    f"layer {layer_index} has {local_end} cached tokens; "
                    f"cannot archive the newest {token_count}"
                )
            layers.append(
                LayerKV(
                    _copy_off_cache(cache["k"][:, local_start:local_end], self.device),
                    _copy_off_cache(cache["v"][:, local_start:local_end], self.device),
                )
            )

        stored_viewmats = None
        if viewmats is not None:
            stored_viewmats = _copy_off_cache(viewmats, self.device)
        stored_Ks = None
        if Ks is not None:
            stored_Ks = _copy_off_cache(Ks, self.device)
        stored_spatial_shape = None
        if spatial_shape is not None:
            stored_spatial_shape = (int(spatial_shape[0]), int(spatial_shape[1]))
            if math.prod(stored_spatial_shape) != int(frame_tokens):
                raise ValueError("dyKV spatial shape must match frame_tokens")
        block = MemoryBlock(
            block_id=len(self.blocks),
            frame_start=int(frame_start),
            frame_count=int(frame_count),
            layers=tuple(layers),
            viewmats=stored_viewmats,
            Ks=stored_Ks,
            spatial_shape=stored_spatial_shape,
        )
        self.blocks.append(block)
        return block

    def evicted_candidates(
        self,
        *,
        current_frame: int,
        recent_frames: int,
        sink_frames: int,
    ) -> list[int]:
        """Return blocks no longer represented by sink or recent live cache."""

        local_start = max(int(sink_frames), int(current_frame) - int(recent_frames))
        return [
            index
            for index, block in enumerate(self.blocks)
            if block.frame_start >= int(sink_frames) and block.frame_end <= local_start
        ]

    def materialize(
        self,
        block_indices: Iterable[int],
        *,
        target_device: torch.device | str,
        chunk_frames: int,
        frame_tokens: int,
        keep_ratio: float,
        compression_mode: str = "yaw_fov",
        current_viewmats: torch.Tensor | None = None,
        current_Ks: torch.Tensor | None = None,
        preserve_input_order: bool = False,
    ) -> list[dict]:
        """Build one compressed retrieval payload per transformer layer."""

        selected = [self.blocks[int(index)] for index in block_indices]
        if not selected:
            return []
        if not preserve_input_order:
            selected.sort(key=lambda block: block.frame_start)
        if any(block.frame_count != int(chunk_frames) for block in selected):
            raise ValueError(
                "retrieval-time compression requires complete model-sized chunks"
            )
        layer_count = len(selected[0].layers)
        if any(len(block.layers) != layer_count for block in selected):
            raise RuntimeError("dyKV bank blocks have inconsistent layer counts")

        if compression_mode not in {"none", "fixed_novelty", "yaw_fov"}:
            raise ValueError(f"unsupported dyKV compression mode: {compression_mode}")

        planned: list[tuple[MemoryBlock, YawCropPlan | None]] = []
        for block in selected:
            plan = None
            if compression_mode == "yaw_fov":
                plan = build_yaw_crop_plan(
                    historical_viewmats=block.viewmats,
                    historical_Ks=block.Ks,
                    current_viewmats=current_viewmats,
                    current_Ks=current_Ks,
                    frame_count=block.frame_count,
                    frame_tokens=frame_tokens,
                    spatial_shape=block.spatial_shape,
                )
                if plan is not None and plan.token_indices.numel() == 0:
                    continue
            planned.append((block, plan))
        if not planned:
            return []

        payloads: list[dict] = []
        device = torch.device(target_device)
        for layer_index in range(layer_count):
            chunk_keys: list[torch.Tensor] = []
            chunk_values: list[torch.Tensor] = []
            chunk_token_lengths: list[int] = []
            modes: list[str] = []
            kept_tokens_per_frame: list[list[int]] = []
            kept_columns_per_frame: list[list[list[int]]] = []
            delta_yaw_degrees: list[list[float]] = []
            horizontal_fov_degrees: list[list[float]] = []
            for block, plan in planned:
                raw_k = block.layers[layer_index].k.to(device=device)
                raw_v = block.layers[layer_index].v.to(device=device)
                if compression_mode == "none":
                    compressed_k, compressed_v = raw_k, raw_v
                    mode = "none"
                    frame_kept = [int(frame_tokens)] * int(block.frame_count)
                    columns = []
                    deltas = []
                    fovs = []
                elif plan is not None:
                    indices = plan.token_indices.to(device=device)
                    compressed_k = raw_k.index_select(1, indices)
                    compressed_v = raw_v.index_select(1, indices)
                    mode = "yaw_fov"
                    frame_kept = list(plan.kept_tokens_per_frame)
                    columns = [list(values) for values in plan.kept_columns_per_frame]
                    deltas = list(plan.delta_yaw_degrees)
                    fovs = list(plan.horizontal_fov_degrees)
                else:
                    compressed_k, compressed_v = compress_retrieved_kv(
                        raw_k,
                        raw_v,
                        chunk_frames=chunk_frames,
                        frame_tokens=frame_tokens,
                        keep_ratio=keep_ratio,
                    )
                    mode = "fixed_novelty_fallback" if compression_mode == "yaw_fov" else "fixed_novelty"
                    keep_tokens = max(1, int(math.ceil(float(keep_ratio) * frame_tokens)))
                    frame_kept = [int(frame_tokens)] + [keep_tokens] * (int(block.frame_count) - 1)
                    columns = []
                    deltas = []
                    fovs = []
                chunk_keys.append(compressed_k)
                chunk_values.append(compressed_v)
                chunk_token_lengths.append(int(compressed_k.shape[1]))
                modes.append(mode)
                kept_tokens_per_frame.append(frame_kept)
                kept_columns_per_frame.append(columns)
                delta_yaw_degrees.append(deltas)
                horizontal_fov_degrees.append(fovs)

            compressed_k = torch.cat(chunk_keys, dim=1)
            compressed_v = torch.cat(chunk_values, dim=1)
            payloads.append(
                {
                    "k": compressed_k,
                    "v": compressed_v,
                    "src_frame_ids": [block.frame_start for block, _ in planned],
                    "chunk_frame_counts": [block.frame_count for block, _ in planned],
                    "chunk_token_lengths": chunk_token_lengths,
                    "compression_modes": modes,
                    "kept_tokens_per_frame": kept_tokens_per_frame,
                    "kept_columns_per_frame": kept_columns_per_frame,
                    "delta_yaw_degrees": delta_yaw_degrees,
                    "horizontal_fov_degrees": horizontal_fov_degrees,
                    "raw_tokens": sum(block.frame_count for block, _ in planned) * int(frame_tokens),
                    "kept_tokens": int(compressed_k.shape[1]),
                }
            )
        return payloads

    def summary(self) -> dict[str, int | str]:
        byte_count = 0
        for block in self.blocks:
            for layer in block.layers:
                byte_count += layer.k.numel() * layer.k.element_size()
                byte_count += layer.v.numel() * layer.v.element_size()
        return {
            "blocks": len(self.blocks),
            "bytes": byte_count,
            "device": str(self.device),
        }
