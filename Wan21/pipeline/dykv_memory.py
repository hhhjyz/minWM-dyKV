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
    sink_frames: int = 4
    # The local region includes both recent cached frames and the current query.
    local_frames: int = 8
    rope_train_frames: int = 20
    compression_keep_ratio: float = 0.5
    bank_device: str = "cpu"
    fov_samples: int = 8192
    fov_radius: float = 8.0
    fov_horizontal_degrees: float = 60.0
    fov_vertical_degrees: float = 35.0

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
        if not 0.0 < self.compression_keep_ratio <= 1.0:
            raise ValueError("dyKV compression_keep_ratio must be in (0, 1]")
        occupied = self.sink_frames + self.memory_frames + self.local_frames
        if occupied != self.rope_train_frames:
            raise ValueError(
                "dyKV regions must exactly fill the trained RoPE range: "
                f"sink({self.sink_frames}) + memory({self.memory_frames}) + "
                f"local_including_current({self.local_frames}) != "
                f"rope_train_frames({self.rope_train_frames})"
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

    @property
    def frame_end(self) -> int:
        return self.frame_start + self.frame_count


def _copy_off_cache(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.detach().to(device=device).contiguous().clone()


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
        block = MemoryBlock(
            block_id=len(self.blocks),
            frame_start=int(frame_start),
            frame_count=int(frame_count),
            layers=tuple(layers),
            viewmats=stored_viewmats,
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
    ) -> list[dict]:
        """Build one compressed retrieval payload per transformer layer."""

        selected = [self.blocks[int(index)] for index in block_indices]
        if not selected:
            return []
        selected.sort(key=lambda block: block.frame_start)
        if any(block.frame_count != int(chunk_frames) for block in selected):
            raise ValueError(
                "retrieval-time compression requires complete model-sized chunks"
            )
        layer_count = len(selected[0].layers)
        if any(len(block.layers) != layer_count for block in selected):
            raise RuntimeError("dyKV bank blocks have inconsistent layer counts")

        payloads: list[dict] = []
        device = torch.device(target_device)
        for layer_index in range(layer_count):
            raw_k = torch.cat([block.layers[layer_index].k for block in selected], dim=1)
            raw_v = torch.cat([block.layers[layer_index].v for block in selected], dim=1)
            raw_k = raw_k.to(device=device)
            raw_v = raw_v.to(device=device)
            compressed_k, compressed_v = compress_retrieved_kv(
                raw_k,
                raw_v,
                chunk_frames=chunk_frames,
                frame_tokens=frame_tokens,
                keep_ratio=keep_ratio,
            )
            tokens_per_chunk = compressed_k.shape[1] // len(selected)
            payloads.append(
                {
                    "k": compressed_k,
                    "v": compressed_v,
                    "src_frame_ids": [block.frame_start for block in selected],
                    "chunk_frame_counts": [block.frame_count for block in selected],
                    "chunk_token_lengths": [tokens_per_chunk] * len(selected),
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
