"""Bounded temporal RoPE helpers for dyKV's three attention regions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TriRegionSpec:
    """Frame counts for contiguous ``sink | retrieval | local`` regions.

    ``local_frames`` includes both the recent cached frames and current query.
    """

    sink_frames: int = 4
    memory_frames: int = 8
    local_frames: int = 8
    rope_train_frames: int = 20

    def query_start(self, query_frames: int) -> int:
        return self.rope_train_frames - int(query_frames)

    def local_start(self, query_frames: int) -> int:
        return self.rope_train_frames - self.local_frames

    def validate(self, query_frames: int) -> None:
        if query_frames <= 0:
            raise ValueError("tri-region query must contain complete frames")
        if query_frames > self.local_frames:
            raise ValueError("tri-region query exceeds the local RoPE region")
        retrieval_end = self.sink_frames + self.memory_frames
        local_start = self.local_start(query_frames)
        if retrieval_end > local_start:
            raise ValueError("tri-region retrieval and local RoPE ranges overlap")
        if retrieval_end < local_start:
            raise ValueError("tri-region retrieval and local RoPE ranges contain a gap")


def shift_roped_time(
    tensor: torch.Tensor,
    freqs: torch.Tensor,
    delta_frames: int,
) -> torch.Tensor:
    """Return already-roped Q/K shifted along the temporal RoPE axis.

    Spatial RoPE channels are unchanged. The input is never modified because a
    cached key can be reused by several denoising steps and future blocks.
    """

    delta_frames = int(delta_frames)
    if delta_frames == 0 or tensor.shape[1] == 0:
        return tensor
    if tensor.ndim != 4 or tensor.shape[-1] % 2:
        raise ValueError("RoPE tensor must have [batch, tokens, heads, even_dim]")

    complex_dim = tensor.shape[-1] // 2
    time_dim = complex_dim - 2 * (complex_dim // 3)
    height_dim = complex_dim // 3
    width_dim = complex_dim // 3
    time_freqs, _, _ = freqs.split([time_dim, height_dim, width_dim], dim=1)
    shift = abs(delta_frames)
    if shift >= time_freqs.shape[0]:
        raise ValueError(
            f"RoPE shift {shift} exceeds frequency table of {time_freqs.shape[0]} positions"
        )
    multiplier = time_freqs[shift]
    if delta_frames < 0:
        multiplier = torch.conj(multiplier)

    output = tensor.clone()
    time_channels = output[..., : 2 * time_dim]
    time_complex = torch.view_as_complex(
        time_channels.to(torch.float64).reshape(*time_channels.shape[:-1], time_dim, 2)
    )
    shifted = time_complex * multiplier.to(device=tensor.device, dtype=time_complex.dtype)
    shifted_real = torch.view_as_real(shifted).flatten(-2)
    output[..., : 2 * time_dim] = shifted_real.to(dtype=tensor.dtype)
    return output


def rebase_query(
    query: torch.Tensor,
    *,
    freqs: torch.Tensor,
    source_start_frame: int,
    query_frames: int,
    spec: TriRegionSpec,
) -> torch.Tensor:
    """Map the current query chunk to the end of the trained RoPE range."""

    spec.validate(query_frames)
    return shift_roped_time(
        query,
        freqs,
        spec.query_start(query_frames) - int(source_start_frame),
    )


def compose_tri_region(
    kv_cache: dict,
    *,
    local_end_index: int,
    frame_tokens: int,
    current_end_frame: int,
    query_frames: int,
    freqs: torch.Tensor,
    retrieval: dict | None,
    spec: TriRegionSpec,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose ``sink | retrieval | local+current`` with bounded RoPE.

    Retrieval payloads may be token-compressed. Legacy payloads shift each chunk
    as a unit. Packed payloads explicitly map each source-frame segment to one
    virtual retrieval slot.
    """

    spec.validate(query_frames)
    frame_tokens = int(frame_tokens)
    if frame_tokens <= 0 or local_end_index % frame_tokens:
        raise ValueError("tri-region cache must be frame-aligned")

    sink_tokens = spec.sink_frames * frame_tokens
    sink_end = min(sink_tokens, int(local_end_index))
    sink_k = kv_cache["k"][:, :sink_end]
    sink_v = kv_cache["v"][:, :sink_end]

    wanted_local_frames = spec.local_frames
    available_local_tokens = max(0, int(local_end_index) - sink_end)
    local_tokens = min(available_local_tokens, wanted_local_frames * frame_tokens)
    if local_tokens % frame_tokens:
        raise ValueError("tri-region local cache must contain complete frames")
    local_start_token = int(local_end_index) - local_tokens
    local_k = kv_cache["k"][:, local_start_token:local_end_index]
    local_v = kv_cache["v"][:, local_start_token:local_end_index]
    local_frame_count = local_tokens // frame_tokens
    local_source_start = int(current_end_frame) - local_frame_count
    past_local_frames = max(0, local_frame_count - int(query_frames))
    local_target_start = spec.query_start(query_frames) - past_local_frames
    local_k = shift_roped_time(
        local_k, freqs, local_target_start - local_source_start
    )

    region_k = [sink_k]
    region_v = [sink_v]
    if retrieval is not None and retrieval.get("k") is not None:
        retrieval_k = retrieval["k"].to(device=device, dtype=dtype)
        retrieval_v = retrieval["v"].to(device=device, dtype=dtype)
        if retrieval.get("frame_token_lengths") is not None:
            source_frames = [
                int(value) for value in retrieval.get("source_frame_ids", [])
            ]
            token_lengths = [
                int(value) for value in retrieval.get("frame_token_lengths", [])
            ]
            virtual_slots = [
                int(value) for value in retrieval.get("virtual_slot_ids", [])
            ]
            if not source_frames or not (
                len(source_frames) == len(token_lengths) == len(virtual_slots)
            ):
                raise ValueError("packed retrieval has invalid frame metadata")
            if any(length <= 0 for length in token_lengths):
                raise ValueError("packed retrieval frame lengths must be positive")
            if sum(token_lengths) != retrieval_k.shape[1] or (
                retrieval_v.shape[1] != retrieval_k.shape[1]
            ):
                raise ValueError("packed retrieval token lengths do not match K/V")
            if sum(token_lengths) > spec.memory_frames * frame_tokens:
                raise ValueError("packed retrieval exceeds the token budget")
            if frame_tokens % 4:
                raise ValueError("packed retrieval frame size must have four atoms")
            atom_tokens = frame_tokens // 4
            if any(
                length > frame_tokens or length % atom_tokens
                for length in token_lengths
            ):
                raise ValueError("packed retrieval frame length is not atom-aligned")
            slot_min = spec.sink_frames
            slot_max = spec.sink_frames + spec.memory_frames - 1
            if virtual_slots != sorted(virtual_slots) or any(
                slot < slot_min or slot > slot_max for slot in virtual_slots
            ):
                raise ValueError("packed retrieval virtual slots are invalid")
            slot_tokens: dict[int, int] = {}
            for slot, length in zip(virtual_slots, token_lengths):
                slot_tokens[slot] = slot_tokens.get(slot, 0) + length
            if any(tokens > frame_tokens for tokens in slot_tokens.values()):
                raise ValueError("packed retrieval virtual slot exceeds frame capacity")

            token_start = 0
            rebased_segments = []
            for source_frame, token_length, virtual_slot in zip(
                source_frames, token_lengths, virtual_slots
            ):
                token_end = token_start + token_length
                rebased_segments.append(
                    shift_roped_time(
                        retrieval_k[:, token_start:token_end],
                        freqs,
                        virtual_slot - source_frame,
                    )
                )
                token_start = token_end
            region_k.append(torch.cat(rebased_segments, dim=1))
            region_v.append(retrieval_v)
        else:
            source_starts = [int(value) for value in retrieval.get("src_frame_ids", [])]
            frame_counts = [int(value) for value in retrieval.get("chunk_frame_counts", [])]
            token_lengths = [int(value) for value in retrieval.get("chunk_token_lengths", [])]
            if not (
                len(source_starts) == len(frame_counts) == len(token_lengths)
                and sum(token_lengths) == retrieval_k.shape[1]
            ):
                raise ValueError("tri-region retrieval payload has invalid chunk metadata")
            if sum(frame_counts) > spec.memory_frames:
                raise ValueError("retrieval payload exceeds the tri-region memory budget")

            target_start = spec.sink_frames
            token_start = 0
            rebased_chunks = []
            for source_start, frame_count, token_length in zip(
                source_starts, frame_counts, token_lengths
            ):
                token_end = token_start + token_length
                rebased_chunks.append(
                    shift_roped_time(
                        retrieval_k[:, token_start:token_end],
                        freqs,
                        target_start - source_start,
                    )
                )
                target_start += frame_count
                token_start = token_end
            if target_start > spec.local_start(query_frames):
                raise ValueError("retrieval region overlaps the local RoPE region")
            if rebased_chunks:
                region_k.append(torch.cat(rebased_chunks, dim=1))
                region_v.append(retrieval_v)

    region_k.append(local_k)
    region_v.append(local_v)
    return torch.cat(region_k, dim=1), torch.cat(region_v, dim=1)
