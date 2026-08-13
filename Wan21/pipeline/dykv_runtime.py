"""Pipeline-side orchestration for the focused dyKV preset."""

from __future__ import annotations

import time

import torch

from .dykv_fov import deterministic_sphere_points, select_fov_blocks
from .dykv_memory import DyKVBank, DyKVConfig


class DyKVRuntime:
    """Own branch-specific banks and turn selected history into layer payloads."""

    def __init__(self, config: DyKVConfig, *, chunk_frames: int) -> None:
        self.config = config.validate(chunk_frames=chunk_frames) if config.enabled else config
        self.chunk_frames = int(chunk_frames)
        self.banks: dict[str, DyKVBank] = {}
        self.events: list[dict] = []
        self.probe_points = None
        if config.enabled:
            self.probe_points = deterministic_sphere_points(
                config.fov_samples, config.fov_radius, device=config.bank_device
            )

    def reset(self) -> None:
        self.banks.clear()
        self.events.clear()

    def bank(self, branch: str) -> DyKVBank:
        if branch not in self.banks:
            self.banks[branch] = DyKVBank(device=self.config.bank_device)
        return self.banks[branch]

    def archive(
        self,
        branch: str,
        caches,
        *,
        frame_start: int,
        frame_count: int,
        frame_tokens: int,
        viewmats,
        Ks=None,
        spatial_shape: tuple[int, int] | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        self.bank(branch).archive_clean_block(
            caches,
            frame_start=frame_start,
            frame_count=frame_count,
            frame_tokens=frame_tokens,
            viewmats=viewmats,
            Ks=Ks,
            spatial_shape=spatial_shape,
        )

    def retrieve(
        self,
        branch: str,
        *,
        current_frame: int,
        current_viewmats,
        frame_tokens: int,
        target_device: torch.device | str,
        current_Ks=None,
    ):
        if not self.config.enabled or current_frame < self.config.rope_train_frames:
            return None
        if current_viewmats is None:
            raise ValueError("dyKV FOV retrieval requires camera view matrices")

        bank = self.bank(branch)
        candidates = bank.evicted_candidates(
            current_frame=current_frame,
            recent_frames=self.config.local_frames - self.chunk_frames,
            sink_frames=self.config.sink_frames,
        )
        started = time.perf_counter()
        selected, ranked_candidates, distances = select_fov_blocks(
            bank,
            candidates,
            current_viewmats=current_viewmats,
            current_Ks=current_Ks,
            memory_frames=self.config.memory_frames,
            probe_points=self.probe_points,
            horizontal_degrees=self.config.fov_horizontal_degrees,
            vertical_degrees=self.config.fov_vertical_degrees,
            radius=self.config.fov_radius,
            fov_source=self.config.retrieval_fov_source,
        )
        payloads = bank.materialize(
            selected,
            target_device=target_device,
            chunk_frames=self.chunk_frames,
            frame_tokens=frame_tokens,
            keep_ratio=self.config.compression_keep_ratio,
            compression_mode=self.config.compression_mode,
            current_viewmats=current_viewmats,
            current_Ks=current_Ks,
            compression_fov_source=self.config.compression_fov_source,
            fixed_horizontal_degrees=self.config.fov_horizontal_degrees,
        ) if selected else None
        if not payloads:
            payloads = None
        diagnostics = payloads[0] if payloads else {}
        self.events.append(
            {
                "branch": branch,
                "current_frame": int(current_frame),
                "candidate_block_ids": [bank.blocks[index].block_id for index in candidates],
                "ranked_candidate_block_ids": [
                    bank.blocks[index].block_id for index in ranked_candidates
                ],
                "selected_block_ids": [bank.blocks[index].block_id for index in selected],
                "selected_frame_starts": [bank.blocks[index].frame_start for index in selected],
                "materialized_frame_starts": diagnostics.get("src_frame_ids", []),
                "distances": distances,
                "retrieved_tokens_per_layer": int(payloads[0]["k"].shape[1]) if payloads else 0,
                "raw_tokens_per_layer": int(diagnostics.get("raw_tokens", 0)),
                "compression_modes": diagnostics.get("compression_modes", []),
                "kept_tokens_per_frame": diagnostics.get("kept_tokens_per_frame", []),
                "kept_columns_per_frame": diagnostics.get("kept_columns_per_frame", []),
                "delta_yaw_degrees": diagnostics.get("delta_yaw_degrees", []),
                "horizontal_fov_degrees": diagnostics.get("horizontal_fov_degrees", []),
                "retrieval_fov_source": self.config.retrieval_fov_source,
                "compression_fov_source": self.config.compression_fov_source,
                "seconds": time.perf_counter() - started,
            }
        )
        return payloads

    def summary(self) -> dict:
        return {
            "enabled": self.config.enabled,
            "compression_mode": self.config.compression_mode,
            "retrieval_fov_source": self.config.retrieval_fov_source,
            "compression_fov_source": self.config.compression_fov_source,
            "branches": {branch: bank.summary() for branch, bank in self.banks.items()},
            "events": list(self.events),
        }
