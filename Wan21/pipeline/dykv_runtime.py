"""Pipeline-side orchestration for the focused dyKV preset."""

from __future__ import annotations

import time

import torch

from .dykv_fov import deterministic_sphere_points, select_fov_blocks
from .dykv_memory import DyKVBank, DyKVConfig
from .dykv_packing import (
    build_fixed_worldkv_retrieval_plan,
    build_packed_retrieval_plan,
    materialize_fixed_worldkv_retrieval,
    materialize_packed_retrieval,
)
from .dykv_predecessor import build_predecessor_retrieval_plan
from .dykv_worldkv import select_worldkv_pose_blocks


class DyKVRuntime:
    """Own branch-specific banks and turn selected history into layer payloads."""

    def __init__(self, config: DyKVConfig, *, chunk_frames: int) -> None:
        self.config = config.validate(chunk_frames=chunk_frames) if config.enabled else config
        self.chunk_frames = int(chunk_frames)
        self.banks: dict[str, DyKVBank] = {}
        self.events: list[dict] = []
        self.probe_points = None
        if config.enabled and config.retrieval_mode == "fov":
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
            raise ValueError("dyKV retrieval requires camera view matrices")
        if (
            self.config.retrieval_mode == "fov"
            or self.config.compression_mode == "yaw_fov"
        ) and current_Ks is None:
            raise ValueError(
                "dyKV geometry retrieval/compression requires camera intrinsics K"
            )

        bank = self.bank(branch)
        candidates = bank.evicted_candidates(
            current_frame=current_frame,
            recent_frames=self.config.local_frames - self.chunk_frames,
            sink_frames=self.config.sink_frames,
        )
        started = time.perf_counter()
        retrieval_diagnostics: dict[str, list[float]] = {}
        if self.config.retrieval_mode == "worldkv_pose":
            (
                selected,
                ranked_candidates,
                distances,
                retrieval_diagnostics,
            ) = select_worldkv_pose_blocks(
                bank,
                candidates,
                current_viewmats=current_viewmats,
                memory_frames=self.config.retrieval_frames,
            )
        else:
            selected, ranked_candidates, distances = select_fov_blocks(
                bank,
                candidates,
                current_viewmats=current_viewmats,
                current_Ks=current_Ks,
                memory_frames=self.config.retrieval_frames,
                probe_points=self.probe_points,
                radius=self.config.fov_radius,
            )
        packing_plan = None
        if self.config.packing_mode == "fixed_worldkv":
            packing_plan = build_fixed_worldkv_retrieval_plan(
                bank,
                selected,
                frame_tokens=frame_tokens,
                memory_frames=self.config.memory_frames,
                sink_frames=self.config.sink_frames,
                retrieval_frames=self.config.retrieval_frames,
                keep_ratio=self.config.compression_keep_ratio,
            )
            payloads = materialize_fixed_worldkv_retrieval(
                bank,
                packing_plan,
                target_device=target_device,
                frame_tokens=frame_tokens,
            )
        elif self.config.packing_mode.startswith("predecessor_"):
            packing_plan = build_predecessor_retrieval_plan(
                bank,
                ranked_candidates,
                distances,
                current_viewmats=current_viewmats,
                current_Ks=current_Ks,
                frame_tokens=frame_tokens,
                memory_frames=self.config.memory_frames,
                sink_frames=self.config.sink_frames,
                include_tail_latents=self.config.packing_mode in {
                    "predecessor_chunks_and_latents",
                    "predecessor_query_backfill",
                },
                query_backfill=(
                    self.config.packing_mode == "predecessor_query_backfill"
                ),
            )
            payloads = materialize_packed_retrieval(
                bank,
                packing_plan,
                target_device=target_device,
                frame_tokens=frame_tokens,
            )
            selected = list(packing_plan.selected_full_blocks)
        elif self.config.packing_mode != "none":
            packing_plan = build_packed_retrieval_plan(
                bank,
                ranked_candidates,
                distances,
                current_viewmats=current_viewmats,
                current_Ks=current_Ks,
                frame_tokens=frame_tokens,
                memory_frames=self.config.memory_frames,
                sink_frames=self.config.sink_frames,
                include_tail_latents=(
                    self.config.packing_mode == "whole_chunks_and_latents"
                ),
            )
            payloads = materialize_packed_retrieval(
                bank,
                packing_plan,
                target_device=target_device,
                frame_tokens=frame_tokens,
            )
            selected = list(packing_plan.selected_full_blocks)
        else:
            payloads = bank.materialize(
                selected,
                target_device=target_device,
                chunk_frames=self.chunk_frames,
                frame_tokens=frame_tokens,
                keep_ratio=self.config.compression_keep_ratio,
                compression_mode=self.config.compression_mode,
                current_viewmats=current_viewmats,
                current_Ks=current_Ks,
            ) if selected else None
        if not payloads:
            payloads = None
        diagnostics = payloads[0] if payloads else {}
        self.events.append(
            {
                "branch": branch,
                "current_frame": int(current_frame),
                "retrieval_mode": self.config.retrieval_mode,
                "candidate_block_ids": [bank.blocks[index].block_id for index in candidates],
                "ranked_candidate_block_ids": [
                    bank.blocks[index].block_id for index in ranked_candidates
                ],
                "selected_block_ids": [bank.blocks[index].block_id for index in selected],
                "selected_frame_starts": [bank.blocks[index].frame_start for index in selected],
                "materialized_frame_starts": diagnostics.get("src_frame_ids", []),
                "distances": distances,
                "worldkv_translation_squared": retrieval_diagnostics.get(
                    "translation_squared", []
                ),
                "worldkv_rotation_degrees": retrieval_diagnostics.get(
                    "rotation_degrees", []
                ),
                "worldkv_translation_normalized": retrieval_diagnostics.get(
                    "translation_normalized", []
                ),
                "worldkv_rotation_normalized": retrieval_diagnostics.get(
                    "rotation_normalized", []
                ),
                "retrieved_tokens_per_layer": int(payloads[0]["k"].shape[1]) if payloads else 0,
                "raw_tokens_per_layer": int(diagnostics.get("raw_tokens", 0)),
                "compression_modes": diagnostics.get("compression_modes", []),
                "kept_tokens_per_frame": diagnostics.get("kept_tokens_per_frame", []),
                "kept_columns_per_frame": diagnostics.get("kept_columns_per_frame", []),
                "delta_yaw_degrees": diagnostics.get("delta_yaw_degrees", []),
                "horizontal_fov_degrees": diagnostics.get("horizontal_fov_degrees", []),
                "predecessor_frame_starts": diagnostics.get(
                    "predecessor_frame_starts", []
                ),
                "incremental_yaw_degrees": diagnostics.get(
                    "incremental_yaw_degrees", []
                ),
                "incremental_fov_ratios": diagnostics.get(
                    "incremental_fov_ratios", []
                ),
                "query_backfill_tokens": diagnostics.get(
                    "query_backfill_tokens", []
                ),
                "packing_mode": self.config.packing_mode,
                "selected_tail_frame_ids": diagnostics.get(
                    "selected_tail_frame_ids", []
                ),
                "source_frame_ids": diagnostics.get("source_frame_ids", []),
                "virtual_slot_ids": diagnostics.get("virtual_slot_ids", []),
                "keep_tiers": diagnostics.get("keep_tiers", []),
                "packing_budget_atoms": diagnostics.get("packing_budget_atoms", 0),
                "packing_used_atoms": diagnostics.get("packing_used_atoms", 0),
                "packing_used_virtual_slots": diagnostics.get(
                    "packing_used_virtual_slots", 0
                ),
                "fixed_keep_ratio": diagnostics.get("fixed_keep_ratio"),
                "fixed_retrieval_frames": diagnostics.get(
                    "fixed_retrieval_frames"
                ),
                "packing_candidate_block_ids": [
                    bank.blocks[index].block_id
                    for index in (
                        getattr(packing_plan, "candidate_block_indices", ())
                        if packing_plan else ()
                    )
                ],
                "seconds": time.perf_counter() - started,
            }
        )
        return payloads

    def summary(self) -> dict:
        return {
            "enabled": self.config.enabled,
            "sink_mode": "fixed",
            "sink_frames": self.config.sink_frames,
            "compression_mode": self.config.compression_mode,
            "retrieval_mode": self.config.retrieval_mode,
            "packing_mode": self.config.packing_mode,
            "retrieval_frames": self.config.retrieval_frames,
            "compression_keep_ratio": self.config.compression_keep_ratio,
            "branches": {branch: bank.summary() for branch, bank in self.banks.items()},
            "events": list(self.events),
        }
