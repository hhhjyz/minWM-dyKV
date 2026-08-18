import importlib.util
import math
import pathlib
import sys
import types
import unittest

import torch


WAN21_ROOT = pathlib.Path(__file__).resolve().parents[1]
PIPELINE_ROOT = WAN21_ROOT / "pipeline"
package = types.ModuleType("dykv_test_pipeline")
package.__path__ = [str(PIPELINE_ROOT)]
sys.modules[package.__name__] = package


def _load(name):
    full_name = f"{package.__name__}.{name}"
    spec = importlib.util.spec_from_file_location(full_name, PIPELINE_ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


_load("dykv_fov")
memory = _load("dykv_memory")
_load("dykv_packing")
_load("dykv_projected_overlap")
_load("dykv_motion_novelty")
_load("dykv_worldkv")
runtime_module = _load("dykv_runtime")


def _w2c(x):
    c2w = torch.eye(4)
    c2w[0, 3] = float(x)
    return torch.linalg.inv(c2w)


def _yaw_w2c(degrees):
    angle = math.radians(float(degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    c2w = torch.eye(4)
    c2w[:3, :3] = torch.tensor(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
    )
    return torch.linalg.inv(c2w)


def _intrinsics(frames=4):
    fx = 0.5 / math.tan(math.radians(30.0))
    K = torch.tensor([[fx, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    return torch.stack([K] * frames).unsqueeze(0)


def _cache(value, tokens=4):
    key = torch.full((1, tokens, 1, 2), float(value))
    return {
        "k": key,
        "v": key + 100,
        "local_end_index": torch.tensor(tokens),
    }


class DyKVRuntimeTest(unittest.TestCase):
    def test_worldkv_pose_runtime_uses_same_bank_budget_and_logs_components(self):
        config = memory.DyKVConfig(
            enabled=True,
            retrieval_mode="worldkv_pose",
            compression_mode="none",
        )
        runtime = runtime_module.DyKVRuntime(config, chunk_frames=4)
        self.assertIsNone(runtime.probe_points)
        for start, x in zip((0, 4, 8, 12, 16), (0, 1, 8, 2, 3)):
            poses = torch.stack([_w2c(x)] * 4).unsqueeze(0)
            runtime.archive(
                "main",
                [_cache(start)],
                frame_start=start,
                frame_count=4,
                frame_tokens=1,
                viewmats=poses,
                Ks=_intrinsics(),
            )

        payloads = runtime.retrieve(
            "main",
            current_frame=20,
            current_viewmats=torch.stack([_w2c(1.1)] * 4).unsqueeze(0),
            frame_tokens=1,
            target_device="cpu",
        )

        self.assertEqual(payloads[0]["src_frame_ids"], [4, 12])
        self.assertEqual(payloads[0]["k"].shape[1], 8)
        event = runtime.summary()["events"][0]
        self.assertEqual(event["retrieval_mode"], "worldkv_pose")
        self.assertEqual(event["selected_frame_starts"], [4, 12])
        self.assertEqual(len(event["worldkv_translation_squared"]), 3)
        self.assertEqual(len(event["worldkv_rotation_degrees"]), 3)
        self.assertEqual(runtime.summary()["retrieval_mode"], "worldkv_pose")

    def test_archive_select_compress_pipeline(self):
        config = memory.DyKVConfig(
            enabled=True,
            memory_frames=8,
            fov_samples=2048,
            compression_keep_ratio=0.5,
        )
        runtime = runtime_module.DyKVRuntime(config, chunk_frames=4)
        for start, x in zip((0, 4, 8, 12, 16), (0, 1, 8, 2, 3)):
            poses = torch.stack([_w2c(x)] * 4).unsqueeze(0)
            runtime.archive(
                "main",
                [_cache(start)],
                frame_start=start,
                frame_count=4,
                frame_tokens=1,
                viewmats=poses,
                Ks=_intrinsics(),
            )

        current = torch.stack([_w2c(1.1)] * 4).unsqueeze(0)
        payloads = runtime.retrieve(
            "main",
            current_frame=20,
            current_viewmats=current,
            current_Ks=_intrinsics(),
            frame_tokens=1,
            target_device="cpu",
        )

        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["src_frame_ids"], [4, 12])
        self.assertEqual(payloads[0]["k"].shape[1], 8)
        event = runtime.summary()["events"][0]
        self.assertEqual(event["selected_frame_starts"], [4, 12])
        self.assertEqual(event["ranked_candidate_block_ids"][:2], [1, 3])
        self.assertEqual(event["retrieved_tokens_per_layer"], 8)

    def test_relevance_order_places_best_chunk_nearest_query(self):
        config = memory.DyKVConfig(
            enabled=True,
            memory_frames=8,
            fov_samples=2048,
            compression_mode="none",
            retrieval_order="relevance_near_query",
        )
        runtime = runtime_module.DyKVRuntime(config, chunk_frames=4)
        for start, x in zip((0, 4, 8, 12, 16), (0, 1, 8, 2, 3)):
            poses = torch.stack([_w2c(x)] * 4).unsqueeze(0)
            runtime.archive(
                "main",
                [_cache(start)],
                frame_start=start,
                frame_count=4,
                frame_tokens=1,
                viewmats=poses,
                Ks=_intrinsics(),
            )

        payloads = runtime.retrieve(
            "main",
            current_frame=20,
            current_viewmats=torch.stack([_w2c(1.1)] * 4).unsqueeze(0),
            current_Ks=_intrinsics(),
            frame_tokens=1,
            target_device="cpu",
        )

        # The same chunks are selected as source-order retrieval. Chunk 4 is
        # the best match, so it is materialized last at virtual frames 8..11.
        self.assertEqual(payloads[0]["src_frame_ids"], [12, 4])
        event = runtime.summary()["events"][0]
        self.assertEqual(event["selected_frame_starts"], [4, 12])
        self.assertEqual(event["materialized_frame_starts"], [12, 4])
        self.assertEqual(event["materialized_virtual_chunk_starts"], [4, 8])
        self.assertEqual(event["retrieval_order"], "relevance_near_query")

    def test_runtime_reports_yaw_crop_diagnostics(self):
        config = memory.DyKVConfig(enabled=True, fov_samples=2048)
        runtime = runtime_module.DyKVRuntime(config, chunk_frames=4)
        historical_poses = torch.stack([_yaw_w2c(0.0)] * 4).unsqueeze(0)
        for start in (0, 4, 8, 12, 16):
            runtime.archive(
                "main",
                [_cache(start, tokens=48)],
                frame_start=start,
                frame_count=4,
                frame_tokens=12,
                viewmats=historical_poses,
                Ks=_intrinsics(),
                spatial_shape=(2, 6),
            )

        current = torch.stack([_yaw_w2c(30.0)] * 4).unsqueeze(0)
        payloads = runtime.retrieve(
            "main",
            current_frame=20,
            current_viewmats=current,
            current_Ks=_intrinsics(),
            frame_tokens=12,
            target_device="cpu",
        )

        self.assertEqual(payloads[0]["compression_modes"], ["yaw_fov", "yaw_fov"])
        self.assertEqual(payloads[0]["kept_tokens"], 48)
        event = runtime.summary()["events"][0]
        self.assertEqual(event["raw_tokens_per_layer"], 96)
        self.assertEqual(event["retrieved_tokens_per_layer"], 48)
        self.assertEqual(event["kept_tokens_per_frame"], [[6] * 4, [6] * 4])

    def test_packed_runtime_can_cover_more_than_eight_source_frames(self):
        config = memory.DyKVConfig(
            enabled=True,
            fov_samples=2048,
            packing_mode="whole_chunks",
        )
        runtime = runtime_module.DyKVRuntime(config, chunk_frames=4)
        historical = torch.stack([_yaw_w2c(0.0)] * 4).unsqueeze(0)
        for start in range(0, 28, 4):
            runtime.archive(
                "main",
                [_cache(start, tokens=64)],
                frame_start=start,
                frame_count=4,
                frame_tokens=16,
                viewmats=historical,
                Ks=_intrinsics(),
                spatial_shape=(2, 8),
            )

        current = torch.stack([_yaw_w2c(45.0)] * 4).unsqueeze(0)
        payloads = runtime.retrieve(
            "main",
            current_frame=28,
            current_viewmats=current,
            current_Ks=_intrinsics(),
            frame_tokens=16,
            target_device="cpu",
        )

        self.assertIsNotNone(payloads)
        payload = payloads[0]
        self.assertLessEqual(payload["k"].shape[1], 8 * 16)
        self.assertGreater(len(payload["source_frame_ids"]), 8)
        self.assertTrue(all(4 <= slot <= 11 for slot in payload["virtual_slot_ids"]))
        event = runtime.summary()["events"][0]
        self.assertEqual(event["packing_mode"], "whole_chunks")
        self.assertEqual(event["packing_used_atoms"], payload["packing_used_atoms"])
        self.assertEqual(event["retrieval_layout"], "source_ordered")

    def test_fixed_worldkv_runtime_retrieves_sixteen_frames_in_eight_frame_budget(self):
        config = memory.DyKVConfig(
            enabled=True,
            fov_samples=2048,
            retrieval_frames=16,
            compression_keep_ratio=1.0 / 3.0,
            compression_mode="fixed_novelty",
            packing_mode="fixed_worldkv",
        )
        runtime = runtime_module.DyKVRuntime(config, chunk_frames=4)
        poses = torch.stack([_yaw_w2c(0.0)] * 4).unsqueeze(0)
        for start in range(0, 36, 4):
            runtime.archive(
                "main",
                [_cache(start, tokens=48)],
                frame_start=start,
                frame_count=4,
                frame_tokens=12,
                viewmats=poses,
                Ks=_intrinsics(),
            )
        payload = runtime.retrieve(
            "main",
            current_frame=36,
            current_viewmats=poses,
            current_Ks=_intrinsics(),
            frame_tokens=12,
            target_device="cpu",
        )[0]

        self.assertEqual(len(payload["source_frame_ids"]), 16)
        self.assertEqual(payload["k"].shape[1], 8 * 12)
        self.assertEqual(payload["packing_used_virtual_slots"], 8)
        event = runtime.summary()["events"][0]
        self.assertEqual(event["fixed_retrieval_frames"], 16)
        self.assertAlmostEqual(event["fixed_keep_ratio"], 1.0 / 3.0)
        self.assertEqual(event["retrieval_layout"], "source_ordered")

    def test_fixed_worldkv_runtime_can_restore_slot_order(self):
        config = memory.DyKVConfig(
            enabled=True,
            fov_samples=2048,
            retrieval_frames=16,
            compression_keep_ratio=1.0 / 3.0,
            compression_mode="fixed_novelty",
            packing_mode="fixed_worldkv",
            retrieval_layout="slot_packed",
        )
        runtime = runtime_module.DyKVRuntime(config, chunk_frames=4)
        poses = torch.stack([_yaw_w2c(0.0)] * 4).unsqueeze(0)
        for start in range(0, 36, 4):
            runtime.archive(
                "main",
                [_cache(start, tokens=48)],
                frame_start=start,
                frame_count=4,
                frame_tokens=12,
                viewmats=poses,
                Ks=_intrinsics(),
            )
        payload = runtime.retrieve(
            "main",
            current_frame=36,
            current_viewmats=poses,
            current_Ks=_intrinsics(),
            frame_tokens=12,
            target_device="cpu",
        )[0]

        self.assertEqual(payload["retrieval_layout"], "slot_packed")
        self.assertEqual(payload["virtual_slot_ids"], sorted(payload["virtual_slot_ids"]))
        self.assertEqual(payload["k"].shape[1], 8 * 12)
        self.assertEqual(
            runtime.summary()["events"][0]["retrieval_layout"], "slot_packed"
        )

    def test_motion_novelty_runtime_emits_flat_source_ordered_diagnostics(self):
        config = memory.DyKVConfig(
            enabled=True,
            fov_samples=2048,
            retrieval_frames=16,
            compression_mode="motion_novelty",
            packing_mode="motion_novelty_flat",
            retrieval_layout="flat_source_ordered",
        )
        runtime = runtime_module.DyKVRuntime(config, chunk_frames=4)
        for start in range(0, 36, 4):
            poses = torch.stack(
                [_yaw_w2c(value) for value in (0.0, 10.0, 20.0, 30.0)]
            ).unsqueeze(0)
            runtime.archive(
                "main",
                [_cache(start, tokens=48)],
                frame_start=start,
                frame_count=4,
                frame_tokens=12,
                viewmats=poses,
                Ks=_intrinsics(),
                spatial_shape=(3, 4),
            )
        current = torch.stack([_yaw_w2c(0.0)] * 4).unsqueeze(0)
        payload = runtime.retrieve(
            "main",
            current_frame=36,
            current_viewmats=current,
            current_Ks=_intrinsics(),
            frame_tokens=12,
            target_device="cpu",
        )[0]
        event = runtime.summary()["events"][0]

        self.assertEqual(payload["retrieval_layout"], "flat_source_ordered")
        self.assertEqual(payload["source_frame_ids"], sorted(payload["source_frame_ids"]))
        self.assertEqual(event["retrieval_layout"], "flat_source_ordered")
        self.assertTrue(event["segments_source_ordered"])
        self.assertEqual(event["base_tokens_total"], payload["k"].shape[1])
        self.assertEqual(event["unique_backfill_tokens_total"], 0)
        self.assertEqual(event["motion_geometry_mode"], "projected_multidepth")
        self.assertEqual(len(event["projection_depths"]), 4)
        self.assertTrue(event["projected_overlap_ratios"])

    def test_motion_novelty_fill_modes_share_selection_and_reference_shape(self):
        outputs = {}
        for packing_mode in (
            "motion_novelty_backfill",
            "motion_novelty_duplicate",
        ):
            with self.subTest(packing_mode=packing_mode):
                config = memory.DyKVConfig(
                    enabled=True,
                    fov_samples=2048,
                    retrieval_frames=16,
                    compression_mode="motion_novelty",
                    packing_mode=packing_mode,
                    retrieval_layout="flat_source_ordered",
                )
                runtime = runtime_module.DyKVRuntime(config, chunk_frames=4)
                for start in range(0, 36, 4):
                    poses = torch.stack(
                        [_yaw_w2c(0.0)] * 4
                    ).unsqueeze(0)
                    runtime.archive(
                        "main",
                        [_cache(start, tokens=48)],
                        frame_start=start,
                        frame_count=4,
                        frame_tokens=12,
                        viewmats=poses,
                        Ks=_intrinsics(),
                        spatial_shape=(3, 4),
                    )
                current = torch.stack([_yaw_w2c(0.0)] * 4).unsqueeze(0)
                payload = runtime.retrieve(
                    "main",
                    current_frame=36,
                    current_viewmats=current,
                    current_Ks=_intrinsics(),
                    frame_tokens=12,
                    target_device="cpu",
                )[0]
                event = runtime.summary()["events"][0]
                self.assertEqual(payload["retrieval_layout"], "flat_source_ordered")
                self.assertEqual(
                    event["final_tokens_total"], event["fill_target_tokens"]
                )
                outputs[packing_mode] = event

        backfill = outputs["motion_novelty_backfill"]
        duplicate = outputs["motion_novelty_duplicate"]
        self.assertEqual(backfill["selected_block_ids"], duplicate["selected_block_ids"])
        self.assertEqual(backfill["base_tokens_total"], duplicate["base_tokens_total"])
        self.assertEqual(backfill["slot_token_loads"], duplicate["slot_token_loads"])
        self.assertGreater(backfill["unique_backfill_tokens_total"], 0)
        self.assertEqual(backfill["duplicate_tokens_total"], 0)
        self.assertEqual(duplicate["unique_backfill_tokens_total"], 0)
        self.assertGreater(duplicate["duplicate_tokens_total"], 0)

if __name__ == "__main__":
    unittest.main()
