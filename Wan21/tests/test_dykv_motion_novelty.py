import importlib.util
import math
import pathlib
import sys
import types
import unittest

import torch


WAN21_ROOT = pathlib.Path(__file__).resolve().parents[1]
PIPELINE_ROOT = WAN21_ROOT / "pipeline"
package = types.ModuleType("dykv_motion_novelty_test_pipeline")
package.__path__ = [str(PIPELINE_ROOT)]
sys.modules[package.__name__] = package


def _load(name):
    full_name = f"{package.__name__}.{name}"
    spec = importlib.util.spec_from_file_location(full_name, PIPELINE_ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


fov = _load("dykv_fov")
memory = _load("dykv_memory")
_load("dykv_projected_overlap")
motion = _load("dykv_motion_novelty")


def _yaw_w2c(degrees):
    angle = math.radians(float(degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    c2w = torch.eye(4)
    c2w[:3, :3] = torch.tensor(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
    )
    return torch.linalg.inv(c2w)


def _intrinsics(frames=4, horizontal=60.0):
    fx = 0.5 / math.tan(math.radians(horizontal) / 2.0)
    K = torch.tensor([[fx, 0.0, 0.5], [0.0, fx, 0.5], [0.0, 0.0, 1.0]])
    return torch.stack([K] * frames).unsqueeze(0)


def _spatial_shape(frame_tokens):
    for height in range(int(math.sqrt(frame_tokens)), 0, -1):
        if frame_tokens % height == 0:
            return height, frame_tokens // height
    raise AssertionError("frame token count has no spatial factorization")


def _cache(block_value, frame_tokens):
    anchor = torch.tensor([[1.0, 0.0]]).repeat(frame_tokens, 1)
    score = torch.linspace(-1.0, 1.0, frame_tokens)
    novelty = torch.stack((score, torch.ones_like(score)), dim=-1)
    frames = [anchor, novelty, novelty.roll(1, 0), novelty.roll(2, 0)]
    key = torch.cat(frames).reshape(1, 4 * frame_tokens, 1, 2)
    key = key + float(block_value) * 0.001
    value = torch.arange(4 * frame_tokens, dtype=torch.float32)
    value = value.reshape(1, 4 * frame_tokens, 1, 1) + float(block_value) * 1000
    return {"k": key, "v": value, "local_end_index": torch.tensor(4 * frame_tokens)}


def _bank(yaws_per_chunk, frame_tokens=20):
    bank = memory.DyKVBank(device="cpu")
    for block_index, yaws in enumerate(yaws_per_chunk):
        poses = torch.stack([_yaw_w2c(value) for value in yaws]).unsqueeze(0)
        bank.archive_clean_block(
            [_cache(block_index, frame_tokens)],
            frame_start=block_index * 4,
            frame_count=4,
            frame_tokens=frame_tokens,
            viewmats=poses,
            Ks=_intrinsics(),
            spatial_shape=_spatial_shape(frame_tokens),
        )
    return bank


class DyKVMotionNoveltyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.points = fov.deterministic_sphere_points(12000, 8.0)

    def test_identical_chunk_keeps_anchor_without_a_minimum_novelty_floor(self):
        bank = _bank([(0, 0, 0, 0)], frame_tokens=20)
        chunk = motion.build_motion_chunk_plan(
            bank.blocks[0],
            block_index=0,
            retrieval_distance=0.1,
            scene_scale=8.0,
            frame_tokens=20,
        )

        self.assertEqual([frame.base_token_count for frame in chunk.frames], [20, 0, 0, 0])
        self.assertEqual([frame.keep_ratio for frame in chunk.frames[1:]], [0.0, 0.0, 0.0])
        for frame in chunk.frames[1:]:
            combined = torch.cat(
                (frame.base_indices, frame.omitted_indices_in_novelty_order)
            ).sort().values
            self.assertTrue(torch.equal(combined, torch.arange(20)))

    def test_exact_overlap_is_not_quantized(self):
        keep_ratio, keep_tokens = motion.motion_keep_ratio_and_token_count(
            0.1234,
            1560,
        )
        self.assertAlmostEqual(keep_ratio, 0.8766)
        self.assertEqual(keep_tokens, math.ceil(0.8766 * 1560))
        self.assertEqual(motion.motion_keep_ratio_and_token_count(1.0, 1560), (0.0, 0))
        self.assertEqual(motion.motion_keep_ratio_and_token_count(0.0, 1560), (1.0, 1560))
        with self.assertRaisesRegex(ValueError, "not finite"):
            motion.motion_keep_ratio_and_token_count(float("nan"), 1560)

    def test_projected_keep_ratio_is_continuous_and_controls_only_token_count(self):
        bank = _bank([(0, 17, 29, 41)], frame_tokens=100)
        chunk = motion.build_motion_chunk_plan(
            bank.blocks[0],
            block_index=0,
            retrieval_distance=0.2,
            scene_scale=8.0,
            frame_tokens=100,
        )

        quantized = {0.0, 0.25, 0.5, 0.75, 1.0}
        self.assertTrue(
            any(
                all(abs(frame.keep_ratio - value) > 1e-4 for value in quantized)
                for frame in chunk.frames[1:]
            )
        )
        for frame in chunk.frames[1:]:
            self.assertEqual(
                frame.base_token_count,
                math.ceil(frame.keep_ratio * 100),
            )
            self.assertTrue(torch.equal(frame.base_indices, frame.base_indices.sort().values))

    def test_sphere_ablation_uses_legacy_overlap_without_projected_diagnostics(self):
        bank = _bank([(0, 17, 29, 41)], frame_tokens=100)
        chunk = motion.build_motion_chunk_plan(
            bank.blocks[0],
            block_index=0,
            retrieval_distance=0.2,
            motion_geometry_mode="sphere_fov",
            probe_points=self.points,
            radius=8.0,
            scene_scale=8.0,
            frame_tokens=100,
        )

        for frame in chunk.frames:
            self.assertIsNone(frame.projected_overlap_ratio)
            self.assertEqual(frame.projection_depths, ())
        self.assertTrue(any(frame.fov_overlap > 0.0 for frame in chunk.frames[1:]))

    def test_flat_plan_can_select_more_chunks_than_slot_capped_plan(self):
        bank = _bank([(0, 32, 32, 32)] * 5, frame_tokens=100)
        ranked = list(range(5))
        distances = [0.01 * index for index in ranked]
        flat = motion.build_motion_retrieval_plan(
            bank,
            ranked,
            distances,
            scene_scale=8.0,
            frame_tokens=100,
            memory_frames=8,
            sink_frames=4,
            slot_capped=False,
        )
        capped = motion.build_motion_retrieval_plan(
            bank,
            ranked,
            distances,
            scene_scale=8.0,
            frame_tokens=100,
            memory_frames=8,
            sink_frames=4,
            slot_capped=True,
        )

        self.assertGreater(len(flat.selected_block_indices), len(capped.selected_block_indices))
        self.assertLessEqual(flat.base_used_tokens, 800)
        self.assertLessEqual(capped.base_used_tokens, 800)
        self.assertEqual(flat.retrieval_layout, "flat_source_ordered")
        self.assertEqual(capped.retrieval_layout, "source_ordered")
        self.assertGreater(max(flat.slot_token_loads), 100)
        self.assertTrue(all(load <= 100 for load in capped.slot_token_loads))

    def test_flat_materialization_is_source_ordered_and_does_not_mutate_bank(self):
        bank = _bank([(0, 17, 29, 41)] * 4, frame_tokens=40)
        original = bank.blocks[0].layers[0].k.clone()
        plan = motion.build_motion_retrieval_plan(
            bank,
            list(range(4)),
            [0.1, 0.2, 0.3, 0.4],
            scene_scale=8.0,
            frame_tokens=40,
            memory_frames=8,
            sink_frames=4,
            slot_capped=False,
        )
        payload = motion.materialize_motion_retrieval(
            bank,
            plan,
            target_device="cpu",
            frame_tokens=40,
        )[0]

        self.assertEqual(payload["source_frame_ids"], sorted(payload["source_frame_ids"]))
        self.assertEqual(payload["retrieval_layout"], "flat_source_ordered")
        self.assertEqual(payload["k"].shape[1], plan.base_used_tokens)
        self.assertEqual(sum(payload["frame_token_lengths"]), plan.base_used_tokens)
        self.assertEqual(payload["unique_backfill_tokens_total"], 0)
        self.assertEqual(payload["duplicate_tokens_total"], 0)
        self.assertTrue(bank.blocks[0].layers[0].k.equal(original))

    def test_capped_plan_keeps_zero_length_frames_as_metadata_only(self):
        bank = _bank([(0, 0, 0, 0)] * 3, frame_tokens=20)
        plan = motion.build_motion_retrieval_plan(
            bank,
            (0, 1, 2),
            (0.1, 0.2, 0.3),
            scene_scale=8.0,
            frame_tokens=20,
            memory_frames=8,
            sink_frames=4,
            slot_capped=True,
        )
        payload = motion.materialize_motion_retrieval(
            bank,
            plan,
            target_device="cpu",
            frame_tokens=20,
        )[0]

        self.assertEqual(len(plan.frames), 12)
        self.assertEqual([frame.base_token_count for frame in plan.frames].count(0), 9)
        self.assertTrue(all(frame.virtual_slot_id >= 4 for frame in plan.frames))
        self.assertEqual(len(payload["source_frame_ids"]), 3)
        self.assertEqual(payload["k"].shape[1], 60)

    def test_invalid_motion_geometry_is_reported_without_fallback(self):
        bank = _bank([(0, 10, 20, 30), (0, 10, 20, 30)], frame_tokens=20)
        invalid = bank.blocks[1]
        bank.blocks[1] = memory.MemoryBlock(
            block_id=invalid.block_id,
            frame_start=invalid.frame_start,
            frame_count=invalid.frame_count,
            layers=invalid.layers,
            viewmats=invalid.viewmats,
            Ks=None,
            spatial_shape=invalid.spatial_shape,
        )
        plan = motion.build_motion_retrieval_plan(
            bank,
            (0, 1),
            (0.1, 0.2),
            scene_scale=8.0,
            frame_tokens=20,
            memory_frames=8,
            sink_frames=4,
            slot_capped=False,
        )
        self.assertEqual(plan.selected_block_indices, (0,))
        self.assertEqual(plan.geometry_invalid_block_indices, (1,))

    def test_backfill_and_duplicate_share_reference_shape_and_slots(self):
        bank = _bank([(0, 17, 29, 41)] * 5, frame_tokens=40)
        common = dict(
            bank=bank,
            ranked_block_indices=list(range(5)),
            ranked_distances=[0.01 * index for index in range(5)],
            scene_scale=8.0,
            frame_tokens=40,
            memory_frames=8,
            sink_frames=4,
            slot_capped=False,
        )
        unfilled = motion.build_motion_retrieval_plan(
            **common,
            fill_mode="unfilled",
        )
        backfill = motion.build_motion_retrieval_plan(
            **common,
            fill_mode="backfill",
        )
        duplicate = motion.build_motion_retrieval_plan(
            **common,
            fill_mode="duplicate",
        )
        backfill_payload = motion.materialize_motion_retrieval(
            bank, backfill, target_device="cpu", frame_tokens=40
        )[0]
        duplicate_payload = motion.materialize_motion_retrieval(
            bank, duplicate, target_device="cpu", frame_tokens=40
        )[0]

        self.assertEqual(
            unfilled.selected_block_indices,
            backfill.selected_block_indices,
        )
        self.assertEqual(
            unfilled.selected_block_indices,
            duplicate.selected_block_indices,
        )
        self.assertEqual(unfilled.base_used_tokens, backfill.base_used_tokens)
        self.assertEqual(unfilled.base_used_tokens, duplicate.base_used_tokens)
        self.assertEqual(backfill.slot_token_loads, duplicate.slot_token_loads)
        self.assertEqual(
            backfill_payload["final_tokens_total"],
            backfill.fill_target_tokens,
        )
        self.assertEqual(
            duplicate_payload["final_tokens_total"],
            backfill_payload["final_tokens_total"],
        )
        self.assertGreater(backfill_payload["unique_backfill_tokens_total"], 0)
        self.assertEqual(backfill_payload["duplicate_tokens_total"], 0)
        self.assertEqual(backfill_payload["max_source_token_multiplicity"], 1)
        self.assertEqual(duplicate_payload["unique_backfill_tokens_total"], 0)
        self.assertGreater(duplicate_payload["duplicate_tokens_total"], 0)
        self.assertGreater(duplicate_payload["max_source_token_multiplicity"], 1)
        self.assertEqual(
            duplicate.duplicate_source_block_indices,
            (duplicate.selected_block_indices[0],),
        )
        self.assertEqual(
            backfill_payload["source_frame_ids"],
            sorted(backfill_payload["source_frame_ids"]),
        )
        self.assertEqual(
            duplicate_payload["source_frame_ids"],
            sorted(duplicate_payload["source_frame_ids"]),
        )
        unfilled_slots = {
            frame.source_frame_id: frame.virtual_slot_id for frame in unfilled.frames
        }
        backfill_slots = {
            frame.source_frame_id: frame.virtual_slot_id for frame in backfill.frames
        }
        self.assertEqual(unfilled_slots, backfill_slots)

        repeat_chunk = duplicate.chunks[0]
        repeat_pool = {
            frame.frame_offset: set(frame.base_indices.tolist())
            for frame in repeat_chunk.frames
        }
        duplicate_segments = [
            segment
            for segment in duplicate.segments
            if segment.selection_kind == "duplicate"
        ]
        self.assertTrue(duplicate_segments)
        for segment in duplicate_segments:
            self.assertEqual(segment.block_index, duplicate.selected_block_indices[0])
            self.assertTrue(
                set(segment.token_indices.tolist()).issubset(
                    repeat_pool[segment.frame_offset]
                )
            )
        expected_quotas = motion._proportional_allocation(
            [frame.base_token_count for frame in repeat_chunk.frames],
            duplicate_payload["duplicate_tokens_total"],
        )
        duplicate_by_source = dict(
            zip(
                [frame.source_frame_id for frame in sorted(
                    duplicate.frames, key=lambda item: item.source_frame_id
                )],
                duplicate.duplicate_tokens_per_frame,
            )
        )
        self.assertEqual(
            [duplicate_by_source[frame.source_frame_id] for frame in repeat_chunk.frames],
            expected_quotas,
        )


if __name__ == "__main__":
    unittest.main()
