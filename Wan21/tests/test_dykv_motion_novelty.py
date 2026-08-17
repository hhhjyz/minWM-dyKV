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
            probe_points=self.points,
            radius=8.0,
            frame_tokens=20,
        )

        self.assertEqual([frame.base_token_count for frame in chunk.frames], [20, 0, 0, 0])
        self.assertEqual([frame.keep_ratio for frame in chunk.frames[1:]], [0.0, 0.0, 0.0])
        for frame in chunk.frames[1:]:
            combined = torch.cat(
                (frame.base_indices, frame.omitted_indices_in_novelty_order)
            ).sort().values
            self.assertTrue(torch.equal(combined, torch.arange(20)))

    def test_fov_keep_ratio_is_continuous_and_controls_only_token_count(self):
        bank = _bank([(0, 17, 29, 41)], frame_tokens=100)
        chunk = motion.build_motion_chunk_plan(
            bank.blocks[0],
            block_index=0,
            retrieval_distance=0.2,
            probe_points=self.points,
            radius=8.0,
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

    def test_flat_plan_can_select_more_chunks_than_slot_capped_plan(self):
        bank = _bank([(0, 32, 32, 32)] * 5, frame_tokens=100)
        ranked = list(range(5))
        distances = [0.01 * index for index in ranked]
        flat = motion.build_motion_retrieval_plan(
            bank,
            ranked,
            distances,
            probe_points=self.points,
            radius=8.0,
            frame_tokens=100,
            memory_frames=8,
            sink_frames=4,
            slot_capped=False,
        )
        capped = motion.build_motion_retrieval_plan(
            bank,
            ranked,
            distances,
            probe_points=self.points,
            radius=8.0,
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
            probe_points=self.points,
            radius=8.0,
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


if __name__ == "__main__":
    unittest.main()
