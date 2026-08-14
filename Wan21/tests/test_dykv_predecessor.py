import importlib.util
import math
import pathlib
import sys
import types
import unittest

import torch


WAN21_ROOT = pathlib.Path(__file__).resolve().parents[1]
PIPELINE_ROOT = WAN21_ROOT / "pipeline"
sys.path.insert(0, str(WAN21_ROOT))
package = types.ModuleType("dykv_predecessor_test_pipeline")
package.__path__ = [str(PIPELINE_ROOT)]
sys.modules[package.__name__] = package


def _load(name):
    full_name = f"{package.__name__}.{name}"
    spec = importlib.util.spec_from_file_location(full_name, PIPELINE_ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


memory = _load("dykv_memory")
_load("dykv_packing")
predecessor = _load("dykv_predecessor")
from wan_utils.camera_trajectory import make_camera_tensors  # noqa: E402


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


def _cache(value, tokens=64):
    key = torch.arange(tokens, dtype=torch.float32).reshape(1, tokens, 1, 1)
    key = key + float(value) * 1000
    return {"k": key, "v": key + 100, "local_end_index": torch.tensor(tokens)}


def _bank_with_yaws(yaws):
    bank = memory.DyKVBank(device="cpu")
    for block_index, yaw in enumerate(yaws):
        bank.archive_clean_block(
            [_cache(block_index)],
            frame_start=block_index * 4,
            frame_count=4,
            frame_tokens=16,
            viewmats=torch.stack([_yaw_w2c(yaw)] * 4).unsqueeze(0),
            Ks=_intrinsics(),
            spatial_shape=(2, 8),
        )
    return bank


class DyKVPredecessorTest(unittest.TestCase):
    def test_float32_inference_trajectory_uses_yaw_geometry(self):
        viewmats, intrinsics = make_camera_tensors(
            "j*7", fx=0.5, fy=0.5, cx=0.5, cy=0.5, dtype=torch.float32
        )
        bank = memory.DyKVBank(device="cpu")
        for block_index in range(2):
            start = block_index * 4
            bank.archive_clean_block(
                [_cache(block_index)],
                frame_start=start,
                frame_count=4,
                frame_tokens=16,
                viewmats=viewmats[:, start:start + 4],
                Ks=intrinsics[:, start:start + 4],
                spatial_shape=(2, 8),
            )

        candidate = predecessor.build_predecessor_candidate(
            bank, block_index=1, distance=0.1, frame_tokens=16
        )

        self.assertIsNone(candidate.fallback_reason)
        self.assertEqual(candidate.keep_tier, 0.25)
        self.assertTrue(
            all(
                frame.compression_mode == "predecessor_incremental_yaw"
                for frame in candidate.chunk_frames
            )
        )

    def test_user_defined_four_tier_boundaries(self):
        values = (0.0, 0.2499, 0.25, 0.4999, 0.5, 0.7499, 0.75, 1.0)
        expected = (0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0)
        self.assertEqual(
            tuple(predecessor.quantize_incremental_ratio(value) for value in values),
            expected,
        )

    def test_crop_is_relative_to_predecessor_and_direction_is_mirrored(self):
        right_bank = _bank_with_yaws((0, 20))
        left_bank = _bank_with_yaws((0, -20))
        right = predecessor.build_predecessor_candidate(
            right_bank,
            block_index=1,
            distance=0.1,
            frame_tokens=16,
        )
        left = predecessor.build_predecessor_candidate(
            left_bank,
            block_index=1,
            distance=0.1,
            frame_tokens=16,
        )

        self.assertEqual(right.keep_tier, 0.5)
        self.assertEqual(left.keep_tier, 0.5)
        self.assertAlmostEqual(right.chunk_frames[0].raw_overlap_ratio, 1.0 / 3.0, places=5)
        self.assertEqual(
            right.chunk_frames[0].kept_columns,
            tuple(7 - column for column in reversed(left.chunk_frames[0].kept_columns)),
        )

    def test_three_quarter_chunks_are_bin_feasible(self):
        bank = _bank_with_yaws((0, 35, 70, 105, 140))
        current = torch.stack([_yaw_w2c(140)] * 4).unsqueeze(0)
        plan = predecessor.build_predecessor_retrieval_plan(
            bank,
            [1, 2, 3, 4],
            [0.1, 0.2, 0.3, 0.4],
            current_viewmats=current,
            current_Ks=_intrinsics(),
            frame_tokens=16,
            memory_frames=8,
            sink_frames=4,
            include_tail_latents=False,
            query_backfill=False,
        )
        payload = predecessor.materialize_packed_retrieval(
            bank, plan, target_device="cpu", frame_tokens=16
        )[0]

        self.assertEqual(len(plan.selected_full_blocks), 2)
        self.assertEqual(plan.used_atoms, 24)
        self.assertTrue(all(tier == 0.75 for tier in payload["keep_tiers"]))
        for slot in set(payload["virtual_slot_ids"]):
            slot_tokens = sum(
                length
                for length, candidate_slot in zip(
                    payload["frame_token_lengths"], payload["virtual_slot_ids"]
                )
                if candidate_slot == slot
            )
            self.assertLessEqual(slot_tokens, 16)

    def test_query_backfill_only_uses_residual_capacity(self):
        bank = _bank_with_yaws((0, 35, 70))
        current = torch.stack([_yaw_w2c(70)] * 4).unsqueeze(0)
        plan = predecessor.build_predecessor_retrieval_plan(
            bank,
            [1, 2],
            [0.2, 0.1],
            current_viewmats=current,
            current_Ks=_intrinsics(),
            frame_tokens=16,
            memory_frames=8,
            sink_frames=4,
            include_tail_latents=True,
            query_backfill=True,
        )
        payload = predecessor.materialize_packed_retrieval(
            bank, plan, target_device="cpu", frame_tokens=16
        )[0]

        self.assertGreater(sum(payload["query_backfill_tokens"]), 0)
        self.assertLessEqual(payload["k"].shape[1], 8 * 16)
        self.assertEqual(sum(payload["frame_token_lengths"]), payload["k"].shape[1])


if __name__ == "__main__":
    unittest.main()
