import importlib.util
import math
import pathlib
import sys
import unittest

import torch


WAN21_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = WAN21_ROOT / "pipeline" / "dykv_memory.py"
SPEC = importlib.util.spec_from_file_location("dykv_dynamic_memory", MODULE_PATH)
memory = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = memory
SPEC.loader.exec_module(memory)


def _w2c(*, yaw_degrees=0.0, x=0.0):
    angle = math.radians(float(yaw_degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    c2w = torch.eye(4)
    c2w[:3, :3] = torch.tensor(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
    )
    c2w[0, 3] = float(x)
    return torch.linalg.inv(c2w)


def _K(horizontal_fov_degrees=60.0):
    fx = 0.5 / math.tan(math.radians(horizontal_fov_degrees) / 2.0)
    return torch.tensor([[fx, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])


def _poses(yaw_degrees, frames=4, *, x=0.0):
    return torch.stack([_w2c(yaw_degrees=yaw_degrees, x=x)] * frames).unsqueeze(0)


def _intrinsics(frames=4, horizontal_fov_degrees=60.0):
    return torch.stack([_K(horizontal_fov_degrees)] * frames).unsqueeze(0)


def _cache(values):
    return {
        "k": values.clone(),
        "v": values.clone() + 1000,
        "local_end_index": torch.tensor(values.shape[1]),
    }


class DyKVDynamicCompressionTest(unittest.TestCase):
    def _plan(self, current_yaw):
        return memory.build_yaw_crop_plan(
            historical_viewmats=_poses(0.0),
            historical_Ks=_intrinsics(),
            current_viewmats=_poses(current_yaw),
            current_Ks=_intrinsics(),
            frame_count=4,
            frame_tokens=12,
            spatial_shape=(2, 6),
        )

    def test_zero_yaw_keeps_every_spatial_token(self):
        plan = self._plan(0.0)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.kept_tokens_per_frame, (12, 12, 12, 12))
        self.assertEqual(plan.token_indices.tolist(), list(range(48)))

    def test_half_fov_crops_directional_half_and_mirrors(self):
        right = self._plan(30.0)
        left = self._plan(-30.0)
        self.assertEqual(right.kept_columns_per_frame[0], (3, 4, 5))
        self.assertEqual(left.kept_columns_per_frame[0], (0, 1, 2))
        self.assertEqual(right.kept_tokens_per_frame, (6, 6, 6, 6))
        self.assertEqual(left.kept_tokens_per_frame, (6, 6, 6, 6))

    def test_one_fov_has_no_visible_columns(self):
        plan = self._plan(60.0)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.token_indices.numel(), 0)
        self.assertEqual(plan.kept_tokens_per_frame, (0, 0, 0, 0))

    def test_fixed_and_intrinsics_fov_produce_separate_crop_cases(self):
        kwargs = dict(
            historical_viewmats=_poses(0.0),
            historical_Ks=_intrinsics(horizontal_fov_degrees=89.424168),
            current_viewmats=_poses(60.0),
            current_Ks=_intrinsics(horizontal_fov_degrees=89.424168),
            frame_count=4,
            frame_tokens=24,
            spatial_shape=(2, 12),
        )
        fixed = memory.build_yaw_crop_plan(**kwargs, fov_source="fixed")
        intrinsic = memory.build_yaw_crop_plan(**kwargs, fov_source="intrinsics")
        self.assertEqual(fixed.token_indices.numel(), 0)
        self.assertGreater(intrinsic.token_indices.numel(), 0)
        self.assertLess(intrinsic.token_indices.numel(), 96)

    def test_full_rotation_wraps_to_same_view(self):
        plan = self._plan(360.0)
        self.assertEqual(plan.kept_tokens_per_frame, (12, 12, 12, 12))

    def test_translation_uses_fixed_compression_fallback(self):
        plan = memory.build_yaw_crop_plan(
            historical_viewmats=_poses(0.0),
            historical_Ks=_intrinsics(),
            current_viewmats=_poses(0.0, x=0.1),
            current_Ks=_intrinsics(),
            frame_count=4,
            frame_tokens=12,
            spatial_shape=(2, 6),
        )
        self.assertIsNone(plan)

    def test_materialize_gathers_same_columns_without_mutating_bank(self):
        values = torch.arange(48, dtype=torch.float32).reshape(1, 48, 1, 1)
        bank = memory.DyKVBank()
        bank.archive_clean_block(
            [_cache(values), _cache(values + 100)],
            frame_start=4,
            frame_count=4,
            frame_tokens=12,
            viewmats=_poses(0.0),
            Ks=_intrinsics(),
            spatial_shape=(2, 6),
        )
        before = bank.blocks[0].layers[0].k.clone()
        payloads = bank.materialize(
            [0],
            target_device="cpu",
            chunk_frames=4,
            frame_tokens=12,
            keep_ratio=0.5,
            compression_mode="yaw_fov",
            current_viewmats=_poses(30.0),
            current_Ks=_intrinsics(),
        )

        expected_indices = []
        for frame in range(4):
            expected_indices.extend(
                frame * 12 + index for index in (3, 4, 5, 9, 10, 11)
            )
        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0]["k"].flatten().tolist(), [float(i) for i in expected_indices])
        self.assertEqual(payloads[1]["k"].flatten().tolist(), [float(i + 100) for i in expected_indices])
        self.assertEqual(payloads[0]["chunk_token_lengths"], [24])
        self.assertEqual(payloads[0]["compression_modes"], ["yaw_fov"])
        self.assertEqual(payloads[0]["raw_tokens"], 48)
        self.assertEqual(payloads[0]["kept_tokens"], 24)
        self.assertTrue(bank.blocks[0].layers[0].k.equal(before))

    def test_materialize_drops_a_block_without_fov_overlap(self):
        values = torch.arange(48, dtype=torch.float32).reshape(1, 48, 1, 1)
        bank = memory.DyKVBank()
        bank.archive_clean_block(
            [_cache(values)],
            frame_start=4,
            frame_count=4,
            frame_tokens=12,
            viewmats=_poses(0.0),
            Ks=_intrinsics(),
            spatial_shape=(2, 6),
        )
        payloads = bank.materialize(
            [0],
            target_device="cpu",
            chunk_frames=4,
            frame_tokens=12,
            keep_ratio=0.5,
            compression_mode="yaw_fov",
            current_viewmats=_poses(90.0),
            current_Ks=_intrinsics(),
        )
        self.assertEqual(payloads, [])


if __name__ == "__main__":
    unittest.main()
