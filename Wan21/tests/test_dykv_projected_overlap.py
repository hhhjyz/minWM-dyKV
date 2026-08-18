import importlib.util
import math
import pathlib
import sys
import unittest

import torch


WAN21_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = WAN21_ROOT / "pipeline" / "dykv_projected_overlap.py"
SPEC = importlib.util.spec_from_file_location("dykv_projected_overlap", MODULE_PATH)
projected = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = projected
SPEC.loader.exec_module(projected)


def _K(fx=0.5, fy=0.5, cx=0.5, cy=0.5):
    return torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )


def _w2c(*, yaw=0.0, pitch=0.0, roll=0.0, xyz=(0.0, 0.0, 0.0)):
    yaw = math.radians(float(yaw))
    pitch = math.radians(float(pitch))
    roll = math.radians(float(roll))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    rz = torch.tensor([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    c2w = torch.eye(4, dtype=torch.float64)
    c2w[:3, :3] = ry @ rx @ rz
    c2w[:3, 3] = torch.tensor(xyz, dtype=torch.float64)
    return torch.linalg.inv(c2w)


def _overlap(current, anchor=None, *, K=None, shape=(30, 52), scale=8.0):
    anchor = _w2c() if anchor is None else anchor
    K = _K() if K is None else K
    return projected.projected_motion_overlap(
        current,
        anchor,
        K,
        K,
        shape,
        scene_scale=scale,
    )


class ProjectedMotionOverlapTest(unittest.TestCase):
    def test_identity_is_exact_and_keeps_no_nonanchor_tokens(self):
        result = _overlap(_w2c())
        self.assertEqual(result.overlap_ratio, 1.0)
        self.assertEqual(result.keep_ratio, 0.0)
        self.assertEqual(result.keep_tokens, 0)
        self.assertEqual(result.forward_overlaps, (1.0,) * 4)
        self.assertEqual(result.backward_overlaps, (1.0,) * 4)

    def test_yaw_pitch_and_roll_are_direction_symmetric_and_monotonic(self):
        for axis in ("yaw", "pitch", "roll"):
            with self.subTest(axis=axis):
                positive = [
                    _overlap(_w2c(**{axis: degrees})).keep_ratio
                    for degrees in (3.0, 6.0, 9.0)
                ]
                negative = [
                    _overlap(_w2c(**{axis: -degrees})).keep_ratio
                    for degrees in (3.0, 6.0, 9.0)
                ]
                self.assertLess(positive[0], positive[1])
                self.assertLess(positive[1], positive[2])
                for lhs, rhs in zip(positive, negative):
                    self.assertAlmostEqual(lhs, rhs, places=12)

    def test_lateral_and_vertical_translation_are_symmetric_and_depth_aware(self):
        for xyz in ((0.24, 0.0, 0.0), (0.0, 0.24, 0.0)):
            with self.subTest(xyz=xyz):
                positive = _overlap(_w2c(xyz=xyz))
                negative = _overlap(_w2c(xyz=tuple(-value for value in xyz)))
                self.assertGreater(positive.keep_ratio, 0.0)
                self.assertAlmostEqual(
                    positive.keep_ratio,
                    negative.keep_ratio,
                    places=12,
                )
                self.assertLess(
                    positive.symmetric_overlaps[0],
                    positive.symmetric_overlaps[-1],
                )

    def test_forward_and_backward_are_nonzero_and_symmetric(self):
        forward = _overlap(_w2c(xyz=(0.0, 0.0, 0.24)))
        backward = _overlap(_w2c(xyz=(0.0, 0.0, -0.24)))
        self.assertGreater(forward.keep_ratio, 0.0)
        self.assertGreater(backward.keep_ratio, 0.0)
        self.assertAlmostEqual(forward.keep_ratio, backward.keep_ratio, places=12)
        self.assertTrue(
            any(abs(a - b) > 0.0 for a, b in zip(
                forward.forward_overlaps,
                forward.backward_overlaps,
            ))
        )

    def test_motion_speed_is_monotonic(self):
        for generator in (
            lambda magnitude: _w2c(yaw=3.0 * magnitude),
            lambda magnitude: _w2c(xyz=(0.0, 0.0, 0.08 * magnitude)),
            lambda magnitude: _w2c(xyz=(0.08 * magnitude, 0.0, 0.0)),
        ):
            ratios = [_overlap(generator(value)).keep_ratio for value in (0.5, 1, 2, 4)]
            self.assertTrue(all(a < b for a, b in zip(ratios, ratios[1:])))

    def test_swapping_frames_preserves_symmetric_overlap(self):
        anchor = _w2c(yaw=-4.0, xyz=(-0.1, 0.03, 0.02))
        current = _w2c(yaw=11.0, pitch=3.0, xyz=(0.2, -0.05, 0.3))
        first = _overlap(current, anchor)
        second = _overlap(anchor, current)
        self.assertAlmostEqual(first.overlap_ratio, second.overlap_ratio, places=12)
        self.assertEqual(first.forward_overlaps, second.backward_overlaps)
        self.assertEqual(first.backward_overlaps, second.forward_overlaps)

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "scene scale"):
            _overlap(_w2c(), scale=0.0)
        with self.assertRaisesRegex(ValueError, "focal lengths"):
            _overlap(_w2c(), K=_K(fx=0.0))
        with self.assertRaisesRegex(ValueError, "spatial shape"):
            projected.projected_motion_overlap(
                _w2c(), _w2c(), _K(), _K(), (10,), scene_scale=8.0
            )


if __name__ == "__main__":
    unittest.main()
