import importlib.util
import pathlib
import sys
import unittest
from types import SimpleNamespace

import torch


WAN21_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = WAN21_ROOT / "pipeline" / "dykv_fov.py"
SPEC = importlib.util.spec_from_file_location("dykv_fov", MODULE_PATH)
dykv_fov = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dykv_fov
SPEC.loader.exec_module(dykv_fov)


def _w2c(*, x=0.0, yaw_degrees=0.0):
    angle = torch.tensor(yaw_degrees * torch.pi / 180.0)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    c2w = torch.eye(4)
    c2w[:3, :3] = torch.tensor(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
    )
    c2w[0, 3] = x
    return torch.linalg.inv(c2w)


class DyKVFOVTest(unittest.TestCase):
    def setUp(self):
        self.points = dykv_fov.deterministic_sphere_points(20000, 8.0)

    def test_probe_points_are_deterministic_and_bounded(self):
        again = dykv_fov.deterministic_sphere_points(20000, 8.0)
        self.assertTrue(self.points.equal(again))
        self.assertLessEqual(float(torch.linalg.vector_norm(self.points, dim=-1).max()), 8.0)

    def test_identical_view_has_greater_overlap_than_opposite_view(self):
        identity = _w2c()
        identical = dykv_fov.fov_overlap(identity, identity, self.points)
        opposite = dykv_fov.fov_overlap(identity, _w2c(yaw_degrees=180), self.points)
        self.assertGreater(float(identical), 0.99)
        self.assertLess(float(opposite), 0.05)

    def test_selector_prefers_returned_camera_view_and_respects_frame_budget(self):
        current = torch.stack([_w2c(x=0.1)] * 4).unsqueeze(0)
        blocks = [
            SimpleNamespace(frame_start=4, frame_count=4, viewmats=torch.stack([_w2c(x=0.1)] * 4).unsqueeze(0)),
            SimpleNamespace(frame_start=8, frame_count=4, viewmats=torch.stack([_w2c(yaw_degrees=180)] * 4).unsqueeze(0)),
            SimpleNamespace(frame_start=12, frame_count=4, viewmats=torch.stack([_w2c(x=0.2)] * 4).unsqueeze(0)),
        ]
        selected, ranked, distances = dykv_fov.select_fov_blocks(
            SimpleNamespace(blocks=blocks),
            [0, 1, 2],
            current_viewmats=current,
            memory_frames=8,
            probe_points=self.points,
        )
        self.assertEqual(selected, [0, 2])
        self.assertEqual(ranked[:2], [0, 2])
        self.assertEqual(len(distances), 3)
        self.assertLess(distances[0], distances[-1])


if __name__ == "__main__":
    unittest.main()
