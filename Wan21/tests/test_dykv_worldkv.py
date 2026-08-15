import importlib.util
import math
import pathlib
import sys
import unittest
from types import SimpleNamespace

import torch


WAN21_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = WAN21_ROOT / "pipeline" / "dykv_worldkv.py"
SPEC = importlib.util.spec_from_file_location("dykv_worldkv_for_test", MODULE_PATH)
worldkv = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worldkv
SPEC.loader.exec_module(worldkv)


def _w2c(*, x=0.0, yaw_degrees=0.0, frames=4):
    angle = math.radians(float(yaw_degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    c2w = torch.eye(4)
    c2w[:3, :3] = torch.tensor(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
    )
    c2w[0, 3] = float(x)
    return torch.linalg.inv(c2w).repeat(1, frames, 1, 1)


class DyKVWorldKVRetrievalTest(unittest.TestCase):
    def test_score_matches_reference_candidate_normalization(self):
        blocks = [
            SimpleNamespace(
                frame_start=4,
                frame_count=4,
                viewmats=_w2c(yaw_degrees=90),
            ),
            SimpleNamespace(frame_start=8, frame_count=4, viewmats=_w2c(x=10)),
            SimpleNamespace(
                frame_start=12,
                frame_count=4,
                viewmats=_w2c(x=5, yaw_degrees=45),
            ),
        ]
        selected, ranked, distances, components = worldkv.select_worldkv_pose_blocks(
            SimpleNamespace(blocks=blocks),
            [0, 1, 2],
            current_viewmats=_w2c(),
            memory_frames=4,
        )

        self.assertEqual(selected, [2])
        self.assertEqual(ranked, [2, 0, 1])
        self.assertAlmostEqual(distances[0], 0.375, places=5)
        self.assertAlmostEqual(components["translation_squared"][0], 25.0, places=5)
        self.assertAlmostEqual(components["rotation_degrees"][0], 45.0, places=4)
        self.assertAlmostEqual(components["translation_normalized"][0], 0.25, places=5)
        self.assertAlmostEqual(components["rotation_normalized"][0], 0.5, places=5)

    def test_equal_scores_prefer_older_then_materialize_chronologically(self):
        blocks = [
            SimpleNamespace(frame_start=12, frame_count=4, viewmats=_w2c(x=1)),
            SimpleNamespace(frame_start=4, frame_count=4, viewmats=_w2c(x=-1)),
        ]
        selected, ranked, distances, _ = worldkv.select_worldkv_pose_blocks(
            SimpleNamespace(blocks=blocks),
            [0, 1],
            current_viewmats=_w2c(),
            memory_frames=8,
        )

        self.assertEqual(ranked, [1, 0])
        self.assertEqual(selected, [1, 0])
        self.assertEqual(distances, [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
