import pathlib
import sys
import unittest

import numpy as np
import torch


WAN21_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WAN21_ROOT))
from wan_utils.camera_trajectory import make_camera_tensors, parse_trajectory  # noqa: E402


class MBenchCameraTrajectoryTest(unittest.TestCase):
    def test_static_segment_preserves_pose(self):
        poses = parse_trajectory("n*7")
        self.assertEqual(poses.shape, (8, 4, 4))
        self.assertTrue(np.allclose(poses, np.eye(4, dtype=np.float32)))

    def test_scaled_yaw_completes_full_rotation(self):
        poses = parse_trajectory("j@3*40")
        self.assertEqual(poses.shape[0], 41)
        self.assertTrue(np.allclose(poses[-1], np.eye(4), atol=1e-5))

    def test_inference_camera_factory_preserves_float32_geometry(self):
        viewmats, intrinsics = make_camera_tensors(
            "j*7", fx=0.5, fy=0.5, cx=0.5, cy=0.5, dtype=torch.float32
        )
        self.assertEqual(viewmats.dtype, torch.float32)
        self.assertEqual(intrinsics.dtype, torch.float32)
        self.assertEqual(viewmats.shape, (1, 8, 4, 4))
        self.assertEqual(intrinsics.shape, (1, 8, 3, 3))


if __name__ == "__main__":
    unittest.main()
