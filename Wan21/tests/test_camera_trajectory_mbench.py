import pathlib
import sys
import unittest

import numpy as np


WAN21_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WAN21_ROOT))
from wan_utils.camera_trajectory import parse_trajectory  # noqa: E402


class MBenchCameraTrajectoryTest(unittest.TestCase):
    def test_static_segment_preserves_pose(self):
        poses = parse_trajectory("n*7")
        self.assertEqual(poses.shape, (8, 4, 4))
        self.assertTrue(np.allclose(poses, np.eye(4, dtype=np.float32)))

    def test_scaled_yaw_completes_full_rotation(self):
        poses = parse_trajectory("j@3*40")
        self.assertEqual(poses.shape[0], 41)
        self.assertTrue(np.allclose(poses[-1], np.eye(4), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
