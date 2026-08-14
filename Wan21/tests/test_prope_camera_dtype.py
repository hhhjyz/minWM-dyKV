import importlib.util
import pathlib
import sys
import unittest

import torch


WAN21_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = WAN21_ROOT / "wan" / "modules" / "prope.py"
SPEC = importlib.util.spec_from_file_location("prope_camera_dtype", MODULE_PATH)
prope = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prope
SPEC.loader.exec_module(prope)


class PRoPECameraDtypeTest(unittest.TestCase):
    def test_float32_camera_geometry_is_cast_at_bfloat16_operator_boundary(self):
        q = torch.randn(1, 1, 2, 4, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        viewmats = torch.eye(4, dtype=torch.float32).repeat(1, 2, 1, 1)
        intrinsics = torch.tensor(
            [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).repeat(1, 2, 1, 1)

        query, key, value, apply_output = prope.prope_qkv(
            q,
            k,
            v,
            viewmats=viewmats,
            Ks=intrinsics,
        )

        self.assertEqual(query.dtype, torch.bfloat16)
        self.assertEqual(key.dtype, torch.bfloat16)
        self.assertEqual(value.dtype, torch.bfloat16)
        self.assertEqual(apply_output(value).dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
