import importlib.util
import pathlib
import sys
import unittest


WAN21_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = WAN21_ROOT / "dykv_cases.py"
SPEC = importlib.util.spec_from_file_location("dykv_cases_for_test", MODULE_PATH)
cases = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cases
SPEC.loader.exec_module(cases)


class DyKVCasesTest(unittest.TestCase):
    def test_registered_case_names_are_stable(self):
        self.assertEqual(
            tuple(cases.DYKV_CASES),
            (
                "baseline",
                "retrieval_no_compression",
                "fixed_novelty",
                "yaw_fixed_fov",
                "yaw_mixed_fov",
                "yaw_intrinsics",
            ),
        )

    def test_default_uses_intrinsics_for_retrieval_and_compression(self):
        preset = cases.get_dykv_case(cases.DEFAULT_DYKV_CASE)
        self.assertTrue(preset.enabled)
        self.assertEqual(preset.compression_mode, "yaw_fov")
        self.assertEqual(preset.retrieval_fov_source, "intrinsics")
        self.assertEqual(preset.compression_fov_source, "intrinsics")

    def test_baseline_is_the_only_disabled_case(self):
        disabled = [case.name for case in cases.DYKV_CASES.values() if not case.enabled]
        self.assertEqual(disabled, ["baseline"])

    def test_unknown_case_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown dyKV case"):
            cases.get_dykv_case("unknown")


if __name__ == "__main__":
    unittest.main()
