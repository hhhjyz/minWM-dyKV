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
                "packed_chunks",
                "packed_chunks_latent",
            ),
        )

    def test_default_uses_intrinsics_for_retrieval_and_compression(self):
        preset = cases.get_dykv_case(cases.DEFAULT_DYKV_CASE)
        self.assertTrue(preset.enabled)
        self.assertEqual(preset.compression_mode, "yaw_fov")
        self.assertEqual(preset.retrieval_fov_source, "intrinsics")
        self.assertEqual(preset.compression_fov_source, "intrinsics")
        self.assertEqual(preset.packing_mode, "none")

    def test_packed_cases_register_whole_chunk_and_tail_modes(self):
        self.assertEqual(
            cases.get_dykv_case("packed_chunks").packing_mode,
            "whole_chunks",
        )
        self.assertEqual(
            cases.get_dykv_case("packed_chunks_latent").packing_mode,
            "whole_chunks_and_latents",
        )

    def test_baseline_is_the_only_disabled_case(self):
        disabled = [case.name for case in cases.DYKV_CASES.values() if not case.enabled]
        self.assertEqual(disabled, ["baseline"])

    def test_every_case_uses_the_same_fixed_four_frame_sink(self):
        for preset in cases.DYKV_CASES.values():
            with self.subTest(case=preset.name):
                self.assertEqual(preset.sink_mode, "fixed")
                self.assertEqual(preset.sink_frames, 4)
                self.assertEqual(
                    preset.sink_frames + preset.local_frames,
                    12 if preset.enabled else 20,
                )

    def test_unknown_case_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown dyKV case"):
            cases.get_dykv_case("unknown")

    def test_invocation_defaults_match_fixed_sink_baseline_and_full_method(self):
        baseline = cases.resolve_dykv_case(None, enabled=False)
        complete = cases.resolve_dykv_case(None, enabled=True)
        self.assertEqual(baseline.name, "baseline")
        self.assertEqual(complete.name, "yaw_intrinsics")
        self.assertEqual((baseline.sink_mode, baseline.sink_frames), ("fixed", 4))
        self.assertEqual((complete.sink_mode, complete.sink_frames), ("fixed", 4))

    def test_case_and_dykv_flag_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "without --dykv"):
            cases.resolve_dykv_case("baseline", enabled=True)
        with self.assertRaisesRegex(ValueError, "with --dykv"):
            cases.resolve_dykv_case("yaw_intrinsics", enabled=False)


if __name__ == "__main__":
    unittest.main()
