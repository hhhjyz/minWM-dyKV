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
        expected = {
                "baseline",
                "retrieval_no_compression",
                "retrieval_no_compression_relevance_order",
                "worldkv_pose_no_compression",
                "yaw_intrinsics",
                "packed_chunks",
                "packed_chunks_latent",
                "retr8_compression_r050",
                "retr12_compression_r050",
                "retr16_r033_slot_packed",
                "retr16_compression_r033",
                "motion_novelty_sphere_unfilled",
                "motion_novelty_slot_capped",
                "motion_novelty_unfilled",
                "motion_novelty_backfill",
                "motion_novelty_duplicate",
                "motion_alloc_cam_4chunk",
                "motion_alloc_cam_content_4chunk",
                "motion_alloc_cam_content_prerope_4chunk",
        }
        self.assertTrue(expected.issubset(cases.DYKV_CASES))

    def test_default_uses_intrinsics_for_retrieval_and_compression(self):
        preset = cases.get_dykv_case(cases.DEFAULT_DYKV_CASE)
        self.assertTrue(preset.enabled)
        self.assertEqual(preset.compression_mode, "yaw_fov")
        self.assertEqual(preset.packing_mode, "none")
        self.assertEqual(preset.retrieval_mode, "fov")

    def test_retrieval_ablation_only_changes_the_ranking_mode(self):
        fov = cases.get_dykv_case("retrieval_no_compression")
        worldkv = cases.get_dykv_case("worldkv_pose_no_compression")
        self.assertEqual(fov.retrieval_mode, "fov")
        self.assertEqual(worldkv.retrieval_mode, "worldkv_pose")
        self.assertEqual(fov.compression_mode, worldkv.compression_mode)
        self.assertEqual(fov.packing_mode, worldkv.packing_mode)
        self.assertEqual(fov.retrieval_frames, worldkv.retrieval_frames)
        self.assertEqual(fov.local_frames, worldkv.local_frames)
        self.assertEqual(fov.sink_frames, worldkv.sink_frames)

    def test_relevance_order_ablation_only_changes_chunk_rope_order(self):
        source_order = cases.get_dykv_case("retrieval_no_compression")
        relevance_order = cases.get_dykv_case(
            "retrieval_no_compression_relevance_order"
        )
        self.assertEqual(source_order.retrieval_order, "source_ordered")
        self.assertEqual(relevance_order.retrieval_order, "relevance_near_query")
        self.assertEqual(
            (
                source_order.retrieval_mode,
                source_order.compression_mode,
                source_order.packing_mode,
                source_order.retrieval_frames,
                source_order.compression_keep_ratio,
            ),
            (
                relevance_order.retrieval_mode,
                relevance_order.compression_mode,
                relevance_order.packing_mode,
                relevance_order.retrieval_frames,
                relevance_order.compression_keep_ratio,
            ),
        )

    def test_no_case_exposes_a_fixed_fov_parameter(self):
        for preset in cases.DYKV_CASES.values():
            self.assertFalse(hasattr(preset, "retrieval_fov_source"))
            self.assertFalse(hasattr(preset, "compression_fov_source"))

    def test_packed_cases_register_whole_chunk_and_tail_modes(self):
        self.assertEqual(
            cases.get_dykv_case("packed_chunks").packing_mode,
            "whole_chunks",
        )

    def test_minwm_back_fixed_budget_cases_are_registered(self):
        expected = {
            "retr8_compression_r050": (8, 0.5),
            "retr12_compression_r050": (12, 0.5),
            "retr16_r033_slot_packed": (16, 1.0 / 3.0),
            "retr16_compression_r033": (16, 1.0 / 3.0),
        }
        for name, (retrieval_frames, keep_ratio) in expected.items():
            with self.subTest(case=name):
                preset = cases.get_dykv_case(name)
                self.assertEqual(preset.packing_mode, "fixed_worldkv")
                self.assertEqual(preset.retrieval_frames, retrieval_frames)
                self.assertAlmostEqual(preset.compression_keep_ratio, keep_ratio)
        slot_order = cases.get_dykv_case("retr16_r033_slot_packed")
        source_order = cases.get_dykv_case("retr16_compression_r033")
        self.assertEqual(slot_order.retrieval_layout, "slot_packed")
        self.assertEqual(source_order.retrieval_layout, "source_ordered")
        self.assertEqual(
            (
                slot_order.retrieval_mode,
                slot_order.compression_mode,
                slot_order.packing_mode,
                slot_order.retrieval_frames,
                slot_order.compression_keep_ratio,
            ),
            (
                source_order.retrieval_mode,
                source_order.compression_mode,
                source_order.packing_mode,
                source_order.retrieval_frames,
                source_order.compression_keep_ratio,
            ),
        )
        self.assertEqual(
            cases.get_dykv_case("packed_chunks_latent").packing_mode,
            "whole_chunks_and_latents",
        )

    def test_baseline_is_the_only_disabled_case(self):
        disabled = [case.name for case in cases.DYKV_CASES.values() if not case.enabled]
        self.assertEqual(disabled, ["baseline", "baseline_honest"])

    def test_motion_allocation_ablation_changes_one_axis_at_a_time(self):
        camera = cases.get_dykv_case("motion_alloc_cam_4chunk")
        content = cases.get_dykv_case("motion_alloc_cam_content_4chunk")
        pre_rope = cases.get_dykv_case(
            "motion_alloc_cam_content_prerope_4chunk"
        )
        self.assertEqual(camera.retrieval_frames, 16)
        self.assertEqual(camera.packing_mode, "motion_alloc_4chunk")
        self.assertEqual(camera.motion_allocation_mode, "camera_budgeted")
        self.assertEqual(content.motion_allocation_mode, "camera_content_budgeted")
        self.assertEqual(camera.novelty_feature_mode, content.novelty_feature_mode)
        self.assertEqual(pre_rope.motion_allocation_mode, content.motion_allocation_mode)
        self.assertEqual(pre_rope.novelty_feature_mode, "pre_rope_k")

    def test_motion_novelty_layout_ablation_is_registered(self):
        capped = cases.get_dykv_case("motion_novelty_slot_capped")
        flat = cases.get_dykv_case("motion_novelty_unfilled")
        self.assertEqual(capped.compression_mode, "motion_novelty")
        self.assertEqual(flat.compression_mode, "motion_novelty")
        self.assertEqual(capped.retrieval_layout, "source_ordered")
        self.assertEqual(flat.retrieval_layout, "flat_source_ordered")
        self.assertEqual(capped.retrieval_mode, flat.retrieval_mode)
        self.assertEqual(capped.retrieval_frames, flat.retrieval_frames)
        backfill = cases.get_dykv_case("motion_novelty_backfill")
        duplicate = cases.get_dykv_case("motion_novelty_duplicate")
        self.assertEqual(backfill.retrieval_layout, "flat_source_ordered")
        self.assertEqual(duplicate.retrieval_layout, "flat_source_ordered")
        self.assertEqual(backfill.compression_mode, "motion_novelty")
        self.assertEqual(duplicate.compression_mode, "motion_novelty")

    def test_motion_geometry_ablation_only_changes_geometry_mode(self):
        sphere = cases.get_dykv_case("motion_novelty_sphere_unfilled")
        projected = cases.get_dykv_case("motion_novelty_unfilled")
        self.assertEqual(sphere.motion_geometry_mode, "sphere_fov")
        self.assertEqual(projected.motion_geometry_mode, "projected_multidepth")
        self.assertEqual(
            (
                sphere.retrieval_mode,
                sphere.compression_mode,
                sphere.packing_mode,
                sphere.retrieval_frames,
                sphere.retrieval_layout,
            ),
            (
                projected.retrieval_mode,
                projected.compression_mode,
                projected.packing_mode,
                projected.retrieval_frames,
                projected.retrieval_layout,
            ),
        )

    def test_every_case_uses_the_same_fixed_four_frame_sink(self):
        for preset in cases.DYKV_CASES.values():
            with self.subTest(case=preset.name):
                self.assertEqual(preset.sink_mode, "fixed")
                self.assertEqual(preset.sink_frames, 4)
                self.assertEqual(
                    preset.sink_frames + preset.local_frames,
                    12 if preset.enabled else 20,
                )

    def test_baseline_uses_empty_retrieval_tri_region_layout(self):
        baseline = cases.get_dykv_case("baseline")
        self.assertEqual(
            (baseline.sink_frames, baseline.memory_frames, baseline.local_frames),
            (4, 0, 16),
        )
        self.assertEqual((baseline.retrieval_mode, baseline.retrieval_frames), ("none", 0))
        for preset in cases.DYKV_CASES.values():
            with self.subTest(case=preset.name):
                self.assertEqual(
                    preset.sink_frames + preset.memory_frames + preset.local_frames,
                    20,
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
