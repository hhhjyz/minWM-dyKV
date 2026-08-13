import argparse
import importlib.util
import json
import pathlib
import shutil
import sys
import tempfile
import unittest


WAN21_ROOT = pathlib.Path(__file__).resolve().parents[1]
RESEARCH_ROOT = WAN21_ROOT.parents[1]
ADAPTER_PATH = WAN21_ROOT / "scripts" / "evaluation" / "mbench_adapter.py"
SPEC = importlib.util.spec_from_file_location("mbench_adapter", ADAPTER_PATH)
adapter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adapter
SPEC.loader.exec_module(adapter)


class MBenchAdapterTest(unittest.TestCase):
    def test_all_actions_produce_exact_requested_pose_count(self):
        sys.path.insert(0, str(WAN21_ROOT))
        from wan_utils.camera_trajectory import parse_trajectory

        for action in sorted(adapter.SUPPORTED_ACTIONS):
            trajectory = adapter.trajectory_for_action(action, 100)
            self.assertEqual(parse_trajectory(trajectory).shape[0], 100, action)

    def test_prepare_uses_mbench_case_contract(self):
        dataset_root = RESEARCH_ROOT / "MBench" / "demo" / "dataset" / "a"
        assignments = dataset_root / "models" / "matrix_game_2" / "samples.jsonl"
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = pathlib.Path(temp_dir)
            adapter.prepare(
                argparse.Namespace(
                    dataset_root=dataset_root,
                    work_dir=work_dir,
                    assignments=assignments,
                    subsets="environment,human",
                    conditions="",
                    num_output_frames=100,
                    limit=None,
                )
            )
            rows = [json.loads(line) for line in (work_dir / "cases.jsonl").read_text().splitlines()]
            self.assertEqual([row["subset"] for row in rows], ["environment", "human"])
            self.assertTrue(all(row["num_output_frames"] == 100 for row in rows))
            self.assertEqual(len((work_dir / "prompts.txt").read_text().splitlines()), 2)

    def test_prepare_rejects_condition_length_mismatch(self):
        dataset_root = RESEARCH_ROOT / "MBench" / "demo" / "dataset" / "a"
        assignments = dataset_root / "models" / "matrix_game_2" / "samples.jsonl"
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "requires 100"):
                adapter.prepare(
                    argparse.Namespace(
                        dataset_root=dataset_root,
                        work_dir=pathlib.Path(temp_dir),
                        assignments=assignments,
                        subsets="",
                        conditions="",
                        num_output_frames=40,
                        limit=1,
                    )
                )

    def test_package_writes_model_centric_samples(self):
        demo_root = RESEARCH_ROOT / "MBench" / "demo" / "dataset" / "a"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            shutil.copy2(demo_root / "dataset.yaml", root / "dataset.yaml")
            video = root / "generated.mp4"
            video.write_bytes(b"test-video")
            cases = root / "cases.jsonl"
            generations = root / "generation_manifest.jsonl"
            case = {
                "prompt_index": 0,
                "subset": "environment",
                "sample_id": "sample",
                "condition_id": "left_then_right_25s",
                "trajectory": "j*49,l*49,n*1",
                "num_output_frames": 100,
            }
            cases.write_text(json.dumps(case) + "\n")
            generations.write_text(
                json.dumps({"prompt_index": 0, "output_path": str(video)}) + "\n"
            )
            adapter.package(
                argparse.Namespace(
                    dataset_root=root,
                    cases=cases,
                    generation_manifest=generations,
                    model_id="minwm_dykv_test",
                    link_mode="copy",
                )
            )
            samples = root / "models" / "minwm_dykv_test" / "samples.jsonl"
            row = json.loads(samples.read_text().strip())
            self.assertEqual(row["dataset_id"], "mbencha")
            self.assertEqual(row["model_id"], "minwm_dykv_test")
            packaged = samples.parent / row["media"]["videos"][0]["path"]
            self.assertEqual(packaged.read_bytes(), b"test-video")


if __name__ == "__main__":
    unittest.main()
