import importlib.util
import math
import pathlib
import sys
import types
import unittest

import torch


WAN21_ROOT = pathlib.Path(__file__).resolve().parents[1]
PIPELINE_ROOT = WAN21_ROOT / "pipeline"
package = types.ModuleType("dykv_packing_test_pipeline")
package.__path__ = [str(PIPELINE_ROOT)]
sys.modules[package.__name__] = package


def _load(name):
    full_name = f"{package.__name__}.{name}"
    spec = importlib.util.spec_from_file_location(full_name, PIPELINE_ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


memory = _load("dykv_memory")
packing = _load("dykv_packing")


def _yaw_w2c(degrees):
    angle = math.radians(float(degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    c2w = torch.eye(4)
    c2w[:3, :3] = torch.tensor(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
    )
    return torch.linalg.inv(c2w)


def _intrinsics(frames=4):
    fx = 0.5 / math.tan(math.radians(30.0))
    K = torch.tensor([[fx, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    return torch.stack([K] * frames).unsqueeze(0)


def _cache(value, tokens=64):
    key = torch.arange(tokens, dtype=torch.float32).reshape(1, tokens, 1, 1)
    key = key + float(value) * 1000
    return {"k": key, "v": key + 100, "local_end_index": torch.tensor(tokens)}


class DyKVPackingTest(unittest.TestCase):
    def test_keep_tier_boundaries_and_atom_costs(self):
        self.assertEqual(packing.quantize_keep_tier(0.0), 0.0)
        self.assertEqual(packing.quantize_keep_tier(0.374), 0.25)
        self.assertEqual(packing.quantize_keep_tier(0.375), 0.5)
        self.assertEqual(packing.quantize_keep_tier(0.749), 0.5)
        self.assertEqual(packing.quantize_keep_tier(0.75), 1.0)
        self.assertEqual([packing.tier_atoms(q) for q in (0.25, 0.5, 1.0)], [1, 2, 4])

    def test_quantized_plan_and_materialization_obey_budget(self):
        bank = memory.DyKVBank(device="cpu")
        historical = torch.stack([_yaw_w2c(0)] * 4).unsqueeze(0)
        for start in range(0, 32, 4):
            bank.archive_clean_block(
                [_cache(start)],
                frame_start=start,
                frame_count=4,
                frame_tokens=16,
                viewmats=historical,
                Ks=_intrinsics(),
                spatial_shape=(2, 8),
            )
        current = torch.stack([_yaw_w2c(45)] * 4).unsqueeze(0)
        plan = packing.build_packed_retrieval_plan(
            bank,
            list(range(1, 8)),
            [0.1 + index * 0.01 for index in range(7)],
            current_viewmats=current,
            current_Ks=_intrinsics(),
            frame_tokens=16,
            memory_frames=8,
            sink_frames=4,
            include_tail_latents=True,
            compression_fov_source="intrinsics",
            fixed_horizontal_degrees=60.0,
        )
        payload = packing.materialize_packed_retrieval(
            bank, plan, target_device="cpu", frame_tokens=16
        )[0]

        self.assertEqual(payload["k"].shape[1], plan.used_tokens)
        self.assertLessEqual(plan.used_tokens, 128)
        self.assertGreater(len(plan.frames), 8)
        self.assertEqual(sum(payload["frame_token_lengths"]), plan.used_tokens)
        for slot in set(payload["virtual_slot_ids"]):
            slot_tokens = sum(
                length
                for length, candidate_slot in zip(
                    payload["frame_token_lengths"], payload["virtual_slot_ids"]
                )
                if candidate_slot == slot
            )
            self.assertLessEqual(slot_tokens, 16)

    def test_tail_latents_fill_capacity_left_by_whole_chunk_plan(self):
        bank = memory.DyKVBank(device="cpu")
        for block_index, yaw in enumerate((0, 0, 0, 30)):
            poses = torch.stack([_yaw_w2c(yaw)] * 4).unsqueeze(0)
            bank.archive_clean_block(
                [_cache(block_index)],
                frame_start=block_index * 4,
                frame_count=4,
                frame_tokens=16,
                viewmats=poses,
                Ks=_intrinsics(),
                spatial_shape=(2, 8),
            )
        current = torch.stack([_yaw_w2c(0)] * 4).unsqueeze(0)
        plan = packing.build_packed_retrieval_plan(
            bank,
            [1, 2, 3],
            [0.01, 0.9, 0.01],
            current_viewmats=current,
            current_Ks=_intrinsics(),
            frame_tokens=16,
            memory_frames=8,
            sink_frames=4,
            include_tail_latents=True,
            compression_fov_source="intrinsics",
            fixed_horizontal_degrees=60.0,
        )

        self.assertEqual(plan.selected_full_blocks, (1, 3))
        self.assertEqual(len(plan.selected_tail_frames), 2)
        self.assertEqual(plan.used_atoms, plan.budget_atoms)
        self.assertEqual(
            {frame.selection_kind for frame in plan.frames},
            {"full_chunk", "tail_latent"},
        )

    def test_minwm_back_fixed_cases_match_expected_physical_budgets(self):
        bank = memory.DyKVBank(device="cpu")
        poses = torch.stack([_yaw_w2c(0)] * 4).unsqueeze(0)
        for block_index in range(4):
            bank.archive_clean_block(
                [_cache(block_index, tokens=48)],
                frame_start=block_index * 4,
                frame_count=4,
                frame_tokens=12,
                viewmats=poses,
            )
        cases = (
            ((0, 1), 8, 0.5, 60, 5),
            ((0, 1, 2), 12, 0.5, 90, 8),
            ((0, 1, 2, 3), 16, 1.0 / 3.0, 96, 8),
        )
        for selected, retrieval_frames, keep_ratio, tokens, slots in cases:
            with self.subTest(retrieval_frames=retrieval_frames, keep_ratio=keep_ratio):
                plan = packing.build_fixed_worldkv_retrieval_plan(
                    bank,
                    selected,
                    frame_tokens=12,
                    memory_frames=8,
                    sink_frames=4,
                    retrieval_frames=retrieval_frames,
                    keep_ratio=keep_ratio,
                )
                payload = packing.materialize_fixed_worldkv_retrieval(
                    bank, plan, target_device="cpu", frame_tokens=12
                )[0]
                self.assertEqual(plan.used_tokens, tokens)
                self.assertEqual(plan.used_virtual_slots, slots)
                self.assertEqual(payload["k"].shape[1], tokens)
                self.assertEqual(len(payload["source_frame_ids"]), retrieval_frames)
                self.assertEqual(
                    sum(payload["frame_token_lengths"]), payload["k"].shape[1]
                )


if __name__ == "__main__":
    unittest.main()
