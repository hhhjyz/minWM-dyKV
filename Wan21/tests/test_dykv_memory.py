import importlib.util
import pathlib
import sys
import unittest

import torch


WAN21_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WAN21_ROOT))

MODULE_PATH = WAN21_ROOT / "pipeline" / "dykv_memory.py"
SPEC = importlib.util.spec_from_file_location("dykv_memory", MODULE_PATH)
dykv_memory = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dykv_memory
SPEC.loader.exec_module(dykv_memory)

DyKVBank = dykv_memory.DyKVBank
DyKVConfig = dykv_memory.DyKVConfig
compress_retrieved_kv = dykv_memory.compress_retrieved_kv


def _cache(values: torch.Tensor) -> dict:
    return {
        "k": values.clone(),
        "v": (values + 100).clone(),
        "local_end_index": torch.tensor(values.shape[1]),
    }


class DyKVMemoryTest(unittest.TestCase):
    def test_default_config_is_contiguous_four_eight_eight(self):
        config = DyKVConfig(enabled=True).validate(chunk_frames=4)
        self.assertEqual(
            (config.sink_frames, config.memory_frames, config.local_frames),
            (4, 8, 8),
        )
        self.assertEqual(
            config.sink_frames + config.memory_frames + config.local_frames,
            config.rope_train_frames,
        )
        self.assertEqual(config.retrieval_mode, "fov")
        self.assertFalse(hasattr(config, "fov_horizontal_degrees"))
        self.assertFalse(hasattr(config, "fov_vertical_degrees"))
        self.assertFalse(hasattr(config, "retrieval_fov_source"))
        self.assertFalse(hasattr(config, "compression_fov_source"))

    def test_config_rejects_unaligned_memory_budget(self):
        config = DyKVConfig(enabled=True, memory_frames=6)
        with self.assertRaisesRegex(ValueError, "divisible"):
            config.validate(chunk_frames=4)

    def test_config_rejects_a_layout_with_unused_rope_positions(self):
        config = DyKVConfig(enabled=True, memory_frames=4)
        with self.assertRaisesRegex(ValueError, "exactly fill"):
            config.validate(chunk_frames=4)

    def test_config_rejects_unknown_compression_mode(self):
        config = DyKVConfig(enabled=True, compression_mode="unknown")
        with self.assertRaisesRegex(ValueError, "compression_mode"):
            config.validate(chunk_frames=4)

    def test_config_rejects_unknown_retrieval_mode(self):
        config = DyKVConfig(enabled=True, retrieval_mode="unknown")
        with self.assertRaisesRegex(ValueError, "retrieval_mode"):
            config.validate(chunk_frames=4)

    def test_config_rejects_unknown_packing_mode(self):
        config = DyKVConfig(enabled=True, packing_mode="unknown")
        with self.assertRaisesRegex(ValueError, "packing_mode"):
            config.validate(chunk_frames=4)

    def test_config_rejects_unknown_retrieval_layout(self):
        config = DyKVConfig(enabled=True, retrieval_layout="unknown")
        with self.assertRaisesRegex(ValueError, "retrieval_layout"):
            config.validate(chunk_frames=4)

    def test_fixed_worldkv_allows_source_coverage_above_physical_memory(self):
        config = DyKVConfig(
            enabled=True,
            retrieval_frames=16,
            packing_mode="fixed_worldkv",
        )
        self.assertIs(config.validate(chunk_frames=4), config)

    def test_unpacked_source_coverage_cannot_exceed_physical_memory(self):
        config = DyKVConfig(enabled=True, retrieval_frames=12)
        with self.assertRaisesRegex(ValueError, "retrieval_frames"):
            config.validate(chunk_frames=4)

    def test_bank_archives_latest_clean_tokens_and_only_exposes_evicted_blocks(self):
        values = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1, 1)
        bank = DyKVBank()
        block = bank.archive_clean_block(
            [_cache(values)], frame_start=4, frame_count=2, frame_tokens=2
        )
        self.assertEqual(block.layers[0].k.flatten().tolist(), [8.0, 9.0, 10.0, 11.0])
        self.assertEqual(
            bank.evicted_candidates(current_frame=7, recent_frames=4, sink_frames=4), []
        )
        self.assertEqual(
            bank.evicted_candidates(current_frame=10, recent_frames=4, sink_frames=4), [0]
        )

    def test_retrieval_compression_keeps_anchor_and_novel_tokens(self):
        # Two frames, four scalar-key tokens each. The anchor centroid is positive;
        # the most negative second-frame tokens are therefore the most novel.
        k = torch.tensor([1, 1, 1, 1, 4, -3, 2, -2], dtype=torch.float32).reshape(1, 8, 1, 1)
        v = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1)
        compressed_k, compressed_v = compress_retrieved_kv(
            k, v, chunk_frames=2, frame_tokens=4, keep_ratio=0.5
        )
        self.assertEqual(compressed_k.shape[1], 6)
        self.assertTrue(compressed_k[:, :4].equal(k[:, :4]))
        self.assertEqual(compressed_v.flatten().tolist(), [0, 1, 2, 3, 5, 7])

    def test_materialize_compresses_after_selection_without_mutating_bank(self):
        bank = DyKVBank()
        first = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1)
        second = torch.arange(8, 16, dtype=torch.float32).reshape(1, 8, 1, 1)
        bank.archive_clean_block([_cache(first)], frame_start=4, frame_count=2, frame_tokens=4)
        bank.archive_clean_block([_cache(second)], frame_start=6, frame_count=2, frame_tokens=4)

        before = bank.blocks[0].layers[0].k.clone()
        payload = bank.materialize(
            [1, 0], target_device="cpu", chunk_frames=2, frame_tokens=4, keep_ratio=0.5
        )[0]

        self.assertEqual(payload["src_frame_ids"], [4, 6])
        self.assertEqual(payload["chunk_token_lengths"], [6, 6])
        self.assertEqual(payload["k"].shape[1], 12)
        self.assertTrue(bank.blocks[0].layers[0].k.equal(before))


if __name__ == "__main__":
    unittest.main()
