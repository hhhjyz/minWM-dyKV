import importlib.util
import pathlib
import sys
import unittest

import torch


WAN21_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = WAN21_ROOT / "wan" / "modules" / "dykv_rope.py"
SPEC = importlib.util.spec_from_file_location("dykv_rope", MODULE_PATH)
dykv_rope = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dykv_rope
SPEC.loader.exec_module(dykv_rope)

TriRegionSpec = dykv_rope.TriRegionSpec
compose_tri_region = dykv_rope.compose_tri_region
rebase_query = dykv_rope.rebase_query
shift_roped_time = dykv_rope.shift_roped_time


def _freqs(length=32):
    positions = torch.arange(length, dtype=torch.float64)
    rates = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64)
    return torch.polar(torch.ones(length, 3, dtype=torch.float64), positions[:, None] * rates)


class DyKVRoPETest(unittest.TestCase):
    def test_default_layout_is_contiguous_four_eight_eight(self):
        spec = TriRegionSpec()
        self.assertEqual(spec.sink_frames, 4)
        self.assertEqual(spec.memory_frames, 8)
        self.assertEqual(spec.local_frames, 8)
        self.assertEqual(spec.sink_frames + spec.memory_frames, spec.local_start(4))
        self.assertEqual(spec.query_start(4), 16)

    def test_time_shift_round_trip_preserves_spatial_channels(self):
        tensor = torch.randn(1, 5, 2, 6)
        shifted = shift_roped_time(tensor, _freqs(), 7)
        restored = shift_roped_time(shifted, _freqs(), -7)
        self.assertTrue(torch.allclose(tensor, restored, atol=1e-5))
        self.assertTrue(tensor[..., 2:].equal(shifted[..., 2:]))

    def test_query_maps_to_final_trained_positions(self):
        query = torch.randn(1, 8, 1, 6)
        spec = TriRegionSpec(rope_train_frames=20)
        expected = shift_roped_time(query, _freqs(), 16 - 40)
        actual = rebase_query(
            query,
            freqs=_freqs(),
            source_start_frame=40,
            query_frames=4,
            spec=spec,
        )
        self.assertTrue(torch.allclose(expected, actual))

    def test_composition_has_sink_retrieval_local_order_and_does_not_mutate_inputs(self):
        # One token per frame makes region boundaries directly observable in V.
        keys = torch.randn(1, 20, 1, 6)
        values = torch.arange(20, dtype=torch.float32).reshape(1, 20, 1, 1)
        cache = {"k": keys.clone(), "v": values.clone()}
        retrieval_k = torch.randn(1, 4, 1, 6)
        retrieval_v = torch.tensor([100, 101, 102, 103], dtype=torch.float32).reshape(1, 4, 1, 1)
        retrieval = {
            "k": retrieval_k,
            "v": retrieval_v,
            "src_frame_ids": [4],
            "chunk_frame_counts": [4],
            "chunk_token_lengths": [4],
        }
        original_cache_k = cache["k"].clone()
        original_retrieval_k = retrieval_k.clone()

        composed_k, composed_v = compose_tri_region(
            cache,
            local_end_index=20,
            frame_tokens=1,
            current_end_frame=44,
            query_frames=4,
            freqs=_freqs(64),
            retrieval=retrieval,
            spec=TriRegionSpec(),
            dtype=keys.dtype,
            device=keys.device,
        )

        self.assertEqual(composed_k.shape[1], 16)
        self.assertEqual(
            composed_v.flatten().tolist(),
            [0, 1, 2, 3, 100, 101, 102, 103, 12, 13, 14, 15, 16, 17, 18, 19],
        )
        self.assertTrue(cache["k"].equal(original_cache_k))
        self.assertTrue(retrieval_k.equal(original_retrieval_k))

    def test_packed_segments_are_rebased_to_explicit_shared_slots(self):
        keys = torch.randn(1, 20, 1, 6)
        values = torch.arange(20, dtype=torch.float32).reshape(1, 20, 1, 1)
        retrieval_k = torch.randn(1, 6, 1, 6)
        retrieval_v = torch.arange(100, 106, dtype=torch.float32).reshape(1, 6, 1, 1)
        retrieval = {
            "k": retrieval_k,
            "v": retrieval_v,
            "source_frame_ids": [20, 21, 30],
            "frame_token_lengths": [2, 2, 2],
            "virtual_slot_ids": [4, 4, 5],
        }

        composed_k, composed_v = compose_tri_region(
            {"k": keys, "v": values},
            local_end_index=20,
            frame_tokens=4,
            current_end_frame=44,
            query_frames=4,
            freqs=_freqs(64),
            retrieval=retrieval,
            spec=TriRegionSpec(),
            dtype=keys.dtype,
            device=keys.device,
        )

        expected = torch.cat(
            [
                shift_roped_time(retrieval_k[:, :2], _freqs(64), 4 - 20),
                shift_roped_time(retrieval_k[:, 2:4], _freqs(64), 4 - 21),
                shift_roped_time(retrieval_k[:, 4:], _freqs(64), 5 - 30),
            ],
            dim=1,
        )
        self.assertTrue(torch.allclose(composed_k[:, 16:22], expected))
        self.assertEqual(composed_v[:, 16:22].flatten().tolist(), list(range(100, 106)))
        self.assertTrue(retrieval_k.equal(retrieval["k"]))

    def test_packed_slot_capacity_is_enforced(self):
        retrieval = {
            "k": torch.randn(1, 5, 1, 6),
            "v": torch.randn(1, 5, 1, 1),
            "source_frame_ids": [20, 21],
            "frame_token_lengths": [3, 2],
            "virtual_slot_ids": [4, 4],
        }
        with self.assertRaisesRegex(ValueError, "slot exceeds"):
            compose_tri_region(
                {"k": torch.randn(1, 24, 1, 6), "v": torch.randn(1, 24, 1, 1)},
                local_end_index=24,
                frame_tokens=4,
                current_end_frame=44,
                query_frames=4,
                freqs=_freqs(64),
                retrieval=retrieval,
                spec=TriRegionSpec(),
                dtype=torch.float32,
                device=torch.device("cpu"),
            )

    def test_packed_frame_lengths_must_use_quarter_frame_atoms(self):
        retrieval = {
            "k": torch.randn(1, 3, 1, 6),
            "v": torch.randn(1, 3, 1, 1),
            "source_frame_ids": [20],
            "frame_token_lengths": [3],
            "virtual_slot_ids": [4],
            "packing_atom_tokens": 2,
        }
        with self.assertRaisesRegex(ValueError, "atom-aligned"):
            compose_tri_region(
                {"k": torch.randn(1, 24, 1, 6), "v": torch.randn(1, 24, 1, 1)},
                local_end_index=24,
                frame_tokens=8,
                current_end_frame=44,
                query_frames=4,
                freqs=_freqs(64),
                retrieval=retrieval,
                spec=TriRegionSpec(),
                dtype=torch.float32,
                device=torch.device("cpu"),
            )

    def test_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            TriRegionSpec(memory_frames=16).validate(query_frames=4)

    def test_gap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "gap"):
            TriRegionSpec(memory_frames=4).validate(query_frames=4)


if __name__ == "__main__":
    unittest.main()
