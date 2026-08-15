import pathlib
import sys
import unittest

import torch


WAN21_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WAN21_ROOT))
from wan_utils.reproducibility import (  # noqa: E402
    SEED_POLICY,
    derive_pipeline_seed,
    derive_sample_seed,
    initial_noise_fingerprint,
    sample_initial_noise,
)


class ReproducibleVideoSeedTest(unittest.TestCase):
    def test_seed_is_stable_per_prompt_and_does_not_include_case(self):
        self.assertEqual(SEED_POLICY, "base_seed_plus_prompt_index_v1")
        self.assertEqual(derive_sample_seed(7, 0), 7)
        self.assertEqual(derive_sample_seed(7, 3), 10)
        self.assertEqual(derive_sample_seed(7, 3), derive_sample_seed(7, 3))
        self.assertEqual(derive_pipeline_seed(10), (1 << 32) + 10)
        self.assertNotEqual(derive_pipeline_seed(10), 10)

    def test_initial_noise_is_independent_of_generation_order(self):
        seed_for_two = derive_sample_seed(11, 2)
        direct = sample_initial_noise(
            (1, 4, 2), device="cpu", dtype=torch.float32, sample_seed=seed_for_two
        )
        _ = sample_initial_noise(
            (1, 4, 2),
            device="cpu",
            dtype=torch.float32,
            sample_seed=derive_sample_seed(11, 0),
        )
        after_another_sample = sample_initial_noise(
            (1, 4, 2), device="cpu", dtype=torch.float32, sample_seed=seed_for_two
        )

        self.assertTrue(torch.equal(direct, after_another_sample))
        self.assertEqual(
            initial_noise_fingerprint(direct),
            initial_noise_fingerprint(after_another_sample),
        )

    def test_different_prompt_indices_receive_different_noise(self):
        first = sample_initial_noise(
            (16,),
            device="cpu",
            dtype=torch.float32,
            sample_seed=derive_sample_seed(0, 0),
        )
        second = sample_initial_noise(
            (16,),
            device="cpu",
            dtype=torch.float32,
            sample_seed=derive_sample_seed(0, 1),
        )
        self.assertFalse(torch.equal(first, second))
        self.assertNotEqual(
            initial_noise_fingerprint(first), initial_noise_fingerprint(second)
        )

    def test_initial_noise_does_not_consume_global_rng(self):
        torch.manual_seed(99)
        expected = torch.rand(8)
        torch.manual_seed(99)
        _ = sample_initial_noise(
            (16,), device="cpu", dtype=torch.float32, sample_seed=123
        )
        observed = torch.rand(8)
        self.assertTrue(torch.equal(expected, observed))

    def test_negative_prompt_index_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            derive_sample_seed(0, -1)


if __name__ == "__main__":
    unittest.main()
