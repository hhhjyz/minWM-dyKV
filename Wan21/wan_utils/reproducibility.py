"""Per-video random-state helpers for reproducible inference comparisons."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch


# Python's legacy NumPy RandomState, which is reset by wan_utils.misc.set_seed,
# only accepts unsigned 32-bit seeds.  Keep every public seed in that common
# domain so the same value is valid for Python, NumPy, PyTorch, and CUDA.
_SEED_MODULUS = 1 << 32
_PIPELINE_SEED_OFFSET = 1 << 31
SEED_POLICY = "base_seed_plus_prompt_index_v2"


def derive_sample_seed(base_seed: int, prompt_index: int) -> int:
    """Derive a case-independent seed for one stable dataset index."""

    index = int(prompt_index)
    if index < 0:
        raise ValueError("prompt_index must be non-negative")
    return (int(base_seed) + index) % _SEED_MODULUS


def derive_pipeline_seed(sample_seed: int) -> int:
    """Use a stable, disjoint RNG substream for scheduler-side randomness."""

    return (int(sample_seed) + _PIPELINE_SEED_OFFSET) % _SEED_MODULUS


def sample_initial_noise(
    shape: Sequence[int],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    sample_seed: int,
) -> torch.Tensor:
    """Sample initial video noise without consuming the process-global RNG."""

    generator = torch.Generator(device=device)
    generator.manual_seed(int(sample_seed))
    return torch.randn(
        tuple(int(dimension) for dimension in shape),
        device=device,
        dtype=dtype,
        generator=generator,
    )


def initial_noise_fingerprint(noise: torch.Tensor, *, values: int = 2048) -> str:
    """Hash a small exact-position prefix for inexpensive run verification."""

    if int(values) <= 0:
        raise ValueError("fingerprint values must be positive")
    prefix = (
        noise.detach()
        .reshape(-1)[: int(values)]
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
    )
    return hashlib.sha256(prefix.numpy().tobytes()).hexdigest()
