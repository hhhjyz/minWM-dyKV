# Experiment ledger

This file records the common protocol for minWM-dyKV experiments. Result fields are
left explicit rather than silently filled with unverified values.

## Reproducibility contract

Record the following for every run:

- git commit and dirty-worktree status;
- checkpoint path and checksum;
- prompt/case manifest and selected case IDs;
- seed, output latent-frame count, resolution, and camera trajectory;
- GPU model/count, PyTorch/CUDA versions, and peak allocated memory;
- wall-clock generation time and retrieval timing;
- dyKV memory budget and the resolved internal layout;
- output directory and evaluator report directory.

## Core comparisons

| Run | Method | Purpose | Status |
| --- | --- | --- | --- |
| B0 | upstream minWM local cache | quality/speed baseline | pending |
| B1 | dyKV without compression | isolate retrieval benefit | pending |
| B2 | complete dyKV | evaluate memory, speed, and quality | pending |

`B1` is a Python-level diagnostic preset, not another public CLI surface.

## Evaluation groups

1. Loop-closure camera paths: verify recovery of previously observed content.
2. Long monotonic paths: check that retrieval does not destabilize novel views.
3. MBench cases: report benchmark-native metrics and per-case artifacts.
4. Resource profile: peak VRAM, CPU bank bytes, retrieval time, and total latency.

## MBench protocol

- Use MBench-A official assignments; record the exact `samples.jsonl` checksum.
- Use the checkpoint-aligned 40/100 latent poses for 10s/25s cases and report
  their decoded 157/397-frame lengths alongside the official 161/401 targets.
- Keep case assignment, checkpoint, latent length, resolution, and seeds identical
  between B0/B1/B2.
- Register each method and seed as a distinct MBench `model_id`.
- Run contract validation before evaluation and record which metrics were skipped
  for missing DA3/VLM artifacts.

## Result template

| Commit | Run | Cases | Seed | Frames | Quality report | Peak VRAM | Time | Notes |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| pending | B0 | pending | 0 | pending | pending | pending | pending | baseline |
| pending | B1 | pending | 0 | pending | pending | pending | pending | retrieval only |
| pending | B2 | pending | 0 | pending | pending | pending | pending | complete method |

Do not replace `pending` with estimates. Only measured values belong in this table.
