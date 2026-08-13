# MBench-A adaptation

minWM-dyKV supports MBench-A's action-conditioned cases. MBench-T uses changing
text segments over time and is outside the current minWM inference contract.

## What is adapted

`mbench_adapter.py prepare` reads the benchmark's model-centric assignment manifest
and joins each `subset/sample_id/condition_id` with the caption in
`samples/{subset}/{sample_id}/sample.json`. It writes three aligned files:

```text
work_dir/
├── prompts.txt
├── trajectories.txt
└── cases.jsonl
```

The action mapping is:

| MBench-A action | minWM trajectory |
| --- | --- |
| left then right | yaw left, yaw right, optional static pad |
| right then left | yaw right, yaw left, optional static pad |
| forward then backward | forward, backward, optional static pad |
| left/right 360, 720, 1080 | scaled yaw completing the requested angle |
| static | zero camera motion |

The trajectory parser supports `n*N` and scaled steps such as `j@2.5*40`. Every
adapter trajectory is checked by tests to contain exactly the requested number of
latent camera poses.

MBench lengths describe decoded video frames. Wan's VAE expands time by four, while
the causal checkpoint requires a multiple of four latent frames. The closest lower
valid lengths are therefore 40 latent frames (157 decoded frames) for the official
10-second/161-frame condition and 100 latent frames (397 decoded frames) for the
25-second/401-frame condition. The default runner uses 100; record this small
duration difference in benchmark reports.

## Generate and package

```bash
MBENCH_ROOT=/absolute/path/to/MBench-A-Setup \
ASSIGNMENTS=/absolute/path/to/official/samples.jsonl \
MODEL_ID=minwm_dykv_seed0 \
NUM_OUTPUT_FRAMES=100 \
DYKV_MEMORY_FRAMES=8 \
SEED=0 \
bash Wan21/scripts/inference/run_mbench_dykv.sh
```

Useful filters are `SUBSETS`, `CONDITIONS`, and `LIMIT`. If `ASSIGNMENTS` is omitted,
the adapter uses `models/hy_worldplay/samples.jsonl`, the official MBench-A
assignment source. Pass an explicit manifest for the four-case demo or any custom
case set.
The adapter rejects a 10s case unless `NUM_OUTPUT_FRAMES=40`, and a 25s case unless
`NUM_OUTPUT_FRAMES=100`, preventing a mislabeled benchmark package.

Generation writes `generation_manifest.jsonl`. Packaging then creates:

```text
MBench-A-Setup/models/{MODEL_ID}/
├── samples.jsonl
└── outputs/{subset}/{sample_id}/{condition_id}/video.mp4
```

Videos are relative symlinks by default; use `LINK_MODE=hardlink` or `copy` when the
dataset will move to another filesystem.

## Evaluate

From the MBench environment:

```bash
mbench validate "$MBENCH_ROOT" \
  --models minwm_dykv_seed0 \
  --metrics mbencha.entity.human_identity_consistency \
  --limit 2

mbench eval "$MBENCH_ROOT" \
  --models minwm_dykv_seed0 \
  --metrics mbencha.entity.human_identity_consistency,mbencha.environment.rendering_lighting \
  --output runs/minwm_dykv_seed0
```

Spatial, object-geometry, rendering-style, and camera-interaction metrics require
the benchmark's external DA3 artifact at the path declared in `dataset.yaml`.
State-progress metrics may additionally require the configured VLM judge. These
artifacts are evaluation inputs and are deliberately not fabricated by the adapter.

## Current limitation

The four-frame dyKV checkpoint path is T2V, so MBench captions are used as generation
conditions. The benchmark's first-frame assets are preserved in the dataset for
evaluation but are not injected into generation. Report this distinction alongside
results when comparing against first-frame-conditioned world models.
