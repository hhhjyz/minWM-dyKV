# minWM-dyKV documentation

This directory is the source of truth for the dyKV implementation and experiments.
Each implementation module has one focused document; `EXPERIMENTS.md` is the shared
experiment ledger and must be updated whenever an experiment is added or run.

## Design constraints

The old prototype exposed many independent switches. minWM-dyKV instead uses one
coherent preset:

- attention layout: `sink | retrieval | local`;
- evicted clean KV is retained in a CPU bank;
- candidates are ranked by camera FOV overlap;
- retrieved KV is compressed only when it is materialized for attention;
- all three regions are rebased into the model's trained temporal RoPE range.

The public inference interface intentionally contains only two dyKV controls:

- `--dykv`: enable the complete method;
- `--dykv-memory-frames`: set the retrieved-memory budget.

Implementation constants are centralized in one typed configuration object. This
keeps ablations possible from Python without turning every internal choice into a
command-line hyperparameter.

## Module documents

- [`KV_MEMORY.md`](KV_MEMORY.md): eviction storage, retrieval payloads, and retrieval-time compression.
- [`TRI_REGION_ROPE.md`](TRI_REGION_ROPE.md): bounded temporal position layout.
- [`FOV_RETRIEVAL.md`](FOV_RETRIEVAL.md): HY-WorldPlay-compatible FOV scoring and selection.
- [`MBENCH.md`](MBENCH.md): case conversion, generation, and evaluation workflow.
- `EXPERIMENTS.md`: experiment matrix, commands, environment, and recorded results.

## Reference implementations

- `../minWM-back`: prior prototype and experiment history;
- `../WorldKV`: KV-bank retrieval and anchor/novelty compression;
- `../Anchor-Forcing`: bounded tri-region temporal layout;
- `../HY-WorldPlay`: FOV-overlap memory selection;
- `../MBench`: benchmark case and evaluation contracts.

Reference paths above describe the sibling workspace used during development; this
repository does not import code from those projects at runtime.
