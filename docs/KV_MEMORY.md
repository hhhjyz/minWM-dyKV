# KV memory and retrieval-time compression

## Goal

The live causal cache is still fixed-size and fast. Before a clean generated block
can be rolled out of that cache, dyKV copies its per-layer K/V to a CPU-side bank.
Only blocks that have left both the sink and recent-local regions are eligible for
retrieval, preventing duplicate attention to the same frame.

## Lifecycle

1. The model finishes denoising a block.
2. A final clean forward pass commits that block to the live KV cache.
3. `DyKVBank.archive_clean_block` copies the new tail slice for every transformer
   layer to the bank device.
4. Later blocks query `evicted_candidates`; FOV selection chooses from this set.
5. `materialize` moves only selected K/V to the attention device and compresses it.
6. The attention layer consumes the payload as the middle retrieval region.

The bank stores uncompressed clean K/V. This makes selection reversible and avoids
spending compression time or quality on history that is never retrieved.

## Compression

Compression follows WorldKV's anchor-plus-novelty rule. For each generated chunk:

- retain the first latent frame in full as the anchor;
- compute the mean anchor key across spatial tokens and attention heads;
- rank each later frame's tokens by cosine similarity to that centroid;
- retain the least-similar half, which represents content not already covered by
  the anchor.

With a four-frame chunk and ratio `0.5`, the stored 4-frame block is materialized as
`1 + 3 * 0.5 = 2.5` frame-equivalents of attention tokens. The raw memory-frame
budget is unchanged; compression changes the attention cost, not which frames were
selected.

## Public configuration

The CLI exposes only `--dykv` and `--dykv-memory-frames`. The fixed method preset is:

| Setting | Value |
| --- | ---: |
| sink frames | 1 |
| recent frames before current chunk | 4 |
| retrieval compression keep ratio | 0.5 |
| bank device | CPU |

`DyKVConfig.validate` rejects budgets that do not align to the model chunk size or
cannot fit in the trained RoPE range.

## Validation

Unit tests cover block-tail capture, eviction eligibility, anchor preservation,
novelty selection, chronological payload order, and the guarantee that retrieval
compression does not mutate the lossless bank.

## Running

Use the ordinary causal camera runner with the complete preset enabled:

```bash
DYKV=1 DYKV_MEMORY_FRAMES=8 \
  bash Wan21/scripts/inference/run_infer_causal_camera.sh
```

The output folder contains `dykv_summaries.jsonl`, with one record per prompt. Each
record includes bank byte counts, selected block IDs, source frame starts,
compressed token counts, and retrieval wall time.
