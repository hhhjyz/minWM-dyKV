# FOV retrieval

## Metric

dyKV uses the retrieval policy from HY-WorldPlay. For a current camera pose `C` and
historical pose `H`, it samples points in a radius-8 sphere around `C` and computes:

```text
overlap(C, H) = count(points in FOV(C) and FOV(H)) / count(points in FOV(C))
distance(C, H) = 1 - overlap(C, H)
```

The angular frustum uses a 60-degree horizontal and 35-degree vertical field of
view. Historical points farther than radius 8 from the historical camera are not
counted.

For chunk retrieval, each current frame is compared with the first and midpoint
poses of a historical chunk. Those two distances are averaged, followed by an
average over current frames. Complete chunks are selected by ascending distance
until the raw memory-frame budget is filled.

## Deterministic probes

HY-WorldPlay draws random Monte Carlo points. dyKV replaces that sampling call with
a deterministic golden-angle sphere sequence with volume-corrected radii. This
preserves the geometric estimator while ensuring that:

- candidates receive comparable scores within a run;
- seeds used for video generation do not alter retrieval decisions;
- unit tests and experiment reruns are reproducible.

The probe tensor is generated once per inference run and reused across blocks.

## Candidate boundary

FOV scoring is applied only after `DyKVBank.evicted_candidates` removes blocks still
present in the live sink or recent cache. Blocks without camera matrices are skipped
instead of silently falling back to a second retrieval metric.

## Validation

Tests verify deterministic bounded probes, near-complete overlap for identical
cameras, near-zero overlap for an opposite-facing camera, loop-closure preference,
chronological payload order, and strict adherence to the memory-frame budget.
