# Tri-region temporal RoPE

## Layout

Once generation moves beyond the 20-frame training horizon, dyKV maps every
attention operand back into a fixed virtual timeline:

```text
0         1 ........ 8       9..11       12 ... 15       16 ... 19
[ sink ]  [ retrieval ]      [ gap ]     [ recent local ] [ current ]
```

The gap is intentional. It separates selected long-term memory from the immediate
local trajectory and leaves the final current-query positions identical for every
long-horizon step.

Before frame 20, minWM uses its ordinary monotonic RoPE and dyKV retrieval is not
activated. At and after the boundary:

- sink K remains at its original position 0;
- selected memory chunks are packed chronologically from position 1;
- the four recent frames are mapped to positions 12--15;
- a four-frame current query and its K are mapped to positions 16--19.

## Rebase operation

Cached K and the current Q have already received RoPE. Rebasing therefore multiplies
only their complex temporal channels by the relative rotation for
`target_position - source_position`. Spatial height/width channels are unchanged.
All operations clone their inputs, preventing repeated denoising calls from
accumulating rotations in the cache or memory bank.

Compressed retrieval chunks are rebased as units. Each retained token keeps the
temporal offset encoded before compression, while the chunk's source origin moves
to its assigned virtual origin.

## Invariants

- every live-cache slice is frame-aligned;
- query, recent, retrieval, and sink ranges cannot overlap;
- raw selected memory cannot exceed the configured memory-frame budget;
- shifts beyond the available RoPE frequency table fail explicitly;
- warm-up and activation use the same 20-frame boundary.

The implementation is based on the prior minWM tri-region prototype and
Anchor-Forcing's bounded long-horizon RoPE policy, with one fixed layout instead of
separate sink/retrieval/rebase switches.
