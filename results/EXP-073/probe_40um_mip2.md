# EXP-073 — 40 µm probe on the near-isotropic (32×32×40 nm) substrate

*A probe, not the registered run. Run via `measure()` on
`data/substrate/c40um_mip2/` on 2026-09-02 while the full-cube mip-2
enumeration was still building. The registered run will land on the 100 µm
mip-2 substrate; this is recorded so the number does not live only in a
scratchpad. Dust floor: synapse-free objects under 0.041 µm³ (1,000 mip-2
voxels) excluded, synapse-carriers exempt.*

708 labelled atoms, 205 MST spanning links, 192,474 objects in the cube, 7,361
of them in some panel (radius 2 µm, cap 20).

| constraint on the bridge | chained recall (h≤3) | precision | reachable labelled (median) | objects allowed as bridge |
|---|---:|---:|---:|---:|
| none | 89.8% | 0.091% | 598 | 7,361 |
| cable (elongation ≥2, extent ≤30 µm) | 30.2% | 0.184% | 86 | 3,885 |
| + through ≥90° | 29.8% | 0.191% | 82 | 3,885 |
| + through ≥120° | 29.8% | 0.195% | 80 | 3,885 |
| + through ≥150° | 28.8% | 0.194% | 78 | 3,885 |
| tight (elongation ≥3, extent ≤15 µm, through ≥120°) | 27.3% | 0.195% | 74 | 1,978 |

**Bar (EXP-072's): recall > 50% AND reachable labelled ≤ 50.** No setting meets
it. Unconstrained meets recall and misses reachability by 12×; every constrained
setting misses recall by ~2× and still misses reachability.

## Reading

The `cable` filter does prune — reachable labelled atoms fall 7× (598 → 86) —
but it prunes **true links three times as fast as it prunes precision improves**:
recall drops 89.8% → 30.2% while precision only doubles (0.09% → 0.18%). Object
elongation does not separate a real bridge from an incidental neighbour; it
mostly removes the small objects that were doing the bridging in the first
place.

The `through` angle — the grammar rule proper, "the two fragments attach at
opposite ends of the bridge" — adds **nothing** on top of `cable`: recall 30.2 →
29.8%, reachable 86 → 82, across every threshold from 90° to 150°. At the object
level, with a centroid and a principal axis for the bridge, the attachment
geometry carries no usable signal.

## What this does and does not say

It falsifies the *cheap, object-level* form of the structural-constraint
hypothesis: shape descriptors of the bridge cloud (elongation, extent) and the
attachment angle at its centroid do not make the chained panel sparse at any
useful recall.

It does **not** test the skeleton-level constraints — tangent continuity at the
cut face, caliber continuity across the join, tree-ness of the assembled
result — which need the fragment's own skeleton and its endpoint, not a
centroid. Those remain the untested part of the hypothesis, and they are the
part that would have to work.

Together with EXP-071 and EXP-072, the arc on this substrate is: the population
omitted the connective cable (necessary fix); distance over the complete object
set collapses (not sufficient); cheap object-level structure does not rescue it.
The next lever is either an identity signal with no radius (EXP-057C, SegCLR)
or the skeleton-level constraints, computed only for the objects a small panel
implicates rather than for the whole cube.
