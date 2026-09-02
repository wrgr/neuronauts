# EXP-072 — dust-floor sweep on the near-isotropic (32×32×40 nm) 40 µm substrate

*A probe, not the registered run: `measure()` on `data/substrate/c40um_mip2/`,
2026-09-02, sweeping the physical dust floor while the full-cube mip-2
enumeration was still building. Recorded here so the numbers do not live only
in a scratchpad. Radius 2 µm, panel cap 20, chained recall within 3 hops; 708
labelled atoms, 205 MST spanning links.*

The question was whether the panel's collapse — recall high, precision near
zero, most labelled atoms mutually reachable — was being driven by debris. The
enumeration touches 192,474 v117 root ids in this 40 µm cube, most of them
sub-micron specks that own no synapse. A floor in **physical** units (so it
means the same thing at any read mip) drops synapse-free objects smaller than
the threshold; synapse-carrying objects stay regardless, because a synapse is
label-blind evidence of real structure.

| floor on synapse-free objects | objects kept | widened: recall | precision | reachable labelled (median) | population-only: recall / precision / reachable |
|---|---:|---:|---:|---:|---|
| none | 192,474 (100%) | 85.4% | 0.09% | 566 | 92.2% / 0.09% / 634 |
| 0.01 µm³ | 52,131 (27.1%) | 88.3% | 0.09% | 589 | same |
| **0.041 µm³ = 1,000 voxels at 32×32×40 nm** | 35,610 (18.5%) | 89.8% | 0.09% | 598 | same |
| 0.1 µm³ | 30,071 (15.6%) | 90.7% | 0.09% | 608 | same |
| 0.5 µm³ | 24,890 (12.9%) | 92.2% | 0.09% | 624 | same |

Direct recall at cap 20 rose from 11.7% (no floor) to 16.6% (≥0.5 µm³) — the
population-only figure — because specks stopped taking panel slots.

## Reading

**Dust was not the cause.** Removing 87% of the objects in the cube moves
precision by nothing — 0.09% at every floor — and the reachable set by 10%. At
0.5 µm³ only 860 synapse-free objects survive and the widened substrate becomes
the population-only substrate to the digit: 92.2% / 0.09% / 624 vs 634.

What collapses the panel is the **synapse-carrying population itself**. At a
2 µm radius in dense neuropil, the 24,030 objects that own a synapse are enough
to connect nearly every labelled atom to nearly every other within three hops
of panel edges. That is the quantity no size floor can touch.

**The floor stays.** 0.041 µm³ (1,000 voxels at 32×32×40 nm) is the canonical
setting in `exp072_object_proposal.MIN_VOLUME_UM3` and is inherited by
EXP-073: it is correct hygiene, it recovers ~5 points of direct recall for
free, and it costs nothing. It is not a lever on the result this experiment
is measuring.

## Together with the rest of the day

EXP-071 showed the population omits the connective cable (necessary fix).
EXP-072 showed distance over the complete object set collapses (not
sufficient). This sweep shows the collapse is not debris. EXP-073's probe on
the same substrate showed cheap object-level structure (elongation of the
bridge, attachment angle) prunes true links as fast as false ones. The panel
problem on this substrate is not a substrate-hygiene problem; it needs either
an identity signal with no radius (EXP-057C) or the skeleton-level constraints
computed only for a small panel's members.
