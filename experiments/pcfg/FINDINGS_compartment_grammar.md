# Compartment-augmented PCFG — findings

Goal: extend the PCFG global grammar with (a) real Google **SegCLR** local
embeddings and (b) compartment production rules (axon↔dendrite crossing,
multi-soma) to detect false merges and propose splits. See the approved plan for
the full design.

## Exp 0 — SegCLR-only value probe (the "is this worth building?" test)

**Question (user's framing):** retrieve the MICrONS SegCLR embeddings and, *using
just SegCLR*, see the top-K for split errors — how valuable is SegCLR alone at
localizing the seam of a false merge?

**What was built.** `neuronauts/segclr.py` — a dependency-free loader for the
public SegCLR release (no `connectomics` package, no auth). It reads the CSV-zip
shards over HTTPS byte-range requests (only the ZIP central directory + one
segment's CSV per fetch, not the ~220 MB shard), reproduces the sharding
(`md5_shard`, 10 000 shards, **bytewidth = 64 bytes** — verified against real
shard membership), parses 64-dim embeddings with nm coordinates, caches per
segment, and assigns embeddings to points/vertices spatially.

**Verified facts (not assumed):**
- Sharding scheme confirmed by checking that every member of shards 0/1/2319/777
  hashes back to its shard with `bytewidth=64` (the `..._BYTEWIDTH64` scheme).
  `bytewidth=8` — the naive reading — is wrong and was ruled out empirically.
- CSV row layout: `node_id, x, y, z, e0..e63` (68 cols); coords in nm.
- **"m343" is a 2022 segmentation snapshot, NOT a queryable public
  materialization** (public exposes v1300+). So SegCLR segment ids do not map
  trivially to current CAVE roots — the version footgun is real. Exp 0 therefore
  runs fully inside m343 (SegCLR point clouds only); the m343→current bridge
  (via `seg_m343` volume or chunkedgraph) is the next milestone.

**Setup.** Take real m343 neurons (large embedding point clouds), union pairs
whose arbors actually pass near each other, build a spatial kNN graph, and label
each edge as a *seam* edge (cross-cell contact ≤1.5 µm) or *within* edge. Score
every edge purely by SegCLR discontinuity `1 − cos(emb_i, emb_j)`. Command:

```
python -m experiments.pcfg.run_compartment_grammar --exp0 \
    --shards 0 1 2 3 4 5 --n-neurons 24 --subsample 30000
```

**Result (7 real-contact pairs):**

| metric | value | reading |
|---|---|---|
| AUC (raw per-point) | **0.92 mean / 0.95 median** | SegCLR discontinuity is a *strong ranking signal* for seams |
| best-seam-edge percentile | **~2.8% median** | the first true seam edge sits ~top 3% of all edges |
| edge hit@100 / hit@500 | 0.00 / 0.29 | the exact seam edge is rarely in the absolute top-100 |
| **site hit@1..10** | **0.00** | the seam *neighbourhood* is not among the top ~40 high-discontinuity sites |
| euclid-pooled AUC | 0.22 | euclidean pooling **crosses** the seam and destroys signal — pool geodesically |

**Verdict.** SegCLR alone is a **strong global ranking signal** (AUC ≈ 0.95) but
**not a sufficient stand-alone top-K localizer**: within-cell embedding variation
(compartment transitions, caliber changes, myelination) produces the highest
discontinuities, so the true seam is buried ~3% deep and its site is never in the
top candidate sites. This is exactly the case for the **compartment-augmented
grammar** — the structural rules (A↔D-not-via-soma, multi-soma) and geodesic
(non-seam-crossing) pooling are needed to suppress the within-cell false
positives SegCLR cannot. SegCLR should enter as a strong *corroborating* term,
not the sole detector.

**Caveats (per CLAUDE.md — not over-claimed):** synthetic merges (real neurons,
real contacts, but geometric ground truth rather than a proofread seam); small n
(7 contacting pairs); 30k subsample lowers contact density; no real
proofread-merge validation yet (blocked on the m343→current bridge). The
numbers are indicative of value, not conclusive.

### Next
1. m343→current bridge (`seg_m343` via cloudvolume, or chunkedgraph `get_latest_roots`
   on m343 roots to find real split-corrected merges) → replace synthetic seams
   with real proofread ground truth.
2. Skeleton-based grammar (M1+ in the plan): compartment labels from synapse
   polarity + soma table, A↔D and multi-soma productions, geodesic-window pooling,
   split via `skeleton_cut_op`.
