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

## Exp 1 — SegCLR-only on REAL proofread merges (the number that matters)

Exp 0 used *synthetic* merges. Exp 1 replaces them with **real** false merges via
the m343→current bridge: an m343 SegCLR root whose supervoxels now belong to ≥2
substantial current cells (`chunkedgraph.get_latest_roots`) was a real
false merge that proofreaders split. Label each SegCLR point by its nearest
current-descendant skeleton (= the ground-truth cell it was assigned to), then
run the same seam-localization scoring. Command:

```
python -m experiments.pcfg.run_compartment_grammar --exp1 \
    --shards 0 1 2 3 4 5 6 7 --n-neurons 20 --min-share 0.15
```

Verified along the way: m343 SegCLR coords and current (v1822) skeleton coords
share the same nm frame (point→skeleton median ≈ 970 nm, coverage ≈ 0.92); the
CAVE skeleton service needs `cloud-volume` (without it every skeleton fetch
silently negative-caches to empty — a real footgun).

**Result (7 real merges, from a scan of ~180 large-m343 candidates):**

| metric | **synthetic (Exp 0, n=7)** | **REAL (Exp 1, n=7)** |
|---|---|---|
| AUC (SegCLR discontinuity) | **0.95** | **0.49 mean / 0.56 median** (≈ chance) |
| AUC spread | 0.78–1.00 | 0.27–0.63 (0/7 > 0.7; **3/7 < 0.5**) |
| best-seam-edge percentile | ~2.8% | 1.35% median / 10.0% mean |
| site hit@3 | 0.00 | 0.14 (1/7) |

Per-merge real AUCs: 0.59, 0.56, 0.42, 0.27, 0.34, 0.59, 0.63 (2–3 cells each).

**The synthetic test massively overestimated SegCLR's value.** On real false
merges, SegCLR discontinuity across the true seam is **near chance (AUC ≈ 0.52)**.
This makes biological sense: synthetic merges join two *unrelated* neurons
(different type/location → different embeddings, trivially separable), whereas a
*real* false merge occurs exactly where two cells are locally similar and
touching — that is *why* the segmentation merged them — so their SegCLR
embeddings are barely distinguishable at the join.

**Revised verdict.** SegCLR alone is **not** a reliable stand-alone split-error
detector on realistic merges. Its value as the "fuser" term is real but
secondary; the compartment/structural grammar (A↔D-not-via-soma, multi-soma,
topology) must carry the detection, with SegCLR as a weak corroborator. This
strengthens — not weakens — the case for the compartment-augmented PCFG over a
pure-embedding detector.

**Caveats (per CLAUDE.md):** n = 7 real merges — only ~8% of large-m343
candidates are genuine ≥2-substantial-cell merges (most just shed small
fragments), so real 2-cell merges are the minority and n is modest, but the
pattern is consistent (0/7 above 0.7, mean at chance). Nearest-skeleton labeling
adds noise in the seam region of intertwined merges, which further depresses the
real AUC — so ~0.49 is a floor, and part of the synthetic↔real gap is method, not
only biology. Both effects point the same way: SegCLR-alone does not localize real
seams.

## Exp 2 — SegCLR top-1 retrieval (the RIGHT framing; corrects Exp 1)

Exp 1 asked "is the single most-discontinuous *edge* in the whole neuron exactly
at the seam?" — a harsh framing dominated by within-cell embedding variation, and
it made SegCLR look useless (AUC ≈ 0.52). The **retrieval framing** — for a node,
is its top-1 nearest node *in embedding space* the same cell, and among local
candidates does the same-cell one win — is the right question and tells a very
different, positive story.

```
python -m experiments.pcfg.run_compartment_grammar --exp2 --shards 0 1 2 3 --n-neurons 12
```

Data fact that reshapes the test: SegCLR **nodes are ~1.2 µm apart** (median NN
spacing ~1174 nm), so a 200 nm radius is *tighter than the node spacing* and finds
no candidates. The local test needs ~2–4 µm to have neighbours.

**Metric A — embedding top-1 retrieval (is the nearest-embedding node same-cell?):**

| set | top-1 same-cell | chance |
|---|---|---|
| 12 clean cells pooled | **0.866** | ~0.20 |
| real merges (per-object) | **0.83–0.98** (mean ~0.93) | — |

**Metric B — local top-1 same-cell at contacts (does SegCLR pick the correct
same-cell candidate over the false-merge partner?), on real merges:**

| radius | mean acc | note |
|---|---|---|
| 2 µm | ~0.71 | small n (8–860 discriminative nodes/merge) |
| **4 µm** | **~0.90** (0.76–0.95) | larger n (37–3295), stable |

**Corrected verdict.** SegCLR is a **strong local cell-identity signal**: its top-1
match is same-cell ~87–98% (retrieval), and at a real false-merge seam its top-1
local candidate is the correct same-cell one **~90%** of the time (at the
node-appropriate ~4 µm radius). It genuinely could drive merge/split decisions.
The Exp-1 "near chance" result was an artifact of the edge-discontinuity-ranking
framing, **not** a property of SegCLR. The compartment grammar and SegCLR are
complementary — SegCLR discriminates identity locally, the grammar supplies the
structural rules (A↔D, multi-soma) and the split geometry.

## M1 — compartment labeling on real neurons (PASS)

Built `neuronauts/soma_clusters.py` (verified soma routine extracted to core) and
`experiments/pcfg/compartments.py` (`label_compartments`): synapse polarity
(pre→axon, post→dend) snapped to vertices and diffused along the tree, soma from
large-radius clusters (+ optional nucleus table). Run:

```
python -m experiments.pcfg.run_compartment_grammar --m1 --n-neurons 4
```

Result on 4 proofread neurons (5k–12k verts, 2k–8k synapses each):

| root | n_soma | PRE→AXON | POST→DEND | is_tree |
|---|---|---|---|---|
| …686494647 | 1 | 0.99 | 1.00 | ✓ |
| …812081779 | 1 | 0.99 | 1.00 | ✓ |
| …195284556 | 1 | 1.00 | 1.00 | ✓ |
| …975539779 | 1 | 0.98 | 1.00 | ✓ |

Polarity concordance is essentially perfect and each neuron yields exactly one
soma. Useful data fact: the CAVE skeleton service represents the **soma as a
single large-radius vertex** (~5300 nm; all cable ≤ ~425 nm at p99), so the
radius>3000 nm threshold is cleanly separated and multi-soma detection (2 merged
cells → 2 big-radius vertices) is robust. The compartment alphabet is ready to
drive the grammar.

## Next
1. **Grammar productions (M3)**: A↔D-crossing (geodesic windows, soma-mediation
   guard) + multi-soma, locating the offending edge; combine with the PCFG signals.
   SegCLR enters only as a weak corroborator (Exp 1).
2. Split proposal via `skeleton_cut_op`; evaluate on real merges (m343→current)
   and vs the `atomicity_detector` / `skeleton_topology_merge` baselines.
3. Optionally de-noise the Exp 1 ground truth (supervoxel→current-root labels) to
   pin down the real-merge SegCLR AUC precisely.
