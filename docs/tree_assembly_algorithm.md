# A tractable global/hierarchical tree assembly algorithm

> **Status: proposal (2026-08-20).** Synthesizes the Phase 1–2.12 results
> (`STATUS.md`), the lineage approach (`docs/lineage_approach.md`), and the
> roadmap (`docs/roadmap_global_assembly.md`) into a concrete algorithm that
> scales past the single-bbox regime where the current pipeline works.

## 1. The wall, stated precisely

The current pipeline (EdgePartitionGNN + GAEC over an observation graph of
synapses, `treestitch/` + `neuronauts/assemble/edge_partition.py`) works at
bbox scale and fails to extend globally for four distinct, *measured* reasons:

1. **The flat-graph scale wall.** edge_cc is validated on 10–20k synapses per
   bbox. `synapses_pni_2` is ~337M rows. One flat observation graph over the
   volume is neither buildable nor trainable, and the GNN's receptive field is
   bounded anyway. Union-find collapses outright at 110+ neurons (ARI → 0).
   The box ceiling the roadmap diagnosed for CellGNN reappears in treestitch as
   the sparse-box caveat: only 38–72% of assembled shapes are fully connected
   because same-neuron fragments outside the bbox are simply absent (Phase 2.7,
   2.10). **There is still no cross-region stitching in the active path.**

2. **The small-fragment representation wall.** DNA embeddings discriminate
   *individuals* at half-skeleton scale (within-type AUC 0.768→0.829, GNN
   encoder) and collapse at quarter-skeleton scale under every encoder and
   every loss tried (triplet, NT-Xent; Phase 1 / within-type ablations). This
   is a data limitation — small fragments genuinely carry too little
   morphological identity — so no encoder/loss change will fix it.

3. **The frankenmerge generalization wall.** In-sample fk_split reaches
   0.73–1.00, but the leak-fixed out-of-sample number is **0.000** (Phase 2.8,
   2.11). Whether a v117 root is a frankenmerge is a fact about the local
   proofreading history, not a spatially transferable signature. Learning to
   *detect and cut* frankenmerges out-of-region is a dead end on current
   evidence.

4. **The see-through wall.** Purely pairwise evidence cannot connect fragments
   A and C through an ambiguous middle fragment B (`NEXT_STEPS.md` §4). Any
   flat pairwise formulation inherits this.

Meanwhile, four things are *proven* and should be kept:

- **ARI/merge_P generalize spatially.** Out-of-sample ARI 0.75–0.92 with
  merge_P ≈ 0.95–0.98 across 7 test locations; merge_P std = 0.01 (Phase
  2.11–2.12). The *merge* decision transfers; the *cut* decision does not.
- **Endpoint-adjacency evidence is transformative at fine scale**
  (ARI 0.09→0.42 with skeleton pieces; 0.31→0.84 when L2 skeletons restored
  real endpoints). Geometry, not DNA, is the strong signal between small
  pieces.
- **GAEC's structural refusal to over-merge** (over-merge rate ~13–15× lower
  than union-find in every regime).
- **Kruskal endpoint stitching guarantees trees** (is_tree = 1.000
  unconditionally, `treestitch/assemble.py`).

## 2. The organizing idea: match evidence to scale, merge-only, tree-constrained

Every wall above is a scale-mismatch: we ask fine-scale objects for
coarse-scale evidence (DNA identity from quarter-skeletons), ask a local model
for a global decision (flat graph), and ask a region-specific signal to
transfer (frankenmerge cuts). The fix is a hierarchy in which each level uses
only the evidence class that is *proven* to work at that level's object size,
and in which **cuts happen only once, at the bottom** — everything above is
monotone agglomeration.

```
Level −1  ATOMIZE      v117 roots → branch-level atoms (L2 skeleton splits)
                       cuts happen here and never again
Level 0   TILE ASSEMBLY EdgePartitionGNN + GAEC per overlapping tile
                       evidence: same-fragment lineage, spatial kNN,
                       endpoint adjacency  (the proven bbox pipeline)
Level k   SEAM STITCH   super-fragments from level k−1; tiles 2× linear size
  (k≥1)                evidence: exact halo identity, endpoint matching,
                       pooled DNA (objects now big enough), soma cannot-links
                       inference: constrained maximum-weight forest (Kruskal)
```

### 2.1 Level −1: atomization converts the split problem into a merge problem

Split every v117 root into **branch atoms** along its L2-cache skeleton (cut at
branch points and at unusually long skeleton edges — the exact places
frankenmerges are glued). Consequences:

- A frankenmerge no longer needs to be *detected*: its two halves become
  separate atoms, and they simply fail to re-merge under the same
  merge-evidence model that *does* generalize (ARI transfers; fk_split
  doesn't). Bar 3 stops being a detection problem and becomes an abstention
  outcome of the merge model. This is the classic over-segment-then-agglomerate
  move, chosen here because our measurements say merges transfer and cuts
  don't.
- Above level −1 the algorithm is **monotone**: no operation ever needs to undo
  a previous one. Under-merge at level k is recoverable at level k+1 with more
  context; over-merge never is. Monotone agglomeration matches the asymmetric
  error cost (false merge ≈ 5–10× a missed merge) structurally, not just via a
  bias knob.
- Cost of atomization: same-root atoms that *should* re-merge now depend on the
  fine-scale merge evidence. That evidence is precisely the strong one at this
  scale: atoms from one root share supervoxel lineage (near-free positive
  evidence) and have adjacent skeleton endpoints (the +0.33/+0.53 ARI signal).

### 2.2 Level 0: the existing bbox pipeline, unchanged in kind

Overlapping tiles (~100–150 µm core + ~20 µm halo — the dense-box regime that
passed Bars 1+2 out-of-sample). Within each tile: observation graph over atoms
(edge types 0/1/2 as today), EdgePartitionGNN, GAEC at a conservative bias,
temperature-calibrated (`treestitch/calibration.py`), risk-labelled
(`treestitch/risk.py`). Output per cluster is a **super-fragment**:

- merged L2 skeleton via Kruskal (`merge_fragment_skeletons` — tree by
  construction), with its endpoint set and endpoint tangents;
- pooled DNA (synapse-weighted mean of member-fragment embeddings);
- synapse roster + pre/post polarity counts + spatial extent;
- soma flag (`detect_soma` / nucleus table membership);
- calibrated confidence and abstained edges carried as review items.

### 2.3 Level k ≥ 1: seam stitching as a constrained maximum-weight forest

Nodes are super-fragments from level k−1; tiles double in linear size each
level, so the hierarchy is O(log L) deep. Only two edge sources exist, both
sparse and seam-local:

1. **Exact halo identity (forced merges).** Two super-fragments in adjacent
   tiles that share an atom (same L2 node / supervoxel) in the halo are the
   same object — merge with no model involved. Halos exist precisely to make
   most cross-tile decisions exact. Dedup via the stable synapse/atom keys
   (`experiments/minnie_column/dedup.py` pattern).
2. **Candidate stitch edges.** Endpoint pairs across a seam within radius r
   (10 µm, the validated endpoint-adjacency scale), scored by a small learned
   classifier over pair features:
   `(endpoint_gap, tangent_alignment, pooled_dna_cosine, lineage_agreement,
   size/polarity compatibility, calibrated level-(k−1) confidence)`.
   This is exactly the seam-stitch classifier of
   `docs/roadmap_global_assembly.md` §3.2 — with the crucial change that its
   DNA input is now **pooled over a half-skeleton-or-larger object**, i.e. in
   the regime where DNA is proven to discriminate individuals (wall #2
   dissolves because the hierarchy only asks DNA questions of large objects).

**Inference is not general clustering.** Because every node is a tree with
explicit endpoints, assembly at level k is a **maximum-weight spanning forest
with endpoint-degree constraints**: sort positive-scored candidate edges by
score, accept greedily (Kruskal) subject to
   (a) union-find cycle rejection — the union of trees stays a forest,
   (b) each endpoint used at most once — biological continuation, not a hub,
   (c) hard cannot-link constraints (below).
This is near-linear (α(n) union-find), globally consistent within the level,
and `is_tree = 1.000` holds *by construction* rather than as a post-hoc
property. `neuronauts/assemble/fragment_graph.py` (endpoint stitching, degree
cap, union-find) is the seed implementation.

**Cannot-link constraints — the qualitative checklist becomes hard priors.**
At coarse levels the "does this look like a neuron" checks
(`docs/lineage_approach.md`) stop being diagnostics and become constraints the
forest search must respect:

- **One soma per neuron** (nucleus table): two super-fragments that each
  contain a soma may never merge. This is the single strongest global prior we
  have, it is exact, and it directly caps the worst over-merge chains — the
  failure that killed union-find at scale. It is also the coarse-scale
  frankenmerge backstop.
- **Synapse-count cap** (~50k post-synapses) and **polarity sanity**: reject
  merges producing implausible totals.
- **Spatial compactness**: reject merges whose union is bimodal with a large
  empty gap (over-merged axon-bundle signature).

**See-through resolved by pooling.** A weak middle fragment B no longer blocks
A–C: at the level where A and C's clusters both abut B's region, the decision
is made against *pooled* prototypes (the EM-style running-mean idea of
`NEXT_STEPS.md` §4, realized as the natural node representation of the
hierarchy rather than a separate E/M loop).

### 2.4 Training: same free supervision, one label rule per level

Every level trains from the identical v117→v1718 lineage signal: a pair of
level-k objects is positive iff their majority v1718 roots agree. Level 0 is
the already-trained EdgePartitionGNN. The level-k stitch scorer is a small MLP
over ~10 pair features — cheap, and trainable on seam pairs harvested from a
handful of tile pairs (supervision is free wherever proofreading touched the
seam). Calibrate per level with the existing temperature-scaling module;
the risk layer's CONFIDENT/REVIEW/ABSTAIN decisions apply at every level, so
the human-review product (ranked uncertain stitches) falls out of the same
machinery.

## 3. Why this is tractable

- **Compute.** Per-tile cost is bounded (the validated regime). Levels are
  O(log L); level-k node count shrinks roughly geometrically; seam candidate
  edges are sparse (endpoints within 10 µm of a seam). Total work is
  ~O(N log N) in synapses, embarrassingly parallel per tile at every level,
  and streamable — no step ever holds the global graph.
- **Statistics.** Each decision uses evidence proven at its scale: lineage +
  endpoint geometry between small things, learned edge classification at bbox
  scale, pooled DNA + exact global priors (somata, counts) between big things.
  No component is asked to do the thing it measurably cannot.
- **Failure containment.** Merge-only monotonicity + conservative bias +
  cannot-links means errors accumulate as *recoverable* under-merges (flagged
  as REVIEW stitches), not irreversible over-merges. merge_P — the load-bearing
  metric — is protected at every level, and its measured spatial stability
  (std 0.01) is the property the whole design leans on.

Relation to prior art: levels 0–k are a lifted-multicut / GASP-style
hierarchical agglomeration (Beier et al., Nat. Methods 2017), specialized by
(i) tree-constrained forest inference instead of general clustering, and
(ii) supervision from segmentation version history instead of voxel labels.

## 4. Smallest experiments that answer the question

1. **Two-level stitch demo (the box-ceiling kill shot).** 2×2 adjacent dense
   tiles + halos; run the existing pipeline per tile; one level-1 pass = exact
   halo identity + constrained Kruskal over seam endpoint pairs (score with the
   *existing* `score_edge` geometry before training anything). Metrics:
   fully-connected fraction of cross-seam neurons (expect ~40% → ≥80%),
   ARI over the union, merge_P stays ≥ 0.95. New code: a seam-edge harvester +
   the degree/cycle/cannot-link-constrained Kruskal (~200 lines around
   `fragment_graph.py` / `assemble.py`).

   **→ RUN (2026-08-20), synthetic worlds.** Implemented as
   `treestitch/stitch.py` + `scripts/two_level_stitch.py` (level 0 = the real
   FragmentEncoder → EdgePartitionGNN → GAEC pipeline per tile). Findings:

   | Config (24 obj × 4 pieces) | ΔARI | Δmerge_P | multi-tile assembly |
   |---|---|---|---|
   | 40 µm halo, exact channels only | **+0.031** | **+0.004** | **0% → 100%** |
   | seeds 1/2, same config | +0.021 / +0.045 | −0.042 / 0.000 | 83% / 100% |
   | 40 µm halo + geometry endpoint edges @0.05 | −0.009 | −0.063 | 0% → 100% |
   | zero halo (endpoint channel only) | −0.292 | −0.322 | 0% → 100% |
   | frankenmerge 0.15, shared-obs links only | +0.014 | −0.024 | 0% → 60% |
   | frankenmerge 0.15, + shared-atom links | −0.007 | −0.074 | 0% → 60% |

   Three design claims confirmed, one placeholder falsified:
   - **The exact halo-identity channel alone delivers the box-ceiling win**
     (100% multi-tile assembly at unchanged merge precision) — no model at
     level 1.
   - **Halos are load-bearing twice**: they make the exact channel dense *and*
     give tiles the context to partition well (zero-halo baseline merge_P
     collapses 0.92 → 0.66 before any stitching).
   - **The frankenmerge caveat on atom links is real**: under frankenmerge
     pressure, shared-*atom* forced merges cost 3× more precision than
     shared-*observation* merges (−0.074 vs −0.024). Atomization (level −1)
     or obs-only linking is the answer, as designed.
   - **Geometry-only endpoint scoring is not deployable** (stitch-edge
     precision 0.17–0.67): proximity × pooled-DNA-compat cannot separate
     adjacent objects. The endpoint channel needs the learned stitch scorer
     (experiment 4) before it earns weight; until then it defaults to a
     conservative min-score.

   **→ RUN (2026-08-20), real data.** Same script on the Phase-2.x dense bbox
   (x 950–1150k, y 930–1000k, z 780–880k nm; v117→v1718): 20k fetched
   synapses → 1826 after the sliver filter, 288 v117 fragments, 312 neurons,
   **19 real frankenmerges**, L2 skeletons 288/288. 2×2 tiles on x/z, 20 µm
   halo, per-tile training only (~800 obs & 100 epochs per tile — well below
   the Phase-2.6 regime, so level-0 partitions are weaker than the Phase
   numbers; metrics here are all-pairs pairwise P/R, not the Phase scripts'
   edge-level merge_P, and are not directly comparable):

   | Channels | ΔARI | Δmerge_R | Δmerge_P | multi-tile assembly (241 objects) |
   |---|---|---|---|---|
   | shared-obs only | +0.101 | +0.162 | **−0.007** | 0% → 22.0% |
   | + shared-atom | **+0.132** | **+0.331** | −0.105 | 0% → **50.6%** |

   The real-data run confirms the synthetic story end-to-end: the exact
   shared-observation channel improves global assembly at *unchanged*
   precision, while the shared-atom channel trades precision for a much
   larger recall/assembly gain — the cost concentrating exactly where the 19
   frankenmerges and level-0 impurity live. That trade is the concrete,
   measured argument for level −1 atomization: once atoms cannot span
   neurons, the atom channel's recall comes without its precision bill.
   Remaining gap to 100% assembly is level-0 under-merging (847 super-
   fragments for 312 neurons), i.e. more/better level-0 training and the
   learned stitch scorer — not the stitch machinery itself.

   **→ REPLICATION + odd-skip (2026-08-20, second independent run, same
   200 µm box).** All numbers reproduce within noise, and the new
   `frankenmerge_separation` metric quantifies the re-gluing directly:

   | Stitch config | ΔARI | Δmerge_P | assembly | fk re-glued (of 7 sep.) |
   |---|---|---|---|---|
   | obs-only | +0.108 | −0.001 | 23.7% | 1 |
   | + atom links (all) | +0.135 | −0.122 | **54.8%** | **4** |
   | + atom links, **odd-skip** | +0.089 | −0.039 | 25.7% | 1 |

   The three configs are a precision/recall dial, and the atom channel's
   precision bill is now shown to *be* the frankenmerge re-gluing. odd-skip
   keeps obs-only's safety but — because the current oddness thresholds flag
   85% of real fragments — it drops 597 of 704 atom links and with them most
   of the assembly gain. Calibrating oddness on real data (so it flags ~the
   true frankenmerge rate, ~11% here) is the single knob expected to move
   odd-skip toward atom-links' recall at obs-only's precision.
2. **Atomization A/B.** Rerun the region benchmark with v117 roots pre-split
   into L2-branch atoms. Success: out-of-sample frankenmerge halves end up in
   different clusters (the Bar-3 outcome) *without* any fk-detection features,
   at ≤0.02 ARI cost from over-fragmentation.

   **→ RUN (2026-08-20), synthetic (24 obj × 4 pieces, frankenmerge 0.15,
   7 franken fragments).** Implemented as `treestitch/atomize.py` with three
   strategies, because branch-splitting alone is *not* enough: the real
   frankenmerge glue is an abnormally long L2-MST bridge edge sitting
   mid-path, and in the synthetic generator it is a disconnected component —
   neither is at a branch point. `oddness_scores` flags both signatures
   (odd-edge and multi-component), label-free. `frankenmerge_separation`
   (fk_sep) measures whether franken halves end in disjoint clusters:

   | Treatment | ARI (stitched) | Δmerge_P | assembly | fk_sep |
   |---|---|---|---|---|
   | none | 0.670 | −0.035 | 20% | **0.429** ← atom links glue halves |
   | `--odd-skip` (distrust odd ids) | 0.679 | −0.006 | 20% | **1.000** |
   | `--atomize branch` | 0.957 | +0.004 | 100% | 1.000 |
   | `--atomize shatter` (branch + odd edges) | 0.957 | +0.004 | 100% | 1.000 |
   | `--atomize odd` (split only flagged, 89→96 atoms) | **0.957** | **+0.004** | **100%** | **1.000** |

   Atomization also lifts the *level-0 baseline* (ARI 0.67 → 0.93): cleaner
   same-atom evidence helps before any stitching. The surgical `odd` mode —
   shatter only the fragments the label-free detector flags — matches full
   shatter at 1/5th the atom count. And `odd-skip` alone (no splitting)
   recovers fk_sep 0.429 → 1.000 at nearly-zero precision cost, so "skip odd
   components" is the right treatment when re-splitting is off the table.

   **→ RUN (2026-08-20), real 600 µm box** (x 750–1350k nm, 882 v117
   fragments, 1,047 neurons, 5,617 post-sliver synapses, **96 real
   frankenmerges**; 2×2 tiles, 30+60 epochs — same budget as the 200 µm
   runs). Two honest negative findings:

   | Variant | baseline ARI / merge_P | stitched ΔARI | assembly (645 multi-tile obj) |
   |---|---|---|---|
   | base | 0.221 / 0.258 | +0.003 | 0% → 2.2% |
   | blanket shatter (±odd-skip) | 0.062 / 0.062 | −0.013 | ~0% |

   1. **Blanket shatter fails on real data**, opposite of synthetic. The
      odd-edge thresholds (4× median, 10 µm floor) flag **746/882 (85%)** of
      real L2-MST fragments — long edges are routine in real skeletons — so
      "shatter" became blanket atomization (882 → 4,998 atoms), deleting the
      same-v117-parent prior that demonstrably carries real level 0. Fix
      before retrying: (a) keep a **same-parent soft edge type** between
      atoms (demote the parent relationship from identity to evidence, never
      delete it), (b) recalibrate oddness for real data (≥8× / ≥20 µm, or a
      relative/percentile rule).
   2. **At 3.4× the neuron count and the same training budget, level-0
      collapses** (merge_P 0.75 → 0.26 for base) and stitching cannot
      recover what level 0 never found — ΔARI +0.003 versus +0.132 on the
      200 µm box. The binding constraint at scale is per-tile partition
      quality (training budget/model capacity per tile), not the stitch
      machinery — the same "evidence quality is the lever" conclusion the
      repo reached in Phases 2.2–2.3. The scale path is therefore: keep
      tiles at the size where level 0 is strong (~200 µm), and add *more
      tiles* — not bigger ones — which is what the hierarchy is for.

   Operational notes from the same runs: the level-1 stitch needed
   memory-bounded candidate generation (per-endpoint k-NN, endpoint cap) —
   an all-pairs radius query over dense-box super-fragment endpoints OOMs;
   and level-0 shows substantial run-to-run variance at fixed seed across
   thread counts (same tile: 310 vs 694 clusters), so multi-seed reporting
   is needed for small deltas.

   **→ FIX CONFIRMED (2026-08-21): 600 µm rerun at the proven tile size.**
   Same data, 6×2 grid (~100 µm tiles) and the proven 40+100-epoch budget:

   | 600 µm | baseline ARI / merge_P | stitched ΔARI | Δmerge_P | assembly |
   |---|---|---|---|---|
   | 2×2, 30+60 ep (above) | 0.221 / 0.258 | +0.003 | −0.041 | 2.2% |
   | 6×2, 40+100, obs-only | 0.310 / 0.497 | **+0.118** | −0.121 | 20.9% |
   | 6×2, 40+100, odd-skip | 0.314 / 0.504 | +0.072 | −0.204 | **29.0%** |

   Scaling by adding right-sized tiles works: the stitch gain grows 40×
   (+0.003 → +0.118) and assembly 10× on the identical volume. The
   remaining gap to the 200 µm regime (baseline merge_P 0.50 vs 0.75) is
   level-0 purity, and here even the obs channel pays precision for it
   (−0.121 vs −0.001 at 200 µm): forced merges faithfully propagate tile
   impurity, so the stitch is exactly as clean as its tiles — the direct
   motivation for the scaffold-first operator of §5. fk_sep on the 96 real
   frankenmerges: 0.54–0.57 baseline → 0.42 stitched in both configs.
3. **Soma cannot-link A/B.** Add the nucleus-table constraint to level-0 GAEC
   and level-1 Kruskal on the densest bbox (T4, where ARI dropped to 0.287).
   Success: ARI recovery at unchanged merge_P.
4. **Pooled-DNA stitch scorer.** Train the level-1 MLP on seam pairs from 3
   tile pairs, evaluate on a held-out seam (seam-buffer protocol of Phase
   2.11). Success: beats the geometry-only scorer of experiment 1 on stitch
   precision/recall.

Each experiment is a day-scale run on existing data paths and directly falsifies
one load-bearing claim of the design before any large build.

## 5. Tiered identity resolution — the corrected frame (2026-08-21)

> Supersedes the scaffold sketch below where they differ. The correction
> (from review): a scaffold story that celebrates the easy dendritic 98% is
> naive — that part was never the bottleneck, and at graph scale *any*
> false merge is quadratic poison (every synapse pair across it is a false
> edge; cf. the old pipeline's 85% pairwise accuracy vs 0.27 line-graph
> F1). What has value is **100% precision on any defensible subset** —
> even dendrites alone — and never asserting identity we cannot certify.

**The product is a fixed synapse graph with progressively resolved
identities.** Synapses are invariant nodes (the coassign framing); identity
claims attach to each synapse's two sides; an unresolved side is an
*anonymous node*, never a guess. The graph is complete in edges from day
one and is never corrupted, because false identity is the single forbidden
move. Per-root tiers, by exact lineage soma containment:

- **NAMED** (exactly 1 contained nucleus): identity = that soma's neuron.
  Target: certify to 100% precision. Dendrites live here, so post-sides of
  edges certify first — half of every edge done exactly.
- **MULTI** (≥2 nuclei): a merge error by arithmetic — the catastrophic
  editor's caseload, found without learning.
- **BIG-NOSOMA** (no nucleus, high synapse mass): external neuron or
  must-merge candidate. Deferred as its own labeled node; deferral is free,
  wrong merging is not.
- **ANON** (no nucleus, small): orphan axon/dendrite fragments. For axons,
  **attribution, not reconstruction** — the path is irrelevant; only which
  output identity its synapses carry, priority-ordered by synapse count.
  Unattributed fragments remain valid anonymous pre-nodes.

Priorities follow synapse mass throughout. The catastrophic editor's prices
become tier prices — touching a NAMED claim costs near-infinity; merging
two ANON nodes is cheap-ish and reversible in effect (it consolidates
anonymous rows, never corrupts a named one). The metric is
**certified-edge fraction at verified 100% precision, synapse-mass-
weighted** — coverage grows monotonically, precision never leaves 100.
`scripts/tier_census.py` measures the day-one tier matrix (post × pre) on
the dual-side world, the verified precision of NAMED claims, and the
ranked ANON attribution worklist.

## 5b. Scaffold-first accretion (coarse-to-fine) — original sketch

An alternative inference operator inside the same tile/halo/cannot-link
skeleton, motivated by the Phase-2.3 measurement that real v117 structure is
"one dominant trunk + slivers" (88% of soma neurons already a single v117
root; ~93% of mass in the trunk) and by the 600 µm finding that *clustering*
difficulty grows with neuron count while *attachment* difficulty does not.

**Round 0 — verified scaffold.** Select v117 roots that pass label-free
gates (not odd, ≤ 1 soma, plausible size/compactness, optionally calibrated
model confidence). These are the trunks. A trunk root spans tiles, so
scaffold mass needs **no stitching at all** — same root id is global
identity. The two-level stitch machinery is demoted to the sliver tail.

**Rounds 1..k — probabilistic accretion.** Each unattached fragment/atom is
scored against nearby scaffolds by aggregating the edge classifier's
probabilities fragment→scaffold (net evidence, star topology) plus endpoint
continuity and pooled-prototype DNA. Soft assignment with an annealed commit
threshold; the scaffold's prototype (pooled DNA, shape, partner stats)
updates as it grows — the "soft bias toward previous joins." Slivers never
merge with each other directly; flat posteriors go to the review queue, and
the per-sliver posterior IS the K-materializations product.

**Priced constraints, not absolutes — the catastrophic editor.** Frozen
cores make round-0 errors permanent, so no constraint is absolute; instead
one objective prices three move classes: accretion (cheap), shell revision
(moderate), and catastrophic edits (λ_cat): scaffold split at the highest-
tension point, scaffold–scaffold merge, cohort eviction. λ_cat calibrates to
the measured error asymmetry (false merge ≈ 5–10× missed merge). Tension
signals: bimodal shape, two-soma pressure, DNA heterogeneity, contradictory
attachment cohorts. Catastrophic edits the optimizer proposes but cannot
afford are emitted as the top of the human-review queue.

**Load-bearing claims to falsify first** (`scripts/scaffold_census.py`):
1. Scaffold purity: label-free gates select v117 roots that are ≳99% pure
   against v1718 (and exclude most real frankenmerges).
2. Coverage: scaffold roots hold ~85–90% of synapse mass, so the
   probabilistic layer only handles the tail.
3. Accretion beats clustering: one attachment round ≥ edge_cc merge_P at
   comparable recall on the same graph.
4. Rounds don't compound: attachment precision per round is non-decreasing
   (else the join bias needs a smaller prototype-update weight).

**→ Census v1 (2026-08-21, subsampled-synapse substrate, both boxes):**

| Gate | 200 µm purity / fk-excl / mass | 600 µm purity / fk-excl / mass |
|---|---|---|
| none (base rate) | 0.934 / 0% / 100% | 0.891 / 0% / 100% |
| not-odd (4×/10 µm) | **0.971 / 94.7%** / 11.1% | **0.993 / 99.0%** / 13.9% |
| not-odd (8×/20 µm) | 0.942 / 84.2% / 18.5% | 0.964 / 93.8% / 17.6% |

**Claim 1 confirmed**: the loose oddness gate alone reaches 97–99% fragment
purity and excludes 95–99% of real frankenmerges, label-free. The mass
numbers (11–14%) do **not** test claim 2: on this substrate (synapse cap →
~5 obs/neuron) trunks are skeleton-huge but synapse-sparse, so nearly every
trunk carries a long L2 edge and flags odd. Claim 2 requires the L2-node
substrate (`--l2-substrate`, mass ∝ arbor) — census v2. The ≤1-soma gate
needed distance-to-skeleton rather than bbox proximity (fixed).

**→ Census v3 (2026-08-21, L2-node substrate, 200 µm box; 395 fragments /
700k L2 nodes / 16 frankenmerges).** Base rate first: the raw v117
segmentation is already **95.9% pure by fragment and 98.2% pure by mass** —
the trunk hypothesis at full strength; the scaffold task is excluding ~4%
bad mass, not finding a rare good subset. And one gate does it:

| Gate (label-free) | frags | mass | purity | mass purity | fk excl. |
|---|---|---|---|---|---|
| **≤1 contained soma (lineage)** | 84.1% | **78.9%** | **0.997** | **1.000** | 93.8% |
| MST-not-odd (either setting) | 15–17% | 7–10% | 1.000 | 1.000 | 100% |
| soma ∧ not-odd | 14% | 9.1% | 1.000 | 1.000 | 100% |

**Soma containment — a nucleus's supervoxel resolving to the root, one
batched `roots_at` call — is the primary scaffold gate**: ~79% of arbor
mass at ~100% mass-weighted purity. The 63 multi-soma roots it excludes are
the catastrophic-editor caseload, detected by lineage arithmetic alone.
Oddness demotes to a secondary flag: it is perfectly pure but low-coverage
because **bbox clipping** slices large arbors into disjoint clouds whose
MST bridges mimic frankenmerge glue (fix: score oddness on unclipped
fragments, or discount bridges that span the bbox boundary). Remaining for
claim 2: the ~21% of mass outside the soma gate = sliver tail + orphan
axons — exactly the accretion layer's workload. 600 µm confirmation run in
progress (walk now checkpointed).

## 6. Backlog

- **Neuroglancer visualization harness** (requested 2026-08-20). Extend
  `treestitch/ngl_export.py` (zero-dep NGL JSON state builder, Phase 2.10)
  into a harness that renders every product of this pipeline for visual
  inspection: per-tile partitions (synapses colored by cluster),
  super-fragment skeletons, accepted/rejected seam-stitch edges with scores,
  atomization cuts and odd-flagged fragments, frankenmerge separations
  (halves colored by predicted cluster), and the abstain/REVIEW queue as an
  annotation layer. One URL per tile / seam / neuron, plus an index page.
- **Tile-parallel driver.** Split `two_level_stitch.py` into per-tile workers
  (save partition outputs to disk) + a stitch collector, so tiles fan out
  across processes/machines; on a single 4-core box run variants
  concurrently instead (torch intra-op threading already saturates cores).
- **Learned stitch scorer** (experiment 4) — prerequisite for the endpoint
  channel to carry weight at level 1; geometry-only scoring measured
  insufficient.
