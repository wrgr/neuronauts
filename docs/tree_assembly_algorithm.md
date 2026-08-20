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
2. **Atomization A/B.** Rerun the region benchmark with v117 roots pre-split
   into L2-branch atoms. Success: out-of-sample frankenmerge halves end up in
   different clusters (the Bar-3 outcome) *without* any fk-detection features,
   at ≤0.02 ARI cost from over-fragmentation.
3. **Soma cannot-link A/B.** Add the nucleus-table constraint to level-0 GAEC
   and level-1 Kruskal on the densest bbox (T4, where ARI dropped to 0.287).
   Success: ARI recovery at unchanged merge_P.
4. **Pooled-DNA stitch scorer.** Train the level-1 MLP on seam pairs from 3
   tile pairs, evaluate on a held-out seam (seam-buffer protocol of Phase
   2.11). Success: beats the geometry-only scorer of experiment 1 on stitch
   precision/recall.

Each experiment is a day-scale run on existing data paths and directly falsifies
one load-bearing claim of the design before any large build.
