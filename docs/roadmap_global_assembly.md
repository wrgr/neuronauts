# Roadmap: Learned Global Neuron Assembly via Tree-DNA

> **Status: north-star roadmap (2026-06-05).** This document is now the
> canonical direction for the project. It supersedes the "primary pipeline"
> framing in `README.md`, `program.md`, and `pipeline_state.md` where they
> conflict (see *Reconciliation* below). It extends, rather than replaces,
> `docs/global_topological_merge_plan.md` — the CellGNN work there becomes the
> *within-region* assembler of Stage C.

## TL;DR

Two goals drive this roadmap:

1. **A clear, modular pipeline a team can contribute to in parallel.**
2. **Learned *global* neuron assembly** — assemble whole neurons across space,
   not synapse clusters trapped inside 30 µm boxes.

The unifying technical move that serves both: **make the unit of computation a
*skeleton fragment carrying a learned morphological embedding ("tree-DNA")*,
not a synapse-in-a-box.** Fragments have global coordinates and translation-
invariant identity signatures, so they can be stitched across box seams — which
is exactly what a box-local synapse graph cannot do.

Settled decisions (2026-06-05):

| Decision | Choice |
|---|---|
| Skeleton backbone for tree-DNA | **kimimaro self-skeletonization** of the seg volume (works on all roots, incl. unproofread) |
| First global testbed | **Minnie Column ROI** (`experiments/minnie_column/`) — bounded but full cortical depth |
| Legacy v1 agent/membrane stack | **Quarantine to `neuronauts/legacy/`** (keep history, drop from default imports/CI) |

---

## 1. Diagnosis: why now

### 1.1 The box is a ceiling, not a window

`neuronauts/cell_graph.py::build_synapse_graph` constructs a K-NN proximity
graph over synapse positions **within a single box** (the KDTree is queried over
one box's `iso_positions`; proximity edges are capped per node to stay O(N·K)).
Training (`train_cell_gnn`), inference (`cell_gnn_assembly`), and clustering all
run **per box**. There is **no cross-box stitching anywhere in the codebase.**

Consequence: a neuron larger than the box (every pyramidal cell — apical
dendrites span the whole column) is *structurally un-assemblable*. This is the
most likely reason terminal line-graph F1 sits at **~0.27** (`docs/TODO.md`)
while pairwise merge accuracy is **85%+**: the model is strong locally and
incapable globally.

### 1.2 The representation is leaving signal on the table

`docs/TODO.md` concludes, after a full per-feature ablation, that "the scalar
evidence features are largely redundant; the model is leaving signal on the
table." The 6 scalar edge features (`distance, same_scaffold, grammar_score,
shared_agents, shared_partners, seg_connectivity`) collapsed. Meanwhile the
skeleton machinery that *would* carry morphology —
`precompute_self_skeletons_for_cache` (kimimaro), `PathEdgeEncoder`,
`skeleton_graph.py` — is built and tested but **sidelined**.

### 1.3 Three regimes, no single story

Three architectural regimes coexist and the docs disagree on which is canonical:

| Regime | Where it's described | Status |
|---|---|---|
| v1 — agent/membrane simulation (700 walkers) | `pipeline_state.md` step `[1]`, `model.md` | Not in active path; tested but dead |
| v2 — shared global grammar + GAT | `program.md` ("EM voxels → connectome") | Partially active (grammar), GAT idle |
| active — no-EM box CellGNN | `README.md` | The thing that actually runs |

Plus two monoliths that block parallel work: `scripts/train.py` (3,351 lines,
~28 subcommands) and `cell_graph.py` (3,950 lines doing graph-build + GNN +
clustering + skeletonization + seg-scoring + beam-search + tangledness).

**Your instincts (global + learned skeletal features) map one-to-one onto these
three diagnosed failure modes.** Most pieces already exist; they're sidelined
and box-trapped.

---

## 2. The reframe: tree-DNA

**Tree-DNA** = a learned, per-fragment embedding of local arbor structure
(caliber/radius profile, branching pattern, tortuosity, tangent flow). It is
poolable to a per-neuron signature. It does double duty — this is your
"what goes to what":

- **Within-neuron (assembly).** Two fragments belong to the same neuron when
  their DNA is compatible *and* their skeleton endpoints are geometrically
  continuous. Because fragments carry global nm coordinates and DNA is
  translation-invariant, this decision **works across box seams**. This is the
  concrete mechanism that breaks the box ceiling.
- **Between-neuron (connectome).** A neuron's pooled DNA + its synaptic partners
  predicts cell type and connectivity. This is the `experiments/soma_graph/`
  graph — which today runs on **random placeholder node features**
  (`build_graph.py:97`). Tree-DNA is the feature that slot was waiting for.

Why skeletons specifically: morphology is translation-invariant identity, it is
cheap relative to dense EM grayscale (kimimaro runs on the *segmentation*
volume, ~5 MB/box at MIP3 — not the EM image, and not the membrane U-Net), and
you already have two code paths that produce it.

> **Note on the "no-EM" framing.** Choosing kimimaro self-skeletonization
> retires the README's "no EM volume" purity claim, but only partway: we need
> the **segmentation** volume, not EM grayscale and not agent perception. The
> honest one-line description becomes: *seg-volume + synapses → skeleton
> fragments → tree-DNA → global assembly → connectome.*

---

## 3. Target architecture: staged, with typed contracts

The mechanism that makes the pipeline team-modular: **each stage reads and
writes a typed artifact on disk.** A stage owner needs only the *schema* of the
previous stage's artifact — never its code. Stages can be developed, cached,
tested, and replaced independently.

```
data/        Substrate: CAVE synapses + seg volume + kimimaro skeletons
   │           → Region artifacts                    [owner: Data/infra]
   ▼  Region{ synapses, root_ids(base+target), skeleton fragments, global coords }
represent/   Tree-DNA: per-fragment morphological encoder
   │           (promotes PathEdgeEncoder + skeleton featurization)
   ▼  Fragment{ coords, endpoints, radius_profile, dna_embedding }   [owner: Representation]
assemble/    Within- AND cross-region GLOBAL assembly:
   │           fragment graph + GNN + seam stitching  [owner: Assembly]
   ▼  NeuronHypothesis{ fragment_ids, pooled_dna, synapses, spans_regions }
connectome/  Neuron × neuron graph: cell typing + connectivity refinement
   │           (soma_graph + GlobalAssemblyGAT, real features) [owner: Connectome]
   ▼  ConnectomeGraph
evaluate/    line-graph F1 (column-scale) + global metrics
             (per-neuron completeness/purity, stitch precision/recall) [owner: Eval]
```

Supporting modules: `schemas.py` (the contracts), `legacy/` (quarantined v1).

### 3.1 The contracts (`neuronauts/schemas.py`)

Concrete dataclasses to introduce in Phase 0. These are the team's interfaces;
keep them small and versioned.

```python
@dataclass(frozen=True)
class Region:
    region_id: str
    bbox_nm: tuple[Vec3, Vec3]
    synapses: SynapseTable        # existing type; pre/post pt + root_ids + seg_ids
    seg_version: int              # base materialization the skeletons came from
    label_version: int            # target materialization for supervision/eval

@dataclass(frozen=True)
class Fragment:
    fragment_id: int              # globally unique across regions
    region_id: str
    base_root_id: int             # noisy seg root the fragment came from
    vertices_nm: np.ndarray       # [V,3] GLOBAL coordinates
    edges: np.ndarray             # [E,2] skeleton connectivity
    endpoints_nm: np.ndarray      # tip vertices — the seam-stitch handles
    radius_nm: np.ndarray         # [V] caliber profile
    synapse_indices: list[int]
    dna: np.ndarray | None        # [D] learned embedding (filled by represent/)

@dataclass(frozen=True)
class NeuronHypothesis:
    neuron_id: int
    fragment_ids: list[int]
    pooled_dna: np.ndarray
    synapse_indices: list[int]
    spans_regions: list[str]      # proof of cross-box assembly

@dataclass
class ConnectomeGraph:
    neurons: dict[int, NeuronHypothesis]
    edges: np.ndarray             # [E,2] pre→post neuron indices
    edge_synapse_count: np.ndarray
    node_features: np.ndarray     # pooled dna + connectivity stats
```

### 3.2 The seam-stitch mechanism (the heart of "global")

This is what the box pipeline cannot do and what makes assembly global:

1. Tile the target ROI into **overlapping regions** (core + halo).
2. **Within each region:** self-skeletonize the seg volume → split skeletons at
   branch points into `Fragment`s → encode tree-DNA → build a within-region
   fragment graph → local neuron hypotheses.
3. **Across regions:** two fragments in overlapping/adjacent regions merge when
   - endpoints are within ε **and** tangents align (geometric continuity), **and**
   - DNA cosine similarity is high (morphological compatibility), **and**
   - their base-seg roots agree across the seam (segmentation evidence).
   Run union-find over the **global** fragment set → `NeuronHypothesis`es that
   span regions.
4. The stitch decision is a small **learned** classifier over
   `(endpoint_gap, tangent_angle, dna_cosine, seg_agreement)`, trained on
   same-root fragment pairs that straddle a seam (supervision is free from the
   base→target root mapping you already compute in `cave_root_mapping.py`).

---

## 4. Phases

Each phase has a single owner-able deliverable and a measurable success bar.
Phases 1–3 are the science; Phase 0 is the prerequisite for team velocity;
Phase 4 is scale.

### Phase 0 — Make it legible (refactor only, no model change)

**Goal:** any contributor can find the stage they own and work without
colliding. **No behavior change** — this is pure restructuring + docs.

- [ ] **Reconcile the docs.** One canonical narrative (Section 2 one-liner).
      Mark `README.md` / `program.md` / `pipeline_state.md` as historical where
      they describe the dead regimes; point to this roadmap.
- [x] **Introduce `neuronauts/schemas.py`** (Section 3.1) + `tests/test_schemas.py`.
      Nothing depends on it yet; it's the target interface. *(done)*
- [ ] **Quarantine v1 → `neuronauts/legacy/`.** This is **not** a clean file
      move yet: verified 2026-06-05, `membrane_unet.py` is already deleted, and
      `agent.py`/`fields.py`/`vectorized.py`/`run.py` are each imported by active
      code, so moving them naively breaks imports. `em_corridor.py` is **active**
      (seg-connectivity in `cell_graph.py`) and stays. The untangling sequence
      that must land first — split `merge.py`, extract active helpers out of
      `run.py`, decouple `fields.py` consumers, then move + re-point — is
      specified in [`stage_ownership.md`](stage_ownership.md#legacy-quarantine-plan).
      Each step is its own behavior-preserving PR; mark legacy tests
      `@pytest.mark.legacy` and drop from default CI at the end.
- [ ] **Split the two monoliths along stage boundaries.**
      `cell_graph.py` → `assemble/graph.py` (build), `assemble/gnn.py` (CellGNN),
      `assemble/partition.py` (clustering/beam), `assemble/skeleton.py`
      (skeletonization/paths — moves to `data/`/`represent/` in Phase 1).
      `scripts/train.py` → a thin `cli/` with one module per stage command.
- [x] **`CONTRIBUTING.md` + stage-ownership map** (`docs/stage_ownership.md`) +
      first per-stage smoke test (`tests/test_schemas.py`). *(done)*

**Success:** `pytest` green; import surface in `__init__.py` reflects the five
stages; a new contributor can run one smoke test per stage from a fresh clone.

### Phase 1 — Tree-DNA fragment representation

**Goal:** a learned per-fragment morphological embedding that beats the
collapsed scalar features.

- [ ] **`Fragment` extraction** from kimimaro skeletons (split at branch points;
      attach synapses by nearest-vertex; compute radius profile + endpoints).
      Promote `precompute_self_skeletons_for_cache` from a sidelined precompute
      to a first-class `data/` stage producing `Region` + `Fragment` artifacts.
- [ ] **Tree-DNA encoder** (`represent/dna.py`): generalize `PathEdgeEncoder`
      from an *edge feature* to the *primary node representation*. Input =
      per-vertex skeletal features along a fragment; output = `[D]` embedding.
- [ ] **Supervision:** same-base-root fragments → positives; the existing
      v117→v1412 false-merge / false-split edit pairs (`path_dataset.py`,
      `cave_root_mapping.py`) → hard negatives / hard positives. Contrastive loss.
- [ ] **Ablation vs scalar baseline** on the same boxes.

**Success:** DNA-only same-neuron prediction AUC clears the 6-scalar-feature
CellGNN baseline by a margin that survives the spatial val/test split (the
documented "leaving signal on the table" gap is the bar to beat).

### Phase 2 — Global assembly (break the box)

**Goal:** assemble neurons that span multiple regions on the Minnie Column.

- [ ] **Overlapping-region tiling** of the column (reuse
      `experiments/minnie_column/` bins/tubes; full-depth z stays in one tile).
- [ ] **Fragment-graph assembler** (`assemble/`): nodes = `Fragment`s with DNA;
      edges = endpoint-continuity + DNA-compatibility + synaptic co-occurrence.
      Generalizes the CellGNN; **no box boundary in the node set.**
- [ ] **Learned seam stitch** (Section 3.2) → global `NeuronHypothesis`es.
- [ ] **Global evaluation** (`evaluate/`): line-graph F1 computed at
      **column scale**, plus per-neuron completeness & purity and stitch
      precision/recall.

**Success:** measurable lift in column-scale line-graph F1 over the box-local
CellGNN baseline, with non-trivial `spans_regions` (i.e. real cross-box neurons
assembled), holding on the held-out spatial bin.

### Phase 3 — Connectome graph (what neuron → what neuron)

**Goal:** a neuron × neuron graph with real features for typing + connectivity.

- [ ] Replace `experiments/soma_graph/build_graph.py:97` placeholder features
      with **pooled tree-DNA** per `NeuronHypothesis` + connectivity stats.
- [ ] Train neuron-level edge refinement + a cell-type head (column has type
      labels via the census reference in `minnie_column_paradigm.md`).

**Success:** cell-type accuracy and connectivity-edge F1 on held-out column
neurons beat a degree/position-only baseline.

### Phase 4 — Scale & close the loop

Only once Phases 1–3 give a stable global signal: extend tiling beyond the
column toward full Minnie65; reintroduce the outer optimizer (`program.md`'s
`codex_optimize`) targeting `represent/`/`assemble/` with the global metric as
the keep/revert signal.

---

## 5. Team modularity mechanics

| Mechanism | What it buys |
|---|---|
| **Typed artifacts on disk between stages** (`schemas.py`) | Owners depend on schemas, not code; stages cache & test independently |
| **One package per stage** (`data/ represent/ assemble/ connectome/ evaluate/`) | Clear ownership; small PRs; no monolith merge conflicts |
| **Thin `cli/` per command** | New commands don't touch a 3.3k-line file |
| **One smoke test per stage + per-stage fixtures** | A stage is "green" on its own; CI fails point at an owner |
| **`legacy/` quarantine + `@pytest.mark.legacy`** | Dead regimes stop confusing contributors and docs |
| **Artifact schema versioning** | Stages evolve without lockstep releases |

Suggested ownership (assign real people): Data/infra · Representation (tree-DNA)
· Assembly (global stitch) · Connectome · Eval/validation.

---

## 6. Reconciliation with existing docs

- `docs/global_topological_merge_plan.md` — its CellGNN becomes the
  **within-region** assembler of Stage C. Its Phases 1–4 (cell-level
  plausibility, partition search, edit-tree supervision, top-down proposals)
  remain valid *inside* a region and as stitch supervision.
- `program.md` — the "shared learned representation" thesis survives; the shared
  representation is now **tree-DNA**, and "global assembly" becomes literal
  (cross-region) rather than "global within one box."
- `README.md` / `pipeline_state.md` — update the canonical one-liner; move the
  agent/membrane `[1]` step into the `legacy/` story.

## 7. Open risks / things to watch

- **Self-skel cost & quality at scale.** ~10 s/box is fine for the column; full
  Minnie65 needs the Phase 4 tiling budget. Validate kimimaro skeleton quality
  on thin/unproofread neurites early — bad skeletons poison tree-DNA.
- **Seam double-counting.** Overlapping tiles can ingest the same synapse twice;
  reuse `experiments/minnie_column/dedup.py` stable keys at the global layer.
- **Leakage.** Skeletons must come from the **base** materialization, not the
  target — `skeleton_graph.py::validate_skeleton_graph_config` already enforces
  this; keep that guard wired through the new `data/` stage.
- **Global eval is expensive/blunt.** Keep the sampled-pair line-graph F1 as the
  cheap diagnostic alongside the full column metric.

## 8. Immediate next actions

1. Land Phase 0 schema + `legacy/` quarantine (unblocks parallel work).
2. Stand up the `data/` stage that emits `Region` + `Fragment` from kimimaro on
   a handful of column tiles.
3. Train the first tree-DNA encoder and run the Phase 1 ablation.
</content>
</invoke>
