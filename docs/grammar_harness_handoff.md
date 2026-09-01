# Handoff: grammar test harness (v117 atoms → topology → experiments)

> Written 2026-09-01; updated later the same day. Read this before re-engaging
> in a new thread. The geometry download may still be running; it is safe to
> interrupt and rerun.
>
> **Update (session 2):** the attribute fetch was silently losing whole request
> batches (14.2% of tier-10 L2 nodes). Found, fixed, refetched, verified. The
> per-atom topology view (step 2) is built and validated. See §"What's
> confirmed" #8-#9 and §"In-flight job".

## The one-paragraph state of the world

We are building a **principled, label-blind test harness** for grammar ideas
(PCFG + learned scorers) on real Minnie65 v117 fragments. The goal is not a
final pipeline yet — it is a trustworthy substrate you can train/validate on
for 1–2 hours at a time, with visual checks at every step, that can later
scale to all somata. **Stage A (data foundation) is mostly done.** We have a
label-blind v117 atom population, proofread GT selection, verification plots,
and a resumable L2-topology fetch in flight. **Stage B (topology + skeleton +
GT labels + train/val split) is next.** User direction as of session end:
**topology first**; frankenmerge *cuts* are deferred (mixed-polarity atoms
flag false merges; synapses can be split by polarity in the interim);
**kimimaro** (`cell_graph.py::precompute_self_skeletons_for_cache`) is the
preferred skeleton path when we need real cable geometry and radii — not
`lineage.l2_skeleton` (kNN+MST, 200 nm radii placeholder).

## Objective (what “done” looks like for this harness)

1. **Label-blind atom set**: every v117 root that carries ≥k synapses in a
   region, with no use of proofread labels to define membership.
2. **Per-atom topology**: real L2 adjacency + coordinates (+ caliber attrs)
   for candidate merge/cut decisions and endpoint discovery.
3. **GT overlay (evaluation only)**: attach proofread cell IDs to atoms where
   extended; limit ambiguity to the extended portion; characterize the rest
   separately.
4. **Spatial train/val split** with a seam buffer.
5. **Experiment runner**: PCFG vs learned scorer on identical candidates;
   cable-weighted metrics; 1–2 hour probe runs.

Grammar nonterminals of interest (even at coarse level): axon→axon,
dendrite→spine→synapse; synapse polarity is a free compartment signal.

## Region and scale choices

| Item | Value | Notes |
|------|-------|-------|
| Centre (µm) | `[663, 591, 860]` | Dense gold proofread cube from manifest |
| **Harness region** | **100 µm cube** | `data/substrate/c100um/population.npz` — 279,075 atoms |
| Outer geometry bounds | 200 µm cube | Fixed so caches compose across nested regions |
| 200 µm population | built, not primary | 2,237,337 atoms — too large for current budget |
| Synapse source | offline static table | `scripts/extract_region_synapses.py` → region NPZ |
| GT cells | proofread status v1822 | gold tier via `fetch_proofread_manifest.py` |

Atom counts by synapse threshold (100 µm cube, from `build_population.py`):

| min synapses | atoms | % synapse mass |
|-------------|-------|----------------|
| ≥10 | 20,826 | 76.4% |
| ≥5 | 40,109 | 82.3% |
| ≥1 | 279,075 | 100% |

## Tier ≥10 topology (measured, `results/atom_topology_k10.json`)

| Quantity | Value |
|---|---|
| Atoms | 20,826 |
| L2 nodes / edges | 18,519,922 / 21,030,275 |
| Connected components | 89,751 (**4.31 per atom** — atoms are fragmented inside the bounds) |
| Endpoints (deg 1) | 5,103,160 (**245 per atom**) |
| Branch nodes (deg ≥3) | 5,037,329 |
| Segments (leaf) | 12,650,842 (5,076,213) |
| Total cable | 32.87 m |
| Segments without a length | 734 (0.006%) |

**The design consequence: endpoints are not scarce.** At L2 resolution a
degree-1 node is usually a spine or a small protrusion, not a split site — 245
endpoints per atom means naive endpoint-pair candidate generation is hopeless.
The endpoint table therefore carries leaf-segment length and tip caliber so a
proposer can filter on data. Yield of a joint filter (share of all 5.1M
endpoints):

| filter | endpoints kept |
|---|---|
| leaf ≥1 µm & caliber ≥30 nm | 1,801,646 (35.3%) |
| leaf ≥2 µm & caliber ≥50 nm | 353,396 (6.9%) |
| leaf ≥5 µm & caliber ≥50 nm | 28,104 (0.55%) |
| leaf ≥5 µm & caliber ≥80 nm | 4,861 (0.10%) |

Endpoint caliber (`mean_dt_nm`) percentiles: p10=8, p50=26, p90=56, p99=79 nm.
Leaf-segment length: p10=383, p50=1,478, p90=3,153, p99=9,691 nm.
**Not yet validated against GT** — no proofread labels have been attached, so
these thresholds are descriptive, not calibrated.

## What's confirmed (keep, build on)

| # | Result | Evidence |
|---|--------|----------|
| 1 | **Label-blind population works.** Atoms = v117 roots with synapses; no GT leakage in selection. | `neuronauts/harness/population.py`, `scripts/build_population.py` |
| 2 | **L2 fetch was our bug, not the API.** Integer seg bounds + correct nm transform; pooled threaded attributes ~15–25× faster than serial `l2_skeleton` path. | `scripts/probe_l2_throughput.py`, `probe_v117_leaves_validity.py` |
| 3 | **Stale v117 roots still return L2 geometry** (`/lvl2_graph`, `/leaves`). Atoms remain valid even after merge into proofread cell. | `scripts/probe_v117_geometry_route.py` |
| 4 | **3.6% “unresolved” L2 nodes** (no v117 ancestor on direct map) are post-v117 edit nodes; attributable via supervoxel majority vote. Not a hard ceiling. | `scripts/probe_unresolved_l2.py` |
| 5 | **Polarity → compartment at atom level is strong.** ~95% pure axon or dendrite vs binomial null; usable without GT. | `scripts/viz_polarity_compartments.py`, `results/figures/06_polarity_compartments.png` |
| 6 | **v117 lineage mapping is visually meaningful** (atom contiguity beats scrambled control). | `scripts/viz_verify_substrate.py`, `results/figures/05_contiguity.png` |
| 7 | **Proofread-cell-first substrate strategy is feasible** for GT overlay. | `neuronauts/harness/substrate.py`, `scripts/probe_substrate_pilot.py` |
| 8 | **The l2cache attribute endpoint caps at 600 requests/min** and reports a breach as HTTP 500 wrapping `429 Too Many Requests`. Our 24-worker fetch ran at ~748/min and `go()` swallowed the failures, losing **exactly 1,318 batches x 2,000 = 2,636,000 L2 nodes (14.2%)**. Diagnosed by the loss being an exact multiple of the batch size, twice. Now rate limited to 480/min with failure accounting; refetched to **100% coverage** (18,519,922/18,519,922; 236 nodes genuinely lack `rep_coord`). | `neuronauts/harness/geometry.py`, `/tmp/geom2.log` |
| 9 | **Contracted topology is built and validated.** Degrees, component counts and edge partition match networkx exactly on 68 atoms incl. the 8 largest; 29 unit tests. | `neuronauts/harness/topology.py`, `tests/test_harness_topology.py` |

## Key design decisions (don't re-litigate without reason)

- **Atom** = v117 segmentation object (flood-fill + light proofreading; real
  false merges/splits). Not an L2 tile.
- **L2 nodes** = skeletal resolution and the only surface for geometric cuts.
  We fetch **real adjacency** (`lvl2_graph` / `level2_chunk_graph`), not kNN+MST.
- **Mixed-polarity atom** = likely frankenmerge flag. Interim: split synapses
  by polarity (`treestitch/atomize.py`, `experiments/pcfg/learned_grammar.py`).
  Geometric L2 cut is a separate cleanup pass, not blocking topology experiments.
- **Skeleton for morphology**: user chose **kimimaro TEASAR** on BossDB seg
  (`neuronauts/cell_graph.py:2867`). Needs seg subvolume per box; real topology
  + radii. `lineage.l2_skeleton` remains the old treestitch default but is
  MST + constant 200 nm radii — do not treat as canonical for new harness work.
- **Tiered fetch**: run ≥10, then ≥5, then ≥1 synapses; each tier skips atoms
  already on disk (`scripts/fetch_atom_geometry.py`).

## In-flight job (may still be running)

The original 24-worker run was **stopped and restarted under the fixed code**
(16 workers, rate-limited attributes). Restarting is cheap: topology shards on
disk are the resume marker, so completed tiers re-scan in seconds.

```bash
tail -f /tmp/geom2.log      # session-2 run

# Command (resumable — just rerun it):
.venv/bin/python scripts/fetch_atom_geometry.py \
  --population data/substrate/c100um/population.npz \
  --geom-dir data/substrate/geom \
  --bounds-centre-um 663 591 860 --bounds-side-um 200 \
  --tiers 10 5 1 --workers 16 --batch 2000 \
  --out results/atom_geometry_tiers.json
```

**Status:**

- Tier ≥10: **complete and verified** — 20,826 atoms, 18,519,922 L2 nodes,
  21,030,275 edges, 100% of nodes present in the attribute cache.
- Tier ≥5: **complete** — 40,109 atoms, 20,956,855 L2 nodes.
- Tier ≥1: topology fetch running (238,966 new atoms, ~1.5 h), then its
  attribute pass.
- Shards: `data/substrate/geom/shards/k{10,5,1}_*.npz`; attributes:
  `data/substrate/geom/l2_attributes.npz` (rewritten at the end of each tier —
  don't read it during an attribute phase).

To verify completion — and **do not trust the script's coverage line alone**,
it used to be computed over the cache's own rows and read 100% while 14% was
missing (now fixed to measure against the tier's node set):

```bash
grep -E "tier >=|all tiers complete|!!" /tmp/geom2.log
cat results/atom_geometry_tiers.json
# independent check: every shard node must appear in the attribute cache
```

## Code map (new harness — mostly uncommitted)

| Path | Role |
|------|------|
| `neuronauts/harness/population.py` | Label-blind v117 atom population from region synapses |
| `neuronauts/harness/substrate.py` | Proofread-cell-first substrate (GT overlay path) |
| `neuronauts/harness/geometry.py` | Resumable per-atom L2 topology + pooled attributes |
| `scripts/extract_region_synapses.py` | Offline region synapse extract from static CSV |
| `scripts/fetch_proofread_manifest.py` | Proofread status + dense-region discovery |
| `scripts/build_population.py` | Build + characterize atom population |
| `scripts/fetch_atom_geometry.py` | Tiered topology + attribute fetch driver |
| `neuronauts/harness/topology.py` | Contract L2 adjacency → junctions, segments, endpoint tangents |
| `scripts/build_atom_topology.py` | Build the per-atom topology + endpoint table |
| `tests/test_harness_topology.py` | 29 tests: known shapes + partition invariants |
| `scripts/viz_verify_substrate.py` | Region / arbor / soma / contiguity verification |
| `scripts/viz_polarity_compartments.py` | Polarity purity vs null |
| `scripts/probe_*.py` | One-off feasibility probes (see `results/probe_*.json`) |

**Existing skeleton/split code to reuse (not yet wired to harness):**

See also [Skeleton code survey](a9ab9fd3-c196-4f11-81ba-082ec237dca7) for full inventory.

| Tier | When | Topology | Radii | Entry point |
|------|------|----------|-------|-------------|
| **Kimimaro TEASAR** | Unproofread roots with local seg | Real (BossDB volume) | Real | `cell_graph.py::precompute_self_skeletons_for_cache` — **harness choice** |
| **CAVE skeleton service** | Proofread whole roots | Real (skv4) | Real | `fetch.py::fetch_root_skeleton`, `loaders.py::load_skeleton` |
| **`l2_skeleton`** | v117 fragments, no seg box | kNN+MST over L2 coords | Hardcoded 200 nm | `lineage.py::l2_skeleton` — **current treestitch default, not harness** |

**Important gap (from survey):** `harness/geometry.py` stores real L2 adjacency per atom,
but nothing yet maps that into a `Fragment` skeleton for treestitch/grammar. The old path
still builds MST edges. Harness should use L2 adjacency for **topology/candidate edges** and
kimimaro for **cable morphology + radii** when needed — they serve different roles.

| Path | Role |
|------|------|
| `neuronauts/cell_graph.py::precompute_self_skeletons_for_cache` | Kimimaro on BossDB seg — user's chosen skeleton path |
| `neuronauts/fetch.py::fetch_root_skeleton` | CAVE service; v117 unproofread often returns 1-vertex placeholders |
| `neuronauts/data/lineage.py::l2_skeleton` | Treestitch v117 path (MST, placeholder radii) |
| `treestitch/atomize.py::split_fragment` | Cut at branch points; reassign synapses by nearest vertex |
| `experiments/pcfg/learned_grammar.py::_split_synapses_into_fragments` | Synapse split by skeleton component |
| `experiments/pcfg/close_loop_cut.py`, `seam_detector.py` | Cut existing skeleton; partition synapses by subtree |
| `experiments/fingerprints/cutface/v117_error_relink.py` | False-split detection via real L2 adjacency graph |

**Treestitch consumption chain (for reference when wiring harness):**

```
skeleton (kimimaro / CAVE / l2_skeleton)
 → Fragment (vertices, edges, radii, endpoints, synapse_indices)
 → SkeletonGNN → DNA embedding
 → observation graph → partition → merge_fragment_skeletons
```

## What's next (priority order)

1. ~~Let `fetch_atom_geometry.py` finish; inspect the 1 topology error.~~
   **Done.** The "1 error" was cosmetic: `_fetch_one` set `rec["error"]` on a
   transient exception and never cleared it when the retry succeeded, so a
   fully-fetched atom was counted as an error. Fixed. All 20,826 tier-10 atoms
   have geometry. Tier ≥1 still running.
2. ~~Build topology view per atom.~~ **Done and verified** —
   `data/substrate/topology/k10.npz` (134 MB): per-atom counts, cable length,
   caliber, polarity; plus a 5,103,160-row endpoint table with position,
   outward tangent, and the leaf segment each tip terminates.
   **Rerun it for tiers 5 and 1 once the fetch completes.**
3. **Kimimaro skeleton pass** for atoms in tier ≥10 (or per-synapse bounding
   boxes like `cell_graph.py` does for training boxes). Store per-atom
   `{vertices_nm, edges, radii_nm}` keyed by v117 root id. Decide box sizing:
   existing code assumes ~6 µm synapse-centered boxes; harness may need
   synapse-cloud bbox per atom or small tiling inside the 100 µm region.
4. **Attach GT labels** from proofread status (v1822 gold) — evaluation only.
   Restrict metrics to extended portions where GT is unambiguous.
5. **Spatial train/val split** with seam buffer.
6. **Experiment runner** — PCFG vs learned on identical candidate edges;
   cable-weighted merge/split metrics; 1–2 hour probe budget.
7. **Deferred**: L2-level frankenmerge cuts; mixed-polarity detector as Bar-3
   signal; scale to 200 µm / all somata.

## Open questions for next thread

- **Kimimaro scope**: skeletonize per atom via synapse bbox (like training
  boxes) vs tile the 100 µm region once and slice by seg ID?
- **Candidate edges**: endpoint adjacency on contracted topology vs existing
  `treestitch` stitch candidate generator?
- **Endpoint filter, now the live question**: which of the 5.1M endpoints are
  real split sites? Needs GT (step 4) to calibrate; until then any threshold is
  a guess. Worth checking whether the 4.31 components/atom are bounds-clipping
  artefacts or genuine fragmentation.
- **PCFG entry point**: `experiments/pcfg/skeleton_tokens.py` vs new harness
  runner?
- **The 1 topology fetch error** on tier ≥10 — identify atom id and whether
  retry or stale-root fallback fixes it.

## Figures and probe artifacts

- Figures: `results/figures/01_region_somas.png` … `06_polarity_compartments.png`
- Probes: `results/probe_*.json`
- Tier summary (when done): `results/atom_geometry_tiers.json`

## Context docs

- `program.md` — overall v2 vision, synapse line-graph F1 as terminal metric
- `README.md` — treestitch + grammar tracks
- `docs/tree_assembly_handoff.md` — related assembly work (different thread)
- `CLAUDE.md` — assume bugs are ours; verify counts against trusted queries
