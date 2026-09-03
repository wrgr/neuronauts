# prior_results — real results, superseded as a direction

**Era.** June–August 2026: the tree-DNA identity track and the treestitch global
partition track, before the label-blind harness substrate existed.

**Read this first: nothing here is discredited.** These scripts produced numbers
that are still quoted on the front page of `README.md`. They are in the archive
because they are **not part of the current registered program**, not because a
later run overturned them. `docs/threads/experiment_survey.md` grades the
treestitch partition and the tree-DNA half-skeleton result **REAL**.

## treestitch global partition

| Script | Role |
|---|---|
| `train_l2_partition.py` | Trains the level-2 partition model. Produces `models/neuronauts_l2_partition*.pt`. |
| `multi_region_train.py` | Multi-region training with the seam-buffer leak fix. Source of out-of-sample adjusted Rand index (ARI) 0.752, merge precision 0.951. |
| `two_level_stitch.py` | Tile stitching. ΔARI +0.10 at 100 µm tiles; 300 µm tiles are dead. This is why the harness cube is ~100 µm. |
| `spatial_variance.py` | Spatial variance and out-of-column shape study over 7 test bounding boxes. |
| `real_lineage_partition.py`, `real_region_partition.py` | Neuron-seeded and region-seeded partition baselines on real lineage. |
| `out_of_column_eval.py` | Out-of-column evaluation of a multi-region model. |
| `p1_completeness_benchmark.py` | Builds the P1 completeness benchmark under the synapse and level-2 substrates. |
| `compare_partition_methods.py`, `baseline_seg_id_f1.py`, `spatial_train_test_split.py` | Supporting comparisons and the pre-harness spatial split. |
| `plot_variance.py`, `plot_calibration.py` | Figures for the two above. |

## Tree-DNA identity

A tight sibling group — each script does a bare `from ablate_dna import …`, which
resolves because they sit in one directory. **Move them apart and they break.**

| Script | Role |
|---|---|
| `ablate_dna.py` | The shared ablation driver every other file in this group imports. |
| `half_split_ablation.py` | The half-skeleton identity result: within-type area under the curve (AUC) 0.829 against a random-init 0.768. |
| `within_type_ablation.py` | Same test restricted to one cell type, so cross-type pairs cannot flatter it. |
| `multi_fragment_ablation.py` | Quarter-skeleton granularity, where the same method **fails** (0.599→0.687). Its `split_skeleton_n_parts` is still covered by `tests/test_multi_fragment_split.py`. |
| `half_synapse_ablation.py` | Synapse-side variant of the half split. |
| `global_gnn_ablation.py` | Global graph-network ablation over the same worlds. |
| `fetch_real_skeletons.py` | Fetches the skeleton archive the group runs on. |

## Falsified, and worth keeping falsified

| Script | Role |
|---|---|
| `verify_attribution.py` | *Stayed in `scripts/`* — `docs/tree_assembly_handoff.md` names it as the file the next experiment extends. Listed here only so the reader does not go looking for it. |

**What replaced it.** For the treestitch partition and the tree-DNA half-skeleton
result: **nothing yet.** They are still the best numbers this project has on
their own questions, and `docs/MAP.md` §2 says so. What superseded them is the
*approach*: the registered program measures on one label-blind substrate with one
metric package, and no result counts until it has a row in `results/RESULTS.md`.

**Route back.** EXP-058's baseline ladder already re-runs
`models/neuronauts_l2_partition*.pt` as a rung. A script here returns to
`scripts/` when a registered experiment needs it as an input, not before.
