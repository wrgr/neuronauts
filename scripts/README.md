# scripts — what builds or refreshes the current data

Everything in this directory is on a live path: it writes something under
`data/substrate/` or `data/external/`, refreshes a report, or is a minimal-repro
probe kept deliberately. Old experiment drivers, superseded trainers and one-off
analyses have moved to [`attic/`](../attic/README.md), which is the archive.

If a registered experiment reports "blocked: missing input," one of the builders
below is what produces it. `docs/MAP.md` §1.3 is the same list with one line per
script on what it produces.

## Substrate builders

| Script | Produces |
|---|---|
| `extract_region_synapses.py` | Region synapses from the local static 337M-row table, in one streaming pass. |
| `build_population.py` | The label-blind atom population for a region. |
| `build_atom_labels.py` | Proofread ground truth joined onto the population. |
| `fetch_atom_geometry.py` | Per-atom level-2 geometry, fetched in widening synapse tiers. |
| `build_atom_topology.py` | The contracted adjacency shards (`k1` / `k10` / `kall`). |
| `build_object_geometry.py` | The object point cloud — every level-2 node, not just skeleton tips. |
| `enumerate_region_objects.py` | Every v117 object in the region, synapse-free ones included. |
| `build_object_clouds.py` | Object clouds read straight from the segmentation volume. |
| `build_object_polarity.py` | Per-object synapse polarity — the grammar's one hard constraint. |
| `build_contact_panels.py` | The contact panels behind EXP-075 / EXP-076. |

## Censuses — where the EXP-074 bars came from

`seed_census.py` · `tier_census.py` · `scaffold_census.py`

## The soma-seeded targets thread (current)

`build_cell_cards.py` · `add_human_cell_types.py` · `aggregate_cell_cards.py` ·
`fetch_cell_l2_positions.py` · `fetch_edit_history.py` · `fetch_seed_graphs.py` ·
`fetch_seed_skeletons.py` · `seeded_recut.py` · `build_gallery_payload.py`

See [`docs/threads/soma_seeded_targets.md`](../docs/threads/soma_seeded_targets.md).

## CAVE fetches into `data/external/`

`fetch_cave_boxes.py` · `fetch_proofread_manifest.py` · `fetch_skeletons.py`

## Probes — the repository's minimal-repro habit

`probe_exp077_true_gap.py` · `probe_l2_throughput.py` · `probe_population_scale.py` ·
`probe_seg_mapping.py` · `probe_static_scan_rate.py` · `probe_substrate_pilot.py` ·
`probe_unresolved_l2.py` · `probe_v117_geometry_route.py` ·
`probe_v117_leaves_validity.py`

Each is the smallest call that isolates one question. `CLAUDE.md` asks for one
before any claim about an external cause, which is why they are kept rather than
deleted after use.

## Numbered experiment drivers, `exp0NN_*.py`

Scripts named for the experiment they drive (`exp079_*`, `exp083_*`, …) belong to
whatever run is in flight. They are current by construction: when the experiment
lands a row in `results/RESULTS.md`, its driver either becomes a builder above or
follows the earlier series into `attic/`.

## Verification and views

| Script | Role |
|---|---|
| `viz_verify_substrate.py`, `viz_polarity_compartments.py` | Substrate sanity figures. |
| `verify_attribution.py` | Star and link attribution verifier. `docs/tree_assembly_handoff.md` names it as the file the next experiment extends. |
| `build_reports.py` | Renders `results/reports/` — Markdown, figures and Neuroglancer views for every result. |
| `ngl_view.py`, `mesh_results.py` | Neuroglancer state and mesh serving; imported by `neuronauts/report/` and `neuronauts/meshing/`. |
| `status.py` | Consolidation and program state, derived from disk. Trust it over any prose. |
| `inspect_pipeline.py` | Stage-by-stage pipeline inspection; covered by `tests/test_inspect_pipeline.py`. |

## Cache and outage handling

`warm_cache.py` · `warm_synapses_1M.py` · `wait_for_cave.py`

## Co-assignment demos

`coassign_demo.py` · `v117_coassign.py` — cited by `INTRO.md` and
`neuronauts/coassign/README.md` as the runnable demo of that package. Both now
resolve the CAVE token through `neuronauts.auth.cave_token`, from the
environment or `~/.cloudvolume`, never from source.

## Pinned here by a resolver: `benchmark_exp051`–`056`

`benchmark_exp051_real_dense_soma_grammar.py` · `benchmark_exp052_proofread_anchor_grammar.py` ·
`benchmark_exp053a_checkpoint_bakeoff.py` · `benchmark_exp053b_l2_candidate_panel.py` ·
`benchmark_exp054_fixed_panel_scorers.py` · `benchmark_exp055_conservative_soma_forest.py` ·
`benchmark_exp056_real_root_atomization.py`

The pre-registry real-data series — graded REAL, superseded by the registered
program, and by every other criterion an archive candidate. **They cannot move.**
`neuronauts/report/registry.py:331` resolves a result record's script by globbing
`scripts/benchmark_exp<id>*.py`, and `tests/test_report.py::test_discover_real_results_parse`
asserts that EXP-056's resolves. Moving them sets `script=None` on five records
and fails that test — tried on 2026-09-02, reverted the same hour. Relocating
them means changing that resolver first.

## The one thing here that is not a data builder

`train.py` — 3,391 lines, 17 subcommands, the box-local CellGNN and
shared-grammar training command line. It stays because `README.md`, `INTRO.md`
and `CONTRIBUTING.md` all document it as the pipeline entry point and five test
modules drive it (`test_train_cli`, `test_train_helpers`, `test_root_id_remap`,
`test_pipeline_commands`, `test_multitask_convergence`).
`docs/consolidation_plan.md` §4.3 marks it SPLIT — its data subcommands to a
`data` command line, its CellGNN/GAT training to the archive with that code.
That split is package surgery and is not done.
