# EXP-051 evaluation: real dense v117 soma-seeded grammar run

## Audit conclusion

The requested branch `feat/global-merge-assessment` does not exist. The matching
branch inspected and run was `feat/global-merge-assembly` at commit
`bcebae88a61f5fcbbe285244674788225268563e`.

The previous dense benchmark was not the requested experiment:

- `scripts/benchmark_exp049_dense_subvolume.py` unconditionally invokes its
  generated dense-subvolume fallback.
- `scripts/export_viz_data.py` splits proofread skeletons into synthetic pieces,
  assigns synthetic synapses, and injects frankenmerges before export.
- `viz/sample_connectome_viz.json` contains synthetic IDs and zero links. The
  viewer is rendering its supplied data; the primary defect is upstream.
- The SANTIAGO infiller initializes random matrices at runtime rather than
  loading a trained real-data grammar checkpoint.

## Fail-closed replacement

`scripts/benchmark_exp051_real_dense_soma_grammar.py`:

1. Fetches real synapses in a 30 x 30 x 30 um box from the public spatial Delta
   export and resolves endpoint supervoxels at exact v117/v1412 timestamps.
2. Includes every v117 root at either endpoint as a candidate/confuser.
3. Seeds growth by exact nucleus-supervoxel lineage containment.
4. Builds paths from real endpoint observations and requires the trained
   `models/shared_grammar_raw_skel_50e.pt` checkpoint.
5. Grows competing soma-seeded pathways under a hard one-soma-per-cluster rule.
6. Withholds target labels until evaluation; unknowns remain singleton confusers.

The implementation rejects synthetic counter IDs, missing checkpoints, missing
lineage data, missing soma seeds, and multi-soma atomic roots.

## Recorded 30 um run

| Quantity | Result |
|---|---:|
| Real synapses | 16,932 |
| Synapse-bearing v117 roots | 9,333 |
| Grammar path roots (>=10 observations + soma seeds) | 679 |
| Singleton confusers retained | 8,654 |
| Exact soma seeds | 3 |
| Candidate joins scored | 21,175 |
| v117 roots with mixed v1412 labels | 20 |
| Distinct v1412 labels among active roots | 678 |
| True fragment-merge pairs among active roots | **1** |
| Predicted clusters among active roots | 10 |
| ARI | -0.000009 |
| Merge precision / recall | 0.000 / 0.000 |
| Merge-aware ERL | 0.590 um |
| Circuit F1 | 0.000290 |
| Single-soma compliance | 1.000 |

The zero-logit threshold over-merged 671 of 679 active roots. Thresholds >=3
abstained from all joins and recovered the untouched-v117 baseline, but no
tested threshold recovered the one true merge pair.

## Interpretation

This is a valid negative result, but not a fair positive test of soma-seeded
false-split recovery. Across all 9,333 synapse-bearing roots, the box contains
only three v117-to-v1412 true merge pairs. The repository already documents this
sampling failure in `docs/dataset_seeding_for_edit_pairs.md`: ordinary spatial
boxes frequently contain no proofreading edit signal, and proofread-cell/soma
anchors are required.

The high-level approach remains sensible, but this checkpoint is not calibrated
for blind dense adjacency at a zero threshold and this box has essentially no
positive joins. The next valid experiment must select a 30 um box from a
proofread soma/edit anchor list, assert nonzero merge and split opportunities
before inference, then enumerate the full local population without exposing
those labels to inference.
