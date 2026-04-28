# CellGNN Pipeline — Open Items

Status as of 2026-04-28.  Ordered by current dependency.

---

## Done (this iteration)

- ✅ **Real-data dataset**: 37 CAVE boxes cached at `data/boxes/`,
  v1412 root IDs.
- ✅ **Seg-connectivity edge feature** (`seg_connectivity`,
  6th edge feature).  Single-bbox BossDB fetch per box at MIP 3
  via `precompute_seg_scores_fast`.  Cache at `data/seg_scores.json`
  (37 boxes, 410K edges, 73% with non-neutral signal).
- ✅ **Threshold sweep** on `cell_gnn_seg.pt` (t in 0.90..0.999).
  Best mean F1 = 0.272 at t=0.99; t=0.999 catastrophic.
- ✅ **K-hop ablation** (n_layers in {1,2,3,4,5}, all seed=42):

      K=1: 0.176   K=2: 0.197   K=3: 0.194   K=4: 0.195   K=5: 0.185

  K=1 underfits.  K=2..4 are flat within ±0.003.  Default
  ``CellGNNConfig.n_layers=3`` no longer matches the empirical
  optimum (K=2) but is preserved for back-compat.  See module
  docstring in `neuronauts/cell_graph.py`.

- ✅ **Per-feature ablation** (zero out one of the 6 edge features
  via ``--ablate-feature``).  All 6 deltas vs the baseline 6-feat
  model are within ±0.01 mean F1 — see commit history for the
  full table.  Implication: the scalar evidence features are
  largely redundant; the model is leaving signal on the table.

- ✅ **PathEdgeEncoder** (Option 2 model plane,
  `neuronauts/path_edge_encoder.py`).  Transformer over per-step
  skeleton-path features → fixed-size edge embedding.  Wired into
  ``CellGNN`` via the new ``path_emb_dim`` config field.  Empty
  paths use a learned ``no_path_embedding``.  6 unit tests in
  `tests/test_path_edge_encoder.py`.

- ✅ **Skeleton-path data plane** (Option 2 data plane).  Two
  precompute pipelines:

  1. ``precompute_self_skeletons_for_cache`` — kimimaro
     skeletonization of the BossDB seg volume per box.  37/37
     boxes, ~10 sec/box, 19–63 real skeletons each.  Output at
     `data/skeletons_self/`.
  2. ``precompute_skeleton_paths_for_cache`` — Dijkstra paths
     between proximity-graph synapse pairs through the cached
     skeletons.  Output at `data/skeleton_paths.pkl` (37 boxes,
     410K edges, 0.3% with traced same-root path).

  ``precompute_skeleton_paths_for_cache`` accepts either CAVE-style
  per-(root, version) caches **or** the kimimaro per-box archives.

- ❌ **CAVE skeleton service** (Option 2b alternative source) —
  attempted, abandoned.  CAVE returns 1-vertex placeholder
  skeletons for unproofread roots, which is >99% of roots in 6 µm
  training boxes.  Per-fetch latency ~80 sec for empty results.
  See `scripts/fetch_skeletons.py` (kept; works only when the
  target roots are proofread).

---

## Active

### 1. Train Option-2 model end-to-end on self-skel paths

The wiring is in place; ``CellGNN(path_emb_dim=H)`` consumes
``(path_seq, path_mask, has_path)`` in addition to the existing
6 scalar edge features.  Remaining work:

- [ ] Thread ``--skeleton-paths-cache`` flag through
      ``cmd_train_cell_gnn`` so the path cache is loaded into
      ``SynapseGraph.edge_path_features`` per box during training
- [ ] Convert raw vertex sequences in `data/skeleton_paths.pkl`
      to per-step features via `featurize_path_points` at load
      time (or precompute and re-cache)
- [ ] Train one model with ``path_emb_dim=16`` (Option 2 +
      original 6 scalars), one with ``path_emb_dim=16`` and the
      grammar feature ablated (the 6→5+pathemb test) — to confirm
      that path embeddings recover the signal grammar lost to scalar
      collapse
- [ ] Compare against `models/cell_gnn_seg.pt` baseline on the test
      split at t=0.99

### 2. Calibrated-threshold evaluation across all ablations

`scripts/eval_at_t099.sh` is ready.  Run it once feature ablation
finishes to produce the apples-to-apples F1 table at the
calibrated threshold.

---

## Pending (lower priority)

### 3. Edit-history supervision

`edit_history.py` and ``--edit-pairs-tsv`` are wired but no real
edit-pair TSV has been generated.  Probably less impactful than
Option 2; revisit after path embeddings show signal.

### 4. Hyperparameter sweep on seg + path-encoder model

Current best: ``cell_gnn_seg.pt`` (6-feat scalar) at t=0.99,
mean F1 0.272.  Sweep ``d_model x embedding_dim x path_emb_dim``
once Option 2 is producing baseline numbers worth tuning.

### 5. Scale testing

`scale-test` subcommand exists.  Real-data scale runs on boxes
with ≥200 synapses still pending.  At >500 synapses,
`partition_from_embeddings` becomes O(N²) — switch to ANN/sparse
clustering when needed.
