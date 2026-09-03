# Topology Learning Test Plan

> **Status: Fully implemented and tested.**
> All stages of this test plan are complete:
> - Cluster label derivation: `build_cluster_examples` in `topology_dataset.py`
> - Feature tensor shapes: validated in `tests/test_topology_learning.py`
> - Attention-based atomicity model: `AttentionArborValidator` in `topology_model.py`
> - Shared multitask training: `multitask_train_step` in `shared_grammar_model.py`
> - End-to-end training with real data: `scripts/train.py`
> Total tests covering this area: 329 (full suite).
> This document is retained as a record of the original test plan.

---

## Goal

Validate the first learned inner loop for Neuronauts:

1. export synapse-cluster atomicity examples from MICrONS/CAVE boxes
2. train an attention-based atomicity classifier
3. confirm the exported supervision and training path work end to end

This plan focuses on correctness first, then data quality, then integration.

## Stage 1: Local Correctness Tests

These should run offline and deterministically.

### 1. Cluster label derivation

Verify:

- clusters built from one true root are labeled `atomic=1`
- merged clusters built from two roots are labeled `atomic=0`

### 2. Feature tensor shape

Verify:

- exported padded branch tensors have stable dimensionality
- exported masks align with branch padding
- labels align with examples
- empty datasets are handled safely

### 3. Attention model training

Verify:

- model trains on a toy separable dataset
- output probabilities are finite
- saved and reloaded model produces the same predictions

## Stage 2: MICrONS/CAVE Dataset Export

Run on one or more fixed validation boxes from `neuronauts.run.REAL_BOXES`.

### 4. Export smoke test

Command:

```bash
python attic/superseded_training/export_topology_dataset.py \
  --output data/topology_dataset_smoke.npz \
  --box-indices 0
```

Verify:

- `.npz` dataset exists
- manifest exists
- number of examples > 0
- both positive and negative labels are present

### 5. Cache compatibility

Run export in both:

- `--membrane-source sobel`
- `--membrane-source auto`

Verify:

- export works without membrane cache
- export also works when cached membrane is available

## Stage 3: Training Loop

### 6. Training smoke test

Command:

```bash
python attic/superseded_training/train_topology_model.py \
  --dataset data/topology_dataset_smoke.npz \
  --output models/topology_atomicity_smoke.pt
```

Verify:

- model file exists
- metrics JSON exists
- training loss is finite
- validation accuracy is above random on the smoke dataset

## Stage 4: Data Quality Inspection

### 7. Dataset balance

Inspect:

- number of atomic vs non-atomic clusters
- pre-role vs post-role counts
- cluster size distribution

Goal:

- confirm the dataset is not degenerate

### 8. Feature sanity

Inspect:

- cluster size
- pairwise distance statistics
- membrane means/stds

Goal:

- confirm features are numerically sensible and not constant

## Stage 5: Downstream Integration

Not yet implemented in this pass, but this is the next milestone.

### 9. Validator-as-veto test

Plan:

- score candidate clusters with the trained atomicity model
- suppress or split predicted non-atomic clusters
- rebuild the line graph

Verify:

- false merges decrease
- downstream line-graph F1 changes in a measurable way

## Success Criteria For This Pass

This implementation pass is successful if:

- unit tests pass
- dataset export works on MICrONS/CAVE
- training script produces a saved model and metrics
- the repo has a concrete learned inner loop rather than only heuristic tuning
