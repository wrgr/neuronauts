# Unified Connectome Grammar Program

## Mission

Build a single `neuronauts` system that goes from EM voxels to connectome with
one shared learned representation in the middle.

The system has three layers:

1. `neuronauts` layer 1: EM perception
   - MICRONS/CAVE volume + synapse fetch
   - optional membrane U-Net cache
   - agent navigation
   - fragment proposals
   - pre/post synapse cluster candidates

2. `neuronauts` layer 2: shared grammar
   - `PathEncoder`
   - `MergeScorer`
   - `ArborEncoder` / topology atomicity
   - beam-search global assembly
   - optional LLM oracle for coherence and identity

3. `neuronauts` layer 3: connectome extraction
   - line-graph construction
   - connectome evaluation
   - terminal metric: line-graph F1

This is one paper, one architecture, and one repo.

## Primary Claim

The paper claim is:

> A connectome grammar: a learned coordinate-free representation of neurite path
> structure that simultaneously predicts local merge plausibility, cluster
> atomicity, and global arbor grammaticality, and is optimized against terminal
> connectome correctness.

This means:

- local merge AUC is not the final target
- morphology alone is not the final target
- proofreading decisions alone are not the final target
- line-graph F1 is the final target

## Primary Editable Surface

The main learned model file is:

- [grammar.py](/Users/wgray13/projects/neuronauts/neuronauts/grammar.py)

That is the default file Codex should edit.

It currently contains:

- `PathEncoder`
- `MergeScorer`
- `ArborEncoder`

It should eventually also hold:

- cluster atomicity heads
- global hypothesis scoring
- shared representation logic used by both local and global tasks

Default rule:

- edit `neuronauts/grammar.py`

Secondary edits are allowed only when needed to support that model or its
training/data path.

## Shared Data Path

There are already useful fetch/data helper patterns across the old repos. The
runtime home is now `neuronauts`, so any absorbed helpers should land here
instead of remaining split across siblings.

### `neuronauts` helpers

- [fetch.py](/Users/wgray13/projects/neuronauts/neuronauts/fetch.py)
  - EM box fetch
  - synapse fetch
  - membrane cache load/save
- [export_topology_dataset.py](/Users/wgray13/projects/neuronauts/scripts/export_topology_dataset.py)
  - real MICRONS/CAVE topology examples from root consistency
- future path/fragment fetch and caching helpers should also live under
  `neuronauts/`

Use existing helper patterns as the integration seam, but keep execution and
ownership local to this repo.

## Supervision Sources

There are two complementary supervision sources and they should update the same
shared representation.

### 1. CAVE edit decisions

These supervise local merge quality:

- accepted or rejected join decisions
- hard reversals
- pairwise fragment compatibility

### 2. MICRONS/CAVE root consistency

These supervise global atomicity:

- candidate pre-side or post-side synapse clusters
- `atomic` if all relevant roots agree
- `non_atomic` otherwise

Both should pull on the same `PathEncoder` weights.

## Objective

The primary scalar for the unified system is:

- `line-graph F1`

Secondary diagnostics:

- local merge AUC
- cluster atomicity accuracy / AUROC
- beam-search hypothesis quality
- precision / recall / TP / FP / FN

Important rule:

- local AUC is a proxy
- line-graph F1 is the real target

## Current Inner Learning Loop

The repo now has a real trainable inner loop.

1. Export topology examples from real MICRONS boxes:

```bash
python scripts/export_topology_dataset.py \
  --output data/topology_dataset_smoke.npz \
  --box-indices 0,1,2 \
  --membrane-source auto
```

2. Train the baseline topology model:

```bash
python scripts/train_topology_model.py \
  --dataset data/topology_dataset_smoke.npz \
  --output models/topology_atomicity_smoke.npz
```

This is not the final grammar model, but it establishes the correct inner-loop
pattern:

- build real examples
- train a learned model
- evaluate

## Outer Optimization Loop

The outer Codex loop should be:

1. edit the shared model in `neuronauts/grammar.py`
2. run the relevant training path
3. evaluate local diagnostics
4. evaluate terminal line-graph F1
5. keep or revert
6. continue

Do not center the workflow on repeated 5-minute reruns of unchanged code.
Those are only diagnostic monitors.

The primary optimizer command remains:

```bash
python scripts/codex_optimize.py --repeat-until-interrupt
```

But the target of that optimizer should migrate toward the shared grammar model
in `neuronauts/grammar.py`, not remain only inside `neuronauts/run.py`.

## What To Build Next

The next real milestone is not more parameter tuning. It is model unification.

Priority order:

1. connect the exported topology dataset into unified grammar training
2. train shared weights on both:
   - edit-decision merge supervision
   - topology atomicity supervision
3. extend `PathEncoder` consumers beyond `MergeScorer`
4. introduce a cluster/arbor atomicity head
5. assemble with beam search
6. evaluate with line-graph F1

## What To Avoid

- feature-spreadsheet morphology engineering
- optimizing only local AUC
- adding many new ad hoc heuristics
- splitting the learned representation across many disconnected models
- pretending the LLM is the learned model

The LLM is only the outer research optimizer.

## Success Condition

The system is successful when:

- one shared learned representation supports local merge plausibility,
  cluster atomicity, and global assembly
- it transfers across MICRONS volumes without retraining from scratch
- terminal line-graph F1 improves on real data

That is the target.
