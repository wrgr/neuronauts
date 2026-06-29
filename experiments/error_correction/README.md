# Thread: error_correction

**Goal.** Mine supervision from **proofreading corrections**. Between CAVE
materialization v117 (raw segmentation, 2021) and v1412 (proofread), editors
fixed false merges and false splits. Each edit is a free label:

- **false merge** (hard negative): one v117 root → 2+ v1412 roots. The junction
  between those synapse chains is a real boundary the model must *not* cross.
- **false split** (hard positive): 2+ v117 roots → one v1412 root. The junction
  across those chains is a valid same-cell path the model *should* join.

This is the "transfer function from raw segmentation to proofread truth" that
trains the [tree-DNA](../tree_dna/README.md) and
[grammar](../grammar/README.md) threads. Merge-only training makes the model
distrust every junction; the split positives are what force it to rely on real
trajectory features.

**Status:** active (core thread) — the standard way to generate path-encoder
training pairs.

## Code (lives in core)

| Module | Role |
|--------|------|
| [`neuronauts/edit_history.py`](../../neuronauts/edit_history.py) | CAVE proofreading-lineage queries |
| [`neuronauts/cave_root_mapping.py`](../../neuronauts/cave_root_mapping.py) | `map_roots_between_versions` (v117 ↔ v1412) |
| [`neuronauts/path_dataset.py`](../../neuronauts/path_dataset.py) | `fetch_cave_edit_history`, edit-pair mining, `save_edit_pairs_tsv` |

Background: [`docs/dataset_seeding_for_edit_pairs.md`](../../docs/dataset_seeding_for_edit_pairs.md).

## Run

```bash
# Preferred: mine edit pairs from the local box cache (spatially stratified)
python scripts/train.py fetch-cave-edits-from-cache \
  --cache-dir data/boxes_30um \
  --min-synapses-per-root 8 \
  --output-tsv data/cave_edit_pairs_v3.tsv \
  --output-chains data/cave_edit_chains_v3.npz
```

Needs a CAVE token (see [`docs/CAVE_AUTHENTICATION_SETUP.md`](../../docs/CAVE_AUTHENTICATION_SETUP.md)).

## Checkpoints

None of its own — it produces `data/cave_edit_pairs_v3.tsv` /
`cave_edit_chains_v3.npz`, the supervision other threads consume.

## Graduation

Already core. Next step is breadth: the current v3 mining yields ~25.8k pairs
but few split positives; richer split coverage is the lever for the
fingerprint/grammar threads.
