# Thread: root_neighborhood

**Goal.** Build training caches seeded from **proofread-anchor neighborhoods**
(all synapses near a set of proofread root IDs) rather than spatially random
boxes. Proofread anchors carry real edit signal, so these caches surface the
false-merge / false-split pairs the [error_correction](../../docs/threads/error_correction.md)
and [grammar](../../docs/threads/grammar.md) threads need — where random `boxes_30um`
caches showed `0 merge pairs, 0 split pairs`.

**Status:** incubating (experiment thread).

## Code

| File | Role |
|------|------|
| [`dataset.py`](dataset.py) | `build_root_neighborhood_cache` — fetch + cache synapses around proofread roots |

Invoked through the main CLI's `build-dataset` with the proofread-core strategy
(this is where `scripts/train.py` imports `build_root_neighborhood_cache`):

```bash
python scripts/train.py build-dataset \
  --cache-dir data/proofread_core_v117 \
  --strategy proofread-core --cave-version 117 \
  --proofread-n-roots 50 --proofread-radius-um 40 --proofread-min-anchor-synapses 50
```

(See [`docs/dataset_seeding_for_edit_pairs.md`](../../docs/dataset_seeding_for_edit_pairs.md)
for why proofread anchors carry edit signal.)

## Checkpoints

`shared_grammar_root_neighborhood_run001.pt` — grammar trained on a
root-neighborhood cache. See [`models/README.md`](../../models/README.md).

## Graduation

Promote the *cache-building strategy* into the core `data/` stage if
neighborhood-seeded caches consistently yield better edit-pair supervision than
spatial sampling; otherwise archive once a better seeding strategy supersedes it.
