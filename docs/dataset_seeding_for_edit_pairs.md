# Dataset seeding: why edit-pair extraction needs proofread-cell seeding

## TL;DR

`build-dataset` with the default `--strategy synapse-seeded` (no `--counts-tsv` / `--nucleus-csv`) produces a box cache that **cannot yield CAVE edit pairs**. `fetch-cave-edits-from-cache` will return 0 merge / 0 split pairs even on hundreds of boxes. The path-encoder hard-negative augmentation pipeline silently degrades to synthetic-only negatives.

If you intend to run `train-path-encoder` with `--edit-pairs-tsv` / `--edit-chains-npz`, you **must** seed the box cache from the proofread-cell list. Pass:

```
--counts-tsv   run_logs/synapse_root_counts_static.tsv
--nucleus-csv  data/microns_static/v{N}/nucleus_detection_v0.csv
```

Generate both files with `python neuronauts/synapse_root_counts_static.py --version 1078`.

## Why this happens

`fetch-cave-edits-from-cache` works by:

1. Reading every `pre_seg_id` (supervoxel) from the box cache and grouping by `pre_root_id` (the cache's stored "current" root).
2. Calling `chunkedgraph.get_roots(svid, timestamp=v117_date)` to resolve each supervoxel back to its v117 root.
3. Looking for two patterns:
   - **False split** (hard positive): one current root whose svids span ≥ 2 v117 roots → CV split a real cell, proofreader merged it.
   - **False merge** (hard negative): one v117 root that appears under ≥ 2 current roots → CV merged two cells, proofreader split them.

Both signals require lineage divergence between v117 and the cached version. That divergence only exists in **proofread regions** — primarily the MICrONS column.

`select_synapse_seeded_boxes` (the default) samples synapse-rich centers from the full Minnie65 volume. The MICrONS column is a small fraction of that volume, so spatial sampling rarely lands inside it. Even when boxes contain "current" roots, those roots have identical lineage at v117, v1412, and latest — no edits to extract.

## Concrete diagnostic (recorded on this machine, 2026-05-01)

After `build-dataset --strategy synapse-seeded --n-boxes 300 --box-side-um 30 --no-em --min-synapses 500`:

| cache | svids probed | with v117 ≠ current root |
|-------|-------------:|-------------------------:|
| `boxes_30um` (this build)            | 1000 | **0%**  |
| `boxes_cave40_v1412_20260407`        |  191 | **0%**  |
| `boxes_v117_spatial40_large_20260407`|  200 | **0%**  |
| `root_neighborhoods_v1718_run001`    |  200 | **28%** |

`fetch-cave-edits-from-cache` on `boxes_30um` reported `26,455 chains, 0 merge pairs, 0 split pairs`. The `root_neighborhoods_*` caches were built by `experiments/root_neighborhood/dataset.py` from a proofread-anchor list, which is why they show edit signal.

Probe to reproduce:

```python
import json, os, datetime as dt, numpy as np
from caveclient import CAVEclient
token = json.load(open(os.path.expanduser("~/.cloudvolume/secrets/cave-secret.json")))["token"]
client = CAVEclient("minnie65_phase3_v1", auth_token=token)
npz = np.load("data/<your_cache>/<some_box>.npz", allow_pickle=True)
svids = npz["pre_seg_id"][:500].tolist()
cache_roots = npz["pre_root_id"][:500].tolist()
v117 = client.chunkedgraph.get_roots(svids, timestamp=dt.datetime.fromisoformat("2021-06-11").replace(tzinfo=dt.timezone.utc))
print("disagree:", sum(int(c != int(v)) for c, v in zip(cache_roots, v117)), "/", len(svids))
```

If the disagreement count is ~0, the cache cannot produce edit pairs.

## Correct workflow when edit pairs are required

1. **Download the static MICrONS tables** and compute per-root counts:

   ```
   python neuronauts/synapse_root_counts_static.py \
     --version 1078 \
     --output run_logs/synapse_root_counts_static.tsv
   ```

   This writes:
   - `data/microns_static/v1078/synapses_pni_2_v1_filtered_view.csv.gz` (~20 GB)
   - `data/microns_static/v1078/synapses_pni_2_v1_filtered_view_header.csv`
   - `data/microns_static/v1078/nucleus_detection_v0.csv` (combined from the
     version-matched GCS data + header)
   - `run_logs/synapse_root_counts_static.tsv` (root_id → pre/post counts + has_soma)

   Two prior gotchas fixed in [neuronauts/synapse_root_counts_static.py](../neuronauts/synapse_root_counts_static.py)
   on 2026-05-01:

   - **GCS path layout uses `v{version}/`** (with a `v` prefix), not `{version}/`.
     The earlier hard-coded URL returned 404 on every materialization.
   - **Nucleus must be version-matched** to the synapse CSV. The BossDB
     `nucleus_detection_v0.csv` URL still works but contains v117-era
     `pt_root_id` values; joining those against counts computed from a v1078
     synapse CSV drops nearly every row. The script now pulls
     `nucleus_detection_v0_merged.csv.gz` from the same `v{version}/` GCS
     prefix and writes a single CSV with a header row at the local
     `nucleus_detection_v0.csv` path.

2. **Build the box cache seeded from proofread cells:**

   ```
   python scripts/train.py build-dataset \
     --cache-dir data/boxes_30um \
     --counts-tsv run_logs/synapse_root_counts_static.tsv \
     --nucleus-csv data/microns_static/v1078/nucleus_detection_v0.csv \
     --n-boxes 300 \
     --box-side-um 30 \
     --no-em \
     --min-synapses 500 \
     --seed 42
   ```

   When both `--counts-tsv` and `--nucleus-csv` are supplied, `cmd_build_dataset` calls `select_boxes_from_nucleus_table` (see [scripts/train.py](../scripts/train.py)), which centres each box on a proofread soma and ignores the `--strategy` flag. This guarantees coverage of edited regions.

3. **Then run** `fetch-cave-edits-from-cache` — it should now find thousands of merge/split pairs.

## Alternative: `--strategy proofread-core`

Produces *root neighborhoods* (variable-radius, anchored on proofread roots) rather than fixed-side boxes. Side length and `--counts-tsv`/`--nucleus-csv` are ignored. Output format is a different cache layout (`root_neighborhoods_*` directories on disk). Use this if the downstream training accepts neighborhoods; do not mix with a `boxes_*` pipeline that expects uniform box sides.

## Why this matters for training

- `train-path-encoder` accepts `--edit-pairs-tsv` / `--edit-chains-npz` to **augment** synthetic hard negatives with real CV-error pairs. Without them, the encoder only sees synthetic insert/delete and KD-tree splices — fine for smoke runs but a weaker calibration signal at the v117 → proofread boundary the model is actually deployed against.
- `train` (grammar) and `train-cell-gnn` do not require edit pairs; they use the cache root IDs directly as targets.
- `evaluate` does not require edit pairs.

So the symptom of skipping this step is silent: training completes, F1 numbers come out, but path-encoder calibration is biased toward easy synthetic negatives.

## Sanity check before training

After `fetch-cave-edits-from-cache` completes, the log line

```
[cache-edit] done: <N_chains> chains, <N_merge> merge pairs, <N_split> split pairs
```

should show **non-zero** merge and split counts. Zero on either side is a bug, not a fluke; abort the pipeline and re-seed.
