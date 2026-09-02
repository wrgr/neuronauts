# ConnectomeBench intake — EXP-057B pre-check

> Desk research only. No dataset files were downloaded; the parquet/image
> shards were never fetched. What follows is (a) claims stated in the papers,
> (b) URLs confirmed to resolve, and (c) contents verified directly — via the
> HuggingFace `datasets-server` API (schema, split sizes, three real sample
> rows, and partial-sample column statistics) and the public GitHub source of
> the data-generation pipeline. Each claim below is tagged accordingly. This
> document answers the five questions from the EXP-057B brief; it is not
> itself EXP-057B.

## 0. Bottom line

**It exists, it is not garbled, and it is a real unblock candidate — but the
716,485 figure needs care, there is no fixed segmentation version to anchor
to, and nobody has checked how much of it falls near our region.** Recommend
**GO** on a half-day EXP-057B, scoped exactly as below (§7), not a blind
"import ConnectomeBench2 and hope."

## 1. Identity — two real, distinct benchmarks, correctly distinguished

There are **two** papers in this family, and the plan's naming is correct, not
a garble:

| | ConnectomeBench | ConnectomeBench2 |
|---|---|---|
| arXiv | [2511.05542](https://arxiv.org/abs/2511.05542) | [2606.21116](https://arxiv.org/abs/2606.21116) |
| Submitted | 31 Oct 2025 (b) | 19 Jun 2026 (b) |
| Venue | NeurIPS 2025 Datasets & Benchmarks Track (a; also listed on the [NeurIPS 2025 virtual site](https://neurips.cc/virtual/2025/poster/121838)) | In submission, per its own GitHub citation block (a) |
| Authors | Jeff Brown, Andrew Kirjner, Annika Vivekananthan, Ed Boyden — MIT (a) | Jeff Brown, Tim Farkas (equal contribution), Gleb Razgar (Open University), Edward S. Boyden (MIT / Mindspan Institute / McGovern Institute) (c, from the paper's own header) |
| Scope | 3 proofreading tasks (segment-type ID, split-error correction, merge-error detection), evaluated as an **LLM/VLM benchmark** (Claude 3.7/4, GPT-4.1/4o, o4-mini, InternVL-3, NVLM) against expert-curated MICrONS + FlyWire examples, multiple-choice with orthographic mesh renders | The **716,485-decision corpus** the plan cites: unified multi-species (mouse/MICrONS, human/H01, zebrafish/Fish1, fly/FlyWire) dataset + a trained ViT baseline (a) |
| Data | [github.com/jffbrwn2/ConnectomeBench](https://github.com/jffbrwn2/ConnectomeBench), [huggingface.co/datasets/jeffbbrown2/ConnectomeBench](https://huggingface.co/datasets/jeffbbrown2/ConnectomeBench) — confirmed to resolve (b); schema (c, via datasets-server) shows `option_1_front/side/top_image`, `proofread_root_id`, `current_root_id` — this is the "three orthographic renders, multiple-choice with decoys" the consolidation plan's §6.4 VLM channel refers to | See §2 |

So: **the 716,485-decision figure belongs to ConnectomeBench2**, confirmed
verbatim in its abstract (c, fetched raw HTML of the arXiv page): *"a unified
multi-species dataset of over 716,485 expert-labeled proofreading decisions
with >4,500,000 associated images spanning four major open connectomes
(mouse, human, zebrafish, fly)."* The consolidation plan's attribution is
correct. The rest of this document is about ConnectomeBench2 (mouse/MICrONS
split) unless noted.

**Caveat on "716,485 decisions":** per ConnectomeBench2 §3.1 (c), a meaningful
share of the released samples are not literal individual human proofreader
judgments — `sample_type` includes `merge_edit` and `split_edit` (real edits)
alongside `adjacent_control`, `junction_control`, and `synapse_control`
(algorithmically constructed negative/positive samples built from the same
proofread graph, not each a discrete expert decision). In a partial sample we
pulled to verify schema (train split, n=16,674 of 319,727, `datasets-server`
column statistics, "partial: true"), real edits were `merge_edit` 36.3% +
`split_edit` 19.3% = **55.6%** of rows; the remainder were controls. The
headline number is a real, paper-stated figure, but "716,485 expert decisions"
overstates how many are literally one-off human judgments.

## 2. Data availability

**GitHub (code/pipeline):** [github.com/timfarkas/ConnectomeBench2](https://github.com/timfarkas/ConnectomeBench2)
— confirmed to resolve (b), README fetched in full (c). MIT-licensed data
generation + training pipeline; requires CAVEclient auth to (re)build from
scratch, but the *released* dataset needs no CAVE credentials to download.

**HuggingFace (data):** [huggingface.co/datasets/jeffbbrown2/ConnectomeBench2](https://huggingface.co/datasets/jeffbbrown2/ConnectomeBench2)
— confirmed via the HF Hub API (c). Format: sharded Parquet (`train/`,
`val/`/`validation`, `test/`), plus lightweight `metadata/{train,val,test}.parquet`
and a `demo.parquet`.

**Size (c, via `datasets-server` `/info`):**

| Split | Rows | Bytes |
|---|---:|---:|
| train | 319,727 | 96.5 GB |
| validation | 43,517 | 13.0 GB |
| test | 37,926 | 12.0 GB |
| **Total** | **401,170** | **121.3 GB download / 121.5 GB decompressed** |

This 401,170-row parquet total does **not** equal the paper's 716,485-decision
headline. Plausible reconciliation (not confirmed): the parquet is the
*rendered training-sample* table (one or more image/mask views per
operation), while 716,485 may count something upstream (raw qualifying
operations before some were dropped for rendering failures, or a different
unit of "decision" than "parquet row"). **This gap is unresolved and should
not be silently assumed away** — flag it, don't round it off.

**Version caveat (c, from the HF dataset card's own text):** the card carries
an explicit disclaimer: *"This dataset is currently under active review for
NeurIPS, so the main branch still points to the initially submitted V1 of the
dataset. For the most up-to-date version — the one used by the arXiv version
of the paper — load the v2 revision."* We queried both `main` (default) and
the `v2` ref via `datasets-server`; both returned identical split counts
above, so either they are already identical or `datasets-server`'s parquet
auto-conversion is not honoring the revision parameter — **not disambiguated
here**. A real intake should pin `revision="v2"` explicitly and re-verify the
row count differs (or doesn't) from `main`.

**License (a + c):** Per the paper's Data Access section and the GitHub
README: **MICrONS and H01 data are CC BY 4.0**; **FlyWire and Fish1 are CC
BY-NC 4.0**; ConnectomeBench2 "respects" the upstream licenses (i.e. the
mouse/MICrONS portion we care about is CC BY 4.0 — permissive, attribution
only, no non-commercial restriction). The HF repo's own tag is `license:
other` (not a standard SPDX tag, consistent with a per-source mixed license).
Note: a naive grep of the raw arXiv HTML surfaces a `License: CC BY-NC-SA 4.0`
string near the References section — **that is arXiv's license badge on the
preprint PDF itself, not the dataset's license**; conflating the two would be
the kind of unearned-certainty mistake this project's `CLAUDE.md` warns
against, so it's called out explicitly here.

**Lightweight metadata files (c, via `HEAD` request only, not downloaded):**
`metadata/train.parquet` is 15.4 MB, `demo.parquet` is 7.5 MB — these appear to
carry the `metadata`/id/coordinate columns without the image and geometry
blobs that make the full shards 100+ GB. This is the file a real intake should
start from (§7).

## 3. The MICrONS split — segmentation version, coordinate frame, decision unit

**Segmentation version / materialization: none is pinned.** We searched the
full paper text for `materializ`, `v117`, `v343`, `v661`, `v795`, `v943`,
`v1078`, `v1181`, `v1300` — **zero hits** (c, grepped the raw arXiv HTML
ourselves after an initial automated summary missed this). Confirmed instead
in the pipeline source (`src/data_generation/connectome/utils.py`, fetched
raw, (c)): mouse data comes from `CAVEclient("minnie65_public")`, and
`src/data_generation/connectome/operation_bank.py` builds each record from
**live PCG proofreading-operation history** (`get_operation_details` against
the chunked graph's append-only edit log), not from a snapshot materialization
like v117 or v1822. Each operation carries its own Unix `timestamp` and
`before_root_ids` / `after_root_ids` valid *at that timestamp* — there is no
single "ConnectomeBench2 materialization version" to look up.

**This is the crux of the id-space question the brief asked about.** Our
substrate is anchored at v117 (base) with GT at v1822
(`neuronauts/experiments/exp057_gt_overlay.py`; `V117_TIMESTAMP = 1623399000`
in `neuronauts/data/lineage.py`). ConnectomeBench2's mouse root ids are
anchored at whatever timestamp each individual edit happened. **Mapping a
ConnectomeBench2 decision's root ids onto our v117/v1822 atoms therefore does
require a lineage/timestamp resolution step** — exactly what the task brief
anticipated ("possible but is work"). The good news: this repo already has
the machinery. `neuronauts/data/lineage.py` resolves supervoxel→root at an
arbitrary timestamp via the ChunkedGraph `roots_binary?timestamp=` endpoint
(already used to build the v117→v1718 mapping); extending it to (a) each
decision's own timestamp and (b) the v1822 timestamp is mechanical, not novel
research — but it is real work, one CAVE round-trip per decision (or per
distinct root id, batched), not a lookup.

**Coordinate frame: nanometers, and — verified directly — the *same* frame our
harness already uses.** Confirmed three ways:
1. Pipeline source (c): `operation_bank.py` converts source/sink points from
   segmentation mip-0 voxel space to nm using the dynamically-fetched
   segmentation resolution (comment: *"source/sink coords from
   get_operation_details are in segmentation mip 0 voxel space — we fetch
   resolution dynamically via segmentation_info and convert to nm."*) — for
   minnie65 that resolves to the standard 4×4×40 nm EM voxel.
2. Real sample rows (c, fetched via `datasets-server /rows`): every mouse
   record's `metadata` field carries `interface_point_nm` and
   `render_center_nm` as plain `[x, y, z]` nanometer triples, e.g.
   `[586292.0, 712884.0, 839200.0]`.
3. Our own harness (confirmed by direct code inspection this session,
   `neuronauts/harness/population.py:146`, `neuronauts/harness/substrate.py:137,150`,
   `tests/test_ngl.py:70-74`): the 100 µm cube's center `(663, 591, 860)` µm is
   literally `centre_um * 1000.0` nm, **same origin, same datastack
   (`minnie65_public`)**, no offset.

So: **spatial filtering (which decisions land in or near our cube) needs no
coordinate transform at all** — a direct comparison of `interface_point_nm /
1000` against the cube's µm bounds. Only the **identity** side (linking a
decision's root ids to our v117 atoms / v1822 gold-silver ownership) needs the
lineage-mapping step above. These are separable problems, and the first one
is cheap.

**What one "decision" (parquet row) actually contains** (c, real example row,
`sample_type="merge_edit"`, mouse):

```json
{
  "operation_id": "1340114",
  "before_root_ids": ["864691134588804007", "864691135375625801"],
  "after_root_ids": ["864691136052887283"],
  "segment1_id": "864691134588804007",
  "segment2_id": "864691135375625801",
  "root_id": "864691137199050049",
  "latest_root_id": "864691136052887283",
  "interface_point_nm": [586292.0, 712884.0, 839200.0],
  "render_center_nm": [585970.0, 713252.9, 839140.4],
  "timestamp": 1733125249.0,
  "species": "mouse",
  "strategy": "merge_correction",
  "view_extent_nm": 7500.0
}
```
plus, at the row level (outside `metadata`): `sample_type`, `same_neuron`
(bool), `false_split_correction_label` / `false_merge_identification_label`
(bools), `task_routing` (list of task names this row supervises), `has_em`,
`present_slots`, and binary `geometry` / `geometry_single` (7-channel `.npy`
mesh-derived views) plus `em_xy` / `em_xz` / `em_yz` / `em_best` image columns.
Root ids are the standard 64-bit pychunkedgraph ids, same format ours uses.

## 4. Task schema

Three tasks, stated in the paper (a) and matching the column names actually
present in the data (c):

1. **Split Error Correction** — binary classification: "whether two neuron
   segments are the same neuron and must be merged" (`false_split_correction_label`).
   This is a **segment-pair judgment**, not a location.
2. **Merge Error Classification** — binary classification: "whether a given
   segment contains a false merge that must be split" (`false_merge_identification_label`).
   A **single-segment judgment** (does this root contain a hidden seam at
   all), not yet a location.
3. **Mask Segmentation for Merge Error Correction** — "requires identifying an
   exact split boundary," implemented as a CNN decoder over patch tokens
   producing a 2-channel mask (one channel per resulting piece), with a
   permutation-invariant loss since "the A/B labeling of the two pieces
   produced by a split is arbitrary." **This is task 3, and it is our
   seam-location problem under another name**, as the consolidation plan
   already asserts — confirmed directly from the paper's own task framing (a),
   not an inference on our part.

So decisions are **not** given as raw 3D locations or masks alone — a
decision is a *record* (operation id + before/after root ids + a 3D interface
point + a timestamp), and the three tasks above are three different
supervised views rendered from that record (pair-judgment, single-segment
judgment, or dense mask), each with its own image/geometry crop centered on
`render_center_nm` at `view_extent_nm` (5,000–10,000 nm for mouse, per the
paper's Table 4, jittered ±300 nm).

## 5. Spatial localization near our region — determinable in principle, not determined here

**Yes, in principle:** every decision carries an exact nm-space 3D point in
the same coordinate frame our harness uses (§3), so "how many fall inside a
100 µm cube centered at (663, 591, 860) µm" is a well-posed, directly
answerable question — filtering `interface_point_nm` against
`x∈[613000,713000], y∈[541000,641000], z∈[810000,910000]` (nm) needs no unit
conversion or reprojection.

**Not determined here, on purpose:** the brief said not to download the
dataset, and getting an exact count requires either (a) pulling
`metadata/{train,val,test}.parquet` (confirmed via `HEAD` request to be ~15
MB each, well under "downloading the dataset") and filtering client-side, or
(b) a working HF `datasets-server` `/filter` query — which we attempted and it
**errored** (`"Parameter 'where' contains errors or invalid symbols"` on two
syntax variants), so server-side filtering isn't available off the shelf for
this dataset today. We did not push further into the ~15 MB metadata files,
treating that as the first real step of EXP-057B rather than something to
sneak into this intake check.

**What we *did* check, and its limits:** two real mouse rows pulled to verify
schema had `interface_point_nm` of `(586.3, 712.9, 839.2)` µm and `(1018.9,
385.4, 753.9)` µm — both outside our cube's bounds (the first misses on y by
~72 µm, the second is far away on all axes). This is **n=2, not a density
estimate** — it only confirms the coordinates are in the right general volume
and format, nothing about local density. A napkin calculation (our 100 µm
cube vs. the paper's own spatial-split grid of 1,960 cubes of 80 µm on a side
tiling the ~1 mm³ MICrONS volume — cf. §3.1.4, verified in raw HTML/LaTeX
after an automated summary first mis-transcribed "80 µm" as "800 µm") puts our
cube at roughly 2 of their ~1,960 split-cubes by volume, i.e. **~0.1% of the
mouse split's spatial footprint** if density were uniform — but proofreading
density is emphatically *not* uniform (that is the entire finding of EXP-057:
our own densest gold-proofread cube in the whole v1822 manifest still only
carries 56 seam-positive atoms). A uniform-density estimate is not trustworthy
in either direction and is not reported as a number for that reason.

**One testable hypothesis worth stating plainly:** ConnectomeBench2's mouse
edits are drawn from the same public, cumulative MICrONS proofreading history
that produced our own v1822 manifest. It is plausible — not confirmed — that
the region we already picked as "the densest gold-proofread cube in the v1822
manifest" is *also* one of the denser regions in ConnectomeBench2's mouse
split, because both are sampling the same underlying editing effort. That is
a reason for optimism, and a reason to check the real metadata rather than
guess.

## 6. What a real EXP-057B intake experiment has to do

In order, each step gated on the previous one succeeding:

1. **Pull `metadata/{train,val,test}.parquet`** from the `v2` revision (≈20–45
   MB total, no CAVE auth needed) and filter `species == "mouse"`. Report the
   real per-`sample_type` counts (replacing this document's partial-sample
   estimate) and the real count with `interface_point_nm` inside the harness
   cube, at 100 µm and at a couple of wider radii (200 µm, 500 µm) to see the
   falloff shape.
2. **If the in-cube count is small** (plausible per §5's volume-fraction
   arithmetic), do not treat that as a dead end for the whole benchmark — it
   only means the *exact 100 µm box* is the wrong scope. Re-run the same
   count against the harness's known proofread-dense **column** (or the full
   1 mm³) rather than insisting on literal overlap with EXP-057's cube; the
   value of ConnectomeBench2 is "seam-location supervision anywhere in
   minnie65 we can trust," not "supervision in this exact box."
3. **Lineage-map a candidate batch.** For decisions that pass the spatial
   filter (or a sample, if none do), resolve `before_root_ids` /
   `after_root_ids` to root ids at the v117 and v1822 timestamps using the
   `roots_binary?timestamp=` pattern already implemented in
   `neuronauts/data/lineage.py` (currently wired for v117↔v1718; extending the
   timestamp argument to arbitrary decision timestamps and to v1822 is
   mechanical, not new research). This tells us whether a decision's segments
   land on atoms we already track, and whether they agree or disagree with
   our own v1822 mixed-lineage/proofread-owned labels.
4. **Only then** decide whether to pull the actual geometry/EM shards for the
   matched subset (now the 100+ GB question becomes "for N matched decisions"
   rather than "for the whole corpus").
5. Resolve the two open discrepancies flagged in §2 and §3 before reporting
   any headline number as a repo result: (a) 716,485 vs. 401,170 rows, (b)
   whether `main` and `v2` actually differ.

Budget: the plan's own half-day estimate for 057B looks right for steps 1–2;
step 3 is the "work" the brief flagged and should be costed separately (it is
a CAVE-network-bound loop, subject to this repo's own rate-limiting rules in
`CLAUDE.md`).

## Measured: decisions near the harness region

Everything in this section is **measured** from real downloaded/streamed
bytes of the actual dataset (not desk research, not extrapolation), except
where explicitly marked "inferred." Files live in the session scratchpad
(`cb2/`), not in this repo.

### 0. Revision used, and the `main` vs `v2` question resolved

`main` and `v2` are **confirmed genuinely different**, not an artifact of
`datasets-server` ignoring the revision parameter as the earlier section of
this document worried. Via `HfApi.repo_info(..., files_metadata=True)`:

| File | `main` size / sha256 (first 12 hex) | `v2` size / sha256 (first 12 hex) |
|---|---:|---:|
| `metadata/train.parquet` | 15,402,826 B / `553324...` | 25,421,261 B / `f3518d...` |
| `metadata/val.parquet` | 2,216,569 B / `842d80...` | 4,129,929 B / `21aea4...` |
| `metadata/test.parquet` | 1,940,900 B / `f4038d...` | 5,035,727 B / `60c816...` |

All six files differ in both size and hash. **This intake used `v2` explicitly**
(`revision="v2"` on every download/stream call below), per the card's own
instruction. `main` was not analyzed further.

### 1. Schema of the three lightweight metadata files (as downloaded)

`metadata/{train,val,test}.parquet` (25.4 MB / 4.1 MB / 5.0 MB, `v2`) share
one schema, 13 columns:

```
combined_sample_hash                 string
source_archive_sample_hash           string
source_archive                       string
sample_type                          string
same_neuron                          bool
has_em                                bool
has_after_mask                        bool
task_routing                          list<string>
false_split_correction_label          double
false_merge_identification_label      double
split                                 string
species                               string
shard                                 string
```

Row counts: train 531,734; val 82,822; test 101,929; **total 716,485** — this
**resolves the 401,170-vs-716,485 discrepancy** flagged in §2/§6.5 above:
401,170 was the `main`/V1 total; **716,485, matching the paper's headline
figure exactly, is the `v2` total** (531,734+82,822+101,929 = 716,485,
confirmed twice — once by summing the row counts in the three parquet files
directly, once independently by summing `n_samples` in the repo's own
`shards.csv` manifest, which agrees exactly).

Three real rows (one per species, `train` split):

```json
{"combined_sample_hash": "81c5a636674e76552ed428d6c3f1148c", "source_archive_sample_hash": "88cda73850c3", "source_archive": "unified_fly", "sample_type": "merge_edit", "same_neuron": true, "has_em": true, "has_after_mask": false, "task_routing": ["false_split_correction"], "false_split_correction_label": 1.0, "false_merge_identification_label": null, "split": "train", "species": "fly", "shard": "train/train-00000.parquet"}
{"combined_sample_hash": "8b8301f8499a8c916ea4586aaca884de", "source_archive_sample_hash": "60a789906f67", "source_archive": "unified_controls_fly", "sample_type": "junction_control", "same_neuron": true, "has_em": true, "has_after_mask": false, "task_routing": ["false_merge_identification"], "false_split_correction_label": null, "false_merge_identification_label": 0.0, "split": "train", "species": "fly", "shard": "train/train-00000.parquet"}
{"combined_sample_hash": "f68dc81a0a77a76a2943da4839464cc9", "source_archive_sample_hash": "a8c0215dbc61", "source_archive": "unified_mouse", "sample_type": "split_edit", "same_neuron": false, "has_em": true, "has_after_mask": true, "task_routing": ["false_merge_identification", "split_mask_generation"], "false_split_correction_label": null, "false_merge_identification_label": 1.0, "split": "train", "species": "mouse", "shard": "train/train-00000.parquet"}
```

**Correction to this document's earlier §2 guess:** §2 speculated the
lightweight `metadata/*.parquet` files "appear to carry the metadata/id/
coordinate columns without the image and geometry blobs." **That guess was
wrong, verified directly** — the schema above has no coordinate field at all
(no `interface_point_nm`, `render_center_nm`, root ids, or `operation_id`).
Those live only inside a `metadata` string column (JSON-encoded) present in
the **full** per-shard files (`train/train-NNNNN.parquet`, etc.), alongside
the large binary image/geometry columns — confirmed via `demo.parquet`'s
schema (a 6.6 MB, 20-row full-schema sample) and directly via the full
shards themselves (§2 below). Per this repo's `CLAUDE.md` §0, we do not
report a workaround-free dead end here; we found the actual coordinate
source and used it (next section), rather than declaring the task impossible
because the file the brief named turned out not to hold what was assumed.

### 2. Getting coordinates without downloading the corpus

The full per-shard files are large — `shards.csv` (downloaded, 903 rows,
one per shard) sums to **235,295,923,238 bytes = 235.3 GB** across
`train/` (669 shards) + `val/` (105) + `test/` (129) — this *is* "the 121 GB
of task data" the brief said not to download (in fact ~2× that figure for
`v2`, since `v2` has 1.79× as many rows as `main`).

Parquet stores columns contiguously per row group, and HuggingFace's CDN
supports HTTP byte-range GETs (confirmed directly: `curl -H "Range:
bytes=0-1023"` against a resolved shard URL returned `HTTP/2 206`,
`Content-Range: bytes 0-1023/261047486`, exactly 1024 bytes). So we read the
parquet footer of every shard (via ranged GETs) to get each column-chunk's
byte offsets, then fetched **only** the `combined_sample_hash`, `species`,
`sample_type`, and `metadata` columns' byte ranges — never the `geometry`,
`em_*`, or `split_mask_*` binary image columns that make up the bulk of each
shard.

**Result, measured directly:** across all 903 shards (0 failures, 0 retries
needed after the first working version), this transferred **1,197,462,860
bytes = 1.197 GB — 0.51% of the 235.3 GB full corpus** — while still reading
every one of the 716,485 rows' `species`/`sample_type`/`metadata` fields.
Per-shard row counts matched the `shards.csv` manifest's `n_samples` exactly
for all 903 shards (0 mismatches) — this is the ground-truth check that the
column-projected reads aren't silently dropping or duplicating rows.

Full per-shard schema (21 columns; only 4 were fetched):
```
combined_sample_hash, source_archive_sample_hash, source_archive, sample_type,
same_neuron, task_routing, false_split_correction_label,
false_merge_identification_label, split, species, has_em, has_after_mask,
present_slots, metadata (string, JSON), geometry (binary),
em_xy_before, em_xz_before, em_yz_before, em_best_before (binary structs),
split_mask_front, split_mask_side, split_mask_top (binary structs)
```

A real `metadata` JSON value (mouse, `adjacent_control`):
```json
{"adjacent_root": "864691135499482491", "after_root_ids": null, "base_root": "864691135764441828", "before_root_ids": null, "cutout_timestamp": 1747792546.0, "edit_point_nm": [619856.0, 731404.0, 835020.0], "interface_point_nm": [619616.0, 731296.0, 835160.0], "is_merge": true, "operation_id": "1687135", "render_center_nm": [619218.39, 731780.11, 834222.46], "root_ids": ["864691135764441828", "864691135499482491"], "species": "mouse", "strategy": "adjacent_control", "timestamp": 1747792546, "view_extent_nm": 7500.0}
```

### 3. Dataset-wide species counts (measured, all 716,485 rows, `v2`)

From the lightweight metadata files (top-level `species` column, all three
splits combined):

| Species | Count |
|---|---:|
| fly | 333,905 |
| **mouse (MICrONS)** | **301,162** |
| human | 43,138 |
| zebrafish | 38,280 |
| **Total** | **716,485** |

**Cross-check:** the full-shard scan (§2), which independently re-derives
`species` per row from the big shards while extracting `metadata`, found
**301,162** mouse rows — an exact match to the lightweight-file count above,
with 0 rows missing a coordinate. This is a strong internal-consistency
check that the column-projected read is not corrupting or dropping data.

### 4. Units determination — nanometers, confirmed by direct measurement

All 301,162 mouse rows carry a non-null `interface_point_nm` triple (0
missing; `render_center_nm` was never needed as a fallback). Measured extent
of `interface_point_nm` across all 301,162 mouse decisions:

| Axis | Min (nm) | Max (nm) | Span (nm) | Span (µm) |
|---|---:|---:|---:|---:|
| x | 289,008 | 1,627,424 | 1,338,416 | 1,338 |
| y | 286,080 | 1,151,632 | 865,552 | 866 |
| z | 594,020 | 1,114,240 | 520,220 | 520 |

**These are nanometers, not voxels.** Reasoning: the brief's own reference
extent for minnie65 is ~1.4 × 0.87 × 0.79 mm. Our measured x-span (1.338 mm)
and y-span (0.866 mm) land right on that reference; z-span (0.52 mm) is
smaller than the reference but that is expected — it's the span of *decision
points*, a subset of the volume, so it can only be ≤ the full-volume extent,
never larger. If these values were mip-0 voxel coordinates at 4×4×40 nm, x/y
would read as ~10⁵ (e.g. 1.4 mm / 4 nm ≈ 350,000) and z as ~10⁴ (0.79 mm /
40 nm ≈ 19,750) — two full orders of magnitude off from what we measured
(~10⁶ on all three axes). No conversion factor was applied; this matches the
pipeline source's own comment (§3 above, "converted to nm") and the
harness's own coordinate convention (`centre_um * 1000.0` nm, no offset).

### 5. Counts in/near the harness region (measured)

Harness cube: center `(663000, 591000, 860000)` nm, half-widths 50,000 nm
(100 µm cube) and 100,000 nm (200 µm cube), axis-aligned box filter directly
on `interface_point_nm`.

| Scope | All mouse decisions | merge_edit + split_edit only |
|---|---:|---:|
| **100 µm cube** (half-width 50,000 nm) | **6,079** | **2,514** |
| **200 µm cube** (half-width 100,000 nm) | **39,292** | **17,496** |
| **Whole MICrONS volume** | **301,162** | **145,324** |

The 100 µm cube holds about 2.0% of all mouse decisions and about 1.7% of
mouse merge/split edits — well above the ~0.1% naive-uniform-density
estimate from §5 above, consistent with this doc's stated hypothesis that
our harness region (picked as the densest gold-proofread cube in the v1822
manifest) sits in one of ConnectomeBench2's denser regions too, because both
draw on the same underlying MICrONS proofreading effort. (This is now a
measured finding, not the "reason for optimism" it was framed as earlier.)

### 6. Decision/task-type distribution among MICrONS (mouse) rows

`sample_type` values, measured directly (not the partial-sample estimate
from §1 above):

| sample_type | Whole volume | 100 µm cube | 200 µm cube |
|---|---:|---:|---:|
| split_edit | 100,463 (33.4%) | 1,684 (27.7%) | 11,968 (30.5%) |
| adjacent_control | 82,869 (27.5%) | 1,568 (25.8%) | 10,283 (26.2%) |
| junction_control | 57,084 (19.0%) | 1,736 (28.6%) | 9,727 (24.8%) |
| merge_edit | 44,861 (14.9%) | 830 (13.7%) | 5,528 (14.1%) |
| synapse_control | 15,885 (5.3%) | 261 (4.3%) | 1,786 (4.5%) |
| **Total** | **301,162** | **6,079** | **39,292** |

Real edits (`merge_edit` + `split_edit`) are 48.3% of whole-volume mouse
rows — close to, and higher than, the 55.6%-from-a-partial-sample figure
this document originally estimated in §1 (that estimate mixed species; this
one is mouse-only and is the full population, not a partial sample).

### 7. Bottom line on EXP-057B's bar

The plan's criterion (`docs/consolidation_plan.md` row 366) is **"≥1,000
mapped merge-or-split decisions in or near the cube."** Two separate claims
are bundled in that sentence, and this intake only measured one of them:

- **Spatially in-or-near the cube: MET, with room to spare.** 2,514
  merge-or-split decisions fall inside the exact 100 µm cube (2.5× the bar on
  its own), and 17,496 fall inside the 200 µm cube. Both numbers are
  measured, not estimated, from all 301,162 mouse rows with 0 missing
  coordinates.
- **"Mapped" (i.e., resolved to this repo's v117/v1822 root-id lineage via
  the `roots_binary?timestamp=` step described in §3/§6.3 above): STILL
  UNDETERMINED.** This intake deliberately did not touch CAVE or run any
  lineage resolution — that is real, separate, network-bound work, not a
  parquet-filtering question, and it was out of scope for this pass. A
  decision landing in the cube spatially is not the same as its
  `before_root_ids`/`after_root_ids` resolving cleanly onto atoms this repo
  already tracks at v117 or v1822; some fraction could fail to resolve
  (renamed/merged-away segments, operations after v1822, etc.).

So: **the spatial-availability half of the bar is comfortably cleared, and
by a wide enough margin (2.5× at exact scope, 17.5× at 200 µm) that even a
substantial lineage-mapping failure rate would plausibly still clear
1,000.** But that is an inference, not a measurement — the honest status is
"spatial supply is not the bottleneck; the mapping step is the only thing
left to measure before the bar can be called met outright."

## 7. Go / no-go

**GO**, scoped narrowly. ConnectomeBench2 is real, its MICrONS portion is
CC BY 4.0 and downloadable without CAVE credentials, its coordinate frame is
verified identical to our harness's (same datastack, same nm-per-µm-1000
convention, no transform needed), its GitHub pipeline exposes exactly the
lineage-resolution primitive (`roots_binary` at an arbitrary timestamp) that
this repo already has a working implementation of, and its third task
("mask segmentation for merge error correction") is confirmed — from the
paper's own task framing, not an inference — to be our seam-location problem.

**But**: this intake check could not establish the one number the plan's own
criterion depends on ("≥1,000 mapped merge-or-split decisions in or near the
cube") without downloading data, which was explicitly out of scope here. Do
not treat this document as having cleared that bar — it has cleared the
*existence, access, and format* questions and left the *count* question, plus
two unresolved numeric discrepancies (§2, §3), to the real EXP-057B run in
§6. If step 1 above comes back near zero even at the widened radius, fall
back to relaxing the spatial-overlap requirement (§6.2) before concluding the
corpus doesn't help — a near-zero count in one 100 µm box is not evidence
against a 401k-plus-row, CC-BY-4.0, same-coordinate-frame corpus with the
exact task we need.
