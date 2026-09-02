# Published morphology embeddings for MICrONS: what exists and whether it helps v117 fragments

> Research probe for EXP-057C, run 2026-09-01. Feeds `docs/grammar_literature_directions.md`
> §1.8 and `docs/consolidation_plan.md` §6.3/6.4 (EXP-057C "embedding intake"). Triggered
> because geometric candidate generation is falsified: the median true continuation partner
> is 6.5 µm away and p90 is 56 µm, so neither a proximity ball nor a directed cone reaches
> enough of them at a tractable panel size. Retrieval over a morphology embedding has no
> radius, so it is the remaining option — *if* a usable one exists at the right unit.
>
> **Labelling convention used throughout:** every claim is tagged
> **(a) paper says** — the publication or repo states this, not independently checked;
> **(b) URL confirmed** — I resolved the URL / bucket listing myself (HTTP 200, or a GCS
> JSON listing returning real objects with real sizes) but did not download or parse the
> file contents;
> **(c) content verified** — I fetched and read the actual bytes (a README, a config file,
> a directory listing), not just its existence.
> No file over a few KB was downloaded in full, per the task's "do not download large
> files" rule; all sizes below come from HTTP headers or GCS object metadata, not from
> reading file contents.

## TL;DR

Two of the four candidates in the prompt are worth distinguishing sharply:

- **GraphDINO / Weis et al. 2025** — real, downloadable checkpoint, but the wrong unit for
  our problem on every axis: whole neurons only, fragments explicitly excluded by an SVM
  filter, dendrites only (axons explicitly removed), 3D-coordinate-only node features. It
  cannot address the axon-continuation gap that is actually blocking us, structurally, not
  just as an out-of-distribution risk.
- **SegCLR (Google)** — real, downloadable, per-fragment/per-skeleton-node embeddings
  covering both axon and dendrite, for the exact volume (`minnie65`), with a released
  encoder checkpoint we could run ourselves. This is the much better fit for "atoms are
  fragments, not whole cells," but the precomputed vectors are keyed to public
  materializations v343/v943, not v117, so a supervoxel-based ID crosswalk (or a
  self-run inference pass) is required before this reaches our substrate.

**Recommendation:** run EXP-057C, but retarget it at SegCLR, not GraphDINO. Details and
exact fetch list at the end.

---

## 1. GraphDINO / Weis et al. — self-supervised neuron morphology embeddings

### 1.1 What exists (a — paper/repo claims)

- Method paper: Weis, Hansel, Lüddecke & Ecker, "Self-Supervised Graph Representation
  Learning for Neuronal Morphologies," TMLR 2023 / arXiv:2112.12482
  (https://arxiv.org/abs/2112.12482, code https://github.com/marissaweis/ssl_neuron).
  Introduces the GraphDINO architecture (AC-attention transformer over spatially embedded
  graphs) and reports training on the Allen Brain Atlas (ABA) and Blue Brain Project (BBP)
  datasets in this paper — **not MICrONS**.
- Application paper: Weis et al. 2025, "An unsupervised map of excitatory neuron dendritic
  morphology in the mouse visual cortex," *Nature Communications* 16:3361
  (https://www.nature.com/articles/s41467-025-58763-w, PMC11982532, code
  https://github.com/marissaweis/unsupervised_neuronal_map). This is the paper that
  actually runs GraphDINO on MICrONS (`minnie65`) and is what
  `docs/grammar_literature_directions.md` §1.8 and §3 point 2 refer to.

### 1.2 What is downloadable (b — URL confirmed)

- **Model checkpoint, pretrained on MICrONS:**
  `https://github.com/marissaweis/unsupervised_neuronal_map/blob/main/data/graphdino/ckpts/ckpt_microns.pt`
  — confirmed resolving (HTTP 200 via `raw.githubusercontent.com`), **6,773,495 bytes**
  (~6.5 MB), plus a 714-byte `config.json` alongside it.
- **Training/inference code:** `https://github.com/marissaweis/ssl_neuron` (MIT license),
  containing `demos/model_inference.ipynb`, `demos/load_data.ipynb`,
  `demos/train_model.ipynb`, and a second checkpoint `ssl_neuron/ckpts/ckpt.pt`
  (3,530,567 bytes; presumably the ABA/BBP one used in the method paper's own figures —
  which dataset it corresponds to was not confirmed).
- **The actual per-neuron embedding vectors** (the thing we'd need for a cosine-similarity
  test) are **not in either GitHub repo**. The `unsupervised_neuronal_map` README states:
  "Data is published under 'Source data' with the paper. This includes the learned
  morphological embeddings of the MICrONS neurons as well as morphometrics computed on
  them," linking to `https://www.nature.com/articles/s41467-025-58763-w#Sec40`. Fetching
  that page redirected to Nature's login gate (`idp.nature.com/authorize?...`); **I could
  not confirm the Source Data file's format, size, or exact contents in this pass — not
  confirmed**, only that the paper claims it exists at that location.

### 1.3 Unit of the embedding (c — content verified, from PMC11982532 and the repo configs)

This is the load-bearing finding. Quoted from the paper's Methods (PMC11982532):

- **Whole neuron only, with fragments explicitly filtered out.** Of >54,000 candidate
  somas in the volume, the authors trained an SVM ("94% cross-validated accuracy") to
  separate manually proofread neurons from fragmented ones, then **"removed cells
  predicted to be fragmented (n = 6304) from subsequent analyses,"** arriving at 31,313
  neurons that were embedded and analyzed. The `unsupervised_neuronal_map` repo ships this
  as an explicit step: `evaluation_pipeline/quality_control/remove_l23_missing_apical.ipynb`
  and `remove_l5_fragments.ipynb`.
- **Dendrites only, axons explicitly removed.** Quote: *"Because axons have not been
  reconstructed well in the data yet, we focused on the dendritic skeleton only and
  removed segments labeled as axon."* There is no axon embedding from this source at any
  granularity.
- **Node features are xyz only, soma-centered, no radius.** `ssl_neuron/configs/config.json`
  (fetched directly) has `"feat_dim": 3` and `"n_nodes": 200`; the data README states the
  released checkpoint "was trained after removal of the axons and centering each neuron
  such that the soma coordinate is (0, 0, 0). Only xyz-coordinates were used as node
  features." Skeletons are mesh→skeleton via NEURD, then randomly subsampled to 200 nodes
  while always retaining branch points.

### 1.4 Input requirements / could we run it ourselves? (c for format, a for feasibility)

The `ssl_neuron` custom-data README (fetched) specifies exactly what a sample needs:
a per-sample `features.npy` (N×3 xyz array) and `neighbors.pkl` (node-adjacency dict), and
`model_inference.ipynb` demonstrates loading a checkpoint and running inference. So the
*mechanics* of running the released checkpoint on new skeletons are genuinely available.

But running it on a v117 fragment would be out-of-distribution on three independent axes,
not just one:
1. No soma to center coordinates on (a fragment usually is not the soma-containing piece).
2. The 200-node subsample-while-keeping-branch-points procedure assumes a roughly
   whole-tree input; a small partial arbor does not carry the same structural signal.
3. It is dendrite-trained only — an axon fragment (the population we most need, since
   `docs/consolidation_plan.md` records H3 "endpoint proximity alone identifies
   continuations" as **falsified for axons**, 0/32 adjudicated) is entirely outside what
   this model was ever shown.

### 1.5 Fit to v117 fragments: poor, structurally

Even setting aside the paywalled Source Data, this resource cannot address our actual
blocking problem (axon continuation) because axons were deliberately excluded from
training and evaluation, and it cannot be applied to partial fragments without retraining,
because fragments were deliberately excluded from the paper's own population. Using it
would mean building and validating a new fragment/axon-capable GraphDINO from scratch on
our own atoms — a multi-week research project, not an intake task.

---

## 2. SegCLR (Google) — segmentation-guided contrastive embeddings

### 2.1 What exists (a — paper claims)

Multi-layered maps of neuropil with segmentation-guided contrastive learning,
*Nature Methods* 20:2011–2020 (2023), PMC10703674
(https://www.nature.com/articles/s41592-023-02059-8). Trained on **both** H01 (human
temporal cortex) and **MICrONS `minnie65`** (mouse visual cortex) — our exact volume.

### 2.2 What is downloadable (b — URL confirmed; several files' content also read, so partly c)

All of the following were independently resolved this session (HTTP 200 / real GCS
listings with sizes), not merely cited from documentation:

- **Format README** (content verified, fetched in full —1,088 bytes):
  `gs://iarpa_microns/minnie/minnie65/embeddings_m343/segclr_csvzips/README`
  (`https://storage.googleapis.com/iarpa_microns/minnie/minnie65/embeddings_m343/segclr_csvzips/README`).
  Quoted in full below since it defines the unit (§2.3).
- **Raw per-node embeddings**, voxel coordinates, MD5-sharded (10,000 shards):
  `gs://iarpa_microns/minnie/minnie65/embeddings_m343/segclr/by_id/*.shard` — sampled
  shards confirmed real (`000.shard` = 298,570,303 bytes, `001.shard` = 315,949,061 bytes,
  etc.). Needs a manual coordinate offset (below) to align to public-release space.
- **Cleaner, coordinate-corrected version (recommended entry point):**
  `gs://iarpa_microns/minnie/minnie65/embeddings_m343/segclr_nm_coord_public_offset_csvzips/*.zip`
  — confirmed resolving, shards ~200–300 MB each (e.g. `0.zip` = 220,460,316 bytes).
- **Aggregated, coarser-grained variants** (from the SegCLR wiki page, and independently
  confirmed by direct GCS listing — real objects with real sizes returned):
  `..._aggregated_10um_csvzips/` (e.g. `0.zip` = 203,008,890 bytes) and
  `..._aggregated_25um_csvzips/` (e.g. `0.zip` = 170,562,475 bytes). These pool multiple
  skeleton-node embeddings into fewer rows per segment (roughly one every 10 or 25 µm of
  cable); exact column schema for the aggregated variant was **not confirmed** (no README
  was found at that specific prefix — a 404 was returned when checked).
- **A second full copy keyed to a newer materialization,** `embeddings_m943/` (v943, the
  January-2024 public release), and an un-suffixed `embeddings/` tree that also contains
  `segclr/`, `segclr_csvzips/`, `segclr_e0-32/`, `segclr_e32-64/`, `models/`, and
  `training_data/` subfolders (listed directly via the GCS API).
- **A released encoder checkpoint** (i.e., not just precomputed vectors — the actual
  trained network), confirmed by direct listing:
  `gs://iarpa_microns/minnie/minnie65/embeddings/models/segclr-216000/` containing
  `checkpoint` (135 B), `model.ckpt-216000.training_tf2resaved-1.data-00000-of-00001`
  (135,112,008 bytes ≈ 135 MB), and `.index` (3,784 B) — a TensorFlow 2 checkpoint.
- **Training/inference code:** `https://github.com/google-research/connectomics/tree/main/connectomics/segclr`
  (Apache-2.0), confirmed via directory listing: `model.py`, `inference.py`, `resnet.py`,
  `reader.py`, `encoders.py`, `objective.py`, `model_util.py`, plus `classification/` and
  `tf2/` subdirectories. The project wiki
  (`https://github.com/google-research/connectomics/wiki/SegCLR`) references a companion
  Colab, "Run a pretrained SegCLR embedding model from TensorFlow 2," demonstrating loading
  this checkpoint and running inference on arbitrary EM+segmentation cutouts via
  TensorStore — **this notebook itself was not opened/verified in this pass (a only)**.

### 2.3 Unit of the embedding (c — content verified, from the fetched README)

Quoted verbatim from `.../embeddings_m343/segclr_csvzips/README`:

> "The SegCLR embedding nodes are saved in a sharded set of ZIP archives. Each segment ID
> is assigned to a ZIP archive according to [an md5-based sharding function] with
> `num_shards == 10_000`. Within each ZIP archive, the paths are `segment_id.csv`. Within
> each CSV, each row has the fields: `node_id, x, y, z, e0, e1, e2, ..., e63` ... x, y, z:
> in units of voxels; 32x32x33 nm resolution for the human cortex dataset, 32x32x40 nm
> resolution for the mouse cortex dataset."

This is exactly the granularity the task asked about: **one CSV per segment (fragment)
ID, one row per skeleton node**, 64-dimensional float embedding per row, nodes spaced
roughly 1.5 µm apart along the object (per the paper's stated sampling density). Crucially,
per the PMC10703674 text: embeddings were computed for **"all non-trivial objects"**
(segments ≥1,000 voxels) — **no proofreading requirement, no whole-cell requirement, and
axon and dendrite are both covered** (the contrastive positive pairs are just "nearby
locations on the same segmented cell," regardless of compartment). This is the opposite of
GraphDINO on every axis that matters here.

Also confirmed by direct README read: there is a known coordinate offset for the mouse
dataset between the raw `by_id` embeddings and the public-release coordinate frame
(`+110592, +110592, +592640` nm) — already resolved for you in the `_nm_coord_public_offset_`
variant, which is why that is the recommended entry point over raw `by_id`.

### 2.4 Fit to v117 fragments — the one real gap, and it's tractable

Per PMC10703674 (fetched directly): *"We trained SegCLR models on the public segmentation
version 117 and then upgraded to the public version 343 for all evaluations and
analyses."* So **v117 is literally the segmentation SegCLR itself was originally trained
against** — but the publicly *released* precomputed CSV/ZIP tables are id-keyed to the
later public materializations, **v343 and v943, not v117**. Two consequences:

1. A v117 fragment's segment ID does not directly index into the released embedding
   tables. You need a supervoxel→root crosswalk (CAVE chunkedgraph `get_roots` on the
   fragment's constituent supervoxels, evaluated at the v343 or v943 timestamp) to find
   which `segment_id.csv` file(s) hold embeddings for a given v117 atom's footprint. This
   is the same kind of operation `docs/consolidation_plan.md` §6.4 already relies on for
   the v117→v1822 GT overlay (supervoxel-majority lineage), so the mechanism is known to
   work in this codebase, but it is **not already built for v343/v943** and a v117
   fragment need not map 1:1 onto a v343/v943 object (splits and merges differ across
   materializations) — some atoms will map cleanly, others will straddle multiple v343
   objects or vice versa.
2. Alternatively, because the *encoder checkpoint itself* is released
   (`segclr-216000`) and its input is an EM image + segmentation-mask crop at an arbitrary
   coordinate (not tied to a specific object ID or materialization), we could in principle
   run SegCLR inference ourselves directly on our own v117 objects using the same public
   EM volume (imagery is shared across all segmentation versions) and our own v117
   segmentation. This sidesteps the ID-crosswalk problem entirely, at the cost of
   reproducing a TensorStore-based inference pipeline (documented in the wiki/Colab, but
   not independently verified end-to-end this session) and needing a TF2 environment.

### 2.5 Other SegCLR-adjacent resources found (a — noted for completeness, not the ask)

Under `gs://iarpa_microns/minnie/minnie65/embedding_classification/` there are derived
classifiers built on top of SegCLR embeddings: `subcompartment_10um_BERT_SNGP_20220819/`
(axon/dendrite/soma-type prediction per point) and `celltype13_...`/`celltype3_...`
(cell-type prediction). These are categorical outputs, not embeddings, but could be a
cheap independent compartment-polarity signal if ever needed (existence confirmed by
listing only; contents not inspected).

---

## 3. NEURD

`https://github.com/reimerlab/NEURD` (Celii et al., *Nature* 2025, PMC/DOI
10.1038/s41586-025-08660-5) decomposes meshes into annotated graphs and applies
hand-engineered heuristic rules for proofreading and feature extraction — **not a learned
embedding**. No embedding or feature-table data release was found in the repo (a — repo
description only; no data folder, Zenodo, or figshare link located). Its actual relevance
here is as an *upstream dependency*: NEURD is the tool the Weis et al. GraphDINO pipeline
used to convert MICrONS meshes into skeleton graphs (§1.1), not an independent embedding
source in its own right. `docs/grammar_literature_directions.md` §1.9 separately notes
ConnectomeBench found NEURD "detected zero merge errors" on its test cohort, consistent
with hand-rule brittleness.

## 4. `aibs_metamodel_celltypes` (CAVE table)

Confirmed via the MICrONS tutorial docs (content verified): this table (e.g.
`aibs_metamodel_celltypes_v661`) contains **categorical** cell-type labels only (values
like `"5P-IT"`, `"23P"`, `"BC"`, `"astrocyte"`, `"oligo"`), keyed on `pt_root_id`, one row
per whole cell. It is **not a continuous embedding** and provides no vector to retrieve
against. It could still be useful as the discrete class-conditioning variable
`docs/grammar_literature_directions.md` §1.6 discusses for a compound-PCFG mixture, but it
does not answer this task's question.

---

## Summary table

| Resource | Downloadable artifact confirmed? | Unit | Axon coverage | Fragment coverage | Segmentation version | Fit for v117 fragments |
|---|---|---|---|---|---|---|
| GraphDINO / Weis 2025 checkpoint | Yes, 6.5 MB (b) | Whole neuron | **No — excluded** | **No — explicitly filtered out (SVM, n=6,304 removed)** | Not stated precisely; `minnie65` | Poor — wrong unit on 3 axes |
| GraphDINO / Weis 2025 embeddings (the vectors) | Not confirmed (paywalled Source Data) | Whole neuron | No | No | — | N/A — could not even confirm access |
| SegCLR precomputed embeddings | Yes, per-shard sizes confirmed (b) | Per-segment, per-skeleton-node (64-d) | **Yes** | **Yes — designed for it, ≥1,000-voxel objects** | Trained on v117, released tables keyed to **v343 / v943** | Needs an ID crosswalk (tractable) |
| SegCLR encoder checkpoint | Yes, 135 MB TF2 checkpoint (b) | N/A (produces the above) | Yes | Yes | Version-independent (EM image + any seg mask) | Best fit — bypasses ID crosswalk if we run it ourselves |
| NEURD | Code yes, data no | Fragment/branch (features, not embeddings) | n/a | n/a | n/a | Not an embedding source |
| `aibs_metamodel_celltypes` | Yes (CAVE table, standard) | Whole cell | n/a | n/a | v661 etc. | Categorical only, not an embedding |

---

## Recommendation on EXP-057C

**Run it, but retarget it.** As originally scoped in `docs/consolidation_plan.md` §6.3
("Check whether Weis et al. 2025 release per-root GraphDINO embeddings; if so join to the
gold manifest and test cosine separation"), EXP-057C is not worth running against
GraphDINO: even if the paywalled Source Data turned out to be accessible, the released
resource structurally excludes both fragments and axons — the two properties of our
substrate that are actually causing the geometric-candidate-generation failure. A positive
result there would not transfer to our problem, and a negative result would not be
informative either, since the population never overlapped.

**SegCLR is the resource that matches the brief:** it is explicitly segment/fragment-level
(not whole-cell), covers both axon and dendrite, and was originally trained on the very
segmentation version (v117) our substrate uses, even though the public release is keyed to
later materializations. Redirect the EXP-057C half-day slot to it:

1. **Fetch (small, half-day-sized):** one or two shards of
   `gs://iarpa_microns/minnie/minnie65/embeddings_m343/segclr_nm_coord_public_offset_csvzips/`
   (or the `aggregated_10um`/`aggregated_25um` variant for a coarser, cheaper first pass),
   restricted spatially to whatever shard(s) cover the 100 µm harness cube
   (center 663/591/860 µm) — no need to pull all 10,000 shards.
2. **Crosswalk:** for the 56 mixed-lineage, proofread-owned gold atoms
   (`results/atom_labels_v1822.json`), resolve each atom's constituent supervoxels to their
   v343 (or v943) root ids via the CAVE chunkedgraph, using the same supervoxel-majority
   machinery already used for the v1822 GT overlay.
3. **Test:** exactly the bar already specified in the consolidation plan — cosine
   separation of same-lineage vs. different-lineage fragment pairs on the harness, the same
   test tree-DNA has to pass — but against SegCLR vectors instead of GraphDINO.
4. **Decision rule:** if crosswalk coverage near the cube is too sparse or too many-to-many
   to trust (v117↔v343 object boundaries disagree materially), that itself is a valid,
   reportable negative result, and the fallback is the self-run-inference path (§2.4,
   point 2) — using the released `segclr-216000` TF2 checkpoint directly on our own v117
   EM+segmentation crops, which avoids the crosswalk problem but is a multi-day effort, not
   a half-day probe, and should be scheduled as its own follow-on experiment rather than
   folded into EXP-057C.

No large files were downloaded to produce this report; all existence/size claims above are
from HTTP HEAD requests, GCS JSON bucket listings, or full reads of small (<7 KB) README /
config files.

---

## Spike: can we use these on v117?

> Follow-on network+data spike, run 2026-09-02, on top of the recommendation above. Everything
> in this section is **measured** (fetched bytes, parsed content, live CAVE/CloudVolume calls,
> compared against this repo's own ground truth) unless a line is explicitly marked
> **(inferred)**. All downloads went to the scratchpad, not the repo; nothing here was
> committed.

### Step 1 — fetch one shard (measured)

Pulled `.../embeddings_m343/segclr_nm_coord_public_offset_csvzips/0.zip` over plain HTTPS
(public bucket, no auth needed): **220,460,316 bytes**, byte-for-byte equal to the GCS-listed
object size — not truncated.

No README exists inside `segclr_nm_coord_public_offset_csvzips/` itself (`404` on `README`,
`README.txt`, `README.md`); the offset convention is documented one directory over, in
`segclr_csvzips/README` (fetched, 1,088 bytes, quoted in §2.3 above): raw embeddings are in
**voxels** (32×32×40 nm for mouse) and need **+110592, +110592, +592640 nm** added to align to
the public coordinate frame. I cross-checked that offset independently: the volume's own
Neuroglancer state (`segclr_v2.json`, fetched) carries an annotation-layer transform of
`(13824, 13824, 14816)` **voxels** at 8×8×40 nm resolution for the *raw* `by_id` annotation
source — `13824×8 = 110592` nm, `13824×8 = 110592` nm, `14816×40 = 592640` nm. Exact match to
the README's offset, from a second, independent source. The `_nm_coord_public_offset_`
directory name is accurate: as parsed below, its coordinates come out already in nm and already
in the public frame (verified in Step 3 by falling inside our own cube's known nm bounds; no
further shift needed).

### Step 2 — actual format (measured)

Parsed `0.zip` fully (23,982 entries):

| Quantity | Value |
|---|---:|
| Zip entries (`{segment_id}.csv`) | 23,982 |
| Total rows (skeleton nodes) | 375,526 |
| Distinct segment ids | 23,982 (1 CSV per id, as documented) |
| Embedding width | 64 (`e0..e63`) — confirmed, 0 rows with a different column count |
| Rows/segment | min 1, max 29,575, mean 15.7 |
| x range (nm) | 216,608 – 1,686,560 |
| y range (nm) | 278,720 – 1,239,488 |
| z range (nm) | 596,560 – 1,111,720 |

A row is exactly `node_id, x, y, z, e0..e63` as the sibling README states. Coordinates are
**nm, not voxels** — confirmed two ways: (a) the raw values (e.g. `305888.0, 534496.0,
772200.0`) are on the right order of magnitude for minnie65 in nm and wildly wrong for voxels
at any documented resolution; (b) a volume-density sanity check: this one 1-in-10,000 md5 shard
spans a bounding box of ≈1.47 × 0.96 × 0.52 mm ≈ 0.73 mm³, in the right ballpark for minnie65's
published extent (≈0.88 mm³), and scaling 23,982 segments in ≈0.73 mm³ down to our 0.001 mm³
(100 µm)³ cube predicts ≈27 segments landing inside it — close to the 37 actually measured in
Step 3 (see there for why it isn't geographic sampling and some spread is expected).

### Step 3 — overlap with our cube (measured, no widening needed)

Filtering shard 0 to `|x-663000| ≤ 50000, |y-591000| ≤ 50000, |z-860000| ≤ 50000` (nm) gave
**440 rows across 37 distinct segment ids** directly — the first shard tried already covers our
region, no need to check a wider box or hunt for a different shard.

This is expected once you read the sharding rule correctly: `md5_shard(segment_id, 10000)` is a
hash of the **segment id**, not of its location. Shard 0.zip is not "the region near the
origin" — it's a pseudo-random **1-in-10,000 subsample of every non-trivial object in the
entire dataset**, spatially spread out just like the full population. That's why the very first
shard already contains ~37 objects overlapping any given small cube, matching the density
math above.

### Step 4 — the crosswalk

**A bug I made and caught before trusting the result (per this repo's CLAUDE.md, logged in
full because it's instructive):**

My first attempt used `neuronauts.fetch.MICRONS_SEG_PATH` (`precomputed://https://bossdb-open-
data.s3.amazonaws.com/.../minnie65/seg`, the path other repo code uses for `fetch_seg_volume`)
at mip 2, assuming it returns raw, timestamp-invariant supervoxels. All 37/37 point queries
"succeeded" and fed cleanly into `roots_at(..., V117_TIMESTAMP)` — but the returned "v117 root"
was **identical to the input SegCLR m343 segment id in 37/37 cases**. That is not a plausible
coincidence, and CLAUDE.md's rule 0 says to suspect my own call before anything else. I checked
by treating a few of those ids as roots and calling `root_leaves(id, stop_layer=1)`: they came
back with 13, 75, 22, and 700 supervoxel leaves — real, multi-supervoxel agglomerated objects,
not raw supervoxels. **`MICRONS_SEG_PATH`'s static, unversioned `/seg` export is itself an
already-agglomerated flat segmentation** (apparently frozen at a state coincident with m343 for
never-since-edited components), not the raw watershed layer. Using it for a "get the
supervoxel at this point" step would have been silently wrong — it happened to produce
plausible-looking numbers for this batch of atoms only because they turn out to be unedited.

**Corrected method:** query the *live* graphene segmentation source
(`client.info.segmentation_source()`) at **mip 0** (8×8×40 nm, the true supervoxel layer) with
**`agglomerate=False`**, then `roots_at(sv, V117_TIMESTAMP)` for the real crosswalk.

**Validated against this repo's own ground truth before trusting it on SegCLR data:** 15
random synapses from `data/substrate/c100um/population.npz` (whose `syn_atom_pre`/
`syn_atom_post` were computed independently, from the synapse table's own
`pre_pt_supervoxel_id`/`post_pt_supervoxel_id` columns, not from coordinates) were run through
`xyz → CloudVolume(graphene, mip0, agglomerate=False) → roots_at(V117_TIMESTAMP)`.
**15/15 exact matches** to the already-trusted supervoxel-column-based atom id. Cost: ~700 ms/
item (dominated by the live point query; `roots_at` batches and is cheap).

**Applied to the 37 SegCLR segments found in the cube:** 37/37 raw-supervoxel lookups
succeeded (nonzero), 37/37 resolved to a nonzero v117 root, and **22/37 (59%) landed exactly in
our existing 279,075-atom substrate population** (`data/substrate/c100um/population.npz`). Cost
end-to-end: 246 ms/item (217 ms/item for the point query, `roots_at` amortizes to ~13 ms/item in
one batch POST).

The 15/37 that don't land in the population are not a crosswalk failure: our population is
defined as "every v117 root with a **synapse side whose center** falls in the cube" (label-
blind, per `neuronauts/harness/population.py`), while SegCLR embeds **any non-trivial object**
(≥1,000 voxels) regardless of synaptic contact — glia, myelin, and dendrite/axon stretches
between synapses all get nodes. A SegCLR node landing geometrically inside the cube on such an
object, with zero synapses centered there, correctly resolves to a valid v117 root that is
correctly *absent* from our synapse-anchored atom list. I did not individually confirm all 15
have zero synapses (that would need a further per-root synapse query), but the population's own
inclusion rule fully accounts for the gap without needing to invoke any bug.

**A striking, separately useful finding:** for all 37/37 sampled objects, `v117_root == m343
segment_id`, i.e. these particular fragments were **never edited** between the v117 (2021) and
m343 (Feb 2022) timestamps — their chunkedgraph id simply never changed. This is consistent
with (not independently proof of) EXP-057's finding that 83.8% of this cube's synapse mass sits
on objects no human has proofread: most objects here have no edit history at all across
materializations, so for the *unproofread* majority, "the crosswalk" often reduces to "the id
doesn't change" — the real crosswalk work is concentrated on the proofread minority, which is
exactly the population Step 5 needs.

**Verdict on the crosswalk: tractable, cheap, and now validated end-to-end.** ~700 ms per
xyz→v117-atom resolution, 100% agreement with ground truth on the validation sample.

### Step 5 — does it separate same-owner from different-owner atoms?

**Random sampling from one shard cannot reach this test — a measured, not assumed, limit.**
Of the 22 embedded objects that landed in our population, **0/22 fell in the 4,802-atom
"pure, proofread-owned" gold pool** used for ground-truth ownership
(`data/substrate/c100um/labels_v1822.npz`). That pool is only 1.72% of the 279,075-atom
population (4,802/279,075), and only 59% of embedded-in-cube objects land in the population at
all, so the expected yield is ≈22 × 1.72% ≈ **0.4 gold atoms per random shard** — the measured
0/22 is exactly consistent with that rate, not a sign anything is broken.

**Targeted (not random) fetch, so every download counts:** instead of hoping a random shard
contains a gold atom, I forward-crosswalked known gold atoms to their own m343 root
(`xyz → supervoxel → roots_at(m343_timestamp=1645690200)`, same validated pipeline as Step 4,
now run forward instead of back to v117) and computed which of the 10,000 shard files must hold
it via `md5_shard(m343_root, 10000)` — first validating my reimplementation of that hash against
the 37 segments already known (from Step 3) to live in shard 0: **37/37 correctly predicted
shard 0.** Forward crosswalk succeeded 60/60 times tried (~730 ms/item).

Selected the **top 12 owner cells by atom count** (a disclosed, non-random stratification —
chosen to guarantee same-owner pairs exist in a small sample; it picks *which* atoms to test,
not the embedding values themselves), capped at 3 atoms/owner: **36 atoms, 35 distinct shard
files, ≈8.8 GB.**

**A second bug I made and caught:** the first download batch (6-way parallel `curl` with
`--max-time 120`) silently truncated 9 of the 35 files — under shared bandwidth, some
200–300 MB shards needed more than 120 s, `curl -s` gave no error, and I hadn't checked exit
codes or sizes. Caught by comparing every local file's byte count against the GCS object
listing's `size` field before trusting any of it (per CLAUDE.md: "verify correctness against
ground truth, not vibes" — the object listing *is* ground truth here). Re-fetched the 9
mismatches serially with a longer timeout; all 9 then matched the remote size exactly.

**Result, from the corrected data:** 34/36 atoms got an embedding (per-atom vector = the
embedding row of the node nearest that atom's own synapse point, guarding against a much larger
already-merged m343 parent object diluting the local signal). The other 2/36 are a distinct,
minor, expected miss — the forward-mapped m343 root wasn't present as its own CSV in its
predicted shard (a numerically adjacent id was; plausibly a sub-1,000-voxel object or one that
changed between query time and m343's exact timestamp), not a repeat of the truncation bug.

| | value |
|---|---:|
| Atoms embedded | 34 (12 distinct owners) |
| Same-owner pairs | 33 |
| Different-owner pairs | 528 |
| Same-owner cosine (mean ± sd) | 0.823 ± 0.118 |
| Different-owner cosine (mean ± sd) | 0.842 ± 0.118 |
| **AUC (nearest-node embedding)** | **0.445** |
| AUC (mean of 5 nearest nodes, robustness check) | 0.461 |
| Tree-DNA within-type AUC (the bar) | 0.829 |

Both pooling variants land at essentially chance, slightly on the wrong side of 0.5 (different-
owner pairs are *marginally more* similar on average than same-owner pairs). Per-owner spread
shows this isn't uniform noise — some owner groups cluster tightly (e.g. one owner's 3 atoms:
cosines 0.95/0.93/0.92) while others don't (e.g. another owner's 3 atoms: 0.55/0.55/0.96) — but
whatever identity signal exists is inconsistent enough across owners that it doesn't produce
separation from the cross-owner background at this sample size.

**Caveats on this AUC, stated plainly:**
- **n is small.** 34 atoms in 12 owner-clusters is closer to 12 independent samples than 34;
  wide uncertainty is expected, and I did not compute a formal interval.
- **Type is not controlled.** The 0.829 tree-DNA bar is specifically a *within-type* number
  (`scripts/within_type_ablation.py` exists precisely because cross-type pairs are trivially
  separable and inflate the naive score). My 12 owners were picked only by atom count, not
  matched by cell type — if several are the same type, local-window embeddings failing to
  separate individuals within a type would look exactly like this result, and it would not be
  a fair comparison to 0.829. I did not check owner cell types this pass.
- **Context size varies enormously and unexamined:** the m343 parent segment each embedding
  came from ranged from 1 node (a still-unedited fragment) to 26,230 nodes (an almost-whole,
  heavily proofread neuron) — because "gold" ownership is anchored to the much later v1822
  timestamp, while the embedding is anchored to m343 (Feb 2022); how much proofreading had
  happened to a given cell by m343 varies atom to atom. Untested whether this confound matters.
- **Only one pooling strategy per node was tried per variant** (single nearest node; mean of 5
  nearest). SegCLR's own aggregated 10 µm / 25 µm precomputed tables, or mean-pooling over the
  *entire* parent object rather than just the local neighborhood, were not tried and could
  behave differently — untested, not ruled out.

### Verdict: is EXP-057C worth building as a real experiment?

**Split verdict.** Two independent things were tested here and they point different directions:

1. **The v117↔m343 id crosswalk itself is solid, reusable infrastructure.** 100% agreement
   with ground truth (15/15), ~700 ms/atom, works in both directions (v117→m343 for shard
   targeting, m343→v117 for population membership). This part is done and should be reused by
   any future embedding-intake experiment, SegCLR or otherwise.
2. **The actual signal test, as run here, does not clear the bar and should not be scaled up
   as-is.** AUC 0.44–0.46 against a bar of 0.829, reproduced under two pooling choices, on a
   small but not trivial (34-atom, 12-owner) targeted sample, with a plausible (not proven)
   explanation — type not controlled — for part of the gap.

**Recommendation: do not fund a full-scale EXP-057C around single-node (or 5-node-local-mean)
SegCLR cosine similarity for same-cell/different-cell atom separation on this substrate.** The
gap from 0.829 is large enough, and reproduced across two variants, that it is unlikely to be
closed by more of the same. Before spending the ~25–70 GB / multi-hour fetch a properly powered
version would need (≥150–300 gold atoms at the ~1.72% gold-atom base rate measured here, even
using the efficient targeted-shard method), a **cheap, no-new-data preliminary check** would
settle the "was it just uncontrolled type" question first: pull cell types for these same 12
owners from the already-confirmed-available `aibs_metamodel_celltypes` table and see whether
the 12 owners are mostly one type. If they are, a proper within-type SegCLR test is the fair
comparison to 0.829 and is worth the larger fetch; if they're already mixed types and it still
doesn't separate, that closes the question without spending more on SegCLR.

**If EXP-057C is pursued further despite this, it needs to do all of the following (not just
repeat this spike at larger scale):**
1. Reuse the validated crosswalk exactly as implemented (§Step 4 method) — do not use
   `MICRONS_SEG_PATH`/`fetch_seg_volume`-style static-volume queries for supervoxel lookups
   elsewhere in this repo without checking they're not hitting the same already-agglomerated-
   export trap.
2. Fetch shards via deterministic `md5_shard(m343_root, 10000)` targeting per gold atom, not
   random shards — ~100× the hit rate (100% vs ~1% measured here).
3. Scale to ≥150–300 gold atoms for a trustworthy AUC (this spike's 34/12-owner sample is a
   pilot, not a result to act on alone) — at ~230 MB/shard this is a genuine multi-hour, ~25-70
   GB fetch, not a half-day task.
4. Control for cell type (via `aibs_metamodel_celltypes`) and report both a raw and a
   within-type AUC, so the number is comparable to tree-DNA's own 0.829.
5. Try at least the whole-parent-object mean-pooled embedding and the precomputed
   `aggregated_10um`/`25um` tables as alternate poolings, since the single-node/5-node-local
   variants tested here are the cheapest but not the only reasonable choice.
6. Decision rule: if the type-controlled, better-pooled variant still doesn't clear roughly
   0.7, deprioritize SegCLR for candidate generation and lean on the lineage-anchored-
   scaffolding alternative already identified in `results/EXP-061/evaluation.md` (59.6% of
   synapses at 99.8% purity via containment in a certified soma-owned scaffold, no embedding
   needed).

All data for this section was produced in a session scratchpad under `/tmp` and
has since been copied to `data/external/segclr/` (gitignored, like the rest of
`data/`): `crosswalk_sample_v2.json` (Step 4), `gold_targets.json` /
`selected_targets.json` / `auc_result.json` (Step 5) and the spike scripts under
`scripts/`. The downloaded embedding shards (`0.zip`, `shards/*.zip`, ~8 GB) are
re-fetchable from the public bucket and were copied alongside only if disk
allowed; check `data/external/segclr/` before re-downloading. Nothing from this
spike is committed.
