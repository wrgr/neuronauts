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
