# Cut-face fingerprints: can micro-ultrastructure re-link a severed neurite?

A small, local experiment probing whether EM neurons carry an idiosyncratic
"fingerprint" — a locality-sensitive **hash** of their cross-sectional
ultrastructure — that uniquely re-links the two faces of a segmentation break,
in the spirit of barcode-based identity methods (FISSEQ et al.).

## The idea

EM connectomics has a *self-inflicted* reconstruction problem. To image the
tissue at all, we slice a continuous 3D object into ~40 nm sections and then ask
an algorithm to stitch it back together. **Every true split error is a cut
through what was, physically, one continuous process.** Across that cut the
local ultrastructure — caliber, mitochondria, the axoplasmic / smooth ER,
microtubule packing, membrane texture — was continuous.

So the question becomes concrete and falsifiable:

> Given the top face of a severed neurite, can a hash of its local
> ultrastructure rank the **true** bottom partner as its nearest neighbour,
> against a panel of distractor neurites crossing the same plane — and does it
> beat the trivial spatial-proximity cue that current pipelines already use?

Ground truth is free: the public MICrONS proofread segmentation gives the
*same* id to both sides of the cut, so the true partner of top-face `k` is
bottom-face `k`. We cut a small volume at its z-midplane, remove a gap, hash the
faces on either side, and rank matches.

No CAVE token needed — only the public EM + segmentation precomputed volumes.

## What we measure

Three hashes, plus baselines, all scored by top-1 re-identification:

| Method | What it uses |
|---|---|
| `spatial` (baseline) | xy centroid proximity only — the cue current pipelines lean on |
| `scalar` hash | aggregate ultrastructure summaries (caliber, intensity stats, dark-blob count, eccentricity) — position-free |
| `PATCH` hash | the cross-section **image** itself: a translation-normalised, masked mean-intensity patch — keeps the *arrangement* of internal structure |
| `fused` | standardised spatial + patch distances added |
| `chance` | `1 / n_candidates` |

Three honesty controls:

- **Gap sweep** (40 → 640 nm). Separates a true idiosyncratic *fingerprint*
  from mere local *continuity*. A tiny gap is trivially matchable because
  neurites are locally smooth; the question is how far the signal survives.
- **Per-section normalisation** (`norm` column, `T`/`F`). EM contrast varies
  section-to-section (staining / imaging batch effect). Two nearby faces share
  that batch effect — a *processing-artifact* hash, not biology. We re-score
  with each section z-scored to its own tissue; signal that survives is
  structural.
- **Hard subset.** Cases where spatial proximity picks the **wrong** partner —
  the genuinely ambiguous breaks. `patch_on_hard` is the fraction of those the
  image hash recovers (chance ≈ `1/n`).

## How to run

```bash
pip install numpy scipy cloud-volume matplotlib   # plus a working `cryptography`

# MIP1 (16 nm) — fast smoke run
python -m experiments.fingerprints.fingerprint_break_resolution \
  --mip 1 --size 320 320 80 \
  --out experiments/fingerprints/results_mip1_smoke.json

# MIP0 (8 nm) — resolves finer ultrastructure (ER, microtubule bundles)
python -m experiments.fingerprints.fingerprint_break_resolution \
  --mip 0 --size 512 512 64 \
  --figure experiments/fingerprints/cutface_montage_mip0.png \
  --out experiments/fingerprints/results_mip0_smoke.json
```

## What we found (smoke runs, single ~4–8 µm box)

`results_mip0_smoke.json` (8 nm, 117 candidate neurites at the cut plane):

```
gap_nm norm Ncand chance | top1: spatial  scalar  PATCH  FUSED | hard(N) patch_on_hard
    40   F   117  0.009 |       0.716   0.018   0.220   0.596 |   31   0.161
    80   F   118  0.008 |       0.626   0.009   0.140   0.514 |   40   0.050
   160   F   117  0.009 |       0.485   0.010   0.087   0.330 |   53   0.057
   320   F   121  0.008 |       0.289   0.062   0.082   0.206 |   69   0.058
   640   F   119  0.008 |       0.101   0.038   0.101   0.190 |   71   0.113
```

(MIP1 in `results_mip1_smoke.json` is qualitatively identical, patch top-1
≈ 0.30 at a 40 nm gap.)

**Read-out:**

1. **Cross-section *pattern* carries real identity; scalar summaries do not.**
   The image-patch hash is ~20–30× chance and an order of magnitude above the
   scalar-summary hash. Where you put the mitochondria/ER and what the footprint
   shape is matters; mean intensity and a blob count do not.

2. **It resolves a meaningful slice of the genuinely ambiguous breaks.** On the
   *hard* subset — where proximity picks the wrong partner — the image hash alone
   recovers ~16% at a 40 nm gap (chance < 1%). That is identity information
   beyond position.

3. **But it is a short-range hash, not a long-range barcode.** Matchability
   decays over a few hundred nm. The signal behaves like *local continuity* of
   ultrastructure, not a globally unique molecular barcode. Useful for the
   short gaps that dominate real splits; not a whole-cell identifier.

4. **The signal is substantially structural, not just staining.** Per-section
   normalisation (`norm=T`) reduces but does not kill it — so it is not merely
   the section-level contrast batch effect.

5. **Naive fusion with proximity does not win.** For small artificial gaps
   proximity is near-perfect, so equal-weight fusion adds noise. The payoff is
   as a **tie-breaker** on ambiguous edges, not a global replacement for
   proximity.

![cut-face montage](cutface_montage_mip0.png)

*Each row: query top face · true partner · the hash's top pick. The query faces
are visibly brighter (a per-section staining offset) yet still match — the hash
keys on shape/texture pattern, which it survives.*

## The learned version (`learned_cutface_encoder.py`)

A small CNN trained with a contrastive (NT-Xent) objective to pull the two
faces of the *same* neurite (sampled at different z) together and push different
neurites apart. The embedding is then used exactly like the raw-patch hash.
Train and test boxes are spatially disjoint, so the encoder is scored on
neurites it never saw.

```bash
pip install torch        # CPU is fine
python -m experiments.fingerprints.learned_cutface_encoder \
  --mip 1 --size 320 320 80 --epochs 45 --steps-per-epoch 50 \
  --out experiments/fingerprints/cutface_encoder.pt \
  --metrics experiments/fingerprints/learned_metrics.json
```

Held-out test box (`learned_metrics.json`, 165 candidate neurites):

```
gap_nm norm Ncand chance | top1: spatial   raw   LEARNED | hard(N) learned_on_hard
    40   F   165  0.006 |       0.599  0.212   0.190 |   55   0.182
   160   F   158  0.006 |       0.361  0.084   0.101 |   76   0.066
   320   F   162  0.006 |       0.254  0.035   0.044 |   85   0.047
   640   F   159  0.006 |       0.093  0.010   0.041 |   88   0.034
```

**What the learned hash buys:**

- At the trivial short gap it roughly **matches** the raw patch (0.190 vs
  0.212) but degrades **more gracefully** at longer gaps where the raw patch
  collapses (160 nm: 0.101 vs 0.084; 640 nm: 0.041 vs 0.010). The harder,
  longer-range regime is the one that matters for real proofreading.
- It is a compact 32-d embedding — deployable as an edge feature — that
  generalises to an unseen box.

**Two honest findings:**

1. **It leans partly on the staining "trick".** Under per-section
   normalisation (`norm=T`, staining batch effect removed) the learned hash
   drops more than the raw patch at short gaps. So some of its short-range power
   *is* the per-section contrast cue, not biology. That is an allowed trick —
   it still re-links faces — but it is a trick, and flagged as one.
2. **Rotation/scale augmentation HURTS here.** Training with random rotation
   stalls the loss (~4.3 vs 2.97 without) and lowers accuracy. Because the cut
   faces are *local*, they are barely rotated relative to each other, and the
   footprint orientation is itself discriminative — forcing invariance throws
   that away. Locality removes the need for the invariance; per-patch contrast
   normalisation is the normalisation that actually matters. (`--augment`
   enables it for the ablation.)

## Headline: disambiguating REAL v117 errors on the reconstructed v117 seg

The faithful experiment (`v117_reconstructed.py`) and its result come first
because they are the point of the whole line of work. Everything below it is the
path that got here (including two wrong turns worth remembering).

**Setup.** Take error sites in their **v117 (split) state** and use the later
**v14XX merges as the answer key**:
- Reconstruct the segmentation *as it was at the v117-era (oldest) timestamp*:
  fetch the supervoxel/watershed layer per box and map each supervoxel to its
  root at that timestamp → the box is painted with the *split* fragments (the
  errored "question").
- Map the same supervoxels to their *current* root → two v117 fragments are a
  true merge iff they share a current root (the proofreading answer).
- At each interface: query = a v117 fragment's cut-face; candidates = nearby
  v117 fragments (proximity + direction cone); true partner = the one sharing
  the query's current root. Rank it by cut-face hash similarity.

**Result** (`v117_reconstructed_metrics.json`, N = 78 real error sites, mean 38
candidates/site, **chance top-1 = 0.085**):

| method | top-1 | MRR |
|---|---|---|
| raw patch | 0.205 | 0.333 |
| planar-trained encoder | 0.423 | 0.544 |
| **real-trained encoder** | **0.474** | **0.572** |

**Read-out:**

- At genuine v117 errors the learned cut-face hash picks the correct merge
  partner **top-1 ~47%** of the time vs **8.5% chance** (~5.6×), ranking the
  truth ~2nd on average (MRR 0.57). The fingerprint does real work.
- **Learned ≫ raw** (top-1 0.47 vs 0.21): the cross-section *pattern* matters,
  and it must be *learned*, not hand-summarised.
- **Computing the hash on the right data is the biggest lever — proven twice.**
  The identical method scored ~0 on the intermediate flat seg (which had already
  merged most sites) and **0.47 on the reconstructed v117 seg**. And training the
  encoder on *real* error cross-sections edges out the planar-cut encoder
  (0.474 vs 0.423) — reversing the transfer failure seen on artificial cuts.

**Location vs correction (two different models).** This experiment measures
*correction*: given a real interface and a candidate panel, pick the partner.
The other half — *location* (which edges are errors / candidates at all) — is a
separate model. In deployment the location model is the candidate generator:
flood it (a 2 µm proximity ball ≈ 256 distractors) and correction drowns; feed
it a clean panel (proximity + cone ≈ 38) and correction works. Evaluate the two
separately — location by recall, correction by top-1 given good candidates —
rather than conflating them.

## Earlier: disambiguating at v117 errors on the flat seg (`v117_error_relink.py`)

> **Superseded.** This masks from the *flat* MICrONS seg, an intermediate
> snapshot that already has most merges baked in, so ~all real interfaces look
> already-merged and get skipped (only ~8 scoreable; numbers were noise). It is
> kept for the site-finding machinery (`sites_from_l2_graph`) and the candidate
> generators, which the headline experiment reuses. Use `v117_reconstructed.py`.

The cuts above are *artificial* planar z-cuts. This experiment tests the hash
where it actually matters: at locations the automated segmentation **got wrong**
and a human had to fix. No materialization server needed — only the
chunkedgraph (and a CAVE token in env var `token`).

**Finding a real error site, from the chunkedgraph alone:** a proofread neuron
is one current root `R`. Look up the historical root of each of its level-2
nodes at the oldest timestamp; if they fall into several distinct historical
roots, `R` was *assembled by merging* those fragments — i.e. the v117-era
segmentation falsely **split** the neuron. Each minor fragment's closest
approach to the main arbor is a real false-split interface: the two points a
human glued. (In a scan, ~45% of soma neurons had at least one such split.)

**The test:** take the cross-section "face" at the main-side point as a query
and rank the candidate neurites near the fragment-side point by cut-face hash
similarity. The true continuation is the neurite actually at the fragment-side
point. This is re-identification across the *real* error gap.

```bash
# realistic candidate set: neurites within R of the query tip, optional cone
python -m experiments.fingerprints.v117_error_relink \
  --encoder experiments/fingerprints/cutface_encoder.pt \
  --n-scan 220 --max-neurons 45 \
  --candidate-mode proximity --radius-nm 2000 --direction-cone-deg 45 \
  --out experiments/fingerprints/v117_relink_metrics_cone.json
```

### The candidate set dominates the result

The single most important knob is *what counts as a candidate*. Three
definitions, all on real v117 split sites (~1.5 µm gaps):

| candidate panel | ~cand/site | chance top-1 | raw top-1 / MRR | learned top-1 / MRR | file |
|---|---|---|---|---|---|
| **slab, same-z** (legacy, optimistic) | 55 | 0.036 | 0.20 / 0.32 | **0.49 / 0.54** | `…_slab.json` (n=49) |
| **proximity ball, 2 µm** (dense, harsh) | 256 | 0.004 | 0.00 / 0.05 | 0.00 / 0.025 | `…_proximity.json` (n=10) |
| **proximity + 45° cone** (realistic) | 75 | 0.015 | 0.00 / **0.12** | 0.00 / 0.08 | `…_cone.json` (n=10) |

**Honest read-out — this is the part that corrects the earlier optimism:**

- The strong slab number (learned top-1 0.49) was **inflated by a small, same-z
  panel**. The candidates there were only the neurites crossing one z-slab in a
  modest box, and the true partner sat at that slab.
- A **proximity ball is *denser*, not sparser** — cortical neuropil packs
  ~250–300 cross-sections into a 2 µm ball, most of them pass-through processes
  no proofreader would ever propose merging. There the single-face hash drowns
  (top-1 ≈ 0).
- The **direction cone** removes the perpendicular pass-throughs (256 → 75
  candidates) and roughly **doubles MRR**, but the true partner still ranks
  ~8th–12th, not 1st.
- On the realistic panel the **raw patch ≥ the learned encoder** (MRR 0.12 vs
  0.08). The encoder was trained on *artificial planar* cuts and does **not**
  transfer to real oblique error cross-sections — a concrete retraining target.

**Conclusion.** A single cross-section cut-face hash is a real but **weak-to-
moderate ranking signal** on a realistic candidate set (5–8× chance), **not a
standalone top-1 matcher**. That is exactly the argument for using it as one
**edge feature** fused with proximity + graph context — which inherently won't
propose 256 candidates — rather than as the whole merge decision (the
integration below). Caveat: the proximity rows are **n = 10** (most real gaps
exceed the 2 µm radius and are skipped as unproposable), so those numbers are
noisy; a gap-capped larger run is the obvious follow-up.

## Using it as a proofreading edge feature (`neuronauts/em_corridor.py`)

The encoder is wired into the boundary-edge resolver as a drop-in edge feature.
`em_corridor` stays torch-free by taking the encoder as an injected `embed_fn`:

```python
from experiments.fingerprints.learned_cutface_encoder import load_encoder, make_embed_fn
from neuronauts.em_corridor import batch_cutface_similarity

enc = load_encoder("experiments/fingerprints/cutface_encoder.pt")
embed_fn = make_embed_fn(enc)

# syn_positions_nm: [N,3]; boundary_edges: ambiguous (i,j) pairs from CellGNN
scores = batch_cutface_similarity(syn_positions_nm, boundary_edges, embed_fn)
# {(i,j): cosine_similarity}  -- higher = faces look like one continuous process
```

One bulk EM+seg fetch covers all referenced points; `cross_section_patch`
extracts each translation-normalised face; a single `embed_fn` call embeds them.

## Honest limitations & obvious next steps

- The learned encoder (above) only *matches* the raw patch at short range
  rather than dominating it. It is trained on a single box for ~100 s on CPU;
  more training data (many boxes), a deeper net, and hard-negative mining are
  the obvious levers. The biggest expected win is restricting to small processes
  (thin axons) where real splits actually occur — large dendrite cross-sections
  are mostly uniform interior and dilute the metric.
- "Break" here is a clean planar z-cut. Real splits are at membrane-ambiguous
  3D surfaces; the partner panel and face extraction would change.
- Deployment hook is in place (`batch_cutface_similarity` in
  `neuronauts/em_corridor.py`); the remaining work is to feed its scores into
  `cell_graph.build_synapse_graph` as an edge feature and measure line-graph F1.
- Bigger evaluation: many boxes, report rank distributions, and restrict to the
  hard subset where it actually pays.
