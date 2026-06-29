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

## Honest limitations & obvious next steps

- One box, one cut plane, hand-crafted patch hash. The patch is a raw masked
  mean-intensity crop with **no rotation normalisation** and no learning — it
  almost certainly *under*-reads the available signal. A learned embedding
  (small CNN / contrastive head trained to pull true partners together) is the
  natural next step and is exactly what would exploit ER / microtubule
  arrangement that a raw 8 nm crop is too noisy to expose cleanly.
- "Break" here is a clean planar z-cut. Real splits are at membrane-ambiguous
  3D surfaces; the partner panel and face extraction would change.
- The right place to deploy this in the existing pipeline is the boundary-edge
  resolver in `neuronauts/em_corridor.py`: replace / augment the intensity
  heuristic with a cut-face hash similarity as an edge feature.
- Bigger evaluation: many boxes, report rank distributions, and restrict to the
  hard subset where it actually pays.
