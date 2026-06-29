"""Cut-face fingerprints: can micro-ultrastructure re-link a severed neurite?

Premise
-------
EM connectomics has a *self-inflicted* reconstruction problem: to image the
tissue we slice a continuous 3D object into ~40 nm sections, then ask an
algorithm to stitch it back together.  Every true split error is therefore a
cut through what was, physically, one continuous process.  Across that cut the
local ultrastructure -- caliber, mitochondria, the axoplasmic / smooth
endoplasmic reticulum, microtubule packing, membrane texture -- is *continuous*.

This module tests the FISSEQ-style hypothesis: does a cut face carry enough
idiosyncratic micro-structure to act as a **locality-sensitive hash** that
uniquely re-links it to its true partner across a gap?  Concretely, given the
top face of a severed neurite, can a hash of its ultrastructure rank the true
bottom partner as the nearest neighbour against a panel of distractor neurites
crossing the same plane -- and does it beat the trivial spatial-proximity cue
that current pipelines lean on?

Ground truth comes for free: the public MICrONS proofread segmentation assigns
the *same* id to both sides of the cut, so the true partner of top-face ``k``
is bottom-face ``k``.

The experiment is deliberately small and local (a few microns cubed).  It uses
only the public EM + segmentation precomputed volumes -- no CAVE token.

Key honesty controls (see ``run_experiment``):

* **Gap sweep** -- separates a true idiosyncratic *fingerprint* from mere local
  *continuity*.  A tiny gap is trivially matchable because neurites are locally
  smooth; the question is how far the signal survives.
* **Per-section normalisation** -- EM contrast varies section to section
  (staining / imaging batch effect).  Two faces sampled near each other share
  that batch effect, which is a *processing-artifact* hash, not biology.  We
  re-score with per-section-normalised intensity to see whether structural
  signal survives.
* **Position-free hash** -- the structural hash never sees xy position; the
  spatial baseline uses *only* xy position.  Beating the spatial baseline is
  the whole point: it means there is identity information beyond proximity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

EM_PATH = "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em"
SEG_PATH = "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/seg"

# Center of the minnie65 imaged volume in MIP1 (16,16,40 nm) voxel coords,
# derived from the public `info`: a dense-tissue region good for a smoke run.
DEFAULT_CENTER_MIP1 = (60160, 51968, 21360)


# ---------------------------------------------------------------------------
# Data fetch (public volumes, no CAVE token)
# ---------------------------------------------------------------------------

@dataclass
class Volume:
    em: np.ndarray            # uint8 [X, Y, Z]
    seg: np.ndarray           # uint64 [X, Y, Z]
    resolution_nm: tuple      # (x, y, z) nm per voxel
    origin_vox: tuple         # (x0, y0, z0) voxel offset of this chunk


def fetch_volume(
    center_vox_mip1: tuple = DEFAULT_CENTER_MIP1,
    size_vox: tuple = (320, 320, 80),
    mip: int = 1,
) -> Volume:
    """Fetch a small EM + proofread-seg cube from the public MICrONS volumes.

    ``center_vox_mip1`` is given in MIP1 voxel coordinates and rescaled to the
    requested ``mip`` so the same physical center can be used at any resolution.
    MIP 0/1 (8/16 nm) is required to resolve intracellular ultrastructure
    (mitochondria, axoplasmic reticulum); MIP 2+ blurs it away.
    """
    from cloudvolume import CloudVolume  # heavy, lazy import

    # MIP1 -> requested mip xy scale factor (z stays 40 nm through mip 3).
    xy_factor = 2 ** (mip - 1)
    cx = center_vox_mip1[0] // xy_factor
    cy = center_vox_mip1[1] // xy_factor
    cz = center_vox_mip1[2]

    sx, sy, sz = size_vox
    x0, y0, z0 = cx - sx // 2, cy - sy // 2, cz - sz // 2

    em_cv = CloudVolume(EM_PATH, mip=mip, use_https=True, progress=False, fill_missing=True)
    seg_cv = CloudVolume(SEG_PATH, mip=mip, use_https=True, progress=False, fill_missing=True)

    em = np.squeeze(np.asarray(em_cv[x0:x0 + sx, y0:y0 + sy, z0:z0 + sz])).astype(np.uint8)
    seg = np.squeeze(np.asarray(seg_cv[x0:x0 + sx, y0:y0 + sy, z0:z0 + sz])).astype(np.uint64)
    res = tuple(int(v) for v in em_cv.resolution)
    return Volume(em=em, seg=seg, resolution_nm=res, origin_vox=(x0, y0, z0))


# ---------------------------------------------------------------------------
# Cut-face hash (the "fingerprint")
# ---------------------------------------------------------------------------

# Order of the structural feature vector returned by ``face_hash``.  Position is
# deliberately excluded -- it is carried separately for the spatial baseline.
FEATURE_NAMES = (
    "log_area",        # caliber (log voxels per section)
    "int_mean",        # mean EM intensity inside the process (cytoplasm darkness)
    "int_std",         # intensity spread
    "int_p10",         # dark tail (membranes / organelles)
    "int_p90",         # bright tail
    "dark_frac",       # fraction of dark voxels (organelle / membrane load)
    "grad_mean",       # ultrastructure "busy-ness" (mean gradient magnitude)
    "grad_std",
    "n_dark_blobs",    # mito / ER cross-section count per section
    "eccentricity",    # cross-section shape
)
N_FEATURES = len(FEATURE_NAMES)


PATCH = 48  # patch side in voxels for the cross-section image hash


@dataclass
class Face:
    seg_id: int
    vec: np.ndarray        # structural feature vector, len N_FEATURES
    centroid_xy: np.ndarray  # (x, y) in voxels -- for the spatial baseline only
    n_vox: int
    patch: np.ndarray = field(default=None)  # centered, masked cross-section image hash


def _label_2d(mask2d: np.ndarray) -> int:
    """Count connected dark blobs in a 2D boolean mask (mito/ER cross-sections)."""
    try:
        from scipy.ndimage import label
        _, n = label(mask2d)
        return int(n)
    except Exception:
        # Cheap fallback: number of True pixels as a rough proxy.
        return int(mask2d.sum() > 0)


def face_hash(
    em: np.ndarray,
    seg: np.ndarray,
    grad: np.ndarray,
    z_lo: int,
    z_hi: int,
    *,
    dark_thresh: float,
    min_vox: int = 40,
    per_section_norm: bool = False,
) -> dict[int, Face]:
    """Hash every neurite cross-section in the z-slab ``[z_lo, z_hi)``.

    Returns a ``{seg_id: Face}`` map.  Each face vector is a position-free
    descriptor of the local ultrastructure -- the locality-sensitive hash whose
    near-collisions we hope re-link true partners.
    """
    sub_em = em[:, :, z_lo:z_hi].astype(np.float32)
    sub_seg = seg[:, :, z_lo:z_hi]
    sub_grad = grad[:, :, z_lo:z_hi]
    nz = z_hi - z_lo

    if per_section_norm:
        # Remove per-section contrast/staining offset: z-score each section using
        # its tissue (non-background) voxels.  This strips the imaging batch
        # effect so any surviving match signal is structural, not staining.
        fg = sub_seg > 0
        for k in range(nz):
            m = fg[:, :, k]
            if m.sum() < 16:
                continue
            vals = sub_em[:, :, k][m]
            mu, sd = float(vals.mean()), float(vals.std()) + 1e-3
            sub_em[:, :, k] = (sub_em[:, :, k] - mu) / sd

    # When per_section_norm=True, sub_em is z-scored (mean≈0, std≈1 in tissue);
    # dark_thresh is a raw-EM value and would mark every voxel as dark.
    # Use 0.0 (below tissue mean) as the normalized-space equivalent.
    _dark_thresh = 0.0 if per_section_norm else dark_thresh

    ids, counts = np.unique(sub_seg, return_counts=True)
    faces: dict[int, Face] = {}
    for sid, cnt in zip(ids.tolist(), counts.tolist()):
        if sid == 0 or cnt < min_vox:
            continue
        mask = sub_seg == sid
        em_vals = sub_em[mask]
        grad_vals = sub_grad[mask]

        p10, p50, p90 = np.percentile(em_vals, [10, 50, 90])
        dark_frac = float((em_vals < _dark_thresh).mean())

        # Per-section internal blob count (organelle cross-sections), averaged.
        blob_counts = []
        xs_all, ys_all = [], []
        for k in range(nz):
            m2 = mask[:, :, k]
            if not m2.any():
                continue
            dark2 = m2 & (sub_em[:, :, k] < _dark_thresh)
            blob_counts.append(_label_2d(dark2))
            xs, ys = np.nonzero(m2)
            xs_all.append(xs)
            ys_all.append(ys)
        n_blobs = float(np.mean(blob_counts)) if blob_counts else 0.0

        xs = np.concatenate(xs_all)
        ys = np.concatenate(ys_all)
        cx, cy = float(xs.mean()), float(ys.mean())
        # Eccentricity from the 2D second-moment matrix of the footprint.
        ux, uy = xs - cx, ys - cy
        cxx, cyy, cxy = float((ux * ux).mean()), float((uy * uy).mean()), float((ux * uy).mean())
        tr, det = cxx + cyy, cxx * cyy - cxy * cxy
        disc = max(tr * tr / 4 - det, 0.0)
        l1 = tr / 2 + np.sqrt(disc)
        l2 = max(tr / 2 - np.sqrt(disc), 1e-6)
        ecc = float(np.sqrt(max(1.0 - l2 / l1, 0.0))) if l1 > 1e-6 else 0.0

        # Cross-section image hash: translation-normalised, masked mean-intensity
        # patch.  This keeps the *arrangement* of internal structure (mito / ER /
        # microtubule bundles) and the footprint shape -- the idiosyncratic part
        # the scalar summaries throw away -- while centring removes absolute xy.
        count2d = mask.sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            proj = np.where(count2d > 0, (sub_em * mask).sum(axis=2) / count2d, 0.0)
        ci, cj = int(round(cx)), int(round(cy))
        h = PATCH // 2
        patch = np.zeros((PATCH, PATCH), dtype=np.float64)
        xi0, xi1 = max(ci - h, 0), min(ci + h, proj.shape[0])
        yi0, yi1 = max(cj - h, 0), min(cj + h, proj.shape[1])
        px0, py0 = xi0 - (ci - h), yi0 - (cj - h)
        patch[px0:px0 + (xi1 - xi0), py0:py0 + (yi1 - yi0)] = proj[xi0:xi1, yi0:yi1]

        vec = np.array([
            np.log1p(cnt / nz),
            float(em_vals.mean()),
            float(em_vals.std()),
            float(p10),
            float(p90),
            dark_frac,
            float(grad_vals.mean()),
            float(grad_vals.std()),
            n_blobs,
            ecc,
        ], dtype=np.float64)

        faces[int(sid)] = Face(
            seg_id=int(sid), vec=vec,
            centroid_xy=np.array([cx, cy]), n_vox=int(cnt), patch=patch,
        )
    return faces


def _patch_features(faces: dict[int, Face]) -> tuple[list[int], np.ndarray]:
    """Flatten each face patch into a mean-subtracted, L2-normalised vector."""
    ids = sorted(faces)
    mat = []
    for i in ids:
        p = faces[i].patch.ravel().astype(np.float64)
        p = p - p.mean()
        nrm = np.linalg.norm(p) + 1e-9
        mat.append(p / nrm)
    return ids, np.stack(mat) if mat else np.zeros((0, PATCH * PATCH))


# ---------------------------------------------------------------------------
# Matching / ranking
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    gap_sections: int
    per_section_norm: bool
    n_candidates: int
    top1_hash: float       # scalar ultrastructure summary hash
    top1_patch: float      # cross-section image hash (arrangement-aware)
    top1_spatial: float
    top1_fused: float      # spatial + patch hash combined (the practical number)
    top1_chance: float
    mrr_hash: float
    mrr_patch: float
    mrr_spatial: float
    # Hard subset: query faces whose true partner is NOT the spatially nearest.
    n_hard: int
    top1_hash_hard: float
    top1_patch_hard: float


def _zscore_columns(mat: np.ndarray) -> np.ndarray:
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0) + 1e-9
    return (mat - mu) / sd


def rank_matches(top: dict[int, Face], bottom: dict[int, Face]) -> Optional[MatchResult]:
    """Rank bottom faces against each top face by hash distance and by xy distance."""
    shared = sorted(set(top) & set(bottom))
    if len(shared) < 3:
        return None

    bottom_ids = sorted(bottom)
    B = np.stack([bottom[i].vec for i in bottom_ids])
    Bxy = np.stack([bottom[i].centroid_xy for i in bottom_ids])
    Bz = _zscore_columns(B)
    id_to_col = {i: c for c, i in enumerate(bottom_ids)}
    pids, Bpatch = _patch_features(bottom)
    pcol = {i: c for c, i in enumerate(pids)}

    # z-score top faces with the SAME column stats as bottom for a shared space.
    mu, sd = B.mean(axis=0), B.std(axis=0) + 1e-9
    n = len(bottom_ids)

    ranks_hash, ranks_patch, ranks_spatial = [], [], []
    hits_hash, hits_patch, hits_spatial, hits_fused = [], [], [], []
    hard_hash, hard_patch = [], []

    def _z(v):  # standardise a distance vector within the candidate panel
        return (v - v.mean()) / (v.std() + 1e-9)

    for qid in shared:
        q = (top[qid].vec - mu) / sd
        qxy = top[qid].centroid_xy
        true_col = id_to_col[qid]

        d_hash = np.linalg.norm(Bz - q, axis=1)
        d_spatial = np.linalg.norm(Bxy - qxy, axis=1)

        # Patch image hash: 1 - cosine similarity in the shared patch space.
        qp = top[qid].patch.ravel().astype(np.float64)
        qp = qp - qp.mean()
        qp = qp / (np.linalg.norm(qp) + 1e-9)
        d_patch = 1.0 - Bpatch @ qp
        true_pcol = pcol[qid]

        # Fused: standardise spatial and patch distances and add them.  Patch
        # ids may differ in order from vec ids, so reindex patch dist to bottom_ids.
        d_patch_b = d_patch[[pcol[i] for i in bottom_ids]]
        d_fused = _z(d_spatial) + _z(d_patch_b)

        r_hash = int((d_hash < d_hash[true_col]).sum())      # 0 == top-1
        r_patch = int((d_patch < d_patch[true_pcol]).sum())
        r_spatial = int((d_spatial < d_spatial[true_col]).sum())
        r_fused = int((d_fused < d_fused[true_col]).sum())
        ranks_hash.append(r_hash)
        ranks_patch.append(r_patch)
        ranks_spatial.append(r_spatial)
        hits_hash.append(r_hash == 0)
        hits_patch.append(r_patch == 0)
        hits_spatial.append(r_spatial == 0)
        hits_fused.append(r_fused == 0)

        # "Hard" = spatial baseline does NOT already put the true partner first.
        if r_spatial != 0:
            hard_hash.append(r_hash == 0)
            hard_patch.append(r_patch == 0)

    def mrr(ranks):
        return float(np.mean([1.0 / (r + 1) for r in ranks]))

    return MatchResult(
        gap_sections=-1, per_section_norm=False, n_candidates=n,
        top1_hash=float(np.mean(hits_hash)),
        top1_patch=float(np.mean(hits_patch)),
        top1_spatial=float(np.mean(hits_spatial)),
        top1_fused=float(np.mean(hits_fused)),
        top1_chance=1.0 / n,
        mrr_hash=mrr(ranks_hash), mrr_patch=mrr(ranks_patch), mrr_spatial=mrr(ranks_spatial),
        n_hard=len(hard_hash),
        top1_hash_hard=float(np.mean(hard_hash)) if hard_hash else float("nan"),
        top1_patch_hard=float(np.mean(hard_patch)) if hard_patch else float("nan"),
    )


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------

def run_experiment(
    vol: Volume,
    *,
    slab_width: int = 3,
    gaps=(1, 2, 4, 8, 16),
    per_section_norm_variants=(False, True),
    dark_percentile: float = 25.0,
) -> list[MatchResult]:
    """Cut the volume at its z-midplane and test hash-based re-linking.

    For each gap ``g`` we remove a slab of ``g`` sections about the midplane and
    hash the ``slab_width`` sections on either side, then rank matches.
    """
    em, seg = vol.em, vol.seg
    nzv = em.shape[2]
    z0 = nzv // 2
    dark_thresh = float(np.percentile(em[seg > 0], dark_percentile))

    # Gradient magnitude over the whole volume (xy only -- ultrastructure texture).
    gx, gy = np.gradient(em.astype(np.float32), axis=0), np.gradient(em.astype(np.float32), axis=1)
    grad = np.sqrt(gx * gx + gy * gy)

    results: list[MatchResult] = []
    for norm in per_section_norm_variants:
        for g in gaps:
            half = g // 2
            b_hi = z0 - half
            b_lo = b_hi - slab_width
            t_lo = z0 + (g - half)
            t_hi = t_lo + slab_width
            if b_lo < 0 or t_hi > nzv:
                continue
            bottom = face_hash(em, seg, grad, b_lo, b_hi,
                               dark_thresh=dark_thresh, per_section_norm=norm)
            top = face_hash(em, seg, grad, t_lo, t_hi,
                            dark_thresh=dark_thresh, per_section_norm=norm)
            r = rank_matches(top, bottom)
            if r is None:
                continue
            r.gap_sections = g
            r.per_section_norm = norm
            results.append(r)
    return results


def summarize(results: list[MatchResult], resolution_nm: tuple) -> str:
    zres = resolution_nm[2]
    lines = []
    lines.append("gap_nm norm Ncand chance | top1: spatial  scalar  PATCH  FUSED | "
                 "hard(N) patch_on_hard")
    for r in results:
        lines.append(
            f"{r.gap_sections * zres:6d}  {str(r.per_section_norm)[0]}  "
            f"{r.n_candidates:4d}  {r.top1_chance:5.3f} |      "
            f"{r.top1_spatial:6.3f}  {r.top1_hash:6.3f}  {r.top1_patch:6.3f}  {r.top1_fused:6.3f} | "
            f"{r.n_hard:4d}   {r.top1_patch_hard:.3f}"
        )
    return "\n".join(lines)


def make_figure(vol: Volume, out_png: str, *, gap: int = 2, slab_width: int = 3,
                n_examples: int = 6, dark_percentile: float = 25.0) -> str:
    """Render query cut-faces next to their true partner and the hash's top pick.

    Prioritises "hard" examples where spatial proximity picks the wrong partner
    but the cross-section image hash recovers the true one -- the cases that show
    the fingerprint doing something proximity cannot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    em, seg = vol.em, vol.seg
    nzv = em.shape[2]
    z0 = nzv // 2
    dark_thresh = float(np.percentile(em[seg > 0], dark_percentile))
    gx, gy = np.gradient(em.astype(np.float32), axis=0), np.gradient(em.astype(np.float32), axis=1)
    grad = np.sqrt(gx * gx + gy * gy)

    half = gap // 2
    b_hi, b_lo = z0 - half, z0 - half - slab_width
    t_lo, t_hi = z0 + (gap - half), z0 + (gap - half) + slab_width
    bottom = face_hash(em, seg, grad, b_lo, b_hi, dark_thresh=dark_thresh)
    top = face_hash(em, seg, grad, t_lo, t_hi, dark_thresh=dark_thresh)

    shared = sorted(set(top) & set(bottom))
    pids, Bpatch = _patch_features(bottom)
    pcol = {i: c for c, i in enumerate(pids)}
    bottom_ids = sorted(bottom)
    Bxy = np.stack([bottom[i].centroid_xy for i in bottom_ids])
    bcol = {i: c for c, i in enumerate(bottom_ids)}

    hard, easy = [], []
    for qid in shared:
        qp = top[qid].patch.ravel() - top[qid].patch.mean()
        qp = qp / (np.linalg.norm(qp) + 1e-9)
        d_patch = 1.0 - Bpatch @ qp
        guess = pids[int(np.argmin(d_patch))]
        d_spatial = np.linalg.norm(Bxy - top[qid].centroid_xy, axis=1)
        spat_guess = bottom_ids[int(np.argmin(d_spatial))]
        if spat_guess != qid and guess == qid:
            hard.append((qid, guess))
        elif guess == qid:
            easy.append((qid, guess))
    picks = (hard + easy)[:n_examples]
    if not picks:
        picks = [(q, pids[int(np.argmin(1.0 - Bpatch @ ((top[q].patch.ravel() - top[q].patch.mean()) /
                 (np.linalg.norm(top[q].patch.ravel() - top[q].patch.mean()) + 1e-9))))]) for q in shared[:n_examples]]

    fig, axes = plt.subplots(len(picks), 3, figsize=(7.5, 2.4 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]
    for row, (qid, guess) in enumerate(picks):
        cells = [(top[qid].patch, f"query top  id={qid}"),
                 (bottom[qid].patch, "TRUE partner (bottom)"),
                 (bottom[guess].patch, f"hash top pick id={guess}" +
                  ("  ✓" if guess == qid else "  ✗"))]
        for col, (img, title) in enumerate(cells):
            ax = axes[row, col]
            ax.imshow(img.T, cmap="gray", origin="lower")
            ax.set_title(title, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Cut-face ultrastructure hash @ {gap * vol.resolution_nm[2]} nm gap "
                 f"({vol.resolution_nm[0]} nm/px)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mip", type=int, default=1, help="0=8nm (best ultrastructure), 1=16nm")
    ap.add_argument("--size", type=int, nargs=3, default=(320, 320, 80),
                    metavar=("X", "Y", "Z"), help="chunk size in voxels")
    ap.add_argument("--center", type=int, nargs=3, default=list(DEFAULT_CENTER_MIP1),
                    metavar=("X", "Y", "Z"), help="center in MIP1 voxel coords")
    ap.add_argument("--out", type=str, default=None, help="write results JSON here")
    ap.add_argument("--figure", type=str, default=None, help="write a cut-face montage PNG here")
    ap.add_argument("--figure-gap", type=int, default=2, help="gap (sections) used for the montage")
    args = ap.parse_args()

    print(f"[fetch] mip={args.mip} size={tuple(args.size)} center(mip1)={tuple(args.center)} ...")
    vol = fetch_volume(tuple(args.center), tuple(args.size), mip=args.mip)
    n_ids = int((np.unique(vol.seg) > 0).sum())
    print(f"[fetch] EM {vol.em.shape} res={vol.resolution_nm} nm  neurites={n_ids}")

    results = run_experiment(vol)
    print("\n" + summarize(results, vol.resolution_nm))
    print("\nReading: top1_HASH > top1_spatial means ultrastructure carries identity")
    print("beyond mere proximity. The gap sweep shows how fast that signal decays.")

    if args.out:
        payload = {
            "resolution_nm": vol.resolution_nm,
            "em_shape": list(vol.em.shape),
            "n_neurites": n_ids,
            "feature_names": list(FEATURE_NAMES),
            "results": [asdict(r) for r in results],
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[out] wrote {args.out}")

    if args.figure:
        make_figure(vol, args.figure, gap=args.figure_gap)
        print(f"[out] wrote {args.figure}")


if __name__ == "__main__":
    main()
