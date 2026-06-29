"""Learned cut-face encoder: a contrastive hash for re-linking severed neurites.

This is the learned successor to ``fingerprint_break_resolution.py``.  Instead
of comparing raw masked cross-section patches, we train a small CNN with an
InfoNCE / NT-Xent objective to pull the two faces of the *same* neurite (sampled
at different z) together and push faces of different neurites apart.  The learned
embedding is then used exactly like the raw-patch hash: rank the true partner of
a cut face against a distractor panel.

Pragmatics
----------
We let the encoder use whatever signal works -- including the per-section
staining "trick" (contrast offsets shared by nearby faces).  Biology is better,
but a trick that proofreads is still useful.  To stay honest about *which* it is,
the evaluation reports a per-section-normalised variant: if the learned hash
keeps most of its lead there, it is reading structure; if it collapses, it was
leaning on the staining trick.

Train and test boxes are fetched from spatially disjoint regions so the encoder
is evaluated on neurites it never saw during training.

Run
---
    python -m experiments.fingerprints.learned_cutface_encoder \
        --epochs 30 --out experiments/fingerprints/cutface_encoder.pt \
        --metrics experiments/fingerprints/learned_metrics.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from .fingerprint_break_resolution import (
    Volume,
    fetch_volume,
    face_hash,
    PATCH,
    DEFAULT_CENTER_MIP1,
)


# ---------------------------------------------------------------------------
# Face bank extraction (patches across many z-planes, for training)
# ---------------------------------------------------------------------------

@dataclass
class FaceBank:
    patches: np.ndarray     # [N, PATCH, PATCH] float32
    seg_ids: np.ndarray     # [N] int64
    centroids: np.ndarray   # [N, 2] float32 (xy, voxels) -- spatial baseline
    z_centers: np.ndarray   # [N] int32


def _grad_mag(em: np.ndarray) -> np.ndarray:
    gx = np.gradient(em.astype(np.float32), axis=0)
    gy = np.gradient(em.astype(np.float32), axis=1)
    return np.sqrt(gx * gx + gy * gy)


def extract_face_bank(
    vol: Volume,
    z_centers,
    *,
    slab_width: int = 3,
    min_vox: int = 60,
    dark_percentile: float = 25.0,
    per_section_norm: bool = False,
) -> FaceBank:
    """Extract one cross-section patch per neurite per z-plane in ``z_centers``."""
    em, seg = vol.em, vol.seg
    grad = _grad_mag(em)
    dark_thresh = float(np.percentile(em[seg > 0], dark_percentile))

    patches, ids, cents, zs = [], [], [], []
    for zc in z_centers:
        z_lo, z_hi = int(zc), int(zc) + slab_width
        if z_hi > em.shape[2]:
            continue
        faces = face_hash(em, seg, grad, z_lo, z_hi, dark_thresh=dark_thresh,
                          min_vox=min_vox, per_section_norm=per_section_norm)
        for f in faces.values():
            patches.append(f.patch.astype(np.float32))
            ids.append(f.seg_id)
            cents.append(f.centroid_xy.astype(np.float32))
            zs.append(zc)
    if not patches:
        return FaceBank(np.zeros((0, PATCH, PATCH), np.float32), np.zeros(0, np.int64),
                        np.zeros((0, 2), np.float32), np.zeros(0, np.int32))
    return FaceBank(
        patches=np.stack(patches),
        seg_ids=np.asarray(ids, dtype=np.int64),
        centroids=np.stack(cents),
        z_centers=np.asarray(zs, dtype=np.int32),
    )


def _normalize_patches(p: np.ndarray) -> np.ndarray:
    """Per-patch mean-subtract + unit-std -- the encoder sees contrast, not absolute level."""
    p = p.astype(np.float32)
    mu = p.mean(axis=(1, 2), keepdims=True)
    sd = p.std(axis=(1, 2), keepdims=True) + 1e-4
    return (p - mu) / sd


def _augment_patches(p: np.ndarray, rng, max_rot_deg: float = 45.0) -> np.ndarray:
    """Random rotation / flip / scale to teach the encoder invariance.

    The two faces of a real cut differ by a small rotation, scale (caliber)
    change, and flip ambiguity; locality keeps these small, so we default to a
    *moderate* rotation range -- full 360-degree invariance would discard the
    footprint-orientation cue that is itself discriminative.  Operates on raw
    patches; the caller normalises afterwards.
    """
    from scipy.ndimage import rotate, zoom

    out = np.empty_like(p)
    for k in range(p.shape[0]):
        img = p[k]
        if rng.random() < 0.5:
            img = img[:, ::-1]
        if rng.random() < 0.5:
            img = img[::-1, :]
        angle = float(rng.uniform(-max_rot_deg, max_rot_deg))
        img = rotate(img, angle, reshape=False, order=1, mode="constant", cval=0.0)
        s = float(rng.uniform(0.85, 1.15))
        if abs(s - 1.0) > 1e-3:
            zimg = zoom(img, s, order=1)
            img = _center_fit(zimg, p.shape[1])
        out[k] = img
    return out


def _center_fit(img: np.ndarray, size: int) -> np.ndarray:
    """Center-crop or zero-pad a 2D array to (size, size)."""
    out = np.zeros((size, size), dtype=img.dtype)
    h, w = img.shape
    # crop region of img
    sy = max((h - size) // 2, 0)
    sx = max((w - size) // 2, 0)
    cy = min(h, size)
    cx = min(w, size)
    crop = img[sy:sy + min(size, h - sy), sx:sx + min(size, w - sx)]
    oy = max((size - h) // 2, 0)
    ox = max((size - w) // 2, 0)
    out[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return out


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def build_encoder(embed_dim: int = 32):
    import torch.nn as nn

    class CutFaceEncoder(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(2),                                       # 48 -> 24
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),                                       # 24 -> 12
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),                               # -> 64
            )
            self.head = nn.Linear(64, d)

        def forward(self, x):
            import torch.nn.functional as F
            h = self.net(x).flatten(1)
            z = self.head(h)
            return F.normalize(z, dim=1)

    return CutFaceEncoder(embed_dim)


# ---------------------------------------------------------------------------
# Contrastive training (NT-Xent over same-neurite face pairs)
# ---------------------------------------------------------------------------

def train_encoder(
    bank: FaceBank,
    *,
    embed_dim: int = 32,
    epochs: int = 30,
    batch_ids: int = 96,
    steps_per_epoch: int = 40,
    lr: float = 1e-3,
    temperature: float = 0.2,
    min_z_gap: int = 2,
    augment: bool = True,
    max_rot_deg: float = 45.0,
    seed: int = 0,
    verbose: bool = True,
):
    """Train the encoder; an anchor's positive is another face of the same id."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    patches_raw = bank.patches.astype(np.float32)
    # Group face indices by neurite id; keep ids with >=2 sufficiently separated faces.
    by_id: dict[int, list[int]] = {}
    for idx, sid in enumerate(bank.seg_ids.tolist()):
        by_id.setdefault(sid, []).append(idx)
    usable = [sid for sid, idxs in by_id.items() if len(idxs) >= 2]
    if len(usable) < 4:
        raise ValueError(f"too few neurites with >=2 faces ({len(usable)})")

    enc = build_encoder(embed_dim)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    enc.train()

    losses = []
    for ep in range(epochs):
        ep_loss, n_batch = 0.0, 0
        for _ in range(steps_per_epoch):
            chosen = rng.choice(usable, size=min(batch_ids, len(usable)), replace=False)
            a_idx, p_idx = [], []
            for sid in chosen.tolist():
                idxs = by_id[sid]
                # prefer a pair separated in z (a real "cut", not the same slab)
                i, j = _sample_pair(idxs, bank.z_centers, min_z_gap, rng)
                a_idx.append(i)
                p_idx.append(j)
            a_raw = patches_raw[a_idx]
            p_raw = patches_raw[p_idx]
            if augment:
                # Independent augmentation of each view -> invariance to rot/scale/flip.
                a_raw = _augment_patches(a_raw, rng, max_rot_deg)
                p_raw = _augment_patches(p_raw, rng, max_rot_deg)
            anch = torch.from_numpy(_normalize_patches(a_raw)).unsqueeze(1)
            pos = torch.from_numpy(_normalize_patches(p_raw)).unsqueeze(1)
            za = enc(anch)
            zp = enc(pos)
            loss = _ntxent(za, zp, temperature, torch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach())
            n_batch += 1
        ep_loss = ep_loss / max(n_batch, 1)
        losses.append(ep_loss)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  ntxent={ep_loss:.4f}")
    enc.eval()
    return enc, losses


def _sample_pair(idxs, z_centers, min_z_gap, rng):
    for _ in range(8):
        i, j = rng.choice(idxs, size=2, replace=False)
        if abs(int(z_centers[i]) - int(z_centers[j])) >= min_z_gap:
            return int(i), int(j)
    i, j = rng.choice(idxs, size=2, replace=False)
    return int(i), int(j)


def _ntxent(za, zp, temperature, torch):
    """Symmetric NT-Xent: positives are aligned rows of za/zp, all else negative."""
    B = za.shape[0]
    z = torch.cat([za, zp], dim=0)              # [2B, d]
    sim = z @ z.t() / temperature               # [2B, 2B]
    sim.fill_diagonal_(-1e9)
    targets = torch.arange(2 * B)
    targets = (targets + B) % (2 * B)           # i <-> i+B
    return torch.nn.functional.cross_entropy(sim, targets)


# ---------------------------------------------------------------------------
# Embedding + re-identification evaluation
# ---------------------------------------------------------------------------

def load_encoder(path: str):
    """Load a saved cut-face encoder checkpoint into eval mode."""
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    enc = build_encoder(ckpt.get("embed_dim", 32))
    enc.load_state_dict(ckpt["state_dict"])
    enc.eval()
    return enc


def make_embed_fn(enc):
    """Adapt an encoder into the ``embed_fn`` callback expected by em_corridor.

    Returns ``f(patches[N, P, P] float32) -> embeddings[N, D]``.
    """
    return lambda patches: embed_patches(enc, np.asarray(patches))


def embed_patches(enc, patches: np.ndarray, batch: int = 256) -> np.ndarray:
    import torch
    p = _normalize_patches(patches)
    out = []
    enc.eval()
    with torch.no_grad():
        for b0 in range(0, len(p), batch):
            x = torch.from_numpy(p[b0:b0 + batch]).unsqueeze(1)
            out.append(enc(x).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


@dataclass
class EvalRow:
    gap_nm: int
    per_section_norm: bool
    n_candidates: int
    top1_chance: float
    top1_spatial: float
    top1_rawpatch: float
    top1_learned: float
    top1_learned_on_hard: float
    n_hard: int


def _top1(D: np.ndarray, true_cols: np.ndarray) -> np.ndarray:
    """Boolean hit array: is the true column the column-wise argmin distance?"""
    return D.argmin(axis=1) == true_cols


def evaluate_reid(
    enc,
    vol: Volume,
    *,
    gaps=(1, 2, 4, 8, 16),
    slab_width: int = 3,
    per_section_norm_variants=(False, True),
    dark_percentile: float = 25.0,
) -> list[EvalRow]:
    """Cut the test volume at its midplane and compare learned vs raw vs spatial."""
    em, seg = vol.em, vol.seg
    grad = _grad_mag(em)
    z0 = em.shape[2] // 2
    dark_thresh = float(np.percentile(em[seg > 0], dark_percentile))
    zres = vol.resolution_nm[2]

    rows: list[EvalRow] = []
    for norm in per_section_norm_variants:
        for g in gaps:
            half = g // 2
            b_hi, b_lo = z0 - half, z0 - half - slab_width
            t_lo, t_hi = z0 + (g - half), z0 + (g - half) + slab_width
            if b_lo < 0 or t_hi > em.shape[2]:
                continue
            bottom = face_hash(em, seg, grad, b_lo, b_hi, dark_thresh=dark_thresh,
                              per_section_norm=norm)
            top = face_hash(em, seg, grad, t_lo, t_hi, dark_thresh=dark_thresh,
                           per_section_norm=norm)
            shared = sorted(set(top) & set(bottom))
            if len(shared) < 3:
                continue
            bids = sorted(bottom)
            bcol = {i: c for c, i in enumerate(bids)}
            true_cols = np.array([bcol[i] for i in shared])

            Bxy = np.stack([bottom[i].centroid_xy for i in bids])
            Qxy = np.stack([top[i].centroid_xy for i in shared])
            D_spatial = np.linalg.norm(Qxy[:, None, :] - Bxy[None, :, :], axis=2)

            Braw = np.stack([_flat_norm(bottom[i].patch) for i in bids])
            Qraw = np.stack([_flat_norm(top[i].patch) for i in shared])
            D_raw = 1.0 - Qraw @ Braw.T

            Bemb = embed_patches(enc, np.stack([bottom[i].patch for i in bids]))
            Qemb = embed_patches(enc, np.stack([top[i].patch for i in shared]))
            D_learned = 1.0 - Qemb @ Bemb.T

            hit_spatial = _top1(D_spatial, true_cols)
            hit_raw = _top1(D_raw, true_cols)
            hit_learned = _top1(D_learned, true_cols)
            hard = ~hit_spatial

            rows.append(EvalRow(
                gap_nm=g * zres, per_section_norm=norm, n_candidates=len(bids),
                top1_chance=1.0 / len(bids),
                top1_spatial=float(hit_spatial.mean()),
                top1_rawpatch=float(hit_raw.mean()),
                top1_learned=float(hit_learned.mean()),
                top1_learned_on_hard=float(hit_learned[hard].mean()) if hard.any() else float("nan"),
                n_hard=int(hard.sum()),
            ))
    return rows


def _flat_norm(patch: np.ndarray) -> np.ndarray:
    v = patch.ravel().astype(np.float64)
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-9)


def summarize(rows: list[EvalRow]) -> str:
    out = ["gap_nm norm Ncand chance | top1: spatial   raw   LEARNED | hard(N) learned_on_hard"]
    for r in rows:
        out.append(
            f"{r.gap_nm:6d}  {str(r.per_section_norm)[0]}  {r.n_candidates:4d}  "
            f"{r.top1_chance:5.3f} |      {r.top1_spatial:6.3f}  {r.top1_rawpatch:5.3f}  "
            f"{r.top1_learned:6.3f} | {r.n_hard:4d}   {r.top1_learned_on_hard:.3f}"
        )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    import argparse
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mip", type=int, default=1)
    ap.add_argument("--size", type=int, nargs=3, default=(320, 320, 80),
                    metavar=("X", "Y", "Z"))
    ap.add_argument("--train-center", type=int, nargs=3, default=list(DEFAULT_CENTER_MIP1),
                    metavar=("X", "Y", "Z"), help="train-box center (MIP1 vox)")
    ap.add_argument("--test-offset", type=int, nargs=3, default=(600, 600, 0),
                    metavar=("X", "Y", "Z"),
                    help="test-box center offset from train (MIP1 vox) -> disjoint neurites")
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--steps-per-epoch", type=int, default=50)
    ap.add_argument("--embed-dim", type=int, default=32)
    ap.add_argument("--augment", action="store_true",
                    help="enable rotation/scale/flip augmentation (ablation: it HURTS here, "
                         "because local cut faces are not rotated relative to each other)")
    ap.add_argument("--out", type=str, default=None, help="save encoder weights")
    ap.add_argument("--metrics", type=str, default=None, help="save eval metrics JSON")
    args = ap.parse_args()

    train_center = tuple(args.train_center)
    test_center = tuple(c + o for c, o in zip(train_center, args.test_offset))

    print(f"[fetch] train box @ {train_center} (MIP1 vox), mip={args.mip}")
    train_vol = fetch_volume(train_center, tuple(args.size), mip=args.mip)
    print(f"[fetch] test  box @ {test_center} (MIP1 vox)")
    test_vol = fetch_volume(test_center, tuple(args.size), mip=args.mip)

    train_overlap = set(np.unique(train_vol.seg).tolist())
    test_overlap = set(np.unique(test_vol.seg).tolist())
    shared_ids = (train_overlap & test_overlap) - {0}
    print(f"[data] neurites shared between train/test boxes: {len(shared_ids)} "
          f"(want ~0 for a clean held-out test)")

    # Train faces from many z-planes across the train box.
    nz = train_vol.em.shape[2]
    z_centers = list(range(2, nz - 4, 2))
    bank = extract_face_bank(train_vol, z_centers)
    n_ids = len(set(bank.seg_ids.tolist()))
    print(f"[data] train face bank: {len(bank.patches)} faces from {n_ids} neurites")

    print(f"[train] {args.epochs} epochs (augment={args.augment}) ...")
    enc, losses = train_encoder(bank, embed_dim=args.embed_dim, epochs=args.epochs,
                                steps_per_epoch=args.steps_per_epoch, augment=args.augment)

    print("\n[eval] held-out test box, midplane cut:")
    rows = evaluate_reid(enc, test_vol)
    print(summarize(rows))
    print("\nLEARNED > raw means the encoder extracts more identity than the raw "
          "patch.\nCompare norm=F vs norm=T rows: signal kept under per-section "
          "normalisation is structural, not a staining trick.")

    if args.out:
        torch.save({"state_dict": enc.state_dict(), "embed_dim": args.embed_dim,
                    "patch": PATCH, "losses": losses}, args.out)
        print(f"\n[out] saved encoder -> {args.out}")
    if args.metrics:
        with open(args.metrics, "w") as f:
            json.dump({"train_center": list(train_center), "test_center": list(test_center),
                       "n_train_faces": int(len(bank.patches)),
                       "shared_neurites": len(shared_ids),
                       "final_loss": losses[-1] if losses else None,
                       "rows": [asdict(r) for r in rows]}, f, indent=2)
        print(f"[out] saved metrics -> {args.metrics}")


if __name__ == "__main__":
    main()
