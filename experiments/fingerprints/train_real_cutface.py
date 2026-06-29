"""Fine-tune the cut-face encoder on REAL v117 error cross-sections.

The planar encoder (``learned_cutface_encoder.py``) is trained on *artificial*
z-cuts and does not transfer to the oblique, messy cross-sections at real split
errors (raw patch >= planar encoder on the proximity panel).  This module trains
directly on the real thing:

* **Positive pair** = the two faces of one true split: the query face on the
  main arbor and the face on the merged-in fragment, at a real v117 error site.
* **Negatives** = the cone-filtered proximity distractors at the same site
  (the neurites a merge-proposal generator would actually confuse it with) plus
  the other sites' faces in the batch (in-batch negatives).

Loss is InfoNCE with those pooled negatives.  Train and test neurons are
disjoint roots.  Evaluation ranks the true partner on the realistic
proximity+cone panel, comparing the real-trained encoder, the planar encoder,
and the raw patch.

Requires a CAVE token in env var ``token`` and the public EM + seg volumes.

    python -m experiments.fingerprints.train_real_cutface \
        --n-scan 400 --train-neurons 120 --test-neurons 60 --epochs 40 \
        --out experiments/fingerprints/cutface_encoder_real.pt \
        --metrics experiments/fingerprints/real_relink_metrics.json
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np

from .fingerprint_break_resolution import PATCH
from .learned_cutface_encoder import (
    build_encoder, _normalize_patches, embed_patches, make_embed_fn, load_encoder,
)
from . import v117_error_relink as v


# ---------------------------------------------------------------------------
# Faces at one real error site
# ---------------------------------------------------------------------------

def site_faces(site, *, mip=1, slab=3, radius_nm=2000.0, direction_cone_deg=45.0,
               min_vox=40):
    """Query face + cone-filtered proximity candidate faces at a real site.

    Returns ``{"query", "patches"[C,P,P], "true_idx", "cand_ids"}`` or None.
    """
    vol = v._fetch_box(site.pos_main_nm, site.pos_frag_nm, radius_nm, mip)
    nz = vol.em.shape[2]
    qa_id, idx_main = v._seg_id_at(vol, site.pos_main_nm)
    true_id, _ = v._seg_id_at(vol, site.pos_frag_nm)
    if qa_id == 0 or true_id == 0 or true_id == qa_id or site.gap_nm > radius_nm:
        return None

    za = max(min(v._z_index(vol, site.pos_main_nm[2]), nz - slab), 0)
    q = v._patch_from_slab(vol.em, vol.seg, za, za + slab, qa_id)
    if q is None:
        return None

    prox = v._proximity_candidates(vol, idx_main, radius_nm, qa_id, min_vox)
    if true_id not in prox:
        return None

    vox = np.asarray(vol.resolution_nm, float)
    origin = np.asarray(vol.origin_vox, float)
    pmain = np.asarray(site.pos_main_nm, float)
    tangent = np.asarray(site.tangent_nm, float)
    tn = np.linalg.norm(tangent)
    cone_cos = np.cos(np.deg2rad(direction_cone_deg)) if direction_cone_deg else None

    cand_ids, patches = [], []
    for sid, (nv, _) in prox.items():
        if cone_cos is not None and tn > 1e-6 and sid != true_id:
            d = (origin + nv + 0.5) * vox - pmain
            dn = np.linalg.norm(d)
            if dn > 1e-6 and abs(float(d @ tangent) / (dn * tn)) < cone_cos:
                continue
        zc = max(min(int(nv[2]) - slab // 2, nz - slab), 0)
        p = v._patch_from_slab(vol.em, vol.seg, zc, zc + slab, sid)
        if p is not None:
            cand_ids.append(sid)
            patches.append(p)
    if true_id not in cand_ids or len(cand_ids) < 3:
        return None
    return {"query": q, "patches": np.stack(patches),
            "true_idx": cand_ids.index(true_id), "cand_ids": cand_ids}


# ---------------------------------------------------------------------------
# Collect a training set (anchor, positive, distractors) over neurons
# ---------------------------------------------------------------------------

def collect_training_set(cl, roots, ts, *, mip=1, radius_nm=2000.0,
                         direction_cone_deg=45.0, n_distractors=8, max_sites=10,
                         seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    anchors, positives, distractors = [], [], []
    for n, rt in enumerate(roots):
        try:
            sites = v.sites_from_l2_graph(cl, rt, ts, max_gap_nm=radius_nm, max_sites=max_sites)
        except Exception:
            continue
        for s in sites:
            try:
                f = site_faces(s, mip=mip, radius_nm=radius_nm,
                               direction_cone_deg=direction_cone_deg)
            except Exception:
                f = None
            if f is None:
                continue
            t = f["true_idx"]
            others = [i for i in range(len(f["cand_ids"])) if i != t]
            if not others:
                continue
            pick = rng.choice(others, size=n_distractors,
                              replace=len(others) < n_distractors)
            anchors.append(f["query"])
            positives.append(f["patches"][t])
            distractors.append(f["patches"][pick])
        if verbose and (n % 15 == 0 or n == len(roots) - 1):
            print(f"  collect: {n + 1}/{len(roots)} neurons, {len(anchors)} pairs")
    if not anchors:
        raise RuntimeError("collected no training pairs")
    return (np.stack(anchors).astype(np.float32),
            np.stack(positives).astype(np.float32),
            np.stack(distractors).astype(np.float32))  # [N, k, P, P]


# ---------------------------------------------------------------------------
# Fine-tune with InfoNCE (anchor=main face, positive=true partner, +distractors)
# ---------------------------------------------------------------------------

def _infonce_batch(enc, A, P, D, bi, temperature, torch):
    a = torch.from_numpy(A[bi]).unsqueeze(1)
    p = torch.from_numpy(P[bi]).unsqueeze(1)
    d = torch.from_numpy(D[bi].reshape(-1, PATCH, PATCH)).unsqueeze(1)
    za, zp, zd = enc(a), enc(p), enc(d)
    pool = torch.cat([zp, zd], dim=0)              # [B + B*k, D]
    logits = za @ pool.t() / temperature           # [B, B + B*k]
    target = torch.arange(len(bi))                 # positive i is column i
    loss = torch.nn.functional.cross_entropy(logits, target)
    # per-pair top-1: is the own positive nearest among own positive + own distractors?
    k = D.shape[1]
    own = torch.cat([zp[:, None, :], zd.reshape(len(bi), k, -1)], dim=1)  # [B, 1+k, D]
    sim = (za[:, None, :] * own).sum(-1)            # [B, 1+k]
    top1 = (sim.argmax(1) == 0).float().mean()
    return loss, float(top1)


def finetune(anchors, positives, distractors, *, init_ckpt=None, embed_dim=32,
             epochs=40, batch=32, lr=5e-4, temperature=0.2, seed=0, verbose=True,
             val_frac=0.15, eval_every=2, patience=8, ckpt_path=None):
    """Contrastive fine-tune with validation-based early stopping + best checkpoint.

    Holds out ``val_frac`` of the pairs; every ``eval_every`` epochs evaluates
    val InfoNCE loss and val per-pair top-1, keeps the best-val weights, and stops
    after ``patience`` checks without val-top1 improvement.  ``epochs`` is the
    cap.  When ``ckpt_path`` is set the best checkpoint (weights + optimizer +
    epoch) is written there so training can be resumed/extended.  Returns the
    best encoder and a history dict.
    """
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc = build_encoder(embed_dim)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    start_ep = 0
    if init_ckpt and os.path.exists(init_ckpt):
        ck = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        enc.load_state_dict(ck["state_dict"])
        if "opt_state" in ck:                       # true resume (optimizer + epoch)
            opt.load_state_dict(ck["opt_state"])
            start_ep = int(ck.get("epoch", 0))
        if verbose:
            print(f"  init from {init_ckpt} (start epoch {start_ep})")

    A = _normalize_patches(anchors)
    P = _normalize_patches(positives)
    k = distractors.shape[1]
    D = _normalize_patches(distractors.reshape(-1, PATCH, PATCH)).reshape(len(anchors), k, PATCH, PATCH)
    N = len(A)

    perm = rng.permutation(N)
    n_val = max(batch, int(val_frac * N)) if val_frac > 0 and N > 4 * batch else 0
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    history = {"train_loss": [], "val_loss": [], "val_top1": []}
    best = {"top1": -1.0, "loss": 1e9, "state": None, "epoch": -1}
    since_improve = 0
    for ep in range(start_ep, epochs):
        enc.train()
        order = rng.permutation(len(train_idx))
        ep_loss, nb = 0.0, 0
        for b0 in range(0, len(train_idx), batch):
            bi = train_idx[order[b0:b0 + batch]]
            if len(bi) < 4:
                continue
            loss, _ = _infonce_batch(enc, A, P, D, bi, temperature, torch)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.detach()); nb += 1
        history["train_loss"].append(ep_loss / max(nb, 1))

        if n_val and (ep % eval_every == 0 or ep == epochs - 1):
            enc.eval()
            with torch.no_grad():
                vl, vt, vb = 0.0, 0.0, 0
                for b0 in range(0, n_val, batch):
                    bi = val_idx[b0:b0 + batch]
                    if len(bi) < 4:
                        continue
                    loss, top1 = _infonce_batch(enc, A, P, D, bi, temperature, torch)
                    vl += float(loss); vt += top1; vb += 1
            vloss, vtop1 = vl / max(vb, 1), vt / max(vb, 1)
            history["val_loss"].append((ep, vloss)); history["val_top1"].append((ep, vtop1))
            improved = vtop1 > best["top1"] + 1e-4
            if improved:
                best = {"top1": vtop1, "loss": vloss,
                        "state": {kk: vv.clone() for kk, vv in enc.state_dict().items()},
                        "epoch": ep}
                since_improve = 0
                if ckpt_path:
                    torch.save({"state_dict": enc.state_dict(), "opt_state": opt.state_dict(),
                                "epoch": ep + 1, "embed_dim": embed_dim, "patch": PATCH,
                                "val_top1": vtop1}, ckpt_path)
            else:
                since_improve += 1
            if verbose:
                print(f"  epoch {ep:3d}  train={history['train_loss'][-1]:.4f}  "
                      f"val={vloss:.4f}  val_top1={vtop1:.3f}"
                      f"{'  *best' if improved else ''}", flush=True)
            if since_improve >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep} (best val_top1 {best['top1']:.3f} "
                          f"@ep{best['epoch']})", flush=True)
                break
        elif verbose and ep % 5 == 0:
            print(f"  epoch {ep:3d}  train={history['train_loss'][-1]:.4f}", flush=True)

    if best["state"] is not None:
        enc.load_state_dict(best["state"])
    enc.eval()
    if ckpt_path and (best["state"] is None or not os.path.exists(ckpt_path)):
        # fallback save (e.g. no val split) so a resumable checkpoint always exists
        torch.save({"state_dict": enc.state_dict(), "opt_state": opt.state_dict(),
                    "epoch": epochs, "embed_dim": embed_dim, "patch": PATCH}, ckpt_path)
    return enc, history


# ---------------------------------------------------------------------------
# Held-out evaluation: rank true partner with each method (one fetch per site)
# ---------------------------------------------------------------------------

def evaluate_methods(cl, roots, ts, embedders, *, mip=1, radius_nm=2000.0,
                     direction_cone_deg=45.0, max_sites=10, verbose=True):
    """embedders: {name: embed_fn}.  Returns per-method rank lists + raw patch."""
    ranks = {name: [] for name in embedders}
    ranks["raw"] = []
    ncand = []
    for n, rt in enumerate(roots):
        try:
            sites = v.sites_from_l2_graph(cl, rt, ts, max_gap_nm=radius_nm, max_sites=max_sites)
        except Exception:
            continue
        for s in sites:
            try:
                f = site_faces(s, mip=mip, radius_nm=radius_nm,
                               direction_cone_deg=direction_cone_deg)
            except Exception:
                f = None
            if f is None:
                continue
            q, P, t = f["query"], f["patches"], f["true_idx"]
            ncand.append(len(f["cand_ids"]))
            for name, emb in embedders.items():
                qe = np.asarray(emb(q[None]))[0]
                ce = np.asarray(emb(P))
                qe = qe / (np.linalg.norm(qe) + 1e-9)
                ce = ce / (np.linalg.norm(ce, axis=1, keepdims=True) + 1e-9)
                d = 1.0 - ce @ qe
                ranks[name].append(int((d < d[t]).sum()))
            qf = v._flatnorm(q)
            cf = np.stack([v._flatnorm(P[i]) for i in range(len(P))])
            d = 1.0 - cf @ qf
            ranks["raw"].append(int((d < d[t]).sum()))
        if verbose and ranks["raw"]:
            print(f"  eval: {n + 1}/{len(roots)} neurons, {len(ranks['raw'])} sites")
    return ranks, ncand


def _summary(ranks, ncand):
    out = {}
    for name, rs in ranks.items():
        if not rs:
            continue
        rs = np.asarray(rs)
        out[name] = {"top1": float((rs == 0).mean()),
                     "mrr": float(np.mean(1.0 / (rs + 1.0))),
                     "n": int(len(rs))}
    out["mean_candidates"] = float(np.mean(ncand)) if ncand else 0.0
    out["chance_top1"] = float(np.mean(1.0 / np.asarray(ncand))) if ncand else 0.0
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    import argparse
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-scan", type=int, default=400)
    ap.add_argument("--train-neurons", type=int, default=120)
    ap.add_argument("--test-neurons", type=int, default=60)
    ap.add_argument("--mip", type=int, default=1)
    ap.add_argument("--radius-nm", type=float, default=2000.0)
    ap.add_argument("--direction-cone-deg", type=float, default=45.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max-sites", type=int, default=4, help="cap sites/neuron (bounds EM fetches)")
    ap.add_argument("--init", default="experiments/fingerprints/cutface_encoder.pt",
                    help="planar encoder to initialise from")
    ap.add_argument("--cache", default=None, help="npz cache of collected training patches")
    ap.add_argument("--out", default="experiments/fingerprints/cutface_encoder_real.pt")
    ap.add_argument("--metrics", default="experiments/fingerprints/real_relink_metrics.json")
    args = ap.parse_args()

    cl = v._client()
    print(f"[cave] scanning {args.n_scan} somas for v117-era splits ...")
    roots, ts = v.find_split_neurons(cl, n_scan=args.n_scan)
    print(f"[cave] {len(roots)} split neurons")
    rng = np.random.default_rng(0)
    roots = list(rng.permutation(roots))
    train_roots = roots[:args.train_neurons]
    test_roots = roots[args.train_neurons:args.train_neurons + args.test_neurons]
    print(f"[split] {len(train_roots)} train / {len(test_roots)} test neurons (disjoint)")

    if args.cache and os.path.exists(args.cache):
        z = np.load(args.cache)
        anchors, positives, distractors = z["anchors"], z["positives"], z["distractors"]
        print(f"[cache] loaded {len(anchors)} training pairs from {args.cache}")
    else:
        print("[collect] gathering real error-site faces ...")
        anchors, positives, distractors = collect_training_set(
            cl, train_roots, ts, mip=args.mip, radius_nm=args.radius_nm,
            direction_cone_deg=args.direction_cone_deg, max_sites=args.max_sites)
        if args.cache:
            np.savez(args.cache, anchors=anchors, positives=positives, distractors=distractors)
            print(f"[cache] wrote {len(anchors)} pairs -> {args.cache}")

    print(f"[train] fine-tuning on {len(anchors)} real pairs ({args.epochs} epochs) ...")
    enc, losses = finetune(anchors, positives, distractors,
                           init_ckpt=args.init, epochs=args.epochs)
    torch.save({"state_dict": enc.state_dict(), "embed_dim": 32, "patch": PATCH,
                "losses": losses, "trained_on": "real_v117_sites"}, args.out)
    print(f"[out] saved real-trained encoder -> {args.out}")

    print("[eval] held-out test neurons (proximity + cone) ...")
    embedders = {"real": make_embed_fn(enc)}
    if os.path.exists(args.init):
        embedders["planar"] = make_embed_fn(load_encoder(args.init))
    ranks, ncand = evaluate_methods(cl, test_roots, ts, embedders, mip=args.mip,
                                    radius_nm=args.radius_nm,
                                    direction_cone_deg=args.direction_cone_deg, max_sites=args.max_sites)
    summ = _summary(ranks, ncand)
    print("\nHeld-out v117 re-linking (proximity + cone):")
    print(f"  sites={summ.get('raw',{}).get('n',0)}  mean candidates={summ['mean_candidates']:.1f}  "
          f"chance top-1={summ['chance_top1']:.3f}")
    for name in ("raw", "planar", "real"):
        if name in summ:
            print(f"  {name:7s} top-1 / MRR: {summ[name]['top1']:.3f} / {summ[name]['mrr']:.3f}")

    with open(args.metrics, "w") as f:
        json.dump({"radius_nm": args.radius_nm, "direction_cone_deg": args.direction_cone_deg,
                   "n_train_pairs": int(len(anchors)),
                   "train_neurons": len(train_roots), "test_neurons": len(test_roots),
                   "final_loss": losses[-1] if losses else None, "summary": summ}, f, indent=2)
    print(f"[out] wrote {args.metrics}")


if __name__ == "__main__":
    main()
