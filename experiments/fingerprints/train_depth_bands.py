"""Depth-aware band encoders: retrain on 16 nm 3-section depth stacks.

The headline encoders see one mean-projected slab.  The residual-error
diagnostic showed misses are mostly distant / degenerate partners (not a patch
problem), but the raw-cosine probe showed a 3-section depth stack roughly
*doubles* geom-miss recovery at 16 nm.  This trains depth-aware band encoders to
see whether that survives into the learned combiner and beats the 0.767 headline.

Self-contained (does not touch the single-channel pipeline): a 3-channel CNN,
NT-Xent/InfoNCE training on 3-section synthetic stacks mined from the cached
16 nm boxes (no CAVE), real fine-tune on 3-section v117 pairs, then the learned
confidence combiner over depth-stack faces (`band_faces_depth`).

z is 40 nm at every mip, so the 3-section stack is genuine z-structure the flat
patch threw away; mip stays 1 (16 nm) per the cheap-and-safe choice.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from .fingerprint_break_resolution import PATCH, Volume
from . import v117_error_relink as v
from . import v117_reconstructed as r
from .band_faces_depth import _stack, site_faces_bands_depth
from .train_synthetic_skeleton import _fragment_z_extents
from .train_combiner import _z, train_mlp, _score


# --------------------------------------------------------------------------- #
# 3-channel encoder + contrastive training
# --------------------------------------------------------------------------- #

def build_depth_encoder(embed_dim=32, in_ch=3):
    import torch.nn as nn

    class DepthEncoder(nn.Module):
        def __init__(self, d, c):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(c, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(64, d)

        def forward(self, x):
            import torch.nn.functional as F
            return F.normalize(self.head(self.net(x).flatten(1)), dim=1)

    return DepthEncoder(embed_dim, in_ch)


def _norm_stack(p):
    """Per-(sample,channel) mean-subtract + unit-std on [N, C, P, P]."""
    p = p.astype(np.float32)
    mu = p.mean(axis=(2, 3), keepdims=True)
    sd = p.std(axis=(2, 3), keepdims=True) + 1e-4
    return (p - mu) / sd


def embed_stacks(enc, stacks, batch=256):
    import torch
    p = _norm_stack(np.asarray(stacks))
    out = []
    enc.eval()
    with torch.no_grad():
        for b0 in range(0, len(p), batch):
            out.append(enc(torch.from_numpy(p[b0:b0 + batch])).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def make_stack_embed_fn(enc):
    return lambda stacks: embed_stacks(enc, np.asarray(stacks))


def _infonce(enc, A, P, D, bi, temperature, torch):
    a, p = torch.from_numpy(A[bi]), torch.from_numpy(P[bi])
    k = D.shape[1]
    d = torch.from_numpy(D[bi].reshape(-1, *D.shape[2:]))
    za, zp, zd = enc(a), enc(p), enc(d)
    pool = torch.cat([zp, zd], dim=0)
    logits = za @ pool.t() / temperature
    target = torch.arange(len(bi))
    loss = torch.nn.functional.cross_entropy(logits, target)
    own = torch.cat([zp[:, None, :], zd.reshape(len(bi), k, -1)], dim=1)
    sim = (za[:, None, :] * own).sum(-1)
    return loss, float((sim.argmax(1) == 0).float().mean())


def finetune_depth(anchors, positives, distractors, *, init_ckpt=None, warm_only=False,
                   embed_dim=32, epochs=60, batch=32, lr=1e-3, temperature=0.2, seed=0,
                   val_frac=0.15, eval_every=2, patience=8, ckpt_path=None, verbose=True):
    """InfoNCE with val early-stop + best checkpoint; inputs are [N,C,P,P]."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    in_ch = anchors.shape[1]
    enc = build_depth_encoder(embed_dim, in_ch)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    start_ep = 0
    if init_ckpt and os.path.exists(init_ckpt):
        ck = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        enc.load_state_dict(ck["state_dict"])
        if "opt_state" in ck and not warm_only:
            opt.load_state_dict(ck["opt_state"]); start_ep = int(ck.get("epoch", 0))

    A, P = _norm_stack(anchors), _norm_stack(positives)
    k = distractors.shape[1]
    D = _norm_stack(distractors.reshape(-1, *distractors.shape[2:])).reshape(distractors.shape)
    N = len(A)
    perm = rng.permutation(N)
    n_val = max(batch, int(val_frac * N)) if val_frac > 0 and N > 4 * batch else 0
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    best = {"top1": -1.0, "state": None, "epoch": -1}
    history = {"train_loss": [], "val_top1": []}
    since = 0
    for ep in range(start_ep, epochs):
        enc.train()
        order = rng.permutation(len(train_idx))
        el, nb = 0.0, 0
        for b0 in range(0, len(train_idx), batch):
            bi = train_idx[order[b0:b0 + batch]]
            if len(bi) < 4:
                continue
            loss, _ = _infonce(enc, A, P, D, bi, temperature, torch)
            opt.zero_grad(); loss.backward(); opt.step()
            el += float(loss.detach()); nb += 1
        history["train_loss"].append(el / max(nb, 1))
        if n_val and (ep % eval_every == 0 or ep == epochs - 1):
            enc.eval()
            with torch.no_grad():
                vt, vb = 0.0, 0
                for b0 in range(0, n_val, batch):
                    bi = val_idx[b0:b0 + batch]
                    if len(bi) < 4:
                        continue
                    _, top1 = _infonce(enc, A, P, D, bi, temperature, torch)
                    vt += top1; vb += 1
            vtop1 = vt / max(vb, 1)
            history["val_top1"].append((ep, vtop1))
            if vtop1 > best["top1"] + 1e-4:
                best = {"top1": vtop1, "state": {kk: vv.clone() for kk, vv in enc.state_dict().items()},
                        "epoch": ep}
                since = 0
                if ckpt_path:
                    torch.save({"state_dict": enc.state_dict(), "opt_state": opt.state_dict(),
                                "epoch": ep + 1, "embed_dim": embed_dim, "in_ch": in_ch,
                                "patch": PATCH, "val_top1": vtop1}, ckpt_path)
            else:
                since += 1
            if verbose:
                print(f"  epoch {ep:3d}  train={history['train_loss'][-1]:.4f}  "
                      f"val_top1={vtop1:.3f}{'  *best' if since == 0 else ''}", flush=True)
            if since >= patience:
                break
    if best["state"] is not None:
        enc.load_state_dict(best["state"])
    enc.eval()
    return enc, history


def load_depth_encoder(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    enc = build_depth_encoder(ck.get("embed_dim", 32), ck.get("in_ch", 3))
    enc.load_state_dict(ck["state_dict"]); enc.eval()
    return enc


# --------------------------------------------------------------------------- #
# Synthetic 3-section stack mining (from cached 16 nm boxes, no CAVE)
# --------------------------------------------------------------------------- #

def mine_box_depth(vol, *, n_sections=3, gap_sections=2, sigma=2.0, pairs_per_fragment=2,
                   n_distractors=8, min_vox_per_section=30, max_frags=25, seed=0):
    rng = np.random.default_rng(seed)
    ext = _fragment_z_extents(vol.seg, min_vox_per_section)
    need = 2 * n_sections + gap_sections + 1
    big = [f for f, zs in ext.items() if len(zs) >= need]
    if len(big) < 4:
        return []
    if len(big) > max_frags:
        big = [int(x) for x in rng.choice(big, max_frags, replace=False)]
    nz = vol.em.shape[2]
    # one stack per fragment (mid z) as reusable hard negative
    neg_at = {}
    for f in big:
        zs = ext[f]
        zc = min(zs[len(zs) // 2], nz - n_sections)
        st = _stack(vol, zc, 1, n_sections, f, sigma)
        if st is not None:
            neg_at[f] = st
    valid = [f for f in big if f in neg_at]
    if len(valid) < 4:
        return []
    out = []
    for f in valid:
        zs = ext[f]
        z0lo, z0hi = min(zs), max(zs)
        for _ in range(pairs_per_fragment):
            hi = max(z0lo + 1, z0hi - n_sections - gap_sections - n_sections + 1)
            za = int(rng.integers(z0lo, hi)) if hi > z0lo else z0lo
            zb = za + n_sections + gap_sections
            if zb + n_sections > nz:
                continue
            sa = _stack(vol, za, 1, n_sections, f, sigma)
            sb = _stack(vol, zb, 1, n_sections, f, sigma)
            if sa is None or sb is None:
                continue
            negs = [g for g in valid if g != f]
            idx = rng.choice(len(negs), size=n_distractors, replace=len(negs) < n_distractors)
            nl = np.stack([neg_at[negs[i]][0] for i in idx])     # [k, ns, P, P]
            nh = np.stack([neg_at[negs[i]][1] for i in idx])
            out.append((sa[0], sb[0], nl, sa[1], sb[1], nh))
    return out


def mine_from_cache_depth(cache_dir, exclude_keys, *, want_res=16, n_sections=3,
                          gap_sections=2, sigma=2.0, target_pairs=3000, verbose=True):
    files = sorted(glob.glob(os.path.join(cache_dir, "v117_*.npz")))
    lo_a, lo_p, lo_d, hi_a, hi_p, hi_d = [], [], [], [], [], []
    used = 0
    for fp in files:
        key = os.path.basename(fp)[len("v117_"):-len(".npz")]
        if key in exclude_keys:
            continue
        try:
            z = np.load(fp)
            res = tuple(int(x) for x in z["res"])
            if res[0] != want_res:
                continue
            vol = Volume(em=z["em"], seg=z["seg"], resolution_nm=res,
                         origin_vox=tuple(int(x) for x in z["origin"]))
        except Exception:
            continue
        for (la, lp, ld, ha, hp, hd) in mine_box_depth(
                vol, n_sections=n_sections, gap_sections=gap_sections, sigma=sigma,
                seed=int(key, 16) % 2**31):
            lo_a.append(la); lo_p.append(lp); lo_d.append(ld)
            hi_a.append(ha); hi_p.append(hp); hi_d.append(hd)
        used += 1
        if verbose and used % 25 == 0:
            print(f"  mine: {used} boxes, {len(lo_a)} pairs", flush=True)
        if len(lo_a) >= target_pairs:
            break
    if not lo_a:
        raise RuntimeError("no synthetic depth stacks mined")
    st = lambda L: np.stack(L).astype(np.float32)
    return {"lo_a": st(lo_a), "lo_p": st(lo_p), "lo_d": st(lo_d),
            "hi_a": st(hi_a), "hi_p": st(hi_p), "hi_d": st(hi_d)}


def collect_real_depth(cl, ts, roots, *, id_mip=1, hi_mip=1, n_sections=3, radius_nm=2000.0,
                       direction_cone_deg=45.0, n_distractors=8, max_sites=10, sigma=2.0,
                       seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    lo_a, lo_p, lo_d, hi_a, hi_p, hi_d = [], [], [], [], [], []
    for n, rt in enumerate(roots):
        try:
            ss = v.sites_from_l2_graph(cl, rt, ts, max_gap_nm=radius_nm, max_sites=max_sites)
        except Exception:
            continue
        for s in ss:
            try:
                f = site_faces_bands_depth(cl, ts, s, id_mip=id_mip, hi_mip=hi_mip,
                                           n_sections=n_sections, radius_nm=radius_nm,
                                           direction_cone_deg=direction_cone_deg, sigma=sigma)
            except Exception:
                f = None
            if f is None:
                continue
            it = f["is_true"]; tr = np.where(it)[0]; fa = np.where(~it)[0]
            if len(tr) < 1 or len(fa) < 2:
                continue
            t = int(tr[0])
            pick = rng.choice(fa, size=n_distractors, replace=len(fa) < n_distractors)
            lo_a.append(f["q_low"]); lo_p.append(f["low"][t]); lo_d.append(f["low"][pick])
            hi_a.append(f["q_high"]); hi_p.append(f["high"][t]); hi_d.append(f["high"][pick])
        if verbose and lo_a:
            print(f"  real-pairs: {n + 1}/{len(roots)} neurons, {len(lo_a)} pairs", flush=True)
    if not lo_a:
        return None
    st = lambda L: np.stack(L).astype(np.float32)
    return {"lo_a": st(lo_a), "lo_p": st(lo_p), "lo_d": st(lo_d),
            "hi_a": st(hi_a), "hi_p": st(hi_p), "hi_d": st(hi_d)}


# --------------------------------------------------------------------------- #
# Combiner over depth-stack faces
# --------------------------------------------------------------------------- #

def _sims_stack(q, bank, emb):
    qe = np.asarray(emb(q[None]))[0]
    ce = np.asarray(emb(bank))
    qe = qe / (np.linalg.norm(qe) + 1e-9)
    ce = ce / (np.linalg.norm(ce, axis=1, keepdims=True) + 1e-9)
    return ce @ qe


def collect_combiner_depth(cl, ts, roots, bio_emb, art_emb, *, n_sections=3, radius_nm=2000.0,
                           direction_cone_deg=45.0, max_sites=10, verbose=True):
    sites = []
    for n, rt in enumerate(roots):
        try:
            ss = v.sites_from_l2_graph(cl, rt, ts, max_gap_nm=radius_nm, max_sites=max_sites)
        except Exception:
            continue
        for s in ss:
            try:
                f = site_faces_bands_depth(cl, ts, s, id_mip=1, hi_mip=1, n_sections=n_sections,
                                           radius_nm=radius_nm, direction_cone_deg=direction_cone_deg)
            except Exception:
                f = None
            if f is None:
                continue
            it = f["is_true"].astype(np.float32)
            gd = f["geom_dist"].astype(float)
            arts = _sims_stack(f["q_high"], f["high"], art_emb)
            bios = _sims_stack(f["q_low"], f["low"], bio_emb)
            gz = -_z(gd)
            X = np.stack([gz, _z(arts), _z(bios), arts, bios,
                          (gd == gd.min()).astype(np.float32),
                          (arts == arts.max()).astype(np.float32)], axis=1).astype(np.float32)
            sites.append((X, it, gd, arts))
        if verbose and sites:
            print(f"  collect: {n + 1}/{len(roots)} neurons, {len(sites)} sites", flush=True)
    return sites


def evaluate_combiner(test_sites, net):
    hc, hg, ha = [], [], []
    for X, it, gd, arts in test_sites:
        hc.append(int(bool(it[int(np.argmax(_score(net, X)))])))
        hg.append(int(bool(it[int(np.argmin(gd))])))
        ha.append(int(bool(it[int(np.argmax(arts))])))
    n = len(test_sites)
    return {"n": n, "combiner_top1": float(np.mean(hc)),
            "geom_top1": float(np.mean(hg)), "art_top1": float(np.mean(ha))}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-scan", type=int, default=200)
    ap.add_argument("--test-neurons", type=int, default=20)
    ap.add_argument("--train-neurons", type=int, default=40)
    ap.add_argument("--n-sections", type=int, default=3)
    ap.add_argument("--radius-nm", type=float, default=2000.0)
    ap.add_argument("--max-sites", type=int, default=10)
    ap.add_argument("--target-pairs", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--ft-epochs", type=int, default=40)
    ap.add_argument("--finetune-real", action="store_true")
    ap.add_argument("--out-bio", default="experiments/fingerprints/cutface_bio_depth.pt")
    ap.add_argument("--out-art", default="experiments/fingerprints/cutface_art_depth.pt")
    ap.add_argument("--out", default="experiments/fingerprints/combiner_depth_metrics.json")
    args = ap.parse_args()

    cl = v._client()
    ts = cl.chunkedgraph.get_oldest_timestamp()
    roots, _ = v.find_split_neurons(cl, n_scan=args.n_scan)
    test_roots = roots[:args.test_neurons]
    train_roots = roots[args.test_neurons:args.test_neurons + args.train_neurons]
    print(f"[split] {len(train_roots)} train / {len(test_roots)} test neurons", flush=True)

    box_cache = os.environ.get("V117_BOX_CACHE", "data/v117_box_cache")
    exclude = set()
    for rt in test_roots:
        try:
            for s in v.sites_from_l2_graph(cl, rt, ts, max_gap_nm=args.radius_nm, max_sites=args.max_sites):
                pts = np.asarray([s.pos_main_nm, s.pos_frag_nm], float)
                lo = pts.min(0) - args.radius_nm; hi = pts.max(0) + args.radius_nm
                exclude.add(v._box_key((tuple(lo.tolist()), tuple(hi.tolist())), 1))
        except Exception:
            continue
    print(f"[exclude] {len(exclude)} test box keys held out of mining", flush=True)

    print(f"[mine] 3-section stacks from {box_cache} (16 nm, no CAVE) ...", flush=True)
    data = mine_from_cache_depth(box_cache, exclude, want_res=16, n_sections=args.n_sections,
                                 target_pairs=args.target_pairs)
    print(f"[pretrain] bio on {len(data['lo_a'])} stacks ...", flush=True)
    bio, _ = finetune_depth(data["lo_a"], data["lo_p"], data["lo_d"],
                            epochs=args.epochs, ckpt_path=args.out_bio)
    print(f"[pretrain] art on {len(data['hi_a'])} stacks ...", flush=True)
    art, _ = finetune_depth(data["hi_a"], data["hi_p"], data["hi_d"],
                            epochs=args.epochs, ckpt_path=args.out_art)

    if args.finetune_real:
        print(f"[finetune-real] collecting real depth pairs ...", flush=True)
        rd = collect_real_depth(cl, ts, train_roots, n_sections=args.n_sections,
                                radius_nm=args.radius_nm, max_sites=args.max_sites)
        if rd is not None and len(rd["lo_a"]) >= 8:
            print(f"[finetune-real] {len(rd['lo_a'])} real pairs; adapting ...", flush=True)
            bio, _ = finetune_depth(rd["lo_a"], rd["lo_p"], rd["lo_d"], init_ckpt=args.out_bio,
                                    warm_only=True, epochs=args.ft_epochs, lr=2e-4, ckpt_path=args.out_bio)
            art, _ = finetune_depth(rd["hi_a"], rd["hi_p"], rd["hi_d"], init_ckpt=args.out_art,
                                    warm_only=True, epochs=args.ft_epochs, lr=2e-4, ckpt_path=args.out_art)
        else:
            print(f"[finetune-real] too few real pairs -- skipping", flush=True)

    bio_emb, art_emb = make_stack_embed_fn(bio), make_stack_embed_fn(art)
    print("[collect] combiner train sites ...", flush=True)
    train_sites = collect_combiner_depth(cl, ts, train_roots, bio_emb, art_emb,
                                         n_sections=args.n_sections, radius_nm=args.radius_nm,
                                         max_sites=args.max_sites)
    print("[collect] combiner test sites ...", flush=True)
    test_sites = collect_combiner_depth(cl, ts, test_roots, bio_emb, art_emb,
                                        n_sections=args.n_sections, radius_nm=args.radius_nm,
                                        max_sites=args.max_sites)
    if not train_sites or not test_sites:
        print("insufficient sites"); return
    net = train_mlp(train_sites)
    res = evaluate_combiner(test_sites, net)
    print(f"\n16nm depth-stack combiner: {res['n']} test sites")
    print(f"  geometry   top-1: {res['geom_top1']:.3f}")
    print(f"  art-stack  top-1: {res['art_top1']:.3f}")
    print(f"  COMBINER   top-1: {res['combiner_top1']:.3f}"
          f"{'  <-- beats geometry' if res['combiner_top1'] > res['geom_top1'] else ''}")
    with open(args.out, "w") as fh:
        json.dump({"n_sections": args.n_sections, "n_train_sites": len(train_sites),
                   "n_synth_pairs": int(len(data["lo_a"])), **res}, fh, indent=2)
    print(f"[out] wrote {args.out}")


if __name__ == "__main__":
    main()
