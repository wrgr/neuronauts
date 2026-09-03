#!/usr/bin/env python3
"""Self-supervised generative grammar v2 -- richer model, same framing.

v1 (skel_ssl_grammar.py) plateaued ~0.77: a single global max-pool latent is too coarse to
express multi-scale neuron structure. v2 keeps the framing identical -- denoising
reconstruction of real (noisy) neurons, errors = low grammaticality, reconstruction = the
fix, no labels / no hand features / no synthesized merges -- and only makes the MODEL richer:

  1. multi-scale DGCNN encoder (dynamic-kNN EdgeConv; local->meso->global with depth),
  2. raw attributes we'd been discarding: per-point radius + skeleton-vs-synapse flag,
     fed in AND reconstructed (so anomalous caliber/synapse layout also raises error),
  3. larger latent + more points.

Inputs are still raw: [xyz, log radius, is_synapse]. Trained on the noisy corpus; group-by-cell
CV; do-nothing-relevant precision@k.

    python -m attic.pcfg_one_offs.skel_ssl_grammar_v2 --sidetable data/sidetable_7box.npz
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pcfg.synapse_correction import SideTable  # noqa: E402

SCALE = 50_000.0
N_PTS = 384


def load_skels(skel_dir, version):
    """rid -> (vertices[n,3], radius[n])."""
    out = {}
    pat = re.compile(rf"v{version}_rid(\d+)_skv")
    for f in glob.glob(str(Path(skel_dir) / f"v{version}_rid*.npz")):
        m = pat.search(Path(f).name)
        if not m:
            continue
        d = np.load(f)
        V = d["vertices"].astype(np.float32)
        if len(V) < 8:
            continue
        R = d["radius"].astype(np.float32) if "radius" in d else np.full(len(V), 300.0, np.float32)
        out[int(m.group(1))] = (V, R)
    return out


def obj_points(skel_parts, syn_parts, rng, n=N_PTS, augment=False):
    """Raw 5-channel points: [x,y,z, log1p(radius)/5, is_synapse]."""
    rows = []
    for V, R in skel_parts:
        if len(V):
            rows.append(np.column_stack([V, np.log1p(R) / 5.0, np.zeros(len(V), np.float32)]))
    for S in syn_parts:
        if len(S):
            rows.append(np.column_stack([S, np.zeros(len(S), np.float32), np.ones(len(S), np.float32)]))
    P = np.vstack(rows).astype(np.float32)
    idx = rng.choice(len(P), n, replace=len(P) < n)
    P = P[idx]
    xyz = P[:, :3] - P[:, :3].mean(0)
    if augment:
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3))); xyz = xyz @ Q
        xyz = xyz * rng.uniform(0.85, 1.15)
    P = P.copy(); P[:, :3] = xyz / SCALE
    return P.astype(np.float32)


def build_model():
    import torch
    import torch.nn as nn

    def knn(x, k):
        d = torch.cdist(x, x)
        return d.topk(min(k, x.size(1)), dim=-1, largest=False).indices

    class EdgeConv(nn.Module):
        def __init__(self, cin, cout, k=20):
            super().__init__()
            self.k = k
            self.mlp = nn.Sequential(nn.Linear(2 * cin, cout), nn.ReLU(), nn.Linear(cout, cout), nn.ReLU())

        def forward(self, x):
            B, N, C = x.shape
            idx = knn(x, self.k)
            nb = torch.gather(x.unsqueeze(2).expand(B, N, idx.size(-1), C), 1,
                              idx.unsqueeze(-1).expand(B, N, idx.size(-1), C))
            feat = torch.cat([x.unsqueeze(2).expand(B, N, idx.size(-1), C), nb - x.unsqueeze(2)], -1)
            return self.mlp(feat).max(2).values

    class AE(nn.Module):
        def __init__(self, k=N_PTS, lat=256, cin=5):
            super().__init__()
            self.e1 = EdgeConv(cin, 64); self.e2 = EdgeConv(64, 64); self.e3 = EdgeConv(64, 128)
            self.to_lat = nn.Linear(2 * 256, lat)            # multi-scale concat -> latent
            self.dec = nn.Sequential(nn.Linear(lat, 512), nn.ReLU(), nn.Linear(512, 1024), nn.ReLU(),
                                     nn.Linear(1024, k * 5))
            self.k = k

        def forward(self, x):
            h1 = self.e1(x); h2 = self.e2(h1); h3 = self.e3(h2)
            h = torch.cat([h1, h2, h3], -1)                   # [B,N,256]
            z = self.to_lat(torch.cat([h.max(1).values, h.mean(1)], -1))
            return self.dec(z).view(x.size(0), self.k, 5)

    return AE()


def recon_loss(rec, x):
    """Chamfer on xyz + attribute loss at the input->nearest-recon assignment."""
    import torch
    import torch.nn.functional as F
    rx, ix = rec[:, :, :3], x[:, :, :3]
    d = torch.cdist(ix, rx)                                   # [B,Ni,Nr]
    di, ji = d.min(2)                                          # input -> nearest recon
    dj = d.min(1).values                                      # recon -> nearest input
    chamf = di.mean(1) + dj.mean(1)
    rec_attr = torch.gather(rec[:, :, 3:], 1, ji.unsqueeze(-1).expand(-1, -1, 2))
    attr = F.mse_loss(rec_attr[:, :, 0], x[:, :, 3], reduction="none").mean(1) + \
        F.binary_cross_entropy_with_logits(rec_attr[:, :, 1], x[:, :, 4], reduction="none").mean(1)
    return chamf + 0.1 * attr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sidetable", required=True)
    ap.add_argument("--skel-cache", default="data/skel_v1718")
    ap.add_argument("--version", type=int, default=1718)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    skels = load_skels(args.skel_cache, args.version)
    d = np.load(args.sidetable)
    tab = SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])
    from scipy.spatial import cKDTree
    valid = tab.root_later > 0
    syn_pts_by_later, syn_v117_by_later = defaultdict(list), defaultdict(list)
    by_v117_later, by_later_v117, cnt = defaultdict(set), defaultdict(set), defaultdict(int)
    for i in np.nonzero(valid)[0]:
        a, b = int(tab.root_v117[i]), int(tab.root_later[i])
        syn_pts_by_later[b].append(tab.pt[i]); syn_v117_by_later[b].append(a)
        by_v117_later[a].add(b); by_later_v117[b].add(a); cnt[b] += 1
    syn_by_later = {b: np.array(v, np.float32) for b, v in syn_pts_by_later.items()}

    clean = [b for b, s in by_later_v117.items() if len(s) == 1 and cnt[b] >= 15 and b in skels]
    merges = [sorted(s) for a, s in by_v117_later.items()
              if len(s) >= 2 and sum(p in skels for p in s) >= 2]
    # false SPLITS: a v1718 cell with >=2 v117 sources -> each source is an incomplete fragment.
    splits = [b for b, s in by_later_v117.items() if len(s) >= 2 and b in skels and cnt[b] >= 20]
    print(f"cached skeletons={len(skels)}  clean cells={len(clean)}  "
          f"merges={len(merges)}  split-parents={len(splits)}", flush=True)
    if len(clean) < 30 or len(merges) < 8:
        print("not enough cached objects yet."); return

    gid = {}

    def G(later):
        return gid.setdefault(int(later), len(gid))

    # kind: 0=clean, 1=merge (too much), 2=split fragment (too little)
    objs = [(0, [skels[b]], [syn_by_later.get(b, np.zeros((0, 3), np.float32))], G(b)) for b in clean]
    for parts in merges:
        pp = [p for p in parts if p in skels]
        objs.append((1, [skels[p] for p in pp],
                     [syn_by_later.get(p, np.zeros((0, 3), np.float32)) for p in pp], G(pp[0])))
    for b in splits:  # partition the cell skeleton among its v117 fragments by nearest synapse
        Vb, Rb = skels[b]
        cs = np.array(syn_pts_by_later[b], np.float32); cv = np.array(syn_v117_by_later[b])
        vert_src = cv[cKDTree(cs).query(Vb)[1]]
        for a in set(cv.tolist()):
            vm, sm = vert_src == a, cv == a
            if vm.sum() >= 8 and sm.sum() >= 8:
                objs.append((2, [(Vb[vm], Rb[vm])], [cs[sm]], G(b)))

    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    kind = np.array([o[0] for o in objs]); groups = np.array([o[3] for o in objs])
    y = (kind > 0).astype(int)                      # any-error vs clean
    print(f"objects: clean={int((kind==0).sum())} merge={int((kind==1).sum())} "
          f"split={int((kind==2).sum())}", flush=True)
    gkf = GroupKFold(n_splits=max(2, min(args.folds, len(np.unique(groups)))))
    score = np.full(len(objs), np.nan)

    for fold, (tr, te) in enumerate(gkf.split(objs, y, groups)):
        train_clean = [i for i in tr if objs[i][0] == 0]
        if len(train_clean) < 8:
            continue
        net = build_model(); opt = torch.optim.Adam(net.parameters(), lr=8e-4, weight_decay=1e-5)
        for ep in range(args.epochs):
            net.train(); order = rng.permutation(train_clean)
            for s in range(0, len(order), args.batch):
                ids = order[s:s + args.batch]
                xb = torch.tensor(np.stack([obj_points(objs[i][1], objs[i][2], rng, augment=True) for i in ids]))
                opt.zero_grad(); loss = recon_loss(net(xb), xb).mean(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            for i in te:
                errs = [recon_loss(net(torch.tensor(obj_points(objs[i][1], objs[i][2], rng)[None])),
                                   torch.tensor(obj_points(objs[i][1], objs[i][2], rng)[None])).item()
                        for _ in range(8)]
                score[i] = float(np.mean(errs))
        ok = ~np.isnan(score)
        if ok.sum() > 5 and len(np.unique(y[ok])) == 2:
            print(f"  fold {fold+1}: AUC={roc_auc_score(y[ok], score[ok]):.3f} ({int(ok.sum())} scored)", flush=True)

    ok = ~np.isnan(score)
    print("\n====================================================================")
    print("SSL generative grammar v2 (DGCNN AE, xyz+radius+syn, no labels/features)")
    print("  ONE grammar, BOTH error types: merge=too much structure, split=too little")

    def report(name, mask):
        m = ok & mask
        if (m & (kind == 0)).sum() < 5 or (m & (kind > 0)).sum() < 5:
            print(f"  {name:18s} (too few)"); return
        yy, ss = (kind[m] > 0).astype(int), score[m]
        auc = roc_auc_score(yy, ss)
        r2 = np.random.default_rng(1)
        null = [roc_auc_score(r2.permutation(yy), ss) for _ in range(200)]
        thr = np.quantile(ss, 0.90); fl = ss >= thr
        prec = (yy[fl] == 1).mean() if fl.sum() else float("nan")
        rec = (fl & (yy == 1)).sum() / max(1, (yy == 1).sum())
        print(f"  {name:18s} AUC={auc:.3f} null={np.mean(null):.2f}±{np.std(null):.2f} "
              f"prec@10%={prec:.2f} rec={rec:.2f}  (n={int(m.sum())}, err={int(yy.sum())})")

    report("merge vs clean", (kind == 0) | (kind == 1))
    report("split vs clean", (kind == 0) | (kind == 2))
    report("any error vs clean", np.ones(len(objs), bool))
    print("  ref: v1 PointNet-AE merge=0.774  hand synapse-cloud merge=0.88")
    print("====================================================================")


if __name__ == "__main__":
    main()
