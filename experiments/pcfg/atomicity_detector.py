#!/usr/bin/env python3
"""Phase 3: supervised whole-object detector / atomicity STOP test.

Used two ways by the recursive corrector: (a) flag over-merged objects before the loop, and
(b) the recursion STOP test -- "is this child now ONE clean neuron?". Exposes
`AtomicityDetector.fit(...)` / `.p_merge(pts)` so the loop can score arbitrary sub-objects.

Two backends, compared in __main__:
  * rf  -- supervised RandomForest on the 10 hand global-shape features (global_shape_merge);
           this IS the 0.88 baseline, the safe deployable default.
  * gnn -- a SUPERVISED point classifier (DGCNN over the synapse cloud) trained on the real
           merge labels. Every learned model this session was UNSUPERVISED (reconstruction/NLL)
           and lost to the hand RF; this fills that gap -- can supervised learning beat hand
           features? If not, ship the RF (reuse wins).

Inputs are raw synapse positions (no skeleton needed -> runs on all v117 objects).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pcfg_synapse_partitions.synapse_correction import SideTable, cell_components  # noqa: E402
from experiments.pcfg_synapse_partitions.global_shape_merge import global_features  # noqa: E402

SCALE = 50_000.0
N_PTS = 256


def build_objects(sidetable, min_syn=8):
    d = np.load(sidetable)
    tab = SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])
    comp = cell_components(tab)
    valid = tab.root_later > 0
    pts_by, lat_by = defaultdict(list), defaultdict(list)
    for i in np.nonzero(valid)[0]:
        rv = int(tab.root_v117[i])
        pts_by[rv].append(tab.pt[i]); lat_by[rv].append(int(tab.root_later[i]))
    objs = []
    for rv, pl in pts_by.items():
        if len(pl) < min_syn:
            continue
        P = np.asarray(pl); lat = np.asarray(lat_by[rv])
        objs.append(dict(rv=rv, pts=P, y=int(len(set(lat.tolist())) >= 2),
                         group=comp.get(rv, -1)))
    return objs


# ---------------------------------------------------------------------------
# RF backend (the deployable 0.88 stop test)
# ---------------------------------------------------------------------------
class AtomicityDetector:
    def __init__(self, seed=0):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced",
                                             random_state=seed, n_jobs=-1)

    def fit(self, objs):
        X = np.array([global_features(o["pts"]) for o in objs])
        y = np.array([o["y"] for o in objs])
        self.model.fit(X, y)
        return self

    def p_merge(self, pts):
        return float(self.model.predict_proba(global_features(np.asarray(pts))[None])[0, 1])


# ---------------------------------------------------------------------------
# Supervised point-GNN backend
# ---------------------------------------------------------------------------
def _cloud(P, rng, n=N_PTS, augment=False):
    idx = rng.choice(len(P), n, replace=len(P) < n)
    X = P[idx].astype(np.float32)
    X = X - X.mean(0)
    if augment:
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3))); X = X @ Q
        X = X * rng.uniform(0.85, 1.15)
    return (X / SCALE).astype(np.float32)


def build_gnn():
    import torch
    import torch.nn as nn

    def knn(x, k):
        return torch.cdist(x, x).topk(min(k, x.size(1)), dim=-1, largest=False).indices

    class EdgeConv(nn.Module):
        def __init__(self, ci, co, k=16):
            super().__init__(); self.k = k
            self.mlp = nn.Sequential(nn.Linear(2 * ci, co), nn.ReLU(), nn.Linear(co, co), nn.ReLU())

        def forward(self, x):
            B, N, C = x.shape; idx = knn(x, self.k)
            nb = torch.gather(x.unsqueeze(2).expand(B, N, idx.size(-1), C), 1,
                              idx.unsqueeze(-1).expand(B, N, idx.size(-1), C))
            return self.mlp(torch.cat([x.unsqueeze(2).expand(B, N, idx.size(-1), C),
                                       nb - x.unsqueeze(2)], -1)).max(2).values

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1 = EdgeConv(3, 64); self.e2 = EdgeConv(64, 64); self.e3 = EdgeConv(64, 128)
            self.head = nn.Sequential(nn.Linear(2 * 256, 128), nn.ReLU(), nn.Dropout(0.3),
                                      nn.Linear(128, 1))

        def forward(self, x):
            h1 = self.e1(x); h2 = self.e2(h1); h3 = self.e3(h2)
            h = torch.cat([h1, h2, h3], -1)
            return self.head(torch.cat([h.max(1).values, h.mean(1)], -1)).squeeze(-1)

    return Net()


def eval_gnn(objs, groups, folds, epochs, seed):
    import torch
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    y = np.array([o["y"] for o in objs])
    oof = np.full(len(objs), np.nan)
    gkf = GroupKFold(n_splits=max(2, min(folds, len(np.unique(groups)))))
    for tr, te in gkf.split(objs, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        net = build_gnn(); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())]))
        for ep in range(epochs):
            net.train(); order = rng.permutation(tr)
            for s in range(0, len(order), 16):
                ids = order[s:s + 16]
                xb = torch.tensor(np.stack([_cloud(objs[i]["pts"], rng, augment=True) for i in ids]))
                yb = torch.tensor(y[ids].astype(np.float32))
                opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            for i in te:
                ps = [torch.sigmoid(net(torch.tensor(_cloud(objs[i]["pts"], rng)[None]))).item()
                      for _ in range(6)]
                oof[i] = float(np.mean(ps))
    ok = ~np.isnan(oof)
    auc = roc_auc_score(y[ok], oof[ok])
    thr = np.quantile(oof[ok], 0.98); fl = oof[ok] >= thr
    prec = (y[ok][fl] == 1).mean() if fl.sum() else float("nan")
    rec = (fl & (y[ok] == 1)).sum() / max(1, (y[ok] == 1).sum())
    return auc, prec, rec


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidetable", default="data/sidetable_7box.npz")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-gnn", action="store_true")
    args = ap.parse_args()

    objs = build_objects(args.sidetable)
    groups = np.array([o["group"] for o in objs]); y = np.array([o["y"] for o in objs])
    print(f"objects={len(objs)}  merges={int(y.sum())} ({y.mean():.1%})  cells={len(np.unique(groups))}")

    # RF (hand features) grouped-CV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    X = np.array([global_features(o["pts"]) for o in objs])
    gkf = GroupKFold(n_splits=max(2, min(args.folds, len(np.unique(groups)))))
    oof = np.full(len(objs), np.nan)
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced",
                                   random_state=args.seed, n_jobs=-1).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(oof)
    rf_auc = roc_auc_score(y[ok], oof[ok])
    thr = np.quantile(oof[ok], 0.98); fl = oof[ok] >= thr
    rf_prec = (y[ok][fl] == 1).mean(); rf_rec = (fl & (y[ok] == 1)).sum() / max(1, (y[ok] == 1).sum())
    print(f"\n  RF (hand 10 features)   AUC={rf_auc:.3f}  prec@2%={rf_prec:.2f}  rec={rf_rec:.2f}")

    if not args.skip_gnn:
        g_auc, g_prec, g_rec = eval_gnn(objs, groups, args.folds, args.epochs, args.seed)
        print(f"  GNN (supervised DGCNN)  AUC={g_auc:.3f}  prec@2%={g_prec:.2f}  rec={g_rec:.2f}")
        print(f"\n  {'GNN beats RF' if g_auc > rf_auc else 'RF wins -> ship RF as stop test'}"
              f"  (hand baseline ~0.875 / 0.41)")


if __name__ == "__main__":
    main()
