#!/usr/bin/env python3
"""Self-supervised generative grammar of neuron structure -- no hand features, no labels.

The agreed framing: learn what a neuron IS from real (noisy) neurons; errors fall out as low
grammaticality. This is a denoising point autoencoder over each object's RAW elements
(skeleton vertices + its synapses, xyz only): PointNet encoder -> latent bottleneck ->
decoder, trained with Chamfer reconstruction on the noisy corpus. The bottleneck forces the
latent onto the single-neuron manifold, so:

  * reconstruction error  = grammaticality score (a 2-arbor merge is off-manifold -> high error)
  * the reconstruction     = the proposed correction (pulled toward one-neuron structure)

No labels, no synthesized merge negatives (merges appear ONLY in eval, reconstructed from real
constituent cells), no descriptors. "Noisy data" is fine and is the point: training on the raw
corpus (clean-vs-noisy grammar KL was ~0) buys scale. Group-by-cell CV; do-nothing-relevant
precision@k. Skeletons from --skel-cache (grown by the background corpus fetch); synapses from
the cached SideTable.

    python -m experiments.pcfg_synapse_partitions.skel_ssl_grammar --sidetable data/sidetable_7box.npz
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

from experiments.pcfg_synapse_partitions.synapse_correction import SideTable  # noqa: E402

SCALE = 50_000.0
N_PTS = 256


def load_skels(skel_dir, version):
    out = {}
    pat = re.compile(rf"v{version}_rid(\d+)_skv")
    for f in glob.glob(str(Path(skel_dir) / f"v{version}_rid*.npz")):
        m = pat.search(Path(f).name)
        if not m:
            continue
        V = np.load(f)["vertices"].astype(np.float32)
        if len(V) >= 8:
            out[int(m.group(1))] = V
    return out


def obj_points(skel_parts, syn_parts, rng, n=N_PTS, augment=False):
    """Raw point set = skeleton vertices + synapse positions (xyz only), fixed size."""
    chunks = [p for p in skel_parts if len(p)] + [p for p in syn_parts if len(p)]
    P = np.vstack(chunks).astype(np.float32)
    idx = rng.choice(len(P), n, replace=len(P) < n)
    P = P[idx]
    P = P - P.mean(0)
    if augment:
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3))); P = P @ Q
        P = P * rng.uniform(0.85, 1.15)
    return (P / SCALE).astype(np.float32)


def build_model():
    import torch.nn as nn

    class AE(nn.Module):
        def __init__(self, k=N_PTS, lat=128):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(),
                                     nn.Linear(128, 256), nn.ReLU())
            self.to_lat = nn.Linear(256, lat)
            self.dec = nn.Sequential(nn.Linear(lat, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(),
                                     nn.Linear(512, k * 3))
            self.k = k

        def forward(self, x):                  # x:[B,N,3]
            h = self.enc(x).max(1).values      # global max-pool (PointNet)
            z = self.to_lat(h)
            return self.dec(z).view(x.size(0), self.k, 3)

    return AE()


def chamfer(a, b):
    import torch
    d = torch.cdist(a, b)
    return d.min(2).values.mean(1) + d.min(1).values.mean(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sidetable", required=True)
    ap.add_argument("--skel-cache", default="data/skel_v1718")
    ap.add_argument("--version", type=int, default=1718)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    skels = load_skels(args.skel_cache, args.version)
    d = np.load(args.sidetable)
    tab = SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])
    valid = tab.root_later > 0
    syn_by_later = defaultdict(list)
    by_v117_later, by_later_v117, cnt = defaultdict(set), defaultdict(set), defaultdict(int)
    for i in np.nonzero(valid)[0]:
        a, b = int(tab.root_v117[i]), int(tab.root_later[i])
        syn_by_later[b].append(tab.pt[i]); by_v117_later[a].add(b); by_later_v117[b].add(a); cnt[b] += 1
    syn_by_later = {b: np.array(v, np.float32) for b, v in syn_by_later.items()}

    clean = [b for b, s in by_later_v117.items() if len(s) == 1 and cnt[b] >= 15 and b in skels]
    merges = [sorted(s) for a, s in by_v117_later.items()
              if len(s) >= 2 and sum(p in skels for p in s) >= 2]
    print(f"cached skeletons={len(skels)}  clean cells={len(clean)}  real merges={len(merges)}", flush=True)
    if len(clean) < 30 or len(merges) < 8:
        print("not enough cached objects yet; let the corpus fetch run."); return

    cg = {b: i for i, b in enumerate(clean)}
    objs = [(0, [skels[b]], [syn_by_later.get(b, np.zeros((0, 3), np.float32))], cg[b]) for b in clean]
    for parts in merges:
        pp = [p for p in parts if p in skels]
        sk = [skels[p] for p in pp]
        sy = [syn_by_later.get(p, np.zeros((0, 3), np.float32)) for p in pp]
        objs.append((1, sk, sy, cg.get(pp[0], -1 - len(objs))))

    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    y = np.array([o[0] for o in objs]); groups = np.array([o[3] for o in objs])
    gkf = GroupKFold(n_splits=max(2, min(args.folds, len(np.unique(groups)))))
    score = np.full(len(objs), np.nan)

    for fold, (tr, te) in enumerate(gkf.split(objs, y, groups)):
        train_clean = [i for i in tr if objs[i][0] == 0]      # train on real neurons only (noisy ok)
        if len(train_clean) < 8:
            continue
        net = build_model(); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        for ep in range(args.epochs):
            net.train(); order = rng.permutation(train_clean)
            for s in range(0, len(order), args.batch):
                ids = order[s:s + args.batch]
                xb = torch.tensor(np.stack([obj_points(objs[i][1], objs[i][2], rng, augment=True) for i in ids]))
                opt.zero_grad(); rec = net(xb); loss = chamfer(rec, xb).mean()
                loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            for i in te:
                errs = []
                for _ in range(8):
                    x = torch.tensor(obj_points(objs[i][1], objs[i][2], rng)[None])
                    errs.append(chamfer(net(x), x).item())
                score[i] = float(np.mean(errs))
        ok = ~np.isnan(score)
        if ok.sum() > 5 and len(np.unique(y[ok])) == 2:
            print(f"  fold {fold+1}: AUC={roc_auc_score(y[ok], score[ok]):.3f} ({int(ok.sum())} scored)", flush=True)

    ok = ~np.isnan(score)
    auc = roc_auc_score(y[ok], score[ok])
    r2 = np.random.default_rng(1)
    null = [roc_auc_score(r2.permutation(y[ok]), score[ok]) for _ in range(200)]
    thr = np.quantile(score[ok], 0.90); fl = score[ok] >= thr
    prec = (y[ok][fl] == 1).mean() if fl.sum() else float("nan")
    rec = (fl & (y[ok] == 1)).sum() / max(1, (y[ok] == 1).sum())
    print("\n====================================================================")
    print("SELF-SUPERVISED generative grammar (denoising point AE, raw xyz, no labels)")
    print(f"  objects={int(ok.sum())}  merges={int(y[ok].sum())}  base={y[ok].mean():.1%}")
    print(f"  AUC(merge | reconstruction error) = {auc:.3f}  null={np.mean(null):.3f}±{np.std(null):.3f}")
    print(f"  prec@top10%={prec:.2f}  recall={rec:.2f}")
    print(f"  ref: hand synapse-cloud=0.88  hand skeleton-topology=0.82(~size)")
    print("  (reconstruction error = grammaticality; the reconstruction is the proposed fix)")
    print("====================================================================")


if __name__ == "__main__":
    main()
