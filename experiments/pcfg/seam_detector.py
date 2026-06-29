#!/usr/bin/env python3
"""LEARNED seam-edge detector -- close the merge-correction loop with a real (non-oracle) cut.

The oracle single skeleton-edge cut beats do-nothing by +79%; unsupervised heuristics miss the
seam (-14% to -66%). The gap is a learnable problem: predict WHICH skeleton edge is the
false-merge seam. This is a GraphSAGE GNN over each v117 over-merged skeleton; per object it
scores its tree edges and is trained to point at the oracle's best cut (softmax-over-edges CE).
At inference it cuts the top-scored edge; we re-close the loop and compare net vs do-nothing to
the oracle ceiling and the heuristics.

Inputs are raw: vertex xyz + log radius + local synapse count (the connected-synapse signal).
Grouped-by-cell CV. Targets/eval use v1718 only to define the seam and score cuts, never as
model input.

    python -m experiments.pcfg_synapse_partitions.seam_detector --sidetable data/sidetable_7box.npz
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pcfg_synapse_partitions.synapse_correction import SideTable, cell_components  # noqa: E402
from experiments.pcfg_synapse_partitions.close_loop_cut import (  # noqa: E402
    load_skels, disagreement_from_counts, do_nothing_err, root_and_subtrees,
)

SCALE = 50_000.0


def build_objects(sidetable, skel_dir, min_syn, min_side):
    from scipy.spatial import cKDTree
    d = np.load(sidetable)
    tab = SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])
    comp = cell_components(tab)
    valid = tab.root_later > 0
    pts_by, lat_by = defaultdict(list), defaultdict(list)
    for i in np.nonzero(valid)[0]:
        rv = int(tab.root_v117[i])
        pts_by[rv].append(tab.pt[i]); lat_by[rv].append(int(tab.root_later[i]))
    skels = load_skels(skel_dir, 117)
    objs = []
    for rv, (V, E, R) in skels.items():
        if rv not in pts_by:
            continue
        P = np.asarray(pts_by[rv]); lat = np.asarray(lat_by[rv])
        if len(P) < min_syn or len(set(lat.tolist())) < 2:
            continue
        labs, lab_index = np.unique(lat, return_inverse=True)
        nlab = len(labs); tot = np.bincount(lab_index, minlength=nlab).astype(np.int64)
        dn = do_nothing_err(tot)
        if dn == 0:
            continue
        syn_vert = cKDTree(V).query(P)[1]
        rs = root_and_subtrees(V, E, list(zip(syn_vert.tolist(), lab_index.tolist())), nlab)
        if rs is None:
            continue
        parent, order, sub = rs
        idx = {v: i for i, v in enumerate(order)}
        n = len(order)
        xyz = V[order]; rr = R[order]
        sc = np.zeros(n)
        for vtx in syn_vert.tolist():
            if vtx in idx:
                sc[idx[vtx]] += 1
        feat = np.concatenate([(xyz - xyz.mean(0)) / SCALE,
                               (np.log1p(rr) / 5.0)[:, None],
                               np.log1p(sc)[:, None]], axis=1).astype(np.float32)
        edges, edis, rad = [], [], []
        for v in order:
            p = parent[v]
            if p < 0:
                continue
            s = sub[v]; A = int(s.sum()); B = int(tot.sum()) - A
            if min(A, B) < min_side:
                continue
            edges.append((idx[p], idx[v])); edis.append(disagreement_from_counts(s, tot))
            rad.append(min(R[v], R[p]))
        if len(edges) < 2:
            continue
        edis_a = np.array(edis, np.float64)
        benefit = np.clip((dn - edis_a) / dn, -3.0, 1.0).astype(np.float32)   # frac error reduction
        objs.append(dict(rv=int(rv), order=list(order), parent=dict(parent),
                         feat=feat, edges=np.array(edges), edis=np.array(edis, np.int64),
                         rad=np.array(rad), benefit=benefit, dn=int(dn),
                         group=comp.get(rv, -1)))
    return objs


def build_model(cin=5, d=96):
    import torch
    import torch.nn as nn

    class SAGE(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.lin = nn.Linear(2 * cin, cout)

        def forward(self, h, src, dst):
            n = h.size(0)
            agg = torch.zeros(n, h.size(1)); agg.index_add_(0, dst, h[src])
            deg = torch.zeros(n, 1); deg.index_add_(0, dst, torch.ones(len(dst), 1))
            agg = agg / deg.clamp(min=1)
            return torch.relu(self.lin(torch.cat([h, agg], -1)))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = SAGE(cin, d); self.l2 = SAGE(d, d); self.l3 = SAGE(d, d)
            self.edge = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1))

        def forward(self, feat, edges):
            src = torch.cat([edges[:, 0], edges[:, 1]]); dst = torch.cat([edges[:, 1], edges[:, 0]])
            h = self.l1(feat, src, dst); h = self.l2(h, src, dst); h = self.l3(h, src, dst)
            hp, hv = h[edges[:, 0]], h[edges[:, 1]]
            return self.edge(torch.cat([hp, hv, (hp - hv).abs()], -1)).squeeze(-1)

    return Net()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sidetable", required=True)
    ap.add_argument("--skel-cache", default="data/skel_v117")
    ap.add_argument("--min-syn", type=int, default=8)
    ap.add_argument("--min-side", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    torch.manual_seed(args.seed)

    objs = build_objects(args.sidetable, args.skel_cache, args.min_syn, args.min_side)
    groups = np.array([o["group"] for o in objs])
    print(f"merge objects with skeleton = {len(objs)}  cells = {len(np.unique(groups))}", flush=True)
    if len(objs) < 20:
        print("too few objects; let the v117 fetch run."); return

    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=max(2, min(args.folds, len(np.unique(groups)))))
    err_cut = np.full(len(objs), np.nan)     # actual error if we cut the GNN's top edge
    pred_ben = np.full(len(objs), np.nan)    # predicted benefit of that top edge (for abstention)
    topk_dis = [None] * len(objs)            # disagreements of the top-10 predicted edges
    oracle_rank = np.full(len(objs), 9999)   # rank of the oracle-best edge in the prediction
    rng = np.random.default_rng(args.seed)

    def edges_t(o):
        return torch.tensor(o["edges"], dtype=torch.long)

    for fold, (tr, te) in enumerate(gkf.split(objs, np.zeros(len(objs)), groups)):
        net = build_model(); opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
        for ep in range(args.epochs):
            net.train(); order = rng.permutation(tr); opt.zero_grad(); acc = 0
            for k, i in enumerate(order):
                o = objs[i]
                pred = net(torch.tensor(o["feat"]), edges_t(o))         # per-edge predicted benefit
                loss = F.smooth_l1_loss(pred, torch.tensor(o["benefit"]))
                loss.backward(); acc += 1
                if acc == 8 or k == len(order) - 1:
                    opt.step(); opt.zero_grad(); acc = 0
        net.eval()
        with torch.no_grad():
            for i in te:
                o = objs[i]
                pred = net(torch.tensor(o["feat"]), edges_t(o)).numpy()
                order = np.argsort(-pred)
                b = int(order[0])
                pred_ben[i] = float(pred[b]); err_cut[i] = int(o["edis"][b])
                kk = min(10, len(order))
                topk_dis[i] = o["edis"][order[:kk]].tolist()
                oracle_rank[i] = int(np.where(order == int(o["edis"].argmin()))[0][0])

    dn = np.array([o["dn"] for o in objs]); base = int(dn.sum())
    oracle = np.array([int(o["edis"].min()) for o in objs])
    minrad = np.array([int(o["edis"][int(o["rad"].argmin())]) for o in objs])
    hit = np.mean(err_cut == oracle)
    print(f"\ndo-nothing pair errors = {base:,}   oracle net = {base-int(oracle.sum()):+,d} "
          f"({100*(base-oracle.sum())/max(1,base):.1f}%)   min_radius net = {base-int(minrad.sum()):+,d}")
    print(f"GNN picked an oracle-optimal edge on {hit:.1%} of objects\n")
    print("  LEARNED seam GNN with abstention (cut only if predicted benefit > tau):")
    print(f"  {'tau':>6s}{'cuts':>7s}{'pair errors':>13s}{'net_fixed':>12s}{'% base':>9s}")
    for tau in (-9.9, 0.0, 0.1, 0.2, 0.3, 0.5):
        do_cut = pred_ben > tau
        err = np.where(do_cut, err_cut, dn).astype(np.int64)
        netf = base - int(err.sum())
        print(f"  {tau:>6.1f}{int(do_cut.sum()):>7d}{int(err.sum()):>13,d}{netf:>+12,d}"
              f"{100*netf/max(1,base):>8.1f}%")
    print("\n  abstention bounds downside at do-nothing (net>=0); net>0 == deployable corrector.")

    print("\n  HUMAN-ASSIST: model proposes top-k candidate cut edges, a proofreader picks the best:")
    print(f"  {'k':>4s}{'oracle-in-topk':>16s}{'best-of-k net':>15s}{'%base':>8s}"
          f"{'+abstain net':>14s}{'%base':>8s}")
    for k in (1, 3, 5, 10):
        bok = np.array([min(td[:k]) for td in topk_dis])      # best candidate among top-k
        recall = float(np.mean(oracle_rank < k))
        net_raw = base - int(bok.sum())
        net_ab = base - int(np.minimum(bok, dn).sum())        # human won't apply a harmful cut
        print(f"  {k:>4d}{recall:>15.0%}{net_raw:>+15,d}{100*net_raw/max(1,base):>7.1f}%"
              f"{net_ab:>+14,d}{100*net_ab/max(1,base):>7.1f}%")
    print("  splits proposed = one per merge object; merges (joins) proposed = 0 (cut-only loop)")


if __name__ == "__main__":
    main()
