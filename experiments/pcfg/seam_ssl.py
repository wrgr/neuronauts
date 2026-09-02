#!/usr/bin/env python3
"""Path 2: SELF-SUPERVISED pretraining for the seam GNN via synthetic splice detection.

Label-free. Take CLEAN proofread single neurons (v1718 cell skeletons) and synthetically SPLICE
two of them together at a touching point -> a known false-merge whose seam edge we know because we
made it. Train the per-edge seam GNN (seam_detector.build_model, raw inputs only: xyz, log radius,
raw pre/post synapse channels -- no hand features, see CLAUDE.md) to point at the splice. This
gives unlimited training data and teaches "what two-cables-joined looks like" while the model
LEARNS the cues itself. Then evaluate ZERO-SHOT on the REAL v117 merges and compare to the
supervised-from-scratch seam detector -- especially on the axon side.

    python -m experiments.pcfg.seam_ssl \
        --sidetable data/sidetable_big.npz --n-splice 3000 --epochs 8
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pcfg.synapse_correction import SideTable  # noqa: E402
from experiments.pcfg.close_loop_cut import (  # noqa: E402
    load_skels, disagreement_from_counts, do_nothing_err, root_and_subtrees,
)
from experiments.pcfg.seam_detector import build_model, build_objects, SCALE  # noqa: E402


def load_clean_cells(sidetable, skel_dir, min_syn):
    """Clean v1718 single neurons: (V,E,R, syn_vert[list], syn_side[list 0=pre,1=post])."""
    from scipy.spatial import cKDTree
    d = np.load(sidetable)
    tab = SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])
    pts_by, side_by = defaultdict(list), defaultdict(list)
    for i in np.nonzero(tab.root_later > 0)[0]:
        c = int(tab.root_later[i])
        pts_by[c].append(tab.pt[i]); side_by[c].append(int(tab.side[i]))
    skels = load_skels(skel_dir, 1718)
    cells = []
    for c, (V, E, R) in skels.items():
        if c not in pts_by or len(pts_by[c]) < min_syn:
            continue
        P = np.asarray(pts_by[c]); sd = np.asarray(side_by[c])
        sv = cKDTree(V).query(P)[1]
        cells.append((V, np.asarray(E), R, sv.astype(np.int64), sd.astype(np.int64)))
    return cells


def splice(cellA, cellB, rng, jitter_nm=150.0):
    """Join two clean cells at a random touching point -> synthetic merge object dict (or None)."""
    Va, Ea, Ra, sva, sda = cellA
    Vb, Eb, Rb, svb, sdb = cellB
    na = len(Va)
    a = int(rng.integers(na)); b = int(rng.integers(len(Vb)))
    shift = Va[a] - Vb[b] + rng.normal(0, jitter_nm, 3)
    Vb2 = Vb + shift
    V = np.concatenate([Va, Vb2], 0)
    R = np.concatenate([Ra, Rb], 0)
    E = np.concatenate([Ea, Eb + na, np.array([[a, b + na]])], 0)
    syn_vert = np.concatenate([sva, svb + na]).astype(np.int64)
    syn_side = np.concatenate([sda, sdb]).astype(np.int64)
    lab_index = np.concatenate([np.zeros(len(sva), np.int64), np.ones(len(svb), np.int64)])
    nlab = 2
    tot = np.bincount(lab_index, minlength=nlab).astype(np.int64)
    dn = do_nothing_err(tot)
    if dn == 0:
        return None
    rs = root_and_subtrees(V, E.tolist(), list(zip(syn_vert.tolist(), lab_index.tolist())), nlab)
    if rs is None:
        return None
    parent, order, sub = rs
    idx = {v: i for i, v in enumerate(order)}
    n = len(order)
    xyz = V[order]; rr = R[order]
    pre_sc = np.zeros(n); post_sc = np.zeros(n)
    for vtx, s in zip(syn_vert.tolist(), syn_side.tolist()):
        if vtx in idx:
            (pre_sc if s == 0 else post_sc)[idx[vtx]] += 1
    feat = np.concatenate([(xyz - xyz.mean(0)) / SCALE,
                           (np.log1p(rr) / 5.0)[:, None],
                           np.log1p(pre_sc)[:, None],
                           np.log1p(post_sc)[:, None]], axis=1).astype(np.float32)
    edges, edis = [], []
    for v in order:
        p = parent[v]
        if p < 0:
            continue
        s = sub[v]; A = int(s.sum()); B = int(tot.sum()) - A
        if min(A, B) < 2:
            continue
        edges.append((idx[p], idx[v])); edis.append(disagreement_from_counts(s, tot))
    if len(edges) < 2:
        return None
    edis_a = np.array(edis, np.float64)
    benefit = np.clip((dn - edis_a) / dn, -3.0, 1.0).astype(np.float32)
    return dict(feat=feat, edges=np.array(edges), edis=np.array(edis, np.int64), benefit=benefit, dn=int(dn))


def make_splices(cells, n, rng):
    objs = []
    tries = 0
    while len(objs) < n and tries < n * 5:
        tries += 1
        i, j = int(rng.integers(len(cells))), int(rng.integers(len(cells)))
        if i == j:
            continue
        o = splice(cells[i], cells[j], rng)
        if o is not None:
            objs.append(o)
    return objs


def pretrain(spliced, epochs, seed, cin):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = build_model(cin=cin); opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    for ep in range(epochs):
        net.train(); order = rng.permutation(len(spliced)); opt.zero_grad(); acc = 0
        tot_loss = 0.0
        for k, i in enumerate(order.tolist()):
            o = spliced[i]
            pred = net(torch.tensor(o["feat"]), torch.tensor(o["edges"], dtype=torch.long))
            loss = F.smooth_l1_loss(pred, torch.tensor(o["benefit"]))
            loss.backward(); tot_loss += float(loss.detach()); acc += 1
            if acc == 8 or k == len(order) - 1:
                opt.step(); opt.zero_grad(); acc = 0
        print(f"  pretrain epoch {ep+1}/{epochs}  loss={tot_loss/max(1,len(order)):.4f}", flush=True)
    return net


def evaluate_zeroshot(net, real_objs):
    """Apply the pretrained net to REAL merges: hit-rate, oracle-rank, abstention net%."""
    import torch
    base = sum(o["dn"] for o in real_objs)
    err_cut = np.zeros(len(real_objs), np.int64)
    pred_ben = np.zeros(len(real_objs))
    oracle = np.array([int(o["edis"].min()) for o in real_objs])
    hit = 0
    with torch.no_grad():
        for k, o in enumerate(real_objs):
            pred = net(torch.tensor(o["feat"]), torch.tensor(o["edges"], dtype=torch.long)).numpy()
            top = int(np.argsort(-pred)[0])
            err_cut[k] = int(o["edis"][top]); pred_ben[k] = float(pred[top])
            if o["edis"][top] == o["edis"].min():
                hit += 1
    print(f"\nzero-shot on {len(real_objs)} REAL merges  (do-nothing pair errors = {base:,})")
    print(f"  GNN picked an oracle-optimal edge on {hit/len(real_objs):.1%} of objects")
    print(f"  oracle ceiling net = {base-int(oracle.sum()):+,d} ({100*(base-oracle.sum())/max(1,base):.1f}%)")
    print(f"  {'tau':>6s}{'cuts':>7s}{'net_fixed':>12s}{'% base':>9s}")
    for tau in (-9.9, 0.0, 0.1, 0.2, 0.3, 0.5):
        do_cut = pred_ben > tau
        err = np.where(do_cut, err_cut, [o["dn"] for o in real_objs]).astype(np.int64)
        netf = base - int(err.sum())
        print(f"  {tau:>6.1f}{int(do_cut.sum()):>7d}{netf:>+12,d}{100*netf/max(1,base):>8.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidetable", default="data/sidetable_big.npz")
    ap.add_argument("--clean-skel", default="data/skel_v1718")
    ap.add_argument("--real-skel", default="data/skel_v117")
    ap.add_argument("--min-syn", type=int, default=8)
    ap.add_argument("--n-splice", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="data/seam_ssl_net.pt")
    args = ap.parse_args()
    import torch

    rng = np.random.default_rng(args.seed)
    cells = load_clean_cells(args.sidetable, args.clean_skel, args.min_syn)
    print(f"clean cells = {len(cells)}", flush=True)
    spliced = make_splices(cells, args.n_splice, rng)
    print(f"synthetic spliced merges = {len(spliced)}", flush=True)
    cin = int(spliced[0]["feat"].shape[1])
    net = pretrain(spliced, args.epochs, args.seed, cin)
    torch.save(net.state_dict(), args.save)
    print(f"saved pretrained net -> {args.save}", flush=True)

    real = build_objects(args.sidetable, args.real_skel, min_syn=args.min_syn, min_side=2)
    print(f"real merge objects = {len(real)}", flush=True)
    evaluate_zeroshot(net, real)


if __name__ == "__main__":
    main()
