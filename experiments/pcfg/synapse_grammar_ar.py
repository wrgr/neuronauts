#!/usr/bin/env python3
"""Autoregressive grammar over CONNECTED SYNAPSES -- a literal grammar of neuron structure.

We care primarily about connected synapses: a neuron is a connected sequence of synapses
strung along its cable. This learns P(next connected synapse | trajectory so far) with a
causal transformer -- a real grammar whose likelihood scores grammaticality. No labels, no
hand features, no synthesized merges. The skeleton is ONLY the scaffold that says which
synapses are connected and in what order (nearest-vertex attachment + a connectivity-order
traversal); the synapses are the tokens the grammar generates.

  * merge = the connectivity walk jumps across a seam into a second arbor -> improbable step
  * split = the sequence begins/ends where the grammar expects the cable to continue
  * per-object mean NLL (perplexity) = ungrammaticality

Trained on real (noisy) neurons; group-by-cell CV; both error types scored by the one grammar.

    python -m experiments.pcfg_synapse_partitions.synapse_grammar_ar --sidetable data/sidetable_7box.npz
"""
from __future__ import annotations

import argparse
import glob
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pcfg_synapse_partitions.synapse_correction import SideTable  # noqa: E402
from experiments.pcfg_synapse_partitions.skeleton_topology_merge import bridge_union  # noqa: E402

SCALE = 10_000.0     # nm; displacements between connected synapses are ~micron-scale
MAXSYN = 200


def load_skels(skel_dir, version):
    """rid -> (V[n,3], E[m,2], R[n])."""
    out = {}
    pat = re.compile(rf"v{version}_rid(\d+)_skv")
    for f in glob.glob(str(Path(skel_dir) / f"v{version}_rid*.npz")):
        m = pat.search(Path(f).name)
        if not m:
            continue
        d = np.load(f)
        V = d["vertices"].astype(np.float64)
        E = d["edges"].astype(np.int64) if "edges" in d else np.zeros((0, 2), np.int64)
        if len(V) < 8 or len(E) < 4:
            continue
        R = d["radius"].astype(np.float64) if "radius" in d else np.full(len(V), 300.0)
        out[int(m.group(1))] = (V, E, R)
    return out


def _adj(E, n):
    g = defaultdict(list)
    for a, b in E:
        if a != b:
            g[int(a)].append(int(b)); g[int(b)].append(int(a))
    return g


def _far(g, V, src):
    """Farthest vertex from src by cable distance (tree BFS/DFS)."""
    seen = {src: 0.0}; stack = [src]; best, bd = src, 0.0
    while stack:
        u = stack.pop()
        for w in g[u]:
            if w not in seen:
                seen[w] = seen[u] + float(np.linalg.norm(V[u] - V[w]))
                if seen[w] > bd:
                    bd, best = seen[w], w
                stack.append(w)
    return best


def connected_synapse_tree(V, E, syn_pts):
    """Build a branching synapse TREE and return its parent-relative edge displacements.

    The skeleton's BFS spanning tree from a tip defines connectivity (acyclic -> no
    self-loops). Synapses attach to nearest vertices; a DFS that carries the last synapse
    seen on the path gives each synapse a PARENT synapse. Branching is preserved: a branch
    vertex hands the same parent to each of its children. Returns the sequence of tree-edge
    displacements (child - parent), DFS-ordered; the root edge is 0. No sequential 'pop-back'
    jumps -- a branch return is a small parent-relative step, not a fake self-loop.
    """
    if len(syn_pts) < 6 or len(E) < 1:
        return None
    from collections import deque
    from scipy.spatial import cKDTree
    g = _adj(E, len(V))
    if not g:
        return None
    root_v = _far(g, V, _far(g, V, next(iter(g))))
    # BFS spanning tree from root -> parent-vertex map (guarantees a tree, no cycles)
    pv = {root_v: -1}; dq = deque([root_v]); children = defaultdict(list)
    while dq:
        u = dq.popleft()
        for w in g[u]:
            if w not in pv:
                pv[w] = u; children[u].append(w); dq.append(w)
    syn_vert = cKDTree(V).query(syn_pts)[1]
    syn_at = defaultdict(list)
    for si, vv in enumerate(syn_vert):
        syn_at[int(vv)].append(si)
    # DFS carrying the parent synapse along the path
    pos, par = [], []
    stack = [(root_v, -1)]
    while stack:
        v, psyn = stack.pop()
        cur = psyn
        for si in syn_at.get(v, []):
            pos.append(syn_pts[si]); par.append(cur); cur = len(pos) - 1
        for w in children.get(v, []):
            stack.append((w, cur))
    if len(pos) < 6:
        return None
    pos = np.asarray(pos, np.float64); par = np.asarray(par)
    k = len(pos)
    disp = np.zeros_like(pos); hp = par >= 0
    disp[hp] = pos[hp] - pos[par[hp]]              # tree-edge (parent-relative) displacement
    deg = np.zeros(k, np.int64)                    # number of children = BRANCH DEGREE
    for p in par[hp].tolist():
        deg[p] += 1
    depth = np.zeros(k, np.float64)                # tree depth (parents emitted before children)
    for i in range(k):
        if par[i] >= 0:
            depth[i] = depth[par[i]] + 1
    if k > MAXSYN:
        disp, deg, depth = disp[:MAXSYN], deg[:MAXSYN], depth[:MAXSYN]
    return (disp / SCALE).astype(np.float32), deg.astype(np.int64), depth.astype(np.float32)


DEG_CLASSES = 4   # degree 0 (tip/terminate), 1 (continue), 2, 3+ (branch)


def build_model(d=128, layers=4, heads=4):
    import torch
    import torch.nn as nn

    class AR(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(4, d)            # [parent-edge xyz, depth]
            self.pos = nn.Parameter(torch.zeros(1, MAXSYN, d))
            enc = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, dropout=0.1)
            self.tr = nn.TransformerEncoder(enc, layers)
            self.mu = nn.Linear(d, 3); self.logv = nn.Linear(d, 3)   # next-edge geometry
            self.deg = nn.Linear(d, DEG_CLASSES)                      # this node's branch degree

        def forward(self, x):                       # x:[B,T,4]
            T = x.size(1)
            h = self.embed(x) + self.pos[:, :T]
            mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
            h = self.tr(h, mask=mask)
            return self.mu(h), self.logv(h).clamp(-6, 6), self.deg(h)

    return AR()


def nll_steps(mu, logv, target):
    """Per-step diagonal-Gaussian NLL of the true next displacement."""
    import torch
    var = torch.exp(logv)
    return 0.5 * (((target - mu) ** 2 / var) + logv + math.log(2 * math.pi)).sum(-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sidetable", required=True)
    ap.add_argument("--skel-cache", default="data/skel_v1718")
    ap.add_argument("--version", type=int, default=1718)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from scipy.spatial import cKDTree
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    skels = load_skels(args.skel_cache, args.version)
    d = np.load(args.sidetable)
    tab = SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])
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
    splits = [b for b, s in by_later_v117.items() if len(s) >= 2 and b in skels and cnt[b] >= 20]

    gid = {}

    def G(x):
        return gid.setdefault(int(x), len(gid))

    # object -> connected-synapse sequence; kind: 0 clean, 1 merge, 2 split
    objs = []                                       # (kind, seq, group)

    def add(kind, V, E, syn, grp):
        t = connected_synapse_tree(V, E, syn)
        if t is not None and len(t[0]) >= 6:
            objs.append((kind, t[0], t[1], t[2], grp))    # kind, disp, degree, depth, group

    for b in clean:
        V, E, R = skels[b]; add(0, V, E, syn_by_later[b], G(b))
    for parts in merges:
        pp = [p for p in parts if p in skels]
        V, E, R = bridge_union([skels[p] for p in pp])
        syn = np.vstack([syn_by_later[p] for p in pp]).astype(np.float32)
        add(1, V, E, syn, G(pp[0]))
    for b in splits:                                # partition cell by nearest synapse -> fragments
        V, E, R = skels[b]
        cs = np.array(syn_pts_by_later[b], np.float32); cv = np.array(syn_v117_by_later[b])
        vsrc = cv[cKDTree(cs).query(V)[1]]
        for a in set(cv.tolist()):
            vm = np.nonzero(vsrc == a)[0]; sm = cv == a
            if len(vm) >= 8 and sm.sum() >= 8:
                remap = {int(v): i for i, v in enumerate(vm)}
                Ef = np.array([[remap[int(x)], remap[int(y)]] for x, y in E
                               if int(x) in remap and int(y) in remap], np.int64)
                if len(Ef) >= 4:
                    add(2, V[vm], Ef, cs[sm], G(b))

    kind = np.array([o[0] for o in objs]); groups = np.array([o[4] for o in objs])
    print(f"skeletons={len(skels)}  objects: clean={int((kind==0).sum())} "
          f"merge={int((kind==1).sum())} split={int((kind==2).sum())}", flush=True)
    if (kind == 0).sum() < 30 or (kind == 1).sum() < 8:
        print("not enough objects."); return

    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=max(2, min(args.folds, len(np.unique(groups)))))
    score = np.full(len(objs), np.nan)        # mean per-node NLL
    score_peak = np.full(len(objs), np.nan)   # 90th-pct per-node NLL (un-dilutes a local seam)

    def batch_tensors(idxs):
        items = [(i, objs[i][1], objs[i][2], objs[i][3]) for i in idxs if len(objs[i][1]) >= 4]
        if not items:
            return None
        T = max(len(d) for _, d, _, _ in items); B = len(items)
        Fin = np.zeros((B, T, 4), np.float32)        # [parent-edge xyz, depth]
        Yd = np.zeros((B, T, 3), np.float32); Md = np.zeros((B, T), np.float32)
        Yg = np.zeros((B, T), np.int64); Mg = np.zeros((B, T), np.float32); ids = []
        for r, (i, disp, deg, depth) in enumerate(items):
            L = len(disp)
            Fin[r, :L, :3] = disp; Fin[r, :L, 3] = np.clip(depth, 0, 60) / 60.0
            Yg[r, :L] = np.clip(deg, 0, DEG_CLASSES - 1); Mg[r, :L] = 1.0
            if L >= 2:
                Yd[r, :L - 1] = disp[1:]; Md[r, :L - 1] = 1.0   # predict next edge
            ids.append(i)
        return (torch.tensor(Fin), torch.tensor(Yd), torch.tensor(Md),
                torch.tensor(Yg), torch.tensor(Mg), ids)

    def per_obj_nll(mu, logv, deglog, Yd, Md, Yg, Mg):
        dnll = nll_steps(mu, logv, Yd)                                   # [B,T]
        gce = F.cross_entropy(deglog.reshape(-1, DEG_CLASSES), Yg.reshape(-1),
                              reduction="none").reshape(Yg.shape)         # [B,T]
        num = (dnll * Md).sum(1) + (gce * Mg).sum(1)
        return num / (Md.sum(1) + Mg.sum(1)).clamp(min=1), dnll, gce

    for fold, (tr, te) in enumerate(gkf.split(objs, kind, groups)):
        train_clean = [i for i in tr if objs[i][0] == 0]
        if len(train_clean) < 8:
            continue
        net = build_model(); opt = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-5)
        for ep in range(args.epochs):
            net.train(); order = rng.permutation(train_clean)
            for s in range(0, len(order), args.batch):
                bt = batch_tensors(order[s:s + args.batch])
                if bt is None:
                    continue
                Fin, Yd, Md, Yg, Mg, _ = bt
                mu, logv, deglog = net(Fin)
                dnll = nll_steps(mu, logv, Yd); gce = F.cross_entropy(
                    deglog.reshape(-1, DEG_CLASSES), Yg.reshape(-1), reduction="none").reshape(Yg.shape)
                loss = (dnll * Md).sum() / Md.sum().clamp(min=1) + \
                       (gce * Mg).sum() / Mg.sum().clamp(min=1)
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            for s in range(0, len(te), args.batch):
                bt = batch_tensors(te[s:s + args.batch])
                if bt is None:
                    continue
                Fin, Yd, Md, Yg, Mg, ids = bt
                mu, logv, deglog = net(Fin)
                dnll = nll_steps(mu, logv, Yd)
                gce = F.cross_entropy(deglog.reshape(-1, DEG_CLASSES), Yg.reshape(-1),
                                      reduction="none").reshape(Yg.shape)
                tot = dnll * Md + gce * Mg                      # per-node surprise
                for r, i in enumerate(ids):
                    vmask = Mg[r] > 0
                    nv = tot[r][vmask]
                    if nv.numel() == 0:
                        continue
                    score[i] = float(nv.mean())
                    score_peak[i] = float(torch.quantile(nv, 0.9))
        ok = ~np.isnan(score)
        if ok.sum() > 5 and len(np.unique((kind[ok] > 0))) == 2:
            print(f"  fold {fold+1}: mean-AUC={roc_auc_score((kind[ok]>0).astype(int), score[ok]):.3f} "
                  f"peak-AUC={roc_auc_score((kind[ok]>0).astype(int), score_peak[ok]):.3f} "
                  f"({int(ok.sum())})", flush=True)

    ok = ~np.isnan(score)
    print("\n====================================================================")
    print("AUTOREGRESSIVE connected-synapse grammar (causal transformer, NLL)")
    print("  ONE grammar; score = mean per-step NLL of the connected-synapse trajectory")

    def report(name, mask, sc):
        m = ok & mask
        if (m & (kind == 0)).sum() < 5 or (m & (kind > 0)).sum() < 5:
            print(f"  {name:22s} (too few)"); return
        yy, ss = (kind[m] > 0).astype(int), sc[m]
        auc = roc_auc_score(yy, ss)
        thr = np.quantile(ss, 0.90); fl = ss >= thr
        prec = (yy[fl] == 1).mean() if fl.sum() else float("nan")
        rec = (fl & (yy == 1)).sum() / max(1, (yy == 1).sum())
        print(f"  {name:22s} AUC={auc:.3f} prec@10%={prec:.2f} rec={rec:.2f}  "
              f"(n={int(m.sum())}, err={int(yy.sum())})")

    for label, sc in (("[mean NLL]", score), ("[peak NLL]", score_peak)):
        print(f"  {label}")
        report("  merge vs clean", (kind == 0) | (kind == 1), sc)
        report("  split vs clean", (kind == 0) | (kind == 2), sc)
        report("  any error vs clean", np.ones(len(objs), bool), sc)
    print("  ref: reconstruction-AE grammar merge=0.776 (the objective we're replacing)")
    print("====================================================================")


if __name__ == "__main__":
    main()
