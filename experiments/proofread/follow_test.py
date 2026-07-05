"""Can we *follow* a process by inference (trajectory + geometry), not appearance?

The seam test killed the local-appearance cue (type-confounded).  This tests the
other thing humans use: **trajectory momentum + logical reconnection**.  Fully
offline on cached skeletons — no EM, which is the point (EM appearance was the trap).

Task ("reconnect the cut"): take a real interior point ``v`` on a neuron, note its
**incoming direction** from neighbour ``u``, then **open a realistic gap** by
deleting the neuron's own cable within ``gap_nm`` of ``v``.  Now ``v`` is a cut
endpoint.  Gather every vertex from *every* loaded neuron in the annulus
``[gap_nm, search_radius]`` around ``v`` — this is the real pool of competing
processes.  The **true** continuation is the far side of the gap (same neuron,
graph-reachable from the deleted branch); everything else (other neurons, other
branches) is a **distractor**.  Rank the pool and ask: did we reconnect to the true
continuation?

Scorers:
* **nearest** — pure proximity (the naive baseline).
* **align** — direction only: does the candidate continue the incoming trajectory?
* **learned** — logistic over [align, distance, |Δcaliber|, caliber] trained
  leave-one-neuron-out; the "learn to follow" model.

The interesting number is top-1 on the **hard** subset where nearest picks a
distractor — can trajectory inference recover what proximity gets wrong?

Honest caveat: 433 cached neurons is far sparser than real neuropil, so distractor
density (hence difficulty) is a *lower bound*; report the competing-candidate count.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass

import numpy as np


@dataclass
class Neuron:
    v: np.ndarray        # (n,3) nm
    e: np.ndarray        # (m,2)
    r: np.ndarray        # (n,) radius nm
    adj: list


def load_neurons(pattern="cache/skel_current/*.npz", limit=None):
    out = []
    for f in sorted(glob.glob(pattern))[:limit]:
        try:
            d = np.load(f)
            v = d["vertices"].astype(float); e = np.asarray(d["edges"], int).reshape(-1, 2)
            r = d["radius"].astype(float) if "radius" in d else np.full(len(v), 150.0)
        except Exception:  # noqa: BLE001
            continue
        if len(v) < 150:
            continue
        r = np.where(np.isfinite(r), r, 150.0)   # radius data is patchy -> default
        adj = [[] for _ in range(len(v))]
        for a, b in e:
            adj[a].append(b); adj[b].append(a)
        out.append(Neuron(v, e, r, adj))
    return out


def neuron_tangents(N: Neuron) -> np.ndarray:
    """Unit local cable direction per vertex (sign-arbitrary); robust to branches."""
    T = np.zeros_like(N.v)
    for k in range(len(N.v)):
        nb = N.adj[k]
        if len(nb) == 1:
            t = N.v[nb[0]] - N.v[k]
        elif len(nb) == 2:
            t = N.v[nb[1]] - N.v[nb[0]]
        else:  # branch: the two most-opposed neighbours span the through-cable
            dirs = [(N.v[b] - N.v[k]) / (np.linalg.norm(N.v[b] - N.v[k]) + 1e-9) for b in nb]
            bi, bd = (0, 1), 2.0
            for i in range(len(dirs)):
                for j in range(i + 1, len(dirs)):
                    c = float(dirs[i] @ dirs[j])
                    if c < bd:
                        bd = c; bi = (i, j)
            t = N.v[nb[bi[1]]] - N.v[nb[bi[0]]]
        nrm = np.linalg.norm(t)
        T[k] = t / nrm if nrm > 1e-9 else 0.0
    return T


def _line_gap(v, d, c, t):
    """Closest-approach distance (nm) between the endpoint ray (v,d) and the
    candidate's cable line (c,t): small when they are two halves of one cable."""
    w0 = v - c
    b = float(d @ t); dd = float(d @ w0); e = float(t @ w0)
    denom = 1.0 - b * b
    if denom < 1e-6:                       # near-parallel: use perpendicular offset
        perp = w0 - (w0 @ d) * d
        return float(np.linalg.norm(perp))
    sc = (b * e - dd) / denom; tc = (e - b * dd) / denom
    closest = w0 + sc * d - tc * t
    return float(np.linalg.norm(closest))


def _component_from(nb, adj, removed, cap=4000):
    """Vertices reachable from ``nb`` without entering ``removed`` (BFS, capped)."""
    from collections import deque
    seen = {nb}; q = deque([nb])
    while q and len(seen) < cap:
        x = q.popleft()
        for y in adj[x]:
            if y not in seen and y not in removed:
                seen.add(y); q.append(y)
    return seen


def build_instances(neurons, *, gap_nm=1500.0, search_radius=3500.0,
                    per_neuron=6, seed=0):
    """Yield reconnection instances with a global candidate pool per endpoint."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    # global vertex table (all neurons) for realistic distractors
    allv = np.vstack([n.v for n in neurons])
    allr = np.concatenate([n.r for n in neurons])
    alltan = np.vstack([neuron_tangents(n) for n in neurons])
    owner = np.concatenate([np.full(len(n.v), i) for i, n in enumerate(neurons)])
    local_idx = np.concatenate([np.arange(len(n.v)) for n in neurons])
    tree = cKDTree(allv)
    offset = np.cumsum([0] + [len(n.v) for n in neurons])

    inst = []
    for ni, N in enumerate(neurons):
        deg2 = [k for k in range(len(N.v)) if len(N.adj[k]) == 2]
        if not deg2:
            continue
        picks = rng.choice(deg2, min(per_neuron, len(deg2)), replace=False)
        for v_i in picks:
            u, w = N.adj[v_i]                       # incoming u, outgoing w
            d = N.v[v_i] - N.v[u]
            nd = np.linalg.norm(d)
            if nd < 1e-6:
                continue
            d = d / nd
            # the true continuation is the w-subtree: split it from the u-side by
            # removing only v (skeletons are trees), then the annulus gate below
            # opens the realistic gap (candidates must be >= gap_nm from v).
            true_comp = _component_from(w, N.adj, {v_i})    # far side, same neuron
            # global candidate pool in the annulus around v
            gi = tree.query_ball_point(N.v[v_i], search_radius)
            gi = [g for g in gi if np.linalg.norm(allv[g] - N.v[v_i]) >= gap_nm]
            if not gi:
                continue
            cand = np.asarray(gi)
            c_pos = allv[cand]; c_rad = allr[cand]; c_tan = alltan[cand]
            c_owner = owner[cand]; c_local = local_idx[cand]
            # true = same neuron AND on the far-side component
            is_true = np.array([(c_owner[j] == ni) and (int(c_local[j]) in true_comp)
                                for j in range(len(cand))])
            if not is_true.any():
                continue
            inst.append({
                "v": N.v[v_i], "d": d, "r_v": float(N.r[v_i]),
                "c_pos": c_pos, "c_rad": c_rad, "c_tan": c_tan, "is_true": is_true,
                "neuron": ni, "c_owner": c_owner, "c_local": c_local,
            })
    return inst


# feature layout: 0=align 1=dist 2=dcal 3=caliber | 4=recip 5=tan_agree 6=ray_gap
# Ablation (confusable top-1): base 0.72; +recip 0.89 (the driver); +tan_agree 0.79;
# +ray_gap 0.75 (near-useless — a tight parallel fascicle has a *small* line-gap, so
# it favours the distractor).  So the consequence signal is the single reciprocal-
# trajectory feature: does the far end's cable point back through the gap at the cut.
BASE_COLS = [0, 1, 2, 3]                 # trajectory + caliber (the prior model)
CONSEQ_COLS = [0, 1, 2, 3, 4, 5]         # + reciprocal + tangent-agreement (drop ray_gap)


def _features(inst):
    v, d, r_v = inst["v"], inst["d"], inst["r_v"]
    rel = inst["c_pos"] - v
    dist = np.linalg.norm(rel, axis=1)
    u = rel / (dist[:, None] + 1e-9)
    align = rel @ d / (dist + 1e-9)                    # cos angle to trajectory
    dcal = np.abs(inst["c_rad"] - r_v) / (r_v + 1e-6)
    tan = inst["c_tan"]
    # consequence-of-continuity (bidirectional): does the candidate's cable point
    # back at the endpoint, run parallel to it, and lie on the same extrapolated line?
    recip = np.abs((tan * (-u)).sum(axis=1))           # candidate cable aims at v
    tan_agree = np.abs(tan @ d)                        # cables run in the same line
    ray_gap = np.array([_line_gap(v, d, inst["c_pos"][j], tan[j])
                        for j in range(len(dist))]) / 1000.0
    F = np.column_stack([align, dist / 1000.0, dcal, np.full(len(dist), r_v / 1000.0),
                         recip, tan_agree, ray_gap])
    return F, dist, align


def evaluate(neurons, *, gap_nm=1500.0, search_radius=3500.0, per_neuron=6, seed=0,
             verbose=True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    inst = build_instances(neurons, gap_nm=gap_nm, search_radius=search_radius,
                           per_neuron=per_neuron, seed=seed)
    if not inst:
        return {"error": "no instances"}

    # assemble per-candidate features + labels + instance/neuron ids for LOO training
    X, y, iid, nid = [], [], [], []
    feats_per_inst = []
    for k, it in enumerate(inst):
        F, dist, align = _features(it)
        feats_per_inst.append((F, dist, align, it["is_true"]))
        X.append(F); y.append(it["is_true"].astype(int))
        iid.append(np.full(len(F), k)); nid.append(np.full(len(F), it["neuron"]))
    X = np.vstack(X); y = np.concatenate(y)
    iid = np.concatenate(iid); nid = np.concatenate(nid)

    # leave-one-neuron-out learned scores for two nested feature sets
    def loo_scores(cols):
        s = np.full(len(y), np.nan)
        neurons_u = np.unique(nid)
        if len(neurons_u) < 3:
            return s
        Xc = X[:, cols]
        for tr_n in neurons_u:
            tr = nid != tr_n; te = nid == tr_n
            if len(np.unique(y[tr])) < 2:
                continue
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, class_weight="balanced"))
            m.fit(Xc[tr], y[tr])
            s[te] = m.predict_proba(Xc[te])[:, 1]
        return s

    learned = loo_scores(BASE_COLS)          # trajectory + caliber (prior model)
    learned_conseq = loo_scores(CONSEQ_COLS)  # + bidirectional-consistency inference

    # a "confusable" instance has a DISTRACTOR nearly as aligned as the best true
    # candidate (a parallel process) -- the genuinely hard, fascicle-like case
    def is_confusable(F, is_true):
        al = F[:, 0]
        if not (~is_true).any() or not is_true.any():
            return False
        return al[~is_true].max() >= al[is_true].max() - 0.1

    # per-instance top-1 for each scorer
    def top1(scores_per_inst):
        hits, hard_hits, hard_n, mrr = 0, 0, 0, 0.0
        conf_hits, conf_n = 0, 0
        for (F, dist, align, is_true), sc in zip(feats_per_inst, scores_per_inst):
            order = np.argsort(-sc)
            hit = int(is_true[order[0]])
            hits += hit
            ranks = np.where(is_true[order])[0]
            if len(ranks):
                mrr += 1.0 / (ranks[0] + 1)
            nn = np.argmin(dist)
            if not is_true[nn]:
                hard_n += 1
                hard_hits += hit
            if is_confusable(F, is_true):
                conf_n += 1
                conf_hits += hit
        n = len(feats_per_inst)
        return {"top1": hits / n, "mrr": mrr / n,
                "hard_top1": (hard_hits / hard_n) if hard_n else float("nan"),
                "hard_n": hard_n,
                "confusable_top1": (conf_hits / conf_n) if conf_n else float("nan"),
                "confusable_n": conf_n}

    # build per-instance score arrays for each scorer
    def gather(pick):
        out = []
        for k, (F, dist, align, is_true) in enumerate(feats_per_inst):
            sel = iid == k
            out.append(pick(F, dist, align, learned[sel], learned_conseq[sel]))
        return out

    res = {
        "n_instances": len(inst),
        "mean_candidates": float(np.mean([len(f[0]) for f in feats_per_inst])),
        "mean_true_per_inst": float(np.mean([f[3].sum() for f in feats_per_inst])),
        "chance_top1": float(np.mean([f[3].mean() for f in feats_per_inst])),
        "nearest": top1(gather(lambda F, dist, al, lr, lc: -dist)),
        "align": top1(gather(lambda F, dist, al, lr, lc: al)),
        "learned": top1(gather(lambda F, dist, al, lr, lc: lr)),
        "consequence": top1(gather(lambda F, dist, al, lr, lc: lc)),
    }
    if verbose:
        print(f"instances={res['n_instances']}  mean candidates/inst="
              f"{res['mean_candidates']:.1f}  true/inst={res['mean_true_per_inst']:.1f}")
        print(f"chance top-1 = {res['chance_top1']:.3f}")
        print(f"confusable instances (a parallel/aligned distractor present): "
              f"{res['nearest']['confusable_n']}")
        for name in ("nearest", "align", "learned", "consequence"):
            r = res[name]
            print(f"  {name:12s} top1={r['top1']:.3f}  mrr={r['mrr']:.3f}  "
                  f"hard_top1={r['hard_top1']:.3f} (n={r['hard_n']})  "
                  f"confusable_top1={r['confusable_top1']:.3f} (n={r['confusable_n']})")
    return res
