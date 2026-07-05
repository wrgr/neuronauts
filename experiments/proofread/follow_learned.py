"""Learn to follow from RAW geometry — no hand-engineered align/reciprocal/caliber.

Pushback taken: hand-tuned features aren't how humans do it.  Here the model gets
only **raw coordinates** in a canonical frame (translate to the cut point, rotate so
the incoming trajectory is +x) — the candidate point, a few of its own neighbour
offsets (its local cable, unlabelled), and the two radii — and a small MLP learns the
continuation function itself.  If it matches/beats the hand-feature logistic, the
follow signal is *learned*, not engineered: the net discovers "continue straight"
(from the candidate's canonical position), "the far end must point back" (from its
neighbour offsets), and "caliber must match" (from the radii) on its own.

Same task/eval as ``follow_test.py`` (reconnect a real cut against real distractors),
same leave-one-neuron-out protocol, so the numbers are directly comparable.
"""
from __future__ import annotations

import numpy as np

from experiments.proofread.follow_test import load_neurons, build_instances


def _frame(d):
    """Orthonormal rows [d, e1, e2]; canonical roll from world-up via Gram-Schmidt."""
    x = d / (np.linalg.norm(d) + 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    if abs(x @ up) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    y = up - (up @ x) * x; y /= (np.linalg.norm(y) + 1e-9)
    z = np.cross(x, y)
    return np.stack([x, y, z])                       # (3,3) rows


def _raw_features(inst, neurons, n_nbr=4):
    """Raw canonical-frame coordinates per candidate — nothing hand-derived.

    [ c_canon(3) | up to n_nbr neighbour offsets of the candidate's own cable,
      canonical(3 each) | r_candidate, r_endpoint ]  (all in µm).
    The neighbour offsets are the candidate's raw local skeleton — the net must
    infer direction/reciprocity from them, we don't compute a tangent.
    """
    v = inst["v"]; R = _frame(inst["d"]); r_v = inst["r_v"]
    c_pos = inst["c_pos"]; c_rad = inst["c_rad"]
    c_owner = inst["c_owner"]; c_local = inst["c_local"]
    feats = []
    for j in range(len(c_pos)):
        c_canon = R @ (c_pos[j] - v) / 1000.0
        N = neurons[int(c_owner[j])]; li = int(c_local[j])
        offs = []
        for nb in N.adj[li][:n_nbr]:
            offs.append(R @ (N.v[nb] - c_pos[j]) / 1000.0)
        while len(offs) < n_nbr:
            offs.append(np.zeros(3))
        feats.append(np.concatenate([c_canon, np.concatenate(offs),
                                     [c_rad[j] / 1000.0, r_v / 1000.0]]))
    return np.asarray(feats)


def evaluate_learned(neurons, *, gap_nm=2000.0, search_radius=5000.0, per_neuron=12,
                     n_nbr=4, seed=0, hidden=(48, 24), verbose=True):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold

    inst = build_instances(neurons, gap_nm=gap_nm, search_radius=search_radius,
                           per_neuron=per_neuron, seed=seed)
    if not inst:
        return {"error": "no instances"}

    per, X, y, nid, iid = [], [], [], [], []
    for k, it in enumerate(inst):
        F = _raw_features(it, neurons, n_nbr=n_nbr)
        dist = np.linalg.norm(it["c_pos"] - it["v"], axis=1)
        align = (it["c_pos"] - it["v"]) @ it["d"] / (dist + 1e-9)
        per.append((it["is_true"], dist, align))
        X.append(F); y.append(it["is_true"].astype(int))
        nid.append(np.full(len(F), it["neuron"])); iid.append(np.full(len(F), k))
    X = np.vstack(X); y = np.concatenate(y)
    nid = np.concatenate(nid); iid = np.concatenate(iid)

    # leakage-safe GroupKFold by neuron (5 folds -> fast; no cut's own neuron in train)
    scores = np.full(len(y), np.nan)
    n_groups = len(np.unique(nid))
    gkf = GroupKFold(n_splits=max(2, min(5, n_groups)))
    for tr, te in gkf.split(X, y, nid):
        if len(np.unique(y[tr])) < 2:
            continue
        m = make_pipeline(StandardScaler(),
                          MLPClassifier(hidden_layer_sizes=hidden, max_iter=400,
                                        alpha=1e-3, random_state=seed))
        m.fit(X[tr], y[tr]); scores[te] = m.predict_proba(X[te])[:, 1]

    def is_confusable(is_true, align):
        return (~is_true).any() and is_true.any() and \
            align[~is_true].max() >= align[is_true].max() - 0.1

    hits = hard = hard_n = conf = conf_n = 0
    for k, (is_true, dist, align) in enumerate(per):
        s = scores[iid == k]
        top = int(is_true[np.argmax(s)])
        hits += top
        if not is_true[np.argmin(dist)]:
            hard_n += 1; hard += top
        if is_confusable(is_true, align):
            conf_n += 1; conf += top
    n = len(per)
    res = {"model": "MLP(raw canonical coords)", "n_instances": n,
           "n_features": X.shape[1],
           "top1": hits / n, "hard_top1": hard / hard_n if hard_n else float("nan"),
           "confusable_top1": conf / conf_n if conf_n else float("nan"),
           "hard_n": hard_n, "confusable_n": conf_n}
    if verbose:
        print(f"LEARNED (raw geometry, {X.shape[1]} raw coords/candidate, no hand features)")
        print(f"  instances={n}  top1={res['top1']:.3f}  "
              f"hard_top1={res['hard_top1']:.3f} (n={hard_n})  "
              f"confusable_top1={res['confusable_top1']:.3f} (n={conf_n})")
    return res
