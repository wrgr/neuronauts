"""Does *learned* local EM add to the geometry follow model? (the local-cut thesis)

Not a pretrained appearance cosine (that was type-confounded) — instead sample the
**raw EM along each candidate corridor** and let the model read it.  Intuition (the
user's "neurons were contiguous"): the corridor from the cut to the *true*
continuation stays inside one membrane-bounded process (continuous cytoplasm); the
corridor to a distractor must **cross a membrane** out of the cell and into another,
leaving a dark/bright signature in the raw intensity profile.  No seg mask (that would
leak identity), no hand threshold — K raw intensity samples per corridor, learned.

Compares, on the same reconnection task and leakage-safe GroupKFold:
* geometry-only  (17 raw canonical coords, from ``follow_learned``)
* geometry + EM  (+ K raw corridor-intensity samples)

EM cost is bounded to **one fetch per instance** (a box covering the cut and all its
candidates), so a few hundred instances is a few hundred mip-2 fetches.
"""
from __future__ import annotations

import numpy as np

from experiments.proofread.follow_test import build_instances
from experiments.proofread.follow_learned import _raw_features


def _sample_line(em_vol, a_nm, b_nm, K, rad=1):
    """Raw EM along a->b: mean and (membrane-catching) MIN over a small ball at each
    of K steps.  Returns a length-2K vector [means(K), mins(K)] in [0,1].

    The min over a ±rad-voxel ball robustly catches a thin dark membrane the single
    ray would step over; MICrONS EM is low-contrast (bulk ~115-144, membranes ~100),
    so we keep values raw and let the model scale/learn — no threshold picked by hand.
    """
    vox = np.asarray(em_vol.voxel_size_nm, float)
    origin = np.asarray(em_vol.bbox_voxels[0], float)
    shape = np.asarray(em_vol.data.shape)
    ts = np.linspace(0.0, 1.0, K)
    pts = a_nm[None, :] + ts[:, None] * (b_nm - a_nm)[None, :]
    idx = np.round(pts / vox - origin - 0.5).astype(int)
    means = np.empty(K, np.float32); mins = np.empty(K, np.float32)
    for k in range(K):
        lo = np.clip(idx[k] - rad, 0, shape - 1); hi = np.clip(idx[k] + rad + 1, 0, shape)
        block = em_vol.data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
        means[k] = block.mean() if block.size else 128.0
        mins[k] = block.min() if block.size else 128.0
    return np.concatenate([means, mins]) / 255.0


def em_profiles_for_instance(inst, *, mip=2, K=24, margin_nm=1000.0, em_vol=None):
    """(n_cand, K) raw corridor-intensity profiles for one instance (one EM fetch)."""
    v = inst["v"]; c_pos = inst["c_pos"]
    if em_vol is None:
        from neuronauts.fetch import fetch_volume
        pts = np.vstack([v[None, :], c_pos])
        lo = pts.min(0) - margin_nm; hi = pts.max(0) + margin_nm
        em_vol = fetch_volume((tuple(lo), tuple(hi)), mip=mip)
    return np.stack([_sample_line(em_vol, v, c_pos[j], K) for j in range(len(c_pos))])


def evaluate_em(neurons, *, gap_nm=2000.0, search_radius=3500.0, per_neuron=12,
                mip=2, K=24, max_instances=150, conf_fraction=0.5, seed=0, verbose=True):
    """Geometry-only vs geometry+EM on an EM-fetched, confusable-enriched subset."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold

    rng = np.random.default_rng(seed)
    inst = build_instances(neurons, gap_nm=gap_nm, search_radius=search_radius,
                           per_neuron=per_neuron, seed=seed)
    if not inst:
        return {"error": "no instances"}

    # label confusable instances (a parallel/aligned distractor) to enrich the subset
    def confusable(it):
        d = np.linalg.norm(it["c_pos"] - it["v"], axis=1)
        al = (it["c_pos"] - it["v"]) @ it["d"] / (d + 1e-9)
        t = it["is_true"]
        return (~t).any() and t.any() and al[~t].max() >= al[t].max() - 0.1

    conf_idx = [k for k, it in enumerate(inst) if confusable(it)]
    other_idx = [k for k in range(len(inst)) if k not in set(conf_idx)]
    n_conf = min(len(conf_idx), int(max_instances * conf_fraction))
    n_other = min(len(other_idx), max_instances - n_conf)
    keep = (list(rng.choice(conf_idx, n_conf, replace=False)) +
            list(rng.choice(other_idx, n_other, replace=False)))
    subset = [inst[k] for k in keep]
    if verbose:
        print(f"[em] {len(subset)} instances ({n_conf} confusable) -> {len(subset)} EM fetches")

    Xg, Xe, y, nid, iid, per = [], [], [], [], [], []
    for k, it in enumerate(subset):
        try:
            em = em_profiles_for_instance(it, mip=mip, K=K)
        except Exception as ex:  # noqa: BLE001
            if verbose:
                print(f"  [em {k}] fetch failed: {type(ex).__name__}: {ex}")
            continue
        G = _raw_features(it, neurons)
        Xg.append(G); Xe.append(np.hstack([G, em]))
        y.append(it["is_true"].astype(int))
        nid.append(np.full(len(G), it["neuron"])); iid.append(np.full(len(G), k))
        d = np.linalg.norm(it["c_pos"] - it["v"], axis=1)
        al = (it["c_pos"] - it["v"]) @ it["d"] / (d + 1e-9)
        per.append((it["is_true"], d, al))
        if verbose and (k % 25 == 0):
            print(f"  em {k+1}/{len(subset)}")
    if not Xg:
        return {"error": "no EM features"}
    Xg = np.vstack(Xg); Xe = np.vstack(Xe); y = np.concatenate(y)
    nid = np.concatenate(nid); iid = np.concatenate(iid)
    uremap = {k: i for i, k in enumerate(np.unique(iid))}
    iid = np.array([uremap[v] for v in iid])

    def oof(X):
        s = np.full(len(y), np.nan)
        gkf = GroupKFold(n_splits=max(2, min(5, len(np.unique(nid)))))
        for tr, te in gkf.split(X, y, nid):
            if len(np.unique(y[tr])) < 2:
                continue
            m = make_pipeline(StandardScaler(),
                              MLPClassifier(hidden_layer_sizes=(48, 24), max_iter=400,
                                            alpha=1e-3, random_state=seed))
            m.fit(X[tr], y[tr]); s[te] = m.predict_proba(X[te])[:, 1]
        return s

    def score(s):
        hits = hard = hard_n = conf = conf_n = 0
        for k, (t, dist, al) in enumerate(per):
            sc = s[iid == k]
            if len(sc) == 0 or np.isnan(sc).all():
                continue
            top = int(t[np.argmax(sc)]); hits += top
            if not t[np.argmin(dist)]:
                hard_n += 1; hard += top
            if (~t).any() and t.any() and al[~t].max() >= al[t].max() - 0.1:
                conf_n += 1; conf += top
        n = len(per)
        return {"top1": hits / n, "hard_top1": hard / hard_n if hard_n else float("nan"),
                "confusable_top1": conf / conf_n if conf_n else float("nan"),
                "hard_n": hard_n, "confusable_n": conf_n}

    res = {"n_instances": len(per), "K": K, "mip": mip,
           "geometry": score(oof(Xg)), "geometry_plus_em": score(oof(Xe))}
    if verbose:
        print(f"\n=== LEARNED LOCAL EM vs GEOMETRY (n={len(per)} instances) ===")
        for name in ("geometry", "geometry_plus_em"):
            r = res[name]
            print(f"  {name:18s} top1={r['top1']:.3f}  hard={r['hard_top1']:.3f}  "
                  f"confusable={r['confusable_top1']:.3f} (n={r['confusable_n']})")
    return res
