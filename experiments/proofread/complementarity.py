"""The headline test: does the local-EM cue (Pillar 2) resolve join edits the
global shape/grammar cue (the AutoProof-style baseline) cannot?

Ground truth is the v117->later proofreading divergence (``synapse_correction``):
a **false split** is two v117 roots that a proofreader later merged into one cell
(a JOIN that should happen); a spatially-adjacent pair with *different* later roots
is a distinct-cell pair (a JOIN that should NOT happen).  Each candidate carries
its two synapse-side positions, so we can read the raw EM at the exact join site.

Per candidate we assemble two interpretable streams:

* **shape / grammar** — the existing point-cloud grammar+geometry features
  (``synapse_correction._pair_features``): compartment/caliber grammar, seam-gap
  occupancy, relative geometry.  This is the **AutoProof-style shape+synapse
  baseline** (and the dense cousin of Pillar-1 ``grammar_energy``).
* **local ultrastructure** — Pillar-2 ``local_evidence`` at the join site:
  cut-face cross-section match + membrane barrier from the raw mip-1 EM.

We fit leakage-safe (GroupKFold by cell component) out-of-fold classifiers for
shape-only, local-only, and joint, and report AUC + a precision/coverage
(abstention) curve.  Complementarity = joint beats shape-only, and concrete
candidates the local cue flips correctly where shape is wrong.

CLAUDE.md: precision is always reported *with* coverage; the local barrier is a
labelled first-cut approximation; counts trace back to ``summarize_edits``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from itertools import combinations

from experiments.pcfg.synapse_correction import (
    SideTable, cell_components, _build_root_ctx, _pair_features)
from experiments.proofread.local_evidence import local_evidence


@dataclass
class JoinCandidate:
    pos_a: np.ndarray      # (3,) nm synapse-side position on v117 root A
    pos_b: np.ndarray      # (3,) nm synapse-side position on v117 root B
    label: int             # 1 = same cell (should be together), 0 = distinct cells
    group: int             # cell-component id (leakage-safe CV group)
    shape_feat: np.ndarray # point-cloud grammar+geometry features
    side: int              # 0 = pre stream, 1 = post stream
    rv_a: int
    rv_b: int
    stratum: int           # 0 = SPLIT (within-root, a CUT test); 1 = MERGE (cross-root, a JOIN test)


def build_pair_candidates(tab: SideTable, *, cross_radius_nm: float = 6000.0,
                          cross_k_neighbors: int = 12, max_neg_ratio: float = 2.0,
                          min_synapses: int = 4, max_within_pairs_per_root: int = 60,
                          max_pair_nm: float | None = None, rng=None) -> list[JoinCandidate]:
    """Unified same-cell pair candidates over BOTH strata (positions kept).

    Label ``y = 1`` iff the two synapse-sides share the same nonzero later root
    (belong to one proofread cell).  Two strata, mirroring ``build_correction_pairs``:

    * **SPLIT** (within one v117 root): a ``y=0`` pair is a false MERGE the v117 seg
      made — the two sides should be **cut** apart.
    * **MERGE** (cross v117 root): a ``y=1`` pair is a false SPLIT — the two roots
      should be **joined**.

    So the local-EM + shape cues are tested on both correction directions at once.
    """
    from scipy.spatial import cKDTree
    if rng is None:
        rng = np.random.default_rng(0)
    comp = cell_components(tab)
    out: list[JoinCandidate] = []

    for side_code in (0, 1):
        sel = (tab.side == side_code) & (tab.root_later > 0)
        rows = np.nonzero(sel)[0]
        if len(rows) < 2:
            continue
        sub = tab.mask(sel)
        by_root: dict[int, list[int]] = {}
        for li, rv in enumerate(sub.root_v117.tolist()):
            by_root.setdefault(int(rv), []).append(li)
        ctx_all: dict[int, object] = {}

        def get_ctx(rv: int):
            c = ctx_all.get(rv)
            if c is None:
                c = _build_root_ctx(sub, by_root[rv])
                ctx_all[rv] = c
            return c

        # ---- SPLIT stratum: within-root pairs (a CUT test) ----
        for rv, idxs in by_root.items():
            if len(idxs) < min_synapses:
                continue
            c = get_ctx(rv)
            pairs = list(combinations(range(len(idxs)), 2))
            if len(pairs) > max_within_pairs_per_root:
                pairs = [pairs[p] for p in
                         rng.choice(len(pairs), max_within_pairs_per_root, replace=False)]
            for a, b in pairs:
                ra, rb = idxs[a], idxs[b]
                lbl = int(sub.root_later[ra] == sub.root_later[rb])
                feat = _pair_features(c, c.index_of[ra], c, c.index_of[rb],
                                      same_root=True, na=len(idxs), nb=len(idxs))
                out.append(JoinCandidate(
                    pos_a=sub.pt[ra].copy(), pos_b=sub.pt[rb].copy(), label=lbl,
                    group=comp.get(rv, -1), shape_feat=feat, side=side_code,
                    rv_a=rv, rv_b=rv, stratum=0))

        by_later: dict[int, list[int]] = {}
        for li in range(len(sub)):
            by_later.setdefault(int(sub.root_later[li]), []).append(li)

        # ---- MERGE stratum positives: cross-root NN sharing a later root ----
        pos_pairs: list[tuple[int, int]] = []
        for members in by_later.values():
            if len(members) < 2 or len({int(sub.root_v117[m]) for m in members}) < 2:
                continue
            mpts = sub.pt[members]
            tree = cKDTree(mpts)
            kq = min(cross_k_neighbors + 1, len(members))
            dnn, inn = tree.query(mpts, k=kq, workers=-1)
            seen: set[tuple[int, int]] = set()
            for a in range(len(members)):
                ra = members[a]; rva = int(sub.root_v117[ra])
                for slot in range(1, kq):
                    if dnn[a, slot] > cross_radius_nm:
                        break
                    rb = members[int(inn[a, slot])]
                    if int(sub.root_v117[rb]) == rva:
                        continue
                    key = (min(ra, rb), max(ra, rb))
                    if key not in seen:
                        seen.add(key); pos_pairs.append((ra, rb))

        # ---- spatially-matched hard negatives (adjacent but distinct later root) ----
        neg_pairs: list[tuple[int, int]] = []
        if pos_pairs:
            gtree = cKDTree(sub.pt)
            anchors = list({p for pr in pos_pairs for p in pr})
            kq = min(cross_k_neighbors + 1, len(sub))
            seen_n: set[tuple[int, int]] = set()
            for ra in anchors:
                rva = int(sub.root_v117[ra]); la = int(sub.root_later[ra])
                dnn, inn = gtree.query(sub.pt[ra], k=kq, workers=-1)
                for slot in range(1, kq):
                    if dnn[slot] > cross_radius_nm:
                        break
                    rb = int(inn[slot])
                    if int(sub.root_v117[rb]) == rva or int(sub.root_later[rb]) == la:
                        continue
                    key = (min(ra, rb), max(ra, rb))
                    if key not in seen_n:
                        seen_n.add(key); neg_pairs.append((ra, rb))
        n_neg = min(len(neg_pairs), max(1, int(max(1, len(pos_pairs)) * max_neg_ratio)))
        if len(neg_pairs) > n_neg:
            neg_pairs = [neg_pairs[p] for p in rng.choice(len(neg_pairs), n_neg, replace=False)]

        for (ra, rb), lbl in ([(p, 1) for p in pos_pairs] + [(p, 0) for p in neg_pairs]):
            rva, rvb = int(sub.root_v117[ra]), int(sub.root_v117[rb])
            ca, cb = get_ctx(rva), get_ctx(rvb)
            feat = _pair_features(ca, ca.index_of[ra], cb, cb.index_of[rb],
                                  same_root=False, na=len(by_root[rva]), nb=len(by_root[rvb]))
            out.append(JoinCandidate(
                pos_a=sub.pt[ra].copy(), pos_b=sub.pt[rb].copy(), label=lbl,
                group=comp.get(rva, -1), shape_feat=feat, side=side_code,
                rv_a=rva, rv_b=rvb, stratum=1))

    if max_pair_nm is not None:
        # Keep only genuinely *local* pairs: the cut-face / membrane cue is only
        # meaningful between nearby cross-sections (a real seam / join site), not
        # two arbitrary synapse sides several microns apart.
        out = [c for c in out
               if float(np.linalg.norm(c.pos_a - c.pos_b)) <= max_pair_nm]
    return out


def attach_local_evidence(cands: list[JoinCandidate], embed_fn, *, mip=1,
                          margin_nm=1200.0, verbose=True) -> np.ndarray:
    """Compute Pillar-2 local evidence per candidate (one EM+seg fetch each).

    Returns an ``(N, 3)`` array of ``[cutface_sim, barrier, ok]`` aligned to
    ``cands``.  Individual fetches keep each volume tiny; bounded by the caller's
    candidate count.
    """
    feats = np.zeros((len(cands), 3), float)
    for i, c in enumerate(cands):
        try:
            ev = local_evidence(c.pos_a, c.pos_b, embed_fn, mip=mip, margin_nm=margin_nm)
            feats[i] = [ev.cutface_sim, ev.barrier, 1.0 if ev.ok else 0.0]
        except Exception as e:  # noqa: BLE001 - a single bad site must not kill the run
            if verbose:
                print(f"  [local {i}] failed: {type(e).__name__}: {e}")
            feats[i] = [0.0, 0.0, 0.0]
        if verbose and (i % 10 == 0 or i == len(cands) - 1):
            print(f"  local evidence {i+1}/{len(cands)}  sim={feats[i,0]:+.2f} "
                  f"barrier={feats[i,1]:.2f} ok={feats[i,2]:.0f}")
    return feats


def _oof_logreg(X, y, groups, *, n_splits=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced"))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def precision_coverage(y, p, *, grid=None):
    """Precision & coverage as the accept-threshold sweeps (abstain below it).

    Only the *positive* decisions (accept a join) are scored for precision; coverage
    is the fraction of all candidates accepted.  Reports the operating point at the
    highest threshold reaching >=0.95 precision.
    """
    if grid is None:
        grid = np.linspace(0.5, 0.99, 50)
    scored = ~np.isnan(p)
    y = np.asarray(y)[scored]; p = np.asarray(p)[scored]
    rows = []
    for t in grid:
        acc = p >= t
        if acc.sum() == 0:
            continue
        prec = float(y[acc].mean())
        cov = float(acc.mean())
        rows.append((float(t), prec, cov, int(acc.sum())))
    return rows


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    scored = ~np.isnan(p)
    if len(np.unique(np.asarray(y)[scored])) < 2:
        return float("nan")
    return float(roc_auc_score(np.asarray(y)[scored], np.asarray(p)[scored]))


def run_complementarity(tab: SideTable, embed_fn, *, max_candidates=60,
                        max_pair_nm=6000.0, mip=1, seed=0, verbose=True) -> dict:
    """End-to-end complementarity eval on local edit-site candidates from ``tab``."""
    rng = np.random.default_rng(seed)
    cands = build_pair_candidates(tab, max_pair_nm=max_pair_nm, rng=rng)
    if not cands:
        return {"error": "no candidates (need v117->later divergence)"}
    # Bounded, informative subsample: keep every error-implicating pair (the rare
    # signal) and fill the rest with same-cell pairs.  y=0 within-root = false
    # merge (cut); y=1 cross-root = false split (join).  Both are scarce -> keep all.
    err = [i for i, c in enumerate(cands)
           if (c.stratum == 0 and c.label == 0) or (c.stratum == 1 and c.label == 1)]
    other = [i for i, c in enumerate(cands) if i not in set(err)]
    n_other = min(len(other), max(max_candidates - len(err), max_candidates // 2))
    keep = list(err) + list(rng.choice(other, min(n_other, len(other)), replace=False))
    if len(keep) > max_candidates:  # cap total EM fetches, but never drop an error pair
        extra = [k for k in keep if k in set(other)]
        keep = list(err) + list(rng.choice(extra, max_candidates - len(err), replace=False))
    cands = [cands[i] for i in keep]
    y = np.array([c.label for c in cands])
    groups = np.array([c.group for c in cands])
    strata = np.array([c.stratum for c in cands])
    shape = np.stack([c.shape_feat for c in cands])
    if verbose:
        n_cut = int(((strata == 0) & (y == 0)).sum()); n_join = int(((strata == 1) & (y == 1)).sum())
        print(f"[candidates] {len(cands)} pairs "
              f"(false-merge/cut={n_cut}, false-split/join={n_join}, "
              f"same-cell={int((y==1).sum())} tot pos), {len(np.unique(groups))} cell groups")

    local = attach_local_evidence(cands, embed_fn, mip=mip, verbose=verbose)
    joint = np.hstack([shape, local])

    p_shape = _oof_logreg(shape, y, groups, seed=seed)
    p_local = _oof_logreg(local, y, groups, seed=seed)
    p_joint = _oof_logreg(joint, y, groups, seed=seed)

    res = {
        "n": len(cands), "pos": int(y.sum()), "groups": int(len(np.unique(groups))),
        "auc_shape": _auc(y, p_shape), "auc_local": _auc(y, p_local),
        "auc_joint": _auc(y, p_joint),
        "pc_shape": precision_coverage(y, p_shape),
        "pc_joint": precision_coverage(y, p_joint),
        # arrays for the downstream ranked queue
        "cands": cands, "y": y, "local": local, "p_joint": p_joint, "strata": strata,
    }
    # concrete cases the local cue fixes: shape wrong, joint right
    fixed = []
    for i, c in enumerate(cands):
        if np.isnan(p_shape[i]) or np.isnan(p_joint[i]):
            continue
        shape_pred = int(p_shape[i] >= 0.5); joint_pred = int(p_joint[i] >= 0.5)
        if shape_pred != c.label and joint_pred == c.label:
            fixed.append({"i": i, "label": c.label,
                          "kind": "cut" if c.stratum == 0 else "join",
                          "cutface_sim": float(local[i, 0]),
                          "barrier": float(local[i, 1]), "p_shape": float(p_shape[i]),
                          "p_joint": float(p_joint[i]),
                          "pos_a": c.pos_a.tolist(), "pos_b": c.pos_b.tolist()})
    res["local_fixes"] = fixed
    if verbose:
        print(f"\n=== COMPLEMENTARITY (leakage-safe GroupKFold OOF) ===")
        print(f"  AUC  shape-only={res['auc_shape']:.3f}  local-only={res['auc_local']:.3f}"
              f"  JOINT={res['auc_joint']:.3f}")
        for tag in ("pc_shape", "pc_joint"):
            best = [r for r in res[tag] if r[1] >= 0.95]
            if best:
                t, pr, cov, n = max(best, key=lambda r: r[2])
                print(f"  {tag}: P>=0.95 at thr={t:.2f} -> precision={pr:.3f} "
                      f"coverage={cov:.2f} ({n} accepted)")
            else:
                print(f"  {tag}: never reaches P>=0.95")
        print(f"  local uniquely fixes {len(fixed)} candidate(s) shape got wrong")
    return res
