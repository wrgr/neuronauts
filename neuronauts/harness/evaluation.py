"""Evaluation that stays meaningful at realistic base rates.

Two layers, because a merge scorer can look good at one and fail at the other:

* **Ranking** over labelled candidate pairs -- ROC-AUC, average precision,
  precision at fixed recall and recall at fixed precision. AUC alone is
  misleading when positives are ~1% of the panel, so the operating points
  are reported next to it.
* **Assembly** -- accept every pair above a threshold, take connected
  components (union-find), and score the resulting atom partition against the
  proofread owners: adjusted Rand index over labelled atoms, pairwise merge
  precision / recall with the same strict pair rules as the labels, the
  number of false-merge pairs, the largest cluster, and cable-weighted purity
  and completeness. The largest-cluster column is the collapse detector that
  EXP-053A needed.

Recall in the assembly layer is against *all* same-owner pairs of labelled
atoms in the evaluated set, not only those the panel proposed; the difference
between the two is the panel's own coverage and is reported separately.
Everything is numpy; nothing here touches the network.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from neuronauts.harness.labels import (
    LABEL_NEG, LABEL_POS, TIER_GOLD, TIER_NONE, AtomLabels,
)


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------

def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC with tie averaging; NaN without both classes."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, np.float64)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), np.float64)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def precision_recall_curve(y: np.ndarray, score: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision and recall at every distinct score threshold (descending)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, np.float64)
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    last = np.ones(len(ss), bool)
    last[:-1] = ss[1:] != ss[:-1]
    tp, fp, thr = tp[last], fp[last], ss[last]
    n_pos = int(y.sum())
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(n_pos, 1)
    return precision, recall, thr


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    if y.sum() == 0:
        return float("nan")
    p, r, _ = precision_recall_curve(y, score)
    r_prev = np.concatenate([[0.0], r[:-1]])
    return float(np.sum((r - r_prev) * p))


def rank_metrics(y: np.ndarray, score: np.ndarray, *,
                 recall_targets=(0.5, 0.8, 0.9),
                 precision_targets=(0.9, 0.95, 0.99)) -> dict:
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, np.float64)
    out = {"n_pos": int(y.sum()), "n_neg": int((~y).sum()),
           "auc": roc_auc(y, s), "ap": average_precision(y, s)}
    if out["n_pos"] == 0 or out["n_neg"] == 0:
        for r in recall_targets:
            out[f"precision_at_recall_{r:g}"] = float("nan")
        for p in precision_targets:
            out[f"recall_at_precision_{p:g}"] = float("nan")
        return out
    p, r, thr = precision_recall_curve(y, s)
    for rt in recall_targets:
        ok = r >= rt
        out[f"precision_at_recall_{rt:g}"] = float(p[ok].max()) if ok.any() else 0.0
        out[f"threshold_at_recall_{rt:g}"] = (float(thr[ok][np.argmax(p[ok])])
                                             if ok.any() else float("nan"))
    for pt in precision_targets:
        ok = p >= pt
        out[f"recall_at_precision_{pt:g}"] = float(r[ok].max()) if ok.any() else 0.0
    return out


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def union_find_components(n: int, edges_a: np.ndarray, edges_b: np.ndarray
                          ) -> np.ndarray:
    """Component id per node after joining every listed edge."""
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in zip(np.asarray(edges_a).tolist(), np.asarray(edges_b).tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.fromiter((find(i) for i in range(n)), np.int64, n)
    _, comp = np.unique(roots, return_inverse=True)
    return comp.astype(np.int64)


def adjusted_rand_index(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred)
    true = np.asarray(true)
    n = len(pred)
    if n < 2:
        return float("nan")
    _, pi = np.unique(pred, return_inverse=True)
    _, ti = np.unique(true, return_inverse=True)
    n_p = int(pi.max()) + 1
    joint = np.bincount(ti.astype(np.int64) * n_p + pi, minlength=(int(ti.max()) + 1) * n_p)
    c2 = lambda v: (v * (v - 1) / 2.0)
    sum_ij = c2(joint.astype(np.float64)).sum()
    sum_a = c2(np.bincount(ti).astype(np.float64)).sum()
    sum_b = c2(np.bincount(pi).astype(np.float64)).sum()
    total = n * (n - 1) / 2.0
    expected = sum_a * sum_b / total
    max_index = (sum_a + sum_b) / 2.0
    denom = max_index - expected
    if denom == 0:
        return 1.0 if sum_ij == expected else 0.0
    return float((sum_ij - expected) / denom)


def strict_pair_counts(cluster: np.ndarray, owner: np.ndarray, tier: np.ndarray,
                       pure: np.ndarray) -> dict:
    """Pairwise merge counts under the strict label rules, in closed form.

    Within each predicted cluster, over its pure members: a same-proofread-
    owner pair is a true merge; a different-owner pair is a false merge when
    either member is gold or both are proofread. Any other pair is unknown and
    is not counted. Recall denominators use the same rules over all pairs of
    the evaluated atoms, whether or not they were candidates.
    """
    cluster = np.asarray(cluster)
    owner = np.asarray(owner, np.uint64)
    tier = np.asarray(tier)
    pure = np.asarray(pure, bool)
    m = pure
    cl, ow, tr = cluster[m], owner[m], tier[m]

    def counts_by(keys: np.ndarray, weights_mask: np.ndarray) -> np.ndarray:
        return np.bincount(keys, weights=weights_mask.astype(np.float64))

    # per (cluster, owner) group counts, split by tier
    _, cl_i = np.unique(cl, return_inverse=True)
    _, ow_i = np.unique(ow, return_inverse=True)
    n_ow = int(ow_i.max()) + 1 if len(ow_i) else 1
    key = cl_i.astype(np.int64) * n_ow + ow_i
    ukey, inv = np.unique(key, return_inverse=True)
    g_all = np.bincount(inv).astype(np.float64)
    g_gold = counts_by(inv, tr == TIER_GOLD)
    g_silv = counts_by(inv, (tr > TIER_NONE) & (tr != TIER_GOLD))
    g_pr = g_gold + g_silv
    grp_cl = ukey // n_ow

    n_cl = np.bincount(cl_i).astype(np.float64)          # pure members / cluster
    G_cl = np.bincount(grp_cl, weights=g_gold, minlength=len(n_cl))
    S_cl = np.bincount(grp_cl, weights=g_silv, minlength=len(n_cl))

    tp = float((g_pr * (g_pr - 1) / 2.0).sum())
    # gold members paired with any different-owner member, minus the
    # gold-gold different-owner pairs counted from both sides
    gold_cross = (g_gold * (n_cl[grp_cl] - g_all)).sum() \
        - 0.5 * (g_gold * (G_cl[grp_cl] - g_gold)).sum()
    silver_cross = 0.5 * (g_silv * (S_cl[grp_cl] - g_silv)).sum()
    fp = float(gold_cross + silver_cross)

    # denominators over the whole evaluated set (one big "cluster")
    o_all = np.bincount(ow_i).astype(np.float64)
    o_gold = counts_by(ow_i, tr == TIER_GOLD)
    o_silv = counts_by(ow_i, (tr > TIER_NONE) & (tr != TIER_GOLD))
    o_pr = o_gold + o_silv
    n_all = float(len(cl))
    pos_total = float((o_pr * (o_pr - 1) / 2.0).sum())
    neg_total = float((o_gold * (n_all - o_all)).sum()
                      - 0.5 * (o_gold * (o_gold.sum() - o_gold)).sum()
                      + 0.5 * (o_silv * (o_silv.sum() - o_silv)).sum())
    return {"tp": tp, "fp": fp, "fn": pos_total - tp,
            "pos_total": pos_total, "neg_total": neg_total}


def assembly_metrics(cluster: np.ndarray, labels_idx: np.ndarray,
                     labels: AtomLabels, *,
                     cable_nm: Optional[np.ndarray] = None) -> dict:
    """Score an atom partition against the label table.

    ``cluster`` is a component id per evaluated atom; ``labels_idx`` the row
    of each atom in ``labels`` (-1 when it has no row).
    """
    cluster = np.asarray(cluster)
    idx = np.asarray(labels_idx)
    have = idx >= 0
    owner = np.zeros(len(cluster), np.uint64)
    tier = np.zeros(len(cluster), np.int8)
    pure = np.zeros(len(cluster), bool)
    owner[have] = labels.owner[idx[have]]
    tier[have] = labels.owner_tier[idx[have]]
    pure[have] = labels.pure[idx[have]]

    out = {"n_atoms": int(len(cluster)),
           "n_clusters": int(len(np.unique(cluster))),
           "largest_cluster": int(np.bincount(
               np.unique(cluster, return_inverse=True)[1]).max()) if len(cluster) else 0}
    lab = pure & (tier > TIER_NONE)
    out["n_labelled_atoms"] = int(lab.sum())
    out["n_labelled_neurons"] = int(len(np.unique(owner[lab])))
    out["ari_labelled"] = adjusted_rand_index(cluster[lab], owner[lab]) if lab.sum() >= 2 else float("nan")

    pc = strict_pair_counts(cluster, owner, tier, pure)
    tp, fp, fn = pc["tp"], pc["fp"], pc["fn"]
    out["merge_tp_pairs"] = tp
    out["merge_fp_pairs"] = fp
    out["merge_fn_pairs"] = fn
    out["merge_precision"] = tp / (tp + fp) if tp + fp > 0 else float("nan")
    out["merge_recall"] = tp / pc["pos_total"] if pc["pos_total"] > 0 else float("nan")
    p, r = out["merge_precision"], out["merge_recall"]
    out["merge_f1"] = (2 * p * r / (p + r) if np.isfinite(p) and np.isfinite(r)
                       and p + r > 0 else float("nan"))
    out["gt_positive_pairs"] = pc["pos_total"]
    out["gt_negative_pairs"] = pc["neg_total"]

    # A cluster is contaminated when its pure members do not all share one
    # owner and at least one of them is labelled -- a real false merge, as
    # opposed to a cluster of unlabelled atoms we cannot judge.
    _, ci = np.unique(cluster, return_inverse=True)
    n_c = int(ci.max()) + 1 if len(ci) else 0
    if pure.any():
        _, oi_pure = np.unique(owner[pure], return_inverse=True)
        pair = np.unique(np.stack([ci[pure], oi_pure], axis=1), axis=0)
        n_owner_per_cluster = np.bincount(pair[:, 0], minlength=n_c)
    else:
        n_owner_per_cluster = np.zeros(n_c, np.int64)
    has_label = np.bincount(ci[lab], minlength=n_c) > 0 if n_c else np.zeros(0, bool)
    contaminated = (n_owner_per_cluster >= 2) & has_label
    out["n_clusters_with_label"] = int(has_label.sum())
    out["n_contaminated_clusters"] = int(contaminated.sum())

    if cable_nm is not None:
        cab = np.asarray(cable_nm, np.float64)
        cab = np.where(np.isfinite(cab), cab, 0.0)
        # purity: share of labelled cable that sits with its cluster's majority
        # owner; completeness: for each neuron, share of its cable in its
        # largest cluster
        if lab.sum():
            _, oi = np.unique(owner[lab], return_inverse=True)
            n_o = int(oi.max()) + 1
            key = ci[lab].astype(np.int64) * n_o + oi
            w = np.bincount(key, weights=cab[lab], minlength=n_c * n_o).reshape(n_c, n_o)
            total = w.sum()
            out["cable_purity"] = float(w.max(axis=1).sum() / total) if total > 0 else float("nan")
            out["cable_completeness"] = float(w.max(axis=0).sum() / total) if total > 0 else float("nan")
        else:
            out["cable_purity"] = out["cable_completeness"] = float("nan")
    return out


def assemble_at_threshold(n_atoms: int, pair_ia: np.ndarray, pair_ib: np.ndarray,
                          score: np.ndarray, threshold: float) -> np.ndarray:
    keep = np.isfinite(score) & (score >= threshold)
    return union_find_components(n_atoms, pair_ia[keep], pair_ib[keep])


def threshold_grid(score: np.ndarray, y: np.ndarray, *, n: int = 12) -> np.ndarray:
    """Thresholds spanning the labelled score range, densest where positives are."""
    s = np.asarray(score, np.float64)
    ok = np.isfinite(s)
    if not ok.any():
        return np.zeros(0)
    q = np.linspace(0.0, 1.0, n)
    pos = s[ok & (np.asarray(y) == LABEL_POS)]
    base = s[ok]
    grid = np.quantile(base, q)
    if len(pos):
        grid = np.concatenate([grid, np.quantile(pos, [0.1, 0.25, 0.5, 0.75, 0.9])])
    return np.unique(grid)
