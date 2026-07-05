"""Pillar 3 output — a ranked, abstaining proofreading queue with viewer links.

Turns the per-candidate joint decision (P(same cell) from the two-cue combiner)
into a prioritised worklist: confident errors are proposed as auto-edits, the
ambiguous band is deferred to a ranked human queue, and each item carries a
Neuroglancer URL that resolves to the edit site.  Precision is always reported
with coverage (abstention), following the asymmetric merge/split cost of
``treestitch.risk``.

An entry's proposed action:

* SPLIT candidate (within-root pair), P(same) low  -> **CUT** (false merge)
* MERGE candidate (cross-root pair), P(same) high   -> **JOIN** (false split)
* otherwise                                          -> **ABSTAIN** (queue for review)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class QueueItem:
    rank: int
    kind: str            # CUT | JOIN | ABSTAIN
    priority: float      # expected-loss-weighted urgency (higher = review first)
    p_same: float        # combiner P(two sides are the same cell)
    confident: bool      # outside the abstain band -> auto-edit candidate
    correct: int         # 1 if the proposed action matches ground truth, else 0 (-1 unknown)
    cutface_sim: float
    barrier: float
    pos_a: list
    pos_b: list
    rv_a: int
    rv_b: int
    ngl_url: str = ""


def build_queue(res: dict, *, cut_thresh: float = 0.35, join_thresh: float = 0.65,
                cost_merge: float = 5.0, cost_split: float = 1.0,
                with_urls: bool = True) -> list[QueueItem]:
    """Rank candidates into an abstaining edit/review queue from ``run_complementarity``.

    ``priority`` weights how far the decision sits from the abstain band by the
    asymmetric cost of the implied error (a missed false-merge costs ``cost_merge``,
    a missed false-split ``cost_split``), so the reviewer sees the costliest,
    most-confident errors first.
    """
    cands = res["cands"]; p = np.asarray(res["p_joint"]); local = np.asarray(res["local"])
    y = np.asarray(res["y"])
    items: list[QueueItem] = []
    for i, c in enumerate(cands):
        pi = float(p[i]) if not np.isnan(p[i]) else 0.5
        if c.stratum == 0 and pi <= cut_thresh:
            kind, confident = "CUT", True
            correct = int(y[i] == 0)
            weight = cost_merge * (cut_thresh - pi)      # deeper below band = surer cut
        elif c.stratum == 1 and pi >= join_thresh:
            kind, confident = "JOIN", True
            correct = int(y[i] == 1)
            weight = cost_split * (pi - join_thresh)
        else:
            kind, confident = "ABSTAIN", False
            # urgency of a review = closeness to the decision boundary * cost of the
            # error it would imply (within-root -> a possible merge, cost_merge)
            band_cost = cost_merge if c.stratum == 0 else cost_split
            weight = band_cost * (0.5 - abs(pi - 0.5))
            correct = -1
        items.append(QueueItem(
            rank=0, kind=kind, priority=float(weight), p_same=pi, confident=confident,
            correct=correct, cutface_sim=float(local[i, 0]), barrier=float(local[i, 1]),
            pos_a=[float(x) for x in c.pos_a], pos_b=[float(x) for x in c.pos_b],
            rv_a=int(c.rv_a), rv_b=int(c.rv_b)))
    items.sort(key=lambda it: it.priority, reverse=True)
    for r, it in enumerate(items):
        it.rank = r
        if with_urls:
            it.ngl_url = site_url(np.array(it.pos_a), np.array(it.pos_b))
    return items


def site_url(pos_a: np.ndarray, pos_b: np.ndarray) -> str:
    """A Neuroglancer URL centred on the edit site, with the two sides as points."""
    from treestitch.ngl_export import build_neuroglancer_state, state_to_url
    pre = np.asarray([pos_a], float); post = np.asarray([pos_b], float)
    center = 0.5 * (np.asarray(pos_a, float) + np.asarray(pos_b, float))
    state = build_neuroglancer_state(pre, post, np.array([0], np.int64),
                                     center_nm=center)
    return state_to_url(state)


def write_queue_tsv(items: list[QueueItem], path: str) -> None:
    cols = ["rank", "kind", "priority", "p_same", "confident", "correct",
            "cutface_sim", "barrier", "site_x", "site_y", "site_z", "ngl_url"]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for it in items:
            cx, cy, cz = (0.5 * (np.array(it.pos_a) + np.array(it.pos_b))).tolist()
            f.write("\t".join(str(v) for v in [
                it.rank, it.kind, f"{it.priority:.4f}", f"{it.p_same:.4f}",
                int(it.confident), it.correct, f"{it.cutface_sim:.3f}",
                f"{it.barrier:.3f}", int(cx), int(cy), int(cz), it.ngl_url]) + "\n")


def topk_precision(items: list[QueueItem], k: int) -> dict:
    """Precision of the top-k *confident* auto-edits against ground truth."""
    conf = [it for it in items if it.confident and it.correct >= 0][:k]
    if not conf:
        return {"k": k, "n": 0, "precision": float("nan")}
    return {"k": k, "n": len(conf),
            "precision": float(np.mean([it.correct for it in conf]))}


def queue_summary(items: list[QueueItem]) -> str:
    n_cut = sum(it.kind == "CUT" for it in items)
    n_join = sum(it.kind == "JOIN" for it in items)
    n_abs = sum(it.kind == "ABSTAIN" for it in items)
    conf = [it for it in items if it.confident and it.correct >= 0]
    prec = float(np.mean([it.correct for it in conf])) if conf else float("nan")
    cov = len(conf) / max(1, len(items))
    lines = [f"queue: {len(items)} items  (CUT={n_cut} JOIN={n_join} ABSTAIN={n_abs})",
             f"auto-edit precision={prec:.3f} at coverage={cov:.2f} "
             f"({len(conf)}/{len(items)} confident)"]
    for k in (5, 10, 20):
        r = topk_precision(items, k)
        lines.append(f"  top-{k} auto-edit precision={r['precision']:.3f} (n={r['n']})")
    return "\n".join(lines)
