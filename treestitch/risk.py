"""Risk-aware decision layer for probabilistic neuron partition.

Takes soft_partition output and applies asymmetric expected-loss weighting to
produce a per-observation review priority score and an optimal action
(MERGE / SPLIT / ABSTAIN) under configurable merge/split cost assumptions.

Usage
-----
    from treestitch.risk import decision_layer, review_queue

    soft = partition_observations_soft(model, graph, bias=-2.0)
    decisions = decision_layer(soft, graph, cost_merge=5.0, cost_split=1.0)
    queue = review_queue(decisions, frags, graph.fragment_id, top_k=20)
    for item in queue:
        print(item)

Cost semantics
--------------
cost_merge : float
    Relative cost of a false merge (incorrectly joining two neurons).
    Over-merge is the expensive, hard-to-undo error in connectomics — a merged
    neuron requires tracing back to the merge boundary. Default 5.0.
cost_split : float
    Relative cost of a false split (failing to merge two fragments that belong
    to the same neuron). Under-merge can be corrected by a later stitching pass.
    Default 1.0.

Expected loss per observation
------------------------------
    E[loss_i] = P(wrong_merge_i) × cost_merge + P(wrong_split_i) × cost_split

where:
    P(wrong_merge_i)  = cluster_conf_i < 0  → the observation has stronger
                        evidence for a different cluster than its current one.
                        Proxy: max(0, -cluster_conf_i) / (cost_merge + cost_split)
    P(wrong_split_i)  = entropy_i is high → observation is uncertain about
                        which cluster it belongs to, risk of it being under-merged.

In practice we use the soft partition fields directly:
    risk_merge_i  = max(0, -cluster_conf_i)          (scale 0–1)
    risk_split_i  = entropy_i / log(K)               (normalised, 0–1)
    expected_loss = cost_merge × risk_merge + cost_split × risk_split
    priority_score = expected_loss                    (higher = review first)

Optimal action
--------------
    CONFIDENT_MERGE  : cluster_conf > conf_threshold  → keep assignment
    REVIEW_MERGE     : risk_merge > merge_risk_threshold → flag for human review
    REVIEW_SPLIT     : risk_split > split_risk_threshold → flag as possible under-merge
    ABSTAIN          : abstain_mask is True (observation already unassigned)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ObservationDecision:
    obs_idx: int
    fragment_id: int
    pred_cluster: int
    action: str           # CONFIDENT_MERGE | REVIEW_MERGE | REVIEW_SPLIT | ABSTAIN
    priority_score: float
    cluster_conf: float
    entropy: float
    risk_merge: float
    risk_split: float
    expected_loss: float
    calibrated_conf: float = float("nan")  # set by decision_layer if T is provided


def decision_layer(
    soft: dict,
    *,
    cost_merge: float = 5.0,
    cost_split: float = 1.0,
    conf_threshold: float = 0.3,
    merge_risk_threshold: float = 0.15,
    split_risk_threshold: float = 0.40,
    fragment_ids: np.ndarray | None = None,
    calibrated_confs: np.ndarray | None = None,
) -> list[ObservationDecision]:
    """Apply asymmetric expected-loss weighting to soft partition output.

    Parameters
    ----------
    soft:
        Return value of `partition_observations_soft`.
    cost_merge:
        Relative cost of a false merge. Default 5.0.
    cost_split:
        Relative cost of a false split. Default 1.0.
    conf_threshold:
        cluster_conf above which an observation is considered confident.
    merge_risk_threshold:
        risk_merge above which the observation is flagged REVIEW_MERGE.
    split_risk_threshold:
        normalised entropy above which the observation is flagged REVIEW_SPLIT.
    fragment_ids:
        Optional [N] int64 array of v117 fragment IDs per observation.
    calibrated_confs:
        Optional [N] float array from `treestitch.calibration.calibrated_obs_confidence`.
        When provided, stored as `ObservationDecision.calibrated_conf`.

    Returns
    -------
    List[ObservationDecision] — one per observation, sorted by descending
    priority_score (highest-risk observations first).
    """
    pred = soft["pred"]
    cluster_conf = soft["cluster_conf"]
    entropy = soft["entropy"]
    abstain_mask = soft["abstain_mask"]
    membership = soft.get("membership_probs")

    N = len(pred)
    K = membership.shape[1] if membership is not None and membership.ndim == 2 else 1
    log_K = math.log(max(K, 2))

    results: list[ObservationDecision] = []
    for i in range(N):
        cc = float(cluster_conf[i])
        ent = float(entropy[i])
        abstained = bool(abstain_mask[i]) if abstain_mask is not None else (pred[i] < 0)

        risk_merge = float(max(0.0, -cc))
        risk_split = ent / log_K if log_K > 0 else 0.0
        expected_loss = cost_merge * risk_merge + cost_split * risk_split

        if abstained:
            action = "ABSTAIN"
        elif cc > conf_threshold:
            action = "CONFIDENT_MERGE"
        elif risk_merge > merge_risk_threshold:
            action = "REVIEW_MERGE"
        elif risk_split > split_risk_threshold:
            action = "REVIEW_SPLIT"
        else:
            action = "CONFIDENT_MERGE"

        fid = int(fragment_ids[i]) if fragment_ids is not None else -1
        cal = float(calibrated_confs[i]) if calibrated_confs is not None else float("nan")
        results.append(ObservationDecision(
            obs_idx=i,
            fragment_id=fid,
            pred_cluster=int(pred[i]),
            action=action,
            priority_score=expected_loss,
            cluster_conf=cc,
            entropy=ent,
            risk_merge=risk_merge,
            risk_split=risk_split,
            expected_loss=expected_loss,
            calibrated_conf=cal,
        ))

    results.sort(key=lambda d: d.priority_score, reverse=True)
    return results


def review_queue(
    decisions: list[ObservationDecision],
    *,
    top_k: int = 50,
    actions: tuple[str, ...] = ("REVIEW_MERGE", "REVIEW_SPLIT", "ABSTAIN"),
) -> list[ObservationDecision]:
    """Return the top-k highest-priority observations that need human review.

    Filters to the requested action types and returns the top_k by
    priority_score (descending). Suitable for driving a reviewer interface:
    the first item in the queue is the single observation where the model is
    most uncertain or most likely to be wrong, under the given cost assumptions.
    """
    flagged = [d for d in decisions if d.action in actions]
    return flagged[:top_k]


def fragment_risk_summary(
    decisions: list[ObservationDecision],
) -> dict[int, dict]:
    """Aggregate observation-level risk scores to fragment level.

    Returns {fragment_id: {"mean_priority": float, "n_obs": int,
                            "n_review_merge": int, "n_review_split": int,
                            "max_priority": float}} sorted by max_priority desc.

    Useful for identifying which v117 fragments (not just individual synapses)
    are most likely to be mis-partitioned — drives a fragment-level review queue
    rather than a synapse-level one.
    """
    from collections import defaultdict

    agg: dict[int, dict] = defaultdict(lambda: {
        "priorities": [], "n_review_merge": 0, "n_review_split": 0, "n_obs": 0,
    })
    for d in decisions:
        fid = d.fragment_id
        agg[fid]["priorities"].append(d.priority_score)
        agg[fid]["n_obs"] += 1
        if d.action == "REVIEW_MERGE":
            agg[fid]["n_review_merge"] += 1
        elif d.action == "REVIEW_SPLIT":
            agg[fid]["n_review_split"] += 1

    result = {}
    for fid, data in agg.items():
        p = data["priorities"]
        result[fid] = {
            "mean_priority": float(np.mean(p)),
            "max_priority": float(np.max(p)),
            "n_obs": data["n_obs"],
            "n_review_merge": data["n_review_merge"],
            "n_review_split": data["n_review_split"],
        }

    return dict(sorted(result.items(), key=lambda kv: kv[1]["max_priority"], reverse=True))


def risk_summary_str(decisions: list[ObservationDecision]) -> str:
    """One-line summary of the decision distribution for logging."""
    counts: dict[str, int] = {}
    for d in decisions:
        counts[d.action] = counts.get(d.action, 0) + 1
    total = len(decisions)
    parts = [f"{k}={v} ({v/total:.0%})" for k, v in sorted(counts.items())]
    flagged = counts.get("REVIEW_MERGE", 0) + counts.get("REVIEW_SPLIT", 0) + counts.get("ABSTAIN", 0)
    return f"{total} obs → " + ", ".join(parts) + f" | {flagged} flagged for review"
