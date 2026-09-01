"""
Global Multi-Hypothesis Tree Search & Calibrated Decision Confidence (EXP-044).
Features:
  1. Posterior Merge Probability: C(A -> B) = sigma(S(A -> B)).
  2. Decision Margin (Epistemic Ambiguity): Delta C = C(top1) - C(top2).
  3. Ambiguous Branch Forking: Generates K=3-5 competing global tree hypotheses {T_1, ..., T_K}.
  4. Global Whole-Cell Biological Energy Ranking:
       E(T) = w_cajal * ConductionCost(T) + w_murray * CaliberSmoothness(T) + w_syn * BipartiteConsistency(T).
  5. Global Pareto-Optimal Tree Selection.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from collections import defaultdict


class GlobalMultiHypothesisTreeSearch:
    """
    Manages global multi-hypothesis generation and whole-cell physical ranking.
    """
    def __init__(
        self,
        high_conf_thresh: float = 0.75,
        margin_thresh: float = 0.25,
        k_hypotheses: int = 4,
        w_cajal: float = 1.2,
        w_murray: float = 1.5,
        w_syn: float = 1.0,
        seed: int = 42
    ):
        self.high_conf_thresh = high_conf_thresh
        self.margin_thresh = margin_thresh
        self.k_hypotheses = k_hypotheses
        self.w_cajal = w_cajal
        self.w_murray = w_murray
        self.w_syn = w_syn
        self.rng = np.random.default_rng(seed)

    def compute_decision_confidence(
        self,
        raw_score: float,
        runner_up_score: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Converts raw Bayesian log-odds score into posterior probability C and decision margin Delta C.
        """
        # Sigmoid calibration
        conf = float(1.0 / (1.0 + np.exp(-raw_score)))
        if runner_up_score is not None:
            runner_up_conf = float(1.0 / (1.0 + np.exp(-runner_up_score)))
            margin = float(conf - runner_up_conf)
        else:
            margin = float(conf)
        return conf, margin

    def score_global_tree_hypothesis(
        self,
        links: List[Tuple[str, str]],
        tokens_dict: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Evaluates a complete global tree hypothesis on whole-cell physical invariants:
          1. Cajal Conduction Delay (Distance from soma)
          2. Murray Caliber Monotonicity (Radius step taper)
          3. Synaptic Bipartite Consistency
        """
        if len(links) == 0:
            return 0.0

        total_conduction_cost = 0.0
        caliber_penalty = 0.0
        syn_consistency = 0.0

        for u, v in links:
            tok_u = tokens_dict.get(u)
            tok_v = tokens_dict.get(v)
            if tok_u is None or tok_v is None:
                continue

            coord_u = np.array(tok_u.get("coord_nm", [0.0, 0.0, 0.0]), dtype=np.float32)
            coord_v = np.array(tok_v.get("coord_nm", [0.0, 0.0, 0.0]), dtype=np.float32)
            r_u = float(tok_u.get("radius_nm", 100.0))
            r_v = float(tok_v.get("radius_nm", 100.0))

            dist_nm = float(np.linalg.norm(coord_u - coord_v))
            total_conduction_cost += dist_nm / 10000.0

            # Murray Caliber step penalty (expansion > 1.35x is penalized)
            ratio = r_v / (r_u + 1e-5)
            if ratio > 1.35:
                caliber_penalty += (ratio - 1.35) * 2.0

            # Synaptic consistency
            part_u = set(tok_u.get("syn_partners", []))
            part_v = set(tok_v.get("syn_partners", []))
            if len(part_u) > 0 and len(part_v) > 0:
                inter = len(part_u.intersection(part_v))
                union = len(part_u.union(part_v))
                if union > 0:
                    syn_consistency += (inter / union) * 3.0

        # Global energy score (higher is better)
        global_energy = (
            syn_consistency
            - (self.w_cajal * total_conduction_cost)
            - (self.w_murray * caliber_penalty)
        )
        return float(global_energy)

    def assemble_global_optimal_tree(
        self,
        decision_points: List[Dict[str, Any]],
        tokens_dict: Dict[str, Dict[str, Any]]
    ) -> List[Tuple[str, str]]:
        """
        Builds K candidate global trees across ambiguous branching choices,
        scores them globally, and returns the Pareto-optimal join set.
        """
        # Separate high-confidence deterministic joins from ambiguous choices
        deterministic_joins = []
        ambiguous_branches = []

        for dp in decision_points:
            cands = dp.get("candidates", [])
            if len(cands) == 0:
                continue

            top1 = cands[0]
            top2_score = cands[1]["score"] if len(cands) > 1 else None
            conf, margin = self.compute_decision_confidence(top1["score"], top2_score)

            if conf >= self.high_conf_thresh and margin >= self.margin_thresh:
                deterministic_joins.append((dp["parent_id"], top1["fragment_id"]))
            elif conf >= 0.35:
                ambiguous_branches.append({
                    "parent_id": dp["parent_id"],
                    "options": [c["fragment_id"] for c in cands[:min(len(cands), self.k_hypotheses)]]
                })

        if len(ambiguous_branches) == 0:
            return deterministic_joins

        # Generate K candidate global hypotheses
        hypotheses = []
        
        # Hypothesis 0: Greedy top-1
        hypo_0 = list(deterministic_joins)
        for ab in ambiguous_branches:
            if len(ab["options"]) > 0:
                hypo_0.append((ab["parent_id"], ab["options"][0]))
        hypotheses.append(hypo_0)

        # Hypothesis 1: Conservative (only high-confidence joins)
        hypotheses.append(list(deterministic_joins))

        # Hypotheses 2..K: Explore top-2 and top-3 permutations
        for k_idx in range(1, min(self.k_hypotheses, 3)):
            hypo_k = list(deterministic_joins)
            for ab in ambiguous_branches:
                opt_idx = min(k_idx, len(ab["options"]) - 1)
                hypo_k.append((ab["parent_id"], ab["options"][opt_idx]))
            hypotheses.append(hypo_k)

        # Evaluate and score all K candidate global trees
        best_tree = hypo_0
        best_score = -float("inf")

        for hypo in hypotheses:
            score = self.score_global_tree_hypothesis(hypo, tokens_dict)
            if score > best_score:
                best_score = score
                best_tree = hypo

        return best_tree
