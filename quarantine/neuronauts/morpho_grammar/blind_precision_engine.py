"""
SOTA Precision-Recovery Dual Engine for Blind Connectomics.
Implements:
  1. Reciprocal Bipartite Consistency (Mutual Nearest Matching):
     p -> c* is valid iff c* -> p is also the top backward continuation.
  2. Confident Margin Gating (Delta_margin):
     Only commits merge if Score(Top-1) - Score(Top-2) >= tau_margin.
  3. Biological Terminal / Null Hypothesis (P_term):
     Models A -> Terminal with prior based on distance from soma and branch caliber.
  4. Hard Caliber and Directional Cone Constraints.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from neuronauts.morpho_grammar.blind_pcfg_morphology import BlindMorphologicalPCFG
from neuronauts.morpho_grammar.blind_geodesic_em_tracer import BlindGeodesicEMTracer
from neuronauts.morpho_grammar.cajal_conservation_priors import SantiagoCajalPriors
from neuronauts.morpho_grammar.tree_grammar_infiller import EnhancedTreeGrammarInfiller


class SOTAPrecisionEngine:
    """
    Unified SOTA Precision-Gated Dual Engine.
    """
    def __init__(
        self,
        emb_dim: int = 64,
        geo_weight: float = 2.5,
        cajal_weight: float = 1.5,
        margin_threshold: float = 0.85,
        seed: int = 42
    ):
        self.infiller = EnhancedTreeGrammarInfiller(emb_dim=emb_dim, seed=seed)
        self.cajal = SantiagoCajalPriors()
        self.geo_tracer = BlindGeodesicEMTracer(step_size_nm=32.0)
        self.pcfg = BlindMorphologicalPCFG()
        self.geo_weight = geo_weight
        self.cajal_weight = cajal_weight
        self.margin_threshold = margin_threshold
        self.rng = np.random.default_rng(seed)

    def compute_terminal_score(self, radius_nm: float, dist_from_soma_nm: float, is_axon: bool) -> float:
        """
        Computes the log-odds of a branch reaching a natural biological termination (Null Hypothesis).
        Thin branches far from soma have high termination log-odds.
        """
        if is_axon:
            # Axon terminals are very thin (r < 70 nm) and far from soma (> 150 um)
            p_term = (1.0 / (1.0 + np.exp((radius_nm - 70.0) / 15.0))) * (1.0 - np.exp(-dist_from_soma_nm / 150000.0))
        else:
            # Dendrite tufts taper to r < 120 nm around 120 um
            p_term = (1.0 / (1.0 + np.exp((radius_nm - 120.0) / 25.0))) * (1.0 - np.exp(-dist_from_soma_nm / 100000.0))

        p_term = float(np.clip(p_term, 0.05, 0.95))
        return float(np.log(p_term / (1.0 - p_term + 1e-7)))

    def score_single_pair(
        self,
        src_coord: np.ndarray,
        src_tan: np.ndarray,
        src_r: float,
        dst_coord: np.ndarray,
        dst_tan: np.ndarray,
        dst_r: float,
        p_grammar: float,
        is_axon: bool
    ) -> Dict[str, Any]:
        """
        Scores a candidate pair using continuous geometry, geodesic marching, and Cajal priors.
        """
        disp = dst_coord - src_coord
        d_nm = float(np.linalg.norm(disp))
        v_ray = disp / (d_nm + 1e-7)

        # 1. Hard Directional Cone & Caliber Gating
        align_ray = float(np.dot(src_tan, v_ray))
        align_tangents = float(np.dot(src_tan, dst_tan))
        caliber_ratio = abs(src_r - dst_r) / max(src_r, dst_r, 10.0)

        if align_ray < 0.35 or align_tangents < 0.25 or caliber_ratio > 0.70 or d_nm > 18000.0:
            return {"valid": False, "score": -999.0}

        # 2. 3D Geodesic Fast Marching
        geo_res = self.geo_tracer.trace_blind_geodesic_path(
            src_coord_nm=src_coord,
            dst_coord_nm=dst_coord,
            src_tangent=src_tan,
            dst_tangent=dst_tan,
            src_radius_nm=src_r,
            dst_radius_nm=dst_r
        )

        # 3. Cajal Conservation Priors
        p_time = self.cajal.compute_conduction_time_prior(
            centrifugal_order=2,
            dist_from_soma_nm=d_nm,
            is_axon=is_axon
        )

        p_gram = max(1e-4, p_grammar)
        p_geo = max(1e-4, geo_res["geodesic_score"])
        p_caj = max(1e-4, p_time)

        grammar_log_odds = float(np.log(p_gram / (1.0 - p_gram + 1e-7)))
        geo_log_odds = float(np.log(p_geo / (1.0 - p_geo + 1e-7)))
        cajal_log_odds = float(np.log(p_caj / (1.0 - p_caj + 1e-7)))

        combined_score = (
            grammar_log_odds +
            (self.geo_weight * geo_log_odds) +
            (self.cajal_weight * cajal_log_odds)
        )

        return {
            "valid": True,
            "score": combined_score,
            "p_geo": p_geo,
            "p_gram": p_gram,
            "tortuosity": geo_res["tortuosity"]
        }

    def predict_sota_precision_infill(
        self,
        parent_token: Dict[str, Any],
        mask_token: Dict[str, Any],
        candidate_pool: List[Dict[str, Any]],
        all_parent_endpoints: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Executes SOTA Precision-Gated Infilling with:
          1. Forward candidate scoring.
          2. Confident Margin Gating (Score1 - Score2 >= tau).
          3. Biological Terminal / Null Hypothesis testing.
          4. Reciprocal Backward Verification (Mutual Nearest Matching).
        """
        parent_symbol = parent_token.get("symbol", "[SOMA]")
        admissible_lhs_list = self.pcfg.derive_expected_lhs_from_parent(parent_symbol)
        is_axon = ("AXON" in parent_symbol)

        # Stage 1: Forward Grammar Proposals
        proposals = []
        for lhs in admissible_lhs_list:
            prop = self.infiller.predict_infill(
                context_tokens=[parent_token],
                mask_token=mask_token,
                candidate_pool=candidate_pool,
                expected_lhs=lhs
            )
            for cand in prop.get("ranked_candidates", []):
                proposals.append(cand)

        # Deduplicate and sort
        cands_dict = {}
        for c in proposals:
            cid = c["fragment_id"]
            if cid not in cands_dict or c["prob"] > cands_dict[cid]["prob"]:
                cands_dict[cid] = c
        sorted_cands = sorted(cands_dict.values(), key=lambda x: x["prob"], reverse=True)

        if len(sorted_cands) == 0:
            return {"predicted_id": None, "accepted": False, "top3_ids": []}

        mask_coord = np.array(mask_token.get("coord_nm", [0.0, 0.0, 0.0]), dtype=np.float32)
        mask_tan = np.array(mask_token.get("tangent", [1.0, 0.0, 0.0]), dtype=np.float32)
        r_parent = float(mask_token.get("radius_nm", 100.0))

        # Forward Scoring
        scored_candidates = []
        for cand in sorted_cands[:8]:
            cid = cand["fragment_id"]
            cand_obj = [c for c in candidate_pool if c.get("fragment_id") == cid][0]
            cand_tok = cand_obj.get("token", cand_obj)
            cand_coord = np.array(cand_tok.get("coord_nm", [0.0, 0.0, 0.0]), dtype=np.float32)
            cand_tan = np.array(cand_tok.get("tangent", [1.0, 0.0, 0.0]), dtype=np.float32)
            r_child = float(cand_tok.get("radius_nm", 100.0))

            res = self.score_single_pair(
                src_coord=mask_coord, src_tan=mask_tan, src_r=r_parent,
                dst_coord=cand_coord, dst_tan=cand_tan, dst_r=r_child,
                p_grammar=cand["prob"], is_axon=is_axon
            )

            if res["valid"]:
                scored_candidates.append({
                    "fragment_id": cid,
                    "score": res["score"],
                    "cand_coord": cand_coord,
                    "cand_tan": cand_tan,
                    "cand_r": r_child,
                    "prob": cand["prob"]
                })

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)

        if len(scored_candidates) == 0:
            return {"predicted_id": None, "accepted": False, "top3_ids": []}

        top1 = scored_candidates[0]
        top2_score = scored_candidates[1]["score"] if len(scored_candidates) > 1 else -999.0
        top3_ids = [c["fragment_id"] for c in scored_candidates[:3]]

        # Gate 1: Confident Margin Gating
        margin = top1["score"] - top2_score
        has_sufficient_margin = (len(scored_candidates) == 1) or (margin >= self.margin_threshold)

        # Gate 2: Biological Terminal / Null Hypothesis
        d_from_soma = float(np.linalg.norm(mask_coord - np.array(parent_token.get("coord_nm", mask_coord))))
        term_score = self.compute_terminal_score(r_parent, d_from_soma, is_axon)
        beats_terminal = (top1["score"] > term_score)

        # Gate 3: Reciprocal Backward Consistency (Mutual Nearest Match)
        is_reciprocal = True
        if all_parent_endpoints is not None and len(all_parent_endpoints) > 1:
            # Trace backwards from top1 candidate to all ambient cut interfaces
            backward_scores = []
            for ep in all_parent_endpoints:
                ep_coord = np.array(ep["coord_nm"], dtype=np.float32)
                ep_tan = np.array(ep["tangent"], dtype=np.float32)
                ep_r = float(ep.get("radius_nm", 100.0))
                # Backward ray direction
                back_res = self.score_single_pair(
                    src_coord=top1["cand_coord"], src_tan=-top1["cand_tan"], src_r=top1["cand_r"],
                    dst_coord=ep_coord, dst_tan=-ep_tan, dst_r=ep_r,
                    p_grammar=0.5, is_axon=is_axon
                )
                if back_res["valid"]:
                    backward_scores.append({"ep_id": ep["fragment_id"], "score": back_res["score"]})
            if len(backward_scores) > 0:
                backward_scores.sort(key=lambda x: x["score"], reverse=True)
                # Reciprocal match must choose this parent mask
                is_reciprocal = (backward_scores[0]["ep_id"] == mask_token["fragment_id"])

        # Final Acceptance
        accepted = (has_sufficient_margin and beats_terminal and is_reciprocal and top1["score"] > -1.0)

        return {
            "predicted_id": top1["fragment_id"] if accepted else None,
            "raw_top1_id": top1["fragment_id"],
            "accepted": accepted,
            "top1_score": top1["score"],
            "margin": margin,
            "top3_ids": top3_ids
        }
