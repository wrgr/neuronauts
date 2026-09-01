"""
Agentic Actor-Critic Morphological Infilling & Verification Engine (Calibrated EXP-036).
Combines:
  1. MorphoActor: Generates syntactic derivation hypotheses from 3D PCFG + Transformer Pointer Attention.
  2. MorphoCriticJudge: Validates physical/biological invariants (Geodesic EM Flux, Cajal Space & Material, Syntax).
  3. AgenticConnectomeAssembler: Iterative hypothesis testing and diagnostic refinement loop.
100% blind at inference without ground truth.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from neuronauts.morpho_grammar.blind_pcfg_morphology import BlindMorphologicalPCFG
from neuronauts.morpho_grammar.blind_geodesic_em_tracer import BlindGeodesicEMTracer
from neuronauts.morpho_grammar.cajal_conservation_priors import SantiagoCajalPriors
from neuronauts.morpho_grammar.tree_grammar_infiller import EnhancedTreeGrammarInfiller


class MorphoActor:
    """
    Proposer Agent: Generates candidate continuation subtrees from the 3D morphological grammar.
    """
    def __init__(self, emb_dim: int = 64, seed: int = 42):
        self.infiller = EnhancedTreeGrammarInfiller(emb_dim=emb_dim, seed=seed)
        self.pcfg = BlindMorphologicalPCFG()

    def propose_continuation_beam(
        self,
        parent_token: Dict[str, Any],
        mask_token: Dict[str, Any],
        candidate_pool: List[Dict[str, Any]],
        excluded_ids: Optional[set] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generates Top-K grammatically admissible candidate continuations excluding previously rejected IDs.
        """
        excluded_ids = excluded_ids or set()
        parent_symbol = parent_token.get("symbol", "[SOMA]")
        admissible_lhs_list = self.pcfg.derive_expected_lhs_from_parent(parent_symbol)

        all_proposals = []
        for expected_lhs in admissible_lhs_list:
            prop = self.infiller.predict_infill(
                context_tokens=[parent_token],
                mask_token=mask_token,
                candidate_pool=candidate_pool,
                expected_lhs=expected_lhs
            )
            for cand in prop.get("ranked_candidates", []):
                if cand["fragment_id"] not in excluded_ids:
                    all_proposals.append(cand)

        # Deduplicate and sort by grammar proposal probability
        cand_dict = {}
        for c in all_proposals:
            cid = c["fragment_id"]
            if cid not in cand_dict or c["prob"] > cand_dict[cid]["prob"]:
                cand_dict[cid] = c

        sorted_cands = sorted(cand_dict.values(), key=lambda x: x["prob"], reverse=True)
        return sorted_cands[:top_k]


class MorphoCriticJudge:
    """
    Verifier / Judge Agent: Evaluates physical and biological conservation invariants.
    """
    def __init__(
        self,
        geo_weight: float = 2.5,
        cajal_weight: float = 1.5,
        value_threshold: float = 0.35
    ):
        self.cajal = SantiagoCajalPriors()
        self.geo_tracer = BlindGeodesicEMTracer(step_size_nm=32.0)
        self.geo_weight = geo_weight
        self.cajal_weight = cajal_weight
        self.value_threshold = value_threshold

    def evaluate_hypothesis(
        self,
        parent_token: Dict[str, Any],
        mask_token: Dict[str, Any],
        candidate_token: Dict[str, Any],
        p_grammar: float
    ) -> Dict[str, Any]:
        """
        Multi-head evaluation of candidate hypothesis across orthogonal invariants.
        """
        mask_coord = np.array(mask_token.get("coord_nm", [0.0, 0.0, 0.0]), dtype=np.float32)
        mask_tan = np.array(mask_token.get("tangent", [1.0, 0.0, 0.0]), dtype=np.float32)
        r_parent = float(mask_token.get("radius_nm", 100.0))

        cand_coord = np.array(candidate_token.get("coord_nm", [0.0, 0.0, 0.0]), dtype=np.float32)
        cand_tan = np.array(candidate_token.get("tangent", [1.0, 0.0, 0.0]), dtype=np.float32)
        r_child = float(candidate_token.get("radius_nm", 100.0))

        parent_symbol = parent_token.get("symbol", "[SOMA]")
        is_axon = ("AXON" in parent_symbol)

        critiques = []
        is_hard_rejected = False

        # Invariant 1: Directional Ray Cone
        disp = cand_coord - mask_coord
        d_nm = float(np.linalg.norm(disp))
        v_ray = disp / (d_nm + 1e-7)
        align_ray = float(np.dot(mask_tan, v_ray))

        if align_ray < 0.05:
            critiques.append("BACKWARDS_RAY_ALIGNMENT")
            is_hard_rejected = True

        if d_nm > 38000.0:
            critiques.append("EXCESSIVE_DISTANCE")
            is_hard_rejected = True

        # Invariant 2: Murray Caliber Scaling
        caliber_ratio = abs(r_parent - r_child) / max(r_parent, r_child, 10.0)
        if caliber_ratio > 0.85:
            critiques.append("CALIBER_DISPARITY_ANOMALY")
            if caliber_ratio > 0.95:
                is_hard_rejected = True

        # Invariant 3: 3D Geodesic EM Fast Marching Path Integral
        geo_res = self.geo_tracer.trace_blind_geodesic_path(
            src_coord_nm=mask_coord,
            dst_coord_nm=cand_coord,
            src_tangent=mask_tan,
            dst_tangent=cand_tan,
            src_radius_nm=r_parent,
            dst_radius_nm=r_child
        )

        p_geo = max(1e-4, geo_res["geodesic_score"])
        if p_geo < 0.15:
            critiques.append("MEMBRANE_CROSSING_BREACH")

        # Invariant 4: Cajal Conduction Time Conservation
        p_time = self.cajal.compute_conduction_time_prior(
            centrifugal_order=2,
            dist_from_soma_nm=d_nm,
            is_axon=is_axon
        )
        p_cajal = max(1e-4, p_time)

        # Invariant 5: Synapse Polarity Balance
        n_pre = candidate_token.get("n_syn_pre", 0)
        n_post = candidate_token.get("n_syn_post", 0)
        tot_syn = n_pre + n_post
        if tot_syn > 0:
            p_syn_match = (n_pre / tot_syn) if is_axon else (n_post / tot_syn)
            p_syn = float(np.clip(p_syn_match, 0.20, 0.95))
        else:
            p_syn = 0.50

        # Unified Value Function V(H) via Log-Odds Fusion
        p_gram = max(1e-4, p_grammar)
        g_odds = float(np.log(p_gram / (1.0 - p_gram + 1e-7)))
        geo_odds = float(np.log(p_geo / (1.0 - p_geo + 1e-7)))
        caj_odds = float(np.log(p_cajal / (1.0 - p_cajal + 1e-7)))
        syn_odds = float(np.log(p_syn / (1.0 - p_syn + 1e-7)))

        combined_score = g_odds + (self.geo_weight * geo_odds) + (self.cajal_weight * caj_odds) + (0.5 * syn_odds)
        value_prob = 1.0 / (1.0 + np.exp(-combined_score))

        accepted = (value_prob >= self.value_threshold) and (not is_hard_rejected)

        return {
            "accepted": accepted,
            "value_prob": float(value_prob),
            "combined_score": float(combined_score),
            "p_grammar": p_gram,
            "p_geo": p_geo,
            "p_cajal": p_cajal,
            "d_nm": d_nm,
            "critiques": critiques,
            "tortuosity": geo_res["tortuosity"]
        }


class AgenticConnectomeAssembler:
    """
    Agentic Multi-Turn Hypothesis Testing & Refinement Loop.
    """
    def __init__(
        self,
        emb_dim: int = 64,
        max_iterations: int = 3,
        geo_weight: float = 2.5,
        cajal_weight: float = 1.5,
        value_threshold: float = 0.35,
        seed: int = 42
    ):
        self.actor = MorphoActor(emb_dim=emb_dim, seed=seed)
        self.critic = MorphoCriticJudge(
            geo_weight=geo_weight,
            cajal_weight=cajal_weight,
            value_threshold=value_threshold
        )
        self.max_iterations = max_iterations

    def run_agentic_infill(
        self,
        parent_token: Dict[str, Any],
        mask_token: Dict[str, Any],
        candidate_pool: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Executes multi-turn hypothesis testing:
        Actor proposes -> Critic evaluates -> If rejected, Actor refines candidate search.
        """
        excluded_ids = {parent_token.get("fragment_id")}
        history = []
        top3_pool = []

        for step in range(self.max_iterations):
            proposals = self.actor.propose_continuation_beam(
                parent_token=parent_token,
                mask_token=mask_token,
                candidate_pool=candidate_pool,
                excluded_ids=excluded_ids,
                top_k=3
            )

            if len(proposals) == 0:
                break

            if len(top3_pool) == 0:
                top3_pool = [p["fragment_id"] for p in proposals[:3]]

            # Evaluate top proposed candidate with Critic
            top_cand = proposals[0]
            cid = top_cand["fragment_id"]
            cand_tok = [c for c in candidate_pool if c.get("fragment_id") == cid][0]
            cand_token_obj = cand_tok.get("token", cand_tok)

            eval_res = self.critic.evaluate_hypothesis(
                parent_token=parent_token,
                mask_token=mask_token,
                candidate_token=cand_token_obj,
                p_grammar=top_cand["prob"]
            )

            history.append({
                "step": step + 1,
                "fragment_id": cid,
                "eval_res": eval_res
            })

            if eval_res["accepted"]:
                return {
                    "predicted_id": cid,
                    "accepted": True,
                    "top1_score": eval_res["combined_score"],
                    "value_prob": eval_res["value_prob"],
                    "iterations_used": step + 1,
                    "top3_ids": top3_pool,
                    "history": history
                }
            else:
                # Add to negative constraints and refine
                excluded_ids.add(cid)

        # Fallback: Best candidate from history if above soft floor
        if len(history) > 0:
            history.sort(key=lambda h: h["eval_res"]["combined_score"], reverse=True)
            best_h = history[0]
            accepted = (best_h["eval_res"]["combined_score"] >= -2.0)
            return {
                "predicted_id": best_h["fragment_id"] if accepted else None,
                "accepted": accepted,
                "top1_score": best_h["eval_res"]["combined_score"],
                "value_prob": best_h["eval_res"]["value_prob"],
                "iterations_used": len(history),
                "top3_ids": top3_pool,
                "history": history
            }

        return {
            "predicted_id": None,
            "accepted": False,
            "top1_score": -999.0,
            "value_prob": 0.0,
            "iterations_used": 0,
            "top3_ids": [],
            "history": []
        }
