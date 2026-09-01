"""
Master Blind Cajal-Geodesic Dual-Engine (Strict Zero-Leakage Implementation).
Zero target labels, zero ground-truth compartment cheating, zero target ID shortcuts.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from neuronauts.morpho_grammar.tree_grammar_infiller import EnhancedTreeGrammarInfiller
from neuronauts.morpho_grammar.cajal_conservation_priors import SantiagoCajalPriors
from neuronauts.morpho_grammar.blind_geodesic_em_tracer import BlindGeodesicEMTracer
from neuronauts.morpho_grammar.blind_pcfg_morphology import BlindMorphologicalPCFG


class BlindCajalGeodesicDualEngine:
    """
    Unified 100% Blind Cajal-Geodesic Dual-Engine Pipeline.
    """
    def __init__(
        self,
        emb_dim: int = 64,
        geo_weight: float = 2.2,
        cajal_weight: float = 1.2,
        top_k: int = 5,
        seed: int = 42
    ):
        self.infiller = EnhancedTreeGrammarInfiller(emb_dim=emb_dim, seed=seed)
        self.cajal = SantiagoCajalPriors()
        self.geo_tracer = BlindGeodesicEMTracer(step_size_nm=32.0)
        self.pcfg = BlindMorphologicalPCFG()
        self.geo_weight = geo_weight
        self.cajal_weight = cajal_weight
        self.top_k = top_k
        self.rng = np.random.default_rng(seed)

    def predict_blind_infill(
        self,
        context_tokens: List[Dict[str, Any]],
        mask_token: Dict[str, Any],
        candidate_pool: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Executes 100% blind infilling without target labels:
        1. Infers allowable LHS non-terminals strictly from parent node in context.
        2. Proposes Top-K forward continuations using continuous-discrete pointer attention.
        3. Reranks Top-K via Hermite geodesic path integrals and Cajal conservation priors.
        """
        # Step 1: Infer allowable LHS from parent symbol
        parent_symbol = context_tokens[-1].get("symbol", "[SOMA]") if len(context_tokens) > 0 else "[SOMA]"
        admissible_lhs_list = self.pcfg.derive_expected_lhs_from_parent(parent_symbol)

        # Stage 1: Grammar Proposals across admissible LHS
        best_proposals = []
        for expected_lhs in admissible_lhs_list:
            prop = self.infiller.predict_infill(
                context_tokens=context_tokens,
                mask_token=mask_token,
                candidate_pool=candidate_pool,
                expected_lhs=expected_lhs
            )
            for cand in prop.get("ranked_candidates", []):
                best_proposals.append(cand)

        # Deduplicate and sort by raw proposal probability
        cand_dict = {}
        for c in best_proposals:
            cid = c["fragment_id"]
            if cid not in cand_dict or c["prob"] > cand_dict[cid]["prob"]:
                cand_dict[cid] = c
        sorted_cands = sorted(cand_dict.values(), key=lambda x: x["prob"], reverse=True)

        if len(sorted_cands) == 0:
            return {
                "predicted_id": None,
                "top1_score": -999.0,
                "top3_ids": [],
                "reranked_candidates": [],
                "proposer_top1": None
            }

        top_candidates = sorted_cands[:self.top_k]
        reranked = []

        mask_coord = np.array(mask_token.get("coord_nm", [0.0, 0.0, 0.0]), dtype=np.float32)
        mask_tan = np.array(mask_token.get("tangent", [1.0, 0.0, 0.0]), dtype=np.float32)
        r_parent = float(mask_token.get("radius_nm", 100.0))
        is_axon = ("AXON" in parent_symbol)

        for cand in top_candidates:
            c_id = cand["fragment_id"]
            cand_obj = [c for c in candidate_pool if c.get("fragment_id") == c_id][0]
            cand_tok = cand_obj.get("token", cand_obj)
            cand_coord = np.array(cand_tok.get("coord_nm", [0.0, 0.0, 0.0]), dtype=np.float32)
            cand_tan = np.array(cand_tok.get("tangent", [1.0, 0.0, 0.0]), dtype=np.float32)
            r_child = float(cand_tok.get("radius_nm", 100.0))

            # Stage 2: 100% Blind 3D Geodesic Fast Marching line integral
            geo_res = self.geo_tracer.trace_blind_geodesic_path(
                src_coord_nm=mask_coord,
                dst_coord_nm=cand_coord,
                src_tangent=mask_tan,
                dst_tangent=cand_tan,
                src_radius_nm=r_parent,
                dst_radius_nm=r_child
            )

            # Cajal Conservation Priors
            disp = cand_coord - mask_coord
            d_nm = float(np.linalg.norm(disp))
            p_time = self.cajal.compute_conduction_time_prior(
                centrifugal_order=2,
                dist_from_soma_nm=d_nm,
                is_axon=is_axon
            )

            p_grammar = max(1e-4, cand["prob"])
            p_geo = max(1e-4, geo_res["geodesic_score"])
            p_cajal = max(1e-4, p_time)

            # Bayesian Log-Odds Fusion
            grammar_log_odds = float(np.log(p_grammar / (1.0 - p_grammar + 1e-7)))
            geo_log_odds = float(np.log(p_geo / (1.0 - p_geo + 1e-7)))
            cajal_log_odds = float(np.log(p_cajal / (1.0 - p_cajal + 1e-7)))

            combined_score = (
                grammar_log_odds +
                (self.geo_weight * geo_log_odds) +
                (self.cajal_weight * cajal_log_odds)
            )

            reranked.append({
                "fragment_id": c_id,
                "p_grammar": p_grammar,
                "p_geodesic": p_geo,
                "p_cajal": p_cajal,
                "tortuosity": geo_res["tortuosity"],
                "combined_score": combined_score
            })

        reranked.sort(key=lambda x: x["combined_score"], reverse=True)
        top1 = reranked[0]

        return {
            "predicted_id": top1["fragment_id"],
            "top1_score": top1["combined_score"],
            "top3_ids": [r["fragment_id"] for r in reranked[:3]],
            "reranked_candidates": reranked,
            "proposer_top1": sorted_cands[0]["fragment_id"]
        }
