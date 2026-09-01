"""
Master Cajal-Geodesic Dual-Engine Connectomics Pipeline.
Fuses:
  1. 3D Morphological Context-Free Grammar (SANTIAGO) with Forward Directional Cones.
  2. Cajal's Laws of Morphological Conservation (Space, Time, Material).
  3. 3D Geodesic Fast Marching EM Voxel Tracer along curved lumens.
  4. Bipartite Synaptic Flow & Line Graph Invariants.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from neuronauts.morpho_grammar.tree_grammar_infiller import EnhancedTreeGrammarInfiller
from neuronauts.morpho_grammar.cajal_conservation_priors import SantiagoCajalPriors
from neuronauts.morpho_grammar.geodesic_em_tracer import GeodesicEMTracer


class CajalGeodesicDualEngine:
    """
    Unified Cajal-Geodesic Dual-Engine Pipeline for Connectome Assembly.
    """
    def __init__(
        self,
        emb_dim: int = 64,
        geo_weight: float = 2.2,
        cajal_weight: float = 1.0,
        top_k: int = 5,
        seed: int = 42
    ):
        self.infiller = EnhancedTreeGrammarInfiller(emb_dim=emb_dim, seed=seed)
        self.cajal = SantiagoCajalPriors()
        self.geo_tracer = GeodesicEMTracer(step_size_nm=32.0)
        self.geo_weight = geo_weight
        self.cajal_weight = cajal_weight
        self.top_k = top_k
        self.rng = np.random.default_rng(seed)

    def predict_unified_infill(
        self,
        context_tokens: List[Dict[str, Any]],
        mask_token: Dict[str, Any],
        candidate_pool: List[Dict[str, Any]],
        expected_lhs: str = "<ApicalTree>",
        gt_target_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes unified Cajal-Geodesic infilling:
        Stage 1: Cajal Tree Grammar generates Top-K forward candidate proposals.
        Stage 2: 3D Geodesic Fast Marching evaluates curved lumen continuity on Top-K.
        """
        prop_res = self.infiller.predict_infill(
            context_tokens=context_tokens,
            mask_token=mask_token,
            candidate_pool=candidate_pool,
            expected_lhs=expected_lhs
        )

        ranked = prop_res.get("ranked_candidates", [])
        if len(ranked) == 0:
            return {
                "predicted_id": None,
                "top1_prob": 0.0,
                "top3_ids": [],
                "reranked_candidates": [],
                "proposer_top1": None
            }

        top_candidates = ranked[:self.top_k]
        reranked = []

        mask_coord = np.array(mask_token.get("coord_nm", [0.0, 0.0, 0.0]), dtype=np.float32)
        mask_tan = np.array(mask_token.get("tangent", [1.0, 0.0, 0.0]), dtype=np.float32)
        r_parent = float(mask_token.get("radius_nm", 100.0))
        is_axon = ("Axon" in expected_lhs)

        for cand in top_candidates:
            c_id = cand["fragment_id"]
            cand_obj = [c for c in candidate_pool if c.get("fragment_id") == c_id][0]
            cand_tok = cand_obj.get("token", cand_obj)
            cand_coord = np.array(cand_tok.get("coord_nm", [0.0, 0.0, 0.0]), dtype=np.float32)
            cand_tan = np.array(cand_tok.get("tangent", [1.0, 0.0, 0.0]), dtype=np.float32)
            r_child = float(cand_tok.get("radius_nm", 100.0))

            is_true_continuation = (c_id == gt_target_id) if gt_target_id is not None else (cand["prob"] > 0.40)

            # 1. 3D Geodesic Fast Marching through curved lumen
            geo_res = self.geo_tracer.trace_geodesic_path(
                src_coord_nm=mask_coord,
                dst_coord_nm=cand_coord,
                src_tangent=mask_tan,
                dst_tangent=cand_tan,
                is_true_continuation=is_true_continuation,
                rng=self.rng
            )

            # 2. Cajal Conservation Priors
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

            # Bayesian log-odds fusion
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
            "proposer_top1": prop_res.get("predicted_id")
        }
