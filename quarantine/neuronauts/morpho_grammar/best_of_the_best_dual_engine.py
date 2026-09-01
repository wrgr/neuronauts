"""
Best-of-the-Best Unified Dual-Engine (EXP-034).
Combines:
  1. Closest-Surface-Vertex Candidate Matching (reverts distal leaf error).
  2. Continuous-Discrete 3D Transformer Infiller.
  3. 3D Fast Marching Geodesic EM Tracer with Hermite Spline Flux Line Integrals.
  4. Santiago Ramon y Cajal Conservation Laws (Space, Time, Material).
  5. Bayesian Soft Synapse Polarity Likelihoods (n_pre vs n_post).
  6. Calibrated Smooth Logistic Acceptance.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from neuronauts.morpho_grammar.blind_pcfg_morphology import BlindMorphologicalPCFG
from neuronauts.morpho_grammar.blind_geodesic_em_tracer import BlindGeodesicEMTracer
from neuronauts.morpho_grammar.cajal_conservation_priors import SantiagoCajalPriors
from neuronauts.morpho_grammar.tree_grammar_infiller import EnhancedTreeGrammarInfiller


class BestOfTheBestDualEngine:
    """
    Unified, High-Performance Dual-Engine combining the best mechanisms from EXP-027, EXP-029, and EXP-033.
    """
    def __init__(
        self,
        emb_dim: int = 64,
        geo_weight: float = 2.5,
        cajal_weight: float = 1.5,
        acceptance_threshold: float = -0.5,
        seed: int = 42
    ):
        self.infiller = EnhancedTreeGrammarInfiller(emb_dim=emb_dim, seed=seed)
        self.cajal = SantiagoCajalPriors()
        self.geo_tracer = BlindGeodesicEMTracer(step_size_nm=32.0)
        self.pcfg = BlindMorphologicalPCFG()
        self.geo_weight = geo_weight
        self.cajal_weight = cajal_weight
        self.acceptance_threshold = acceptance_threshold
        self.rng = np.random.default_rng(seed)

    def find_closest_candidate_vertex(
        self,
        cut_pos: np.ndarray,
        cand_verts: np.ndarray,
        cand_radii: np.ndarray,
        cand_edges: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Finds the closest surface vertex on the candidate fragment to the parent cut interface,
        along with its local branch tangent and radius.
        """
        if len(cand_verts) == 0:
            return cut_pos, np.array([1.0, 0.0, 0.0]), 100.0, 99999.0

        diffs = cand_verts - cut_pos
        dists = np.linalg.norm(diffs, axis=1)
        best_idx = int(np.argmin(dists))
        best_dist = float(dists[best_idx])
        best_coord = cand_verts[best_idx]
        best_r = float(cand_radii[best_idx]) if len(cand_radii) > best_idx else 100.0

        # Compute local tangent at best_idx
        conn = cand_edges[(cand_edges[:, 0] == best_idx) | (cand_edges[:, 1] == best_idx)]
        if len(conn) > 0:
            nbr = conn[0, 1] if conn[0, 0] == best_idx else conn[0, 0]
            disp = best_coord - cand_verts[nbr]
            norm_d = np.linalg.norm(disp)
            tan = disp / norm_d if norm_d > 0 else np.array([1.0, 0.0, 0.0])
        else:
            tan = np.array([1.0, 0.0, 0.0])

        return best_coord, tan, best_r, best_dist

    def score_candidate(
        self,
        cut_coord: np.ndarray,
        cut_tan: np.ndarray,
        cut_r: float,
        cand_piece: Dict[str, Any],
        p_grammar: float,
        is_axon_parent: bool
    ) -> Dict[str, Any]:
        """
        Scores a candidate fragment using closest-vertex geometry, geodesic marching, and Cajal priors.
        """
        pv = cand_piece["verts"]
        pr = cand_piece["radii"]
        pe = cand_piece["edges"]

        c_coord, c_tan, c_r, d_nm = self.find_closest_candidate_vertex(cut_coord, pv, pr, pe)

        disp = c_coord - cut_coord
        v_ray = disp / (d_nm + 1e-7)

        # 1. Directional Forward-Cone Gating
        align_ray = float(np.dot(cut_tan, v_ray))
        caliber_ratio = abs(cut_r - c_r) / max(cut_r, c_r, 10.0)

        if align_ray < 0.15 or d_nm > 28000.0 or caliber_ratio > 0.80:
            return {"valid": False, "score": -999.0, "d_nm": d_nm}

        # 2. 3D Fast Marching Geodesic Tracer
        geo_res = self.geo_tracer.trace_blind_geodesic_path(
            src_coord_nm=cut_coord,
            dst_coord_nm=c_coord,
            src_tangent=cut_tan,
            dst_tangent=-c_tan,
            src_radius_nm=cut_r,
            dst_radius_nm=c_r
        )

        # 3. Santiago Ramon y Cajal Priors
        p_time = self.cajal.compute_conduction_time_prior(
            centrifugal_order=2,
            dist_from_soma_nm=d_nm,
            is_axon=is_axon_parent
        )

        # 4. Bayesian Synapse Polarity Prior
        n_pre = int(np.sum(cand_piece["syn_types"] == 0))
        n_post = int(np.sum(cand_piece["syn_types"] == 1))
        tot_syn = n_pre + n_post
        if tot_syn > 0:
            p_syn_match = (n_pre / tot_syn) if is_axon_parent else (n_post / tot_syn)
            p_syn = float(np.clip(p_syn_match, 0.20, 0.95))
        else:
            p_syn = 0.50

        p_gram = max(1e-4, p_grammar)
        p_geo = max(1e-4, geo_res["geodesic_score"])
        p_caj = max(1e-4, p_time)

        g_odds = float(np.log(p_gram / (1.0 - p_gram + 1e-7)))
        geo_odds = float(np.log(p_geo / (1.0 - p_geo + 1e-7)))
        caj_odds = float(np.log(p_caj / (1.0 - p_caj + 1e-7)))
        syn_odds = float(np.log(p_syn / (1.0 - p_syn + 1e-7)))

        combined = g_odds + (self.geo_weight * geo_odds) + (self.cajal_weight * caj_odds) + (0.5 * syn_odds)

        return {
            "valid": True,
            "score": combined,
            "p_geo": p_geo,
            "p_gram": p_gram,
            "d_nm": d_nm,
            "tortuosity": geo_res["tortuosity"]
        }

    def predict_best_of_the_best_infill(
        self,
        parent_token: Dict[str, Any],
        mask_token: Dict[str, Any],
        candidate_tokens: List[Dict[str, Any]],
        candidate_pieces_dict: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Executes unified Best-of-the-Best Infilling with:
          1. Context-Free derivation proposals.
          2. Closest-vertex candidate scoring.
          3. Smooth calibrated acceptance thresholding.
        """
        parent_symbol = parent_token.get("symbol", "[SOMA]")
        admissible_lhs_list = self.pcfg.derive_expected_lhs_from_parent(parent_symbol)
        is_axon = ("AXON" in parent_symbol)

        proposals = []
        for lhs in admissible_lhs_list:
            prop = self.infiller.predict_infill(
                context_tokens=[parent_token],
                mask_token=mask_token,
                candidate_pool=[{"token": t, "fragment_id": t["fragment_id"]} for t in candidate_tokens],
                expected_lhs=lhs
            )
            for cand in prop.get("ranked_candidates", []):
                proposals.append(cand)

        cands_dict = {}
        for c in proposals:
            cid = c["fragment_id"]
            if cid not in cands_dict or c["prob"] > cands_dict[cid]["prob"]:
                cands_dict[cid] = c
        sorted_cands = sorted(cands_dict.values(), key=lambda x: x["prob"], reverse=True)

        if len(sorted_cands) == 0:
            return {"predicted_id": None, "accepted": False, "top3_ids": []}

        cut_coord = np.array(mask_token.get("coord_nm", [0.0, 0.0, 0.0]), dtype=np.float32)
        cut_tan = np.array(mask_token.get("tangent", [1.0, 0.0, 0.0]), dtype=np.float32)
        cut_r = float(mask_token.get("radius_nm", 100.0))

        scored = []
        for cand in sorted_cands[:8]:
            cid = cand["fragment_id"]
            if cid == parent_token.get("fragment_id"):
                continue
            if cid not in candidate_pieces_dict:
                continue

            cand_p = candidate_pieces_dict[cid]
            res = self.score_candidate(
                cut_coord=cut_coord,
                cut_tan=cut_tan,
                cut_r=cut_r,
                cand_piece=cand_p,
                p_grammar=cand["prob"],
                is_axon_parent=is_axon
            )

            if res["valid"]:
                scored.append({
                    "fragment_id": cid,
                    "score": res["score"],
                    "prob": cand["prob"],
                    "d_nm": res["d_nm"]
                })

        scored.sort(key=lambda x: x["score"], reverse=True)

        if len(scored) == 0:
            return {"predicted_id": None, "accepted": False, "top3_ids": []}

        top1 = scored[0]
        top3_ids = [c["fragment_id"] for c in scored[:3]]
        accepted = (top1["score"] >= self.acceptance_threshold)

        return {
            "predicted_id": top1["fragment_id"] if accepted else None,
            "raw_top1_id": top1["fragment_id"],
            "accepted": accepted,
            "top1_score": top1["score"],
            "top3_ids": top3_ids
        }
