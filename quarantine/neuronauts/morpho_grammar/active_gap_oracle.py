"""
Active Grammar-Guided Oracle Engine for Long-Range Voids (> 20 um).
Features:
  1. Principal Fascicle Ray Projection: Projects search rays along exit tangent t_exit.
  2. Sparse Oracle Query Protocol: Queries an oracle only on high-leverage tears (> 20 um)
     with strict query budgeting (<= 5 queries / neuron).
  3. Seamless integration with SANTIAGO-v2 grammar derivations.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class ActiveGapOracleEngine:
    """
    Active grammar-guided query engine for bridging large physical voids (> 20 um).
    """
    def __init__(
        self,
        max_void_search_dist_nm: float = 65000.0,
        max_queries_per_neuron: int = 5,
        oracle_accuracy: float = 0.98,
        seed: int = 42
    ):
        self.max_void_search_dist_nm = max_void_search_dist_nm
        self.max_queries_per_neuron = max_queries_per_neuron
        self.oracle_accuracy = oracle_accuracy
        self.rng = np.random.default_rng(seed)

    def query_long_range_gap(
        self,
        cut_token: Dict[str, Any],
        candidate_tokens: List[Dict[str, Any]],
        gt_target_id: Optional[str] = None
    ) -> Tuple[Optional[str], float, int]:
        """
        Simulates a sparse human-in-the-loop oracle query for large gaps (> 20 um).
        Returns:
          chosen_candidate_id (str or None), confidence (float), queries_consumed (int)
        """
        cut_pos = np.array(cut_token.get("coord_nm", [0.0, 0.0, 0.0]), dtype=np.float32)
        cut_tan = np.array(cut_token.get("tangent", [1.0, 0.0, 0.0]), dtype=np.float32)
        cut_type = cut_token.get("inferred_type", "Dendrite")

        # 1. Project search rays and rank candidate fascicles
        ray_candidates = []
        for cand in candidate_tokens:
            cand_id = cand["fragment_id"]
            if cand_id == cut_token.get("fragment_id"):
                continue

            c_pos = np.array(cand.get("coord_nm", [0.0, 0.0, 0.0]), dtype=np.float32)
            c_tan = np.array(cand.get("tangent", [1.0, 0.0, 0.0]), dtype=np.float32)
            c_type = cand.get("inferred_type", "Dendrite")

            # Polarity invariant
            if (cut_type == "Axon" and c_type != "Axon") or (cut_type != "Axon" and c_type == "Axon"):
                continue
            if cand.get("is_glia", False):
                continue

            disp = c_pos - cut_pos
            dist_nm = float(np.linalg.norm(disp))

            if 20000.0 <= dist_nm <= self.max_void_search_dist_nm:
                ray_dir = disp / (dist_nm + 1e-7)
                cos_ray = float(np.dot(cut_tan, ray_dir))
                cos_opp = float(np.dot(-c_tan, -ray_dir))

                if cos_ray > 0.35 and cos_opp > 0.20:
                    ray_score = (cos_ray + cos_opp) - (dist_nm / 40000.0)
                    ray_candidates.append({
                        "id": cand_id,
                        "dist_nm": dist_nm,
                        "ray_score": ray_score
                    })

        if len(ray_candidates) == 0:
            return None, 0.0, 0

        ray_candidates.sort(key=lambda x: x["ray_score"], reverse=True)
        top_cand_ids = [c["id"] for c in ray_candidates[:3]]

        # Oracle query execution (Simulating human oracle on top-3 candidates)
        if gt_target_id in top_cand_ids:
            # Oracle confirms with 98% accuracy
            if self.rng.random() < self.oracle_accuracy:
                return gt_target_id, 0.99, 1
            else:
                return None, 0.0, 1
        
        return None, 0.0, 1
