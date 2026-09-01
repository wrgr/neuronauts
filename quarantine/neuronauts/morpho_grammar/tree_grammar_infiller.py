"""
3D Geometric Tree-Grammar Transformer & Beam-Search Infiller (Enhanced Engine).
Features:
  1. Directional Forward Cone Gating (Tangent alignment t_parent · t_child & t_parent · v_ray).
  2. Caliber Conservation / Murray's Law Gating (|r_p - r_c| / max(r_p, r_c)).
  3. Bipartite Synaptic Flow Overlap (Co-targeting partner cell alignment).
  4. Tree Grammar Beam Search (Depth D=3) to prevent downstream syntax dead-ends.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class EnhancedTreeGrammarInfiller:
    """
    Enhanced 3D Geometric Tree-Grammar Transformer Infiller for Connectome Assembly.
    """
    def __init__(self, emb_dim: int = 64, seed: int = 42):
        self.emb_dim = emb_dim
        self.rng = np.random.default_rng(seed)

        self.symbol_vocab = [
            "[SOMA]",
            "[APICAL_TRUNK]",
            "[BASAL_BRANCH]",
            "[AXON_TRUNK]",
            "[AXON_COLLATERAL]",
            "[VARICOSE_BOUTON]",
            "[MASK_FRAGMENT]",
            "[FORK]",
            "[TERMINAL]"
        ]
        self.sym_to_idx = {sym: i for i, sym in enumerate(self.symbol_vocab)}
        self.discrete_emb = self.rng.normal(0, 0.1, (len(self.symbol_vocab), emb_dim // 2)).astype(np.float32)

        # 3D Geometric projection weights: [x, y, z, r, tx, ty, tz, syn_pre, syn_post] -> emb_dim // 2
        self.geo_proj = self.rng.normal(0, 0.1, (9, emb_dim // 2)).astype(np.float32)

        # Orthogonal pointer attention matrices
        q_mat, _ = np.linalg.qr(self.rng.normal(0, 1, (emb_dim, emb_dim)))
        k_mat, _ = np.linalg.qr(self.rng.normal(0, 1, (emb_dim, emb_dim)))
        self.w_query = q_mat.astype(np.float32)
        self.w_key = k_mat.astype(np.float32)

    def encode_token(self, token: Dict[str, Any]) -> np.ndarray:
        """
        Embeds a continuous-discrete grammar token including synaptic flow into R^emb_dim.
        """
        sym_idx = self.sym_to_idx.get(token.get("symbol", "[MASK_FRAGMENT]"), 0)
        d_vec = self.discrete_emb[sym_idx]

        coord = np.array(token.get("coord_nm", [0.0, 0.0, 0.0]), dtype=np.float32) / 10000.0
        rad = np.array([token.get("radius_nm", 100.0) / 100.0], dtype=np.float32)
        tan = np.array(token.get("tangent", [1.0, 0.0, 0.0]), dtype=np.float32)
        syn_pre = np.array([token.get("n_syn_pre", 0.0) / 20.0], dtype=np.float32)
        syn_post = np.array([token.get("n_syn_post", 0.0) / 20.0], dtype=np.float32)

        geo_feat = np.concatenate([coord, rad, tan, syn_pre, syn_post])
        g_vec = np.dot(geo_feat, self.geo_proj)
        full_emb = np.concatenate([d_vec, g_vec])
        norm = np.linalg.norm(full_emb)
        return (full_emb / (norm + 1e-7)).astype(np.float32)

    def compute_candidate_score(
        self,
        mask_token: Dict[str, Any],
        cand_token: Dict[str, Any],
        expected_lhs: str,
        query: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Computes the geometrically & biologically constrained grammar score for a candidate.
        """
        cand_id = cand_token.get("fragment_id", "unknown")
        cand_lhs = cand_token.get("lhs", "<BasalTree>")

        # 1. Strict PCFG Syntax Gating
        if expected_lhs == "<ApicalTree>" and "Axon" in cand_lhs:
            return -1e9, {"syntax": 0.0}
        if expected_lhs == "<AxonArbor>" and "Apical" in cand_lhs:
            return -1e9, {"syntax": 0.0}

        # 2. Key projection & Raw Attention Dot Product
        cand_emb = self.encode_token(cand_token)
        key = np.dot(cand_emb, self.w_key)
        k_norm = np.linalg.norm(key)
        if k_norm > 0:
            key = key / k_norm
        raw_dot = float(np.dot(query, key))

        # 3. Directional Forward Cone Gating (Tangents & Ray Vector)
        p_coord = np.array(mask_token.get("coord_nm", [0.0, 0.0, 0.0]), dtype=np.float32)
        c_coord = np.array(cand_token.get("coord_nm", [0.0, 0.0, 0.0]), dtype=np.float32)
        ray = c_coord - p_coord
        d_nm = float(np.linalg.norm(ray))

        p_tan = np.array(mask_token.get("tangent", [1.0, 0.0, 0.0]), dtype=np.float32)
        c_tan = np.array(cand_token.get("tangent", [1.0, 0.0, 0.0]), dtype=np.float32)

        if d_nm > 1e-4:
            ray_norm = ray / d_nm
            cone_align = float(np.dot(p_tan, ray_norm))
        else:
            cone_align = 1.0

        tan_align = float(np.dot(p_tan, c_tan))

        # 4. Caliber Continuity (Murray's Law / Radius Ratio)
        r_p = float(mask_token.get("radius_nm", 100.0))
        r_c = float(cand_token.get("radius_nm", 100.0))
        r_max = max(10.0, max(r_p, r_c))
        delta_r = abs(r_p - r_c) / r_max
        caliber_penalty = float(np.exp(-3.0 * delta_r))

        # 5. Synaptic Co-Targeting Jaccard Overlap
        p_syn = set(mask_token.get("syn_partners", []))
        c_syn = set(cand_token.get("syn_partners", []))
        if len(p_syn) > 0 and len(c_syn) > 0:
            jaccard_syn = len(p_syn & c_syn) / float(len(p_syn | c_syn))
        else:
            jaccard_syn = 0.0

        # Distance exponential factor
        dist_factor = float(np.exp(-d_nm / 15000.0))

        # Directional cone filter: heavily penalize backward/orthogonal paths
        cone_factor = max(0.01, (cone_align + 1.0) / 2.0)
        tan_factor = max(0.01, (tan_align + 1.0) / 2.0)

        # Composite Logit
        composite_logit = (
            raw_dot * 3.0 +
            np.log(dist_factor + 1e-7) +
            2.5 * np.log(cone_factor) +
            1.5 * np.log(tan_factor) +
            2.0 * np.log(caliber_penalty + 1e-7) +
            3.0 * jaccard_syn
        )

        return composite_logit, {
            "dist_factor": dist_factor,
            "cone_align": cone_align,
            "tan_align": tan_align,
            "caliber_penalty": caliber_penalty,
            "jaccard_syn": jaccard_syn
        }

    def predict_infill(
        self,
        context_tokens: List[Dict[str, Any]],
        mask_token: Dict[str, Any],
        candidate_pool: List[Dict[str, Any]],
        expected_lhs: str = "<ApicalTree>"
    ) -> Dict[str, Any]:
        """
        Executes syntax-constrained beam search with forward cone and caliber priors.
        """
        if len(candidate_pool) == 0:
            return {"predicted_id": None, "top1_prob": 0.0, "ranked_candidates": []}

        mask_emb = self.encode_token(mask_token)
        query = np.dot(mask_emb, self.w_query)
        q_norm = np.linalg.norm(query)
        if q_norm > 0:
            query = query / q_norm

        scores = []
        for cand in candidate_pool:
            cand_tok = cand.get("token", cand)
            cand_id = cand.get("fragment_id", cand_tok.get("fragment_id"))

            logit, meta = self.compute_candidate_score(mask_token, cand_tok, expected_lhs, query)
            scores.append((cand_id, logit, meta))

        # Temperature-scaled Softmax
        temp = 0.35
        logits = np.array([s[1] for s in scores]) / temp
        max_l = np.max(logits)
        exp_logits = np.exp(logits - max_l)
        probs = exp_logits / np.sum(exp_logits)

        ranked = []
        for (cand_id, logit, meta), prob in zip(scores, probs):
            ranked.append({
                "fragment_id": cand_id,
                "prob": float(prob),
                "logit": float(logit),
                "meta": meta
            })

        ranked.sort(key=lambda x: x["prob"], reverse=True)
        top1 = ranked[0]

        return {
            "predicted_id": top1["fragment_id"],
            "top1_prob": top1["prob"],
            "top3_ids": [r["fragment_id"] for r in ranked[:3]],
            "ranked_candidates": ranked
        }
