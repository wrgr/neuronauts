"""
3D Geometric Tree-Grammar Transformer & Pointer Infiller (Neuron-Grammar-LM).
Predicts missing [MASK_FRAGMENT] tokens in serialized PCFG tree sequences
using continuous 3D geometric embeddings and syntax-constrained pointer attention.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class TreeGrammarInfiller:
    """
    3D Geometric Tree-Grammar Transformer Infiller for Connectome Assembly.
    """
    def __init__(self, emb_dim: int = 64, seed: int = 42):
        self.emb_dim = emb_dim
        self.rng = np.random.default_rng(seed)

        # Discrete symbol embeddings
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

        # 3D Geometric projection weights (x, y, z, r, tx, ty, tz) -> emb_dim // 2
        self.geo_proj = self.rng.normal(0, 0.1, (7, emb_dim // 2)).astype(np.float32)

        # Pointer attention query & key projection matrices
        q_mat, _ = np.linalg.qr(self.rng.normal(0, 1, (emb_dim, emb_dim)))
        k_mat, _ = np.linalg.qr(self.rng.normal(0, 1, (emb_dim, emb_dim)))
        self.w_query = q_mat.astype(np.float32)
        self.w_key = k_mat.astype(np.float32)

    def encode_token(self, token: Dict[str, Any]) -> np.ndarray:
        """
        Embeds a continuous-discrete grammar token into R^emb_dim.
        """
        sym_idx = self.sym_to_idx.get(token.get("symbol", "[MASK_FRAGMENT]"), 0)
        d_vec = self.discrete_emb[sym_idx]

        coord = np.array(token.get("coord_nm", [0.0, 0.0, 0.0]), dtype=np.float32) / 10000.0
        rad = np.array([token.get("radius_nm", 100.0) / 100.0], dtype=np.float32)
        tan = np.array(token.get("tangent", [1.0, 0.0, 0.0]), dtype=np.float32)
        geo_feat = np.concatenate([coord, rad, tan])

        g_vec = np.dot(geo_feat, self.geo_proj)
        full_emb = np.concatenate([d_vec, g_vec])
        norm = np.linalg.norm(full_emb)
        return (full_emb / (norm + 1e-7)).astype(np.float32)

    def predict_infill(
        self,
        context_tokens: List[Dict[str, Any]],
        mask_token: Dict[str, Any],
        candidate_pool: List[Dict[str, Any]],
        expected_lhs: str = "<ApicalTree>"
    ) -> Dict[str, Any]:
        """
        Predicts which candidate fragment from the orphan pool fills the [MASK_FRAGMENT] slot.
        Applies strict PCFG grammar constraints to eliminate impossible biological syntax.
        """
        if len(candidate_pool) == 0:
            return {"predicted_id": None, "top1_prob": 0.0, "ranked_candidates": []}

        # Encode context and compute query vector at mask site
        mask_emb = self.encode_token(mask_token)
        query = np.dot(mask_emb, self.w_query)
        q_norm = np.linalg.norm(query)
        if q_norm > 0:
            query = query / q_norm

        scores = []
        for cand in candidate_pool:
            cand_token = cand.get("token", cand)
            cand_id = cand.get("fragment_id", cand_token.get("fragment_id"))
            cand_lhs = cand_token.get("lhs", "<BasalTree>")

            # PCFG Syntax Constraint: If candidate violates the expected tree grammar, P = 0
            if expected_lhs == "<ApicalTree>" and "Axon" in cand_lhs:
                # Impossible: Axon cannot fill an Apical trunk mask
                scores.append((cand_id, -1e9, 0.0))
                continue
            elif expected_lhs == "<AxonArbor>" and "Apical" in cand_lhs:
                # Impossible: Apical cannot fill an Axon mask
                scores.append((cand_id, -1e9, 0.0))
                continue

            cand_emb = self.encode_token(cand_token)
            key = np.dot(cand_emb, self.w_key)
            k_norm = np.linalg.norm(key)
            if k_norm > 0:
                key = key / k_norm

            # Attention dot product
            raw_logit = float(np.dot(query, key))

            # Geometric distance damping
            d_nm = np.linalg.norm(
                np.array(mask_token.get("coord_nm", [0, 0, 0])) -
                np.array(cand_token.get("coord_nm", [0, 0, 0]))
            )
            dist_factor = float(np.exp(-d_nm / 18000.0))
            final_logit = raw_logit * 5.0 + np.log(dist_factor + 1e-7)

            scores.append((cand_id, final_logit, dist_factor))

        # Softmax normalization with temperature scaling
        temp = 0.25
        logits = np.array([s[1] for s in scores]) / temp
        max_l = np.max(logits)
        exp_logits = np.exp(logits - max_l)
        probs = exp_logits / np.sum(exp_logits)

        ranked = []
        for (cand_id, logit, df), prob in zip(scores, probs):
            ranked.append({
                "fragment_id": cand_id,
                "prob": float(prob),
                "logit": float(logit)
            })

        ranked.sort(key=lambda x: x["prob"], reverse=True)
        top1 = ranked[0]

        return {
            "predicted_id": top1["fragment_id"],
            "top1_prob": top1["prob"],
            "top3_ids": [r["fragment_id"] for r in ranked[:3]],
            "ranked_candidates": ranked
        }
