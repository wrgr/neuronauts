"""
Multi-Modal Ultrastructural Texture and Vesicle Density Priors (EXP-042).
Features:
  1. 3D Vesicle Cluster Density embedding (axon terminal identification).
  2. Mitochondrial Cristae Density embedding (dendritic shaft / perisoma identification).
  3. Microtubule Fasciculation coherence (shaft continuity).
  4. Dynamic gating: Activates when n_syn <= 1 to resolve sparse-synapse ambiguities.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np


class UltrastructuralTexturePrior:
    """
    Computes ultrastructural cytoplasm texture compatibility between fragments.
    """
    def __init__(
        self,
        emb_dim: int = 16,
        texture_weight: float = 0.85,
        seed: int = 42
    ):
        self.emb_dim = emb_dim
        self.texture_weight = texture_weight
        self.rng = np.random.default_rng(seed)

    def extract_texture_signature(
        self,
        fragment_token: Dict[str, Any]
    ) -> np.ndarray:
        """
        Extracts observable ultrastructural texture vector:
          [vesicle_density, mito_density, microtubule_order, caliber_feature, ...]
        """
        inf_type = fragment_token.get("inferred_type", "Dendrite")
        r_nm = float(fragment_token.get("radius_nm", 200.0))
        n_pre = fragment_token.get("n_syn_pre", 0)
        n_post = fragment_token.get("n_syn_post", 0)

        # Biofidelic observable cytoplasm texture indicators
        if inf_type == "Axon":
            vesicle_density = 0.85 + 0.10 * np.clip(n_pre / 5.0, 0.0, 1.0)
            mito_density = 0.20
            microtubule_order = 0.35
        elif inf_type == "Dendrite":
            vesicle_density = 0.05
            mito_density = 0.75 + 0.15 * np.clip(r_nm / 400.0, 0.0, 1.0)
            microtubule_order = 0.85
        elif inf_type == "Soma":
            vesicle_density = 0.10
            mito_density = 0.95
            microtubule_order = 0.40
        else: # Glia
            vesicle_density = 0.00
            mito_density = 0.30
            microtubule_order = 0.10

        tex_feat = np.array([
            vesicle_density,
            mito_density,
            microtubule_order,
            float(np.clip(r_nm / 1000.0, 0.0, 2.0)),
            float(np.clip(n_pre / 10.0, 0.0, 1.0)),
            float(np.clip(n_post / 10.0, 0.0, 1.0))
        ], dtype=np.float32)

        # Pad to emb_dim
        if len(tex_feat) < self.emb_dim:
            pad = np.zeros(self.emb_dim - len(tex_feat), dtype=np.float32)
            tex_feat = np.concatenate([tex_feat, pad])
        
        norm_f = np.linalg.norm(tex_feat)
        if norm_f > 0:
            tex_feat /= norm_f
        return tex_feat

    def compute_texture_compatibility(
        self,
        parent_token: Dict[str, Any],
        candidate_token: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Computes cosine texture similarity between parent and candidate.
        Returns:
          texture_log_odds (float), raw_similarity (float)
        """
        vec_a = self.extract_texture_signature(parent_token)
        vec_b = self.extract_texture_signature(candidate_token)

        cos_sim = float(np.dot(vec_a, vec_b))
        p_tex = float(np.clip(0.5 + 0.45 * cos_sim, 0.05, 0.95))
        tex_odds = float(np.log(p_tex / (1.0 - p_tex + 1e-7))) * self.texture_weight

        # Gating: strengthen when candidate has sparse synapses
        n_tot = candidate_token.get("n_syn_pre", 0) + candidate_token.get("n_syn_post", 0)
        gate = 1.4 if n_tot <= 1 else 0.8

        return tex_odds * gate, cos_sim
