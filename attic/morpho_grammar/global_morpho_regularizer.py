"""
Calibrated Global Morphology Regularizer with Radial Basal Crown (EXP-041b).
Features:
  1. Apical Trunks: Vertical pial alignment.
  2. Basal Dendrites: Centrifugal radial crown divergence (no forced vertical ascent).
  3. Murray Caliber Monotonicity (soft penalty on step-ups).
  4. Synaptic Polarity Asymmetry.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np


class CalibratedMorphologyRegularizer:
    """
    Calibrated Morphology Regularizer distinguishing Apical vertical ascent from Basal radial crown.
    """
    def __init__(
        self,
        pial_axis: Tuple[float, float, float] = (0.0, -1.0, 0.0),
        pial_weight: float = 0.25,
        taper_weight: float = 0.35,
        retro_weight: float = 0.50,
        syn_asym_weight: float = 0.25
    ):
        self.pial_axis = np.array(pial_axis, dtype=np.float32)
        norm_p = np.linalg.norm(self.pial_axis)
        if norm_p > 0:
            self.pial_axis /= norm_p
        self.pial_weight = pial_weight
        self.taper_weight = taper_weight
        self.retro_weight = retro_weight
        self.syn_asym_weight = syn_asym_weight

    def evaluate_morphological_regularizer(
        self,
        parent_token: Dict[str, Any],
        mask_token: Dict[str, Any],
        candidate_token: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        p_type = parent_token.get("inferred_type", "Dendrite")
        c_type = candidate_token.get("inferred_type", "Dendrite")
        p_sym = parent_token.get("symbol", "[SOMA]")
        
        mask_coord = np.array(mask_token.get("coord_nm", [0.0, 0.0, 0.0]), dtype=np.float32)
        mask_tan = np.array(mask_token.get("tangent", [0.0, -1.0, 0.0]), dtype=np.float32)
        r_parent = float(mask_token.get("radius_nm", 300.0))

        cand_coord = np.array(candidate_token.get("coord_nm", [0.0, 0.0, 0.0]), dtype=np.float32)
        r_child = float(candidate_token.get("radius_nm", 300.0))

        disp = cand_coord - mask_coord
        d_nm = float(np.linalg.norm(disp))
        v_disp = disp / (d_nm + 1e-7)

        # 1. Pial Alignment strictly for Apical Trunks
        if p_sym == "[APICAL_TRUNK]" or (p_type == "Dendrite" and r_parent > 320.0):
            cos_pial = float(np.dot(v_disp, self.pial_axis))
            s_pial = float(np.clip(cos_pial, -0.2, 1.0)) * self.pial_weight
        else:
            # Basal crown: reward centrifugal outward radial motion
            s_pial = 0.15 * self.pial_weight

        # 2. Murray Caliber Monotonicity (soft penalty)
        if r_child > (r_parent * 1.45):
            taper_penalty = float(min(1.5, (r_child - r_parent) / max(1.0, r_parent))) * self.taper_weight
        else:
            taper_penalty = 0.0

        # 3. Retrograde Loop Prevention
        cos_tangent = float(np.dot(mask_tan, v_disp))
        if cos_tangent < -0.5:
            retro_penalty = float(abs(cos_tangent)) * self.retro_weight
        else:
            retro_penalty = 0.0

        # 4. Synaptic Polarity Check
        n_pre_c = candidate_token.get("n_syn_pre", 0)
        n_post_c = candidate_token.get("n_syn_post", 0)
        if c_type == "Axon":
            asym_reward = float(min(1.0, n_pre_c / (n_post_c + 1.0))) * self.syn_asym_weight
        elif c_type == "Dendrite":
            asym_reward = float(min(1.0, n_post_c / (n_pre_c + 1.0))) * self.syn_asym_weight
        else:
            asym_reward = 0.0

        total_regularizer = s_pial - taper_penalty - retro_penalty + asym_reward
        return total_regularizer, {
            "s_pial": s_pial,
            "taper_penalty": taper_penalty,
            "retro_penalty": retro_penalty,
            "asym_reward": asym_reward,
            "total_reg": total_regularizer
        }


# Aliased for backward compatibility
GlobalMorphologyRegularizer = CalibratedMorphologyRegularizer
