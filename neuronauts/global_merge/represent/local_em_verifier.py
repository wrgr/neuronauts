"""
Selective Local Micro-EM Verification Module.
Simulates/queries targeted local 3D EM cross-sections along candidate bridge rays
for ambiguous edges (0.30 <= P <= 0.70) without processing full voxel volumes.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np


class LocalEMVerifier:
    """
    Selective Local EM Verifier for ambiguous skeleton bridges.
    Only queries local EM cutouts along the candidate vector v_src -> v_dst.
    """
    def __init__(self, em_noise_std: float = 0.12):
        self.em_noise_std = em_noise_std

    def verify_bridge_ray(
        self,
        src_coord_nm: np.ndarray,
        dst_coord_nm: np.ndarray,
        is_true_continuation: bool,
        rng: np.random.Generator
    ) -> Dict[str, float]:
        """
        Micro-queries local 3D EM cross-section along the trajectory between src and dst.
        Returns continuous visual membrane continuity evidence.
        """
        dist_nm = float(np.linalg.norm(dst_coord_nm - src_coord_nm))
        dist_factor = np.exp(-dist_nm / 18000.0)
        
        if is_true_continuation:
            # Tubular continuity along neurite path
            signal = 0.88 * dist_factor
            noise = rng.normal(0.0, self.em_noise_std)
            p_em = float(np.clip(signal + noise, 0.10, 0.98))
        else:
            # Transverse plasma membrane barrier / extracellular space
            signal = 0.12
            noise = rng.normal(0.0, self.em_noise_std)
            p_em = float(np.clip(signal + noise, 0.02, 0.90))

        # Log-odds update
        log_odds = float(np.log(p_em / (1.0 - p_em + 1e-7)))
        
        return {
            "em_score": p_em,
            "em_log_odds": log_odds,
            "dist_nm": dist_nm
        }
