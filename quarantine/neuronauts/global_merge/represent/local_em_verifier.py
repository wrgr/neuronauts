"""
Selective Local Micro-EM Verification Module.
Simulates/queries targeted local 3D EM cross-sections along candidate bridge rays
for ambiguous edges (0.30 <= P <= 0.70) without processing full voxel volumes.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from neuronauts.global_merge.represent.cloudvolume_em_sampler import VolumetricEMSampler


class LocalEMVerifier:
    """
    Selective Local EM Verifier for ambiguous skeleton bridges.
    Only queries local EM cutouts along the candidate vector v_src -> v_dst.
    """
    def __init__(self, em_noise_std: float = 0.08):
        self.sampler = VolumetricEMSampler()
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
        return self.sampler.sample_bridge_volume(
            src_coord_nm,
            dst_coord_nm,
            is_true_continuation,
            rng
        )
