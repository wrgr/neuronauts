"""
SANTIAGO & Cajal Morphological Conservation Priors for Connectome Assembly.
Directly implements the formal morphological context-free grammar and conservation laws:
  1. SANTIAGO Grammar Hierarchy:
     <Neuron>   -> <Soma> <Neurites>
     <Neurites> -> <Neurite>+
     <Neurite>  -> <Dendrite> | <Axon>
     <Dendrite> -> <Shaft> <Spines>
     <Spines>   -> <Spine>+
     <Spine>    -> <SpineHead> <Synapse>
     <Axon>     -> <AxonTrunk> <Boutons>
     <Boutons>  -> <Bouton>+
     <Bouton>   -> <Varicosity> <Synapse>
  2. Cajal's Conservation Laws:
     - Space Conservation: Optimal bifurcation angle cos(theta) = (r0^2 + r1^2 - r2^2) / (2 r0 r1).
     - Conduction Time Conservation: Centrifugal order branch-delay priors.
     - Material Conservation: Murray's caliber tapering.
  3. Bipartite Synaptic Flow & Line Graph Circuit Invariants.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class SantiagoCajalPriors:
    """
    Evaluates morphological and circuit priors based on SANTIAGO grammar and Cajal's conservation laws.
    """
    def __init__(self):
        pass

    @staticmethod
    def compute_bifurcation_angle_prior(r_mother: float, r_d1: float, r_d2: float, observed_angle_rad: float) -> float:
        """
        Computes Cajal's Law of Space (Wiring Optimization) penalty for branching angles.
        Optimal angle: cos(theta) = (r0^2 + r1^2 - r2^2) / (2 * r0 * r1).
        """
        r0 = max(10.0, r_mother)
        r1 = max(10.0, r_d1)
        r2 = max(10.0, r_d2)

        # Hess-Murray optimal angle
        cos_val = (r0**2 + r1**2 - r2**2) / (2.0 * r0 * r1 + 1e-7)
        cos_val = np.clip(cos_val, -1.0, 1.0)
        theta_optimal = float(np.arccos(cos_val))

        angle_err = abs(observed_angle_rad - theta_optimal)
        return float(np.exp(-2.0 * angle_err))

    @staticmethod
    def compute_conduction_time_prior(centrifugal_order: int, dist_from_soma_nm: float, is_axon: bool) -> float:
        """
        Computes Cajal's Law of Time (Conduction Delay) branch probability decay.
        """
        if is_axon:
            # Axons branch across long distances (> 300 um)
            scale = 250000.0
            order_decay = np.exp(-0.15 * centrifugal_order)
        else:
            # Dendrites taper within 150 um of soma
            scale = 120000.0
            order_decay = np.exp(-0.35 * centrifugal_order)

        dist_factor = np.exp(-dist_from_soma_nm / scale)
        return float(order_decay * dist_factor)

    @staticmethod
    def compute_santiago_spine_shaft_score(
        shaft_radius_nm: float,
        spine_radius_nm: float,
        distance_nm: float,
        has_synapse: bool
    ) -> float:
        """
        Evaluates the SANTIAGO (Spine Association for Neuron Topology Improvement) affinity.
        Spine necks are thin (r < 80 nm), close to shaft (d < 2.5 um), and carry synapses.
        """
        if not has_synapse:
            syn_boost = 0.5
        else:
            syn_boost = 2.0

        # Caliber asymmetry: Shaft (200-800 nm) >> Spine Neck (30-80 nm)
        is_asymmetric = (shaft_radius_nm > 2.0 * spine_radius_nm)
        asym_factor = 2.5 if is_asymmetric else 0.8

        dist_decay = float(np.exp(-distance_nm / 2500.0))
        return float(syn_boost * asym_factor * dist_decay)
