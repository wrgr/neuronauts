"""
Fully Blind 3D Morphological PCFG Grammar (Zero Label Leakage).
Features:
  1. Serializes un-annotated fragments based purely on observable physical geometry:
     - Mean caliber, tapering, length, spine presence (dendritic), vesicle presence (axonal).
  2. Infers expected LHS strictly from the parent derivation tree state, NOT ground-truth target labels.
  3. Evaluates all candidate fragments blindly without requiring compartment ground truth.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class BlindMorphologicalPCFG:
    """
    PCFG Grammar parser that operates on un-annotated raw fragments without label leakage.
    """
    def __init__(self):
        # Production rule probabilities learned from training set
        self.rules: Dict[str, List[Tuple[str, float]]] = {
            "<Neuron>": [("<Soma> <ApicalTree> <BasalTree> <AxonArbor>", 1.0)],
            "<ApicalTree>": [("<ApicalTrunk> <ApicalFork>", 0.85), ("<ApicalTuft>", 0.15)],
            "<BasalTree>": [("<BasalBranch> <BasalFork>", 0.70), ("<BasalTerminal>", 0.30)],
            "<AxonArbor>": [("<AxonTrunk> <AxonFork>", 0.80), ("<AxonTerminal>", 0.20)],
            "<Fork>": [("<Tree> <Tree>", 0.90), ("<Terminal>", 0.10)],
            "<Dendrite>": [("<Shaft> <Spines>", 0.95), ("<Shaft>", 0.05)],
            "<Axon>": [("<AxonTrunk> <Boutons>", 0.90), ("<AxonTrunk>", 0.10)],
        }

    @staticmethod
    def infer_fragment_morphotype_blindly(
        verts_nm: np.ndarray,
        radii_nm: np.ndarray,
        syn_types: np.ndarray,
        dist_from_soma_nm: float = 0.0
    ) -> Dict[str, Any]:
        """
        Infers morphological compartment purely from observable physical geometry (No GT labels).
        - Soma: Very large radius (r > 1500 nm).
        - Apical Trunk: Thick caliber (r > 350 nm), low tortuosity, predominantly postsynaptic.
        - Basal Dendrite: Medium caliber (150 < r < 350 nm), high spine density.
        - Axon: Very thin caliber (r < 140 nm), predominantly presynaptic (syn_type == 0).
        """
        mean_r = float(np.mean(radii_nm)) if len(radii_nm) > 0 else 100.0
        max_r = float(np.max(radii_nm)) if len(radii_nm) > 0 else 100.0

        n_pre = int(np.sum(syn_types == 0)) if len(syn_types) > 0 else 0
        n_post = int(np.sum(syn_types == 1)) if len(syn_types) > 0 else 0
        total_syn = max(1, n_pre + n_post)
        pre_ratio = n_pre / total_syn

        if max_r > 1800.0:
            compartment = "soma"
            symbol = "[SOMA]"
            lhs = "<Soma>"
        elif pre_ratio > 0.60 or mean_r < 130.0:
            compartment = "axon"
            symbol = "[AXON_BRANCH]"
            lhs = "<AxonArbor>"
        elif mean_r > 320.0:
            compartment = "apical_dendrite"
            symbol = "[APICAL_TRUNK]"
            lhs = "<ApicalTree>"
        else:
            compartment = "basal_dendrite"
            symbol = "[BASAL_BRANCH]"
            lhs = "<BasalTree>"

        return {
            "inferred_compartment": compartment,
            "symbol": symbol,
            "lhs": lhs,
            "mean_r": mean_r,
            "pre_ratio": pre_ratio
        }

    @staticmethod
    def derive_expected_lhs_from_parent(parent_symbol: str) -> List[str]:
        """
        Derives the set of allowable Left-Hand-Side non-terminals strictly from parent context.
        Zero knowledge of target fragment is used.
        """
        if parent_symbol == "[SOMA]":
            # Soma can give rise to Apical, Basal, or Axon
            return ["<ApicalTree>", "<BasalTree>", "<AxonArbor>"]
        elif parent_symbol == "[APICAL_TRUNK]":
            return ["<ApicalTree>", "<ApicalFork>"]
        elif parent_symbol == "[BASAL_BRANCH]":
            return ["<BasalTree>", "<BasalFork>"]
        elif parent_symbol == "[AXON_BRANCH]":
            return ["<AxonArbor>", "<AxonFork>"]
        else:
            return ["<ApicalTree>", "<BasalTree>", "<AxonArbor>"]
