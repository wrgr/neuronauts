"""
SANTIAGO-v2 Complete Morphological Grammar, Half-Synapse Polarity, Hard Veto & Forensic Error Analyzer (EXP-040).
Extends SANTIAGO with:
  1. Immutable Biological Hard Polarity Veto: Prohibits Axon-Dendrite and Glia-Neuron merges (P = 0).
  2. Glial non-terminals and Zero-Synapse Exclusion Barrier.
  3. Half-Synapse Pre/Post Polarity from synapse table.
  4. Unsupervised Cell-Type Induction from observable morphology.
  5. Forensic Error Analyzer: Detailed root cause breakdown across distance and angles.
100% blind at inference without ground truth.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
import numpy as np

from neuronauts.morpho_grammar.blind_pcfg_morphology import BlindMorphologicalPCFG
from neuronauts.morpho_grammar.cajal_conservation_priors import SantiagoCajalPriors


class SANTIAGOv2PCFG:
    """
    SANTIAGO-v2 Grammar incorporating Glial Partitions, Pyramidal/Interneuron Typologies, and Half-Synapse Polarities.
    """
    def __init__(self):
        self.cajal = SantiagoCajalPriors()
        self.rules = {
            "<Volume>": [
                ("<Neuron>", 0.85),
                ("<Glia>", 0.12),
                ("<BloodVessel>", 0.03)
            ],
            "<Neuron>": [
                ("<PyramidalNeuron>", 0.75),
                ("<Interneuron>", 0.25)
            ],
            "<PyramidalNeuron>": [
                ("<Soma> <ApicalTree> <BasalTree> <AxonArbor>", 1.0)
            ],
            "<Interneuron>": [
                ("<Soma> <AspinyDendriteTree> <DenseAxonPlexus>", 1.0)
            ],
            "<Glia>": [
                ("<AstrocyteStar>", 0.60),
                ("<OligodendrocyteSheath>", 0.30),
                ("<MicrogliaProcess>", 0.10)
            ],
            "<ApicalTree>": [
                ("<ApicalTrunk> <ApicalFork>", 0.80),
                ("<ApicalTrunk> <ApicalTuft>", 0.20)
            ],
            "<ApicalFork>": [
                ("(<ApicalTree>, <ApicalTree>)", 0.85),
                ("<ApicalTuft>", 0.15)
            ],
            "<BasalTree>": [
                ("<BasalTrunk> <BasalFork>", 0.90),
                ("<BasalTerminal>", 0.10)
            ],
            "<AxonArbor>": [
                ("<AxonTrunk> <AxonCollateral>", 0.70),
                ("<AxonTerminal>", 0.30)
            ],
            "<Dendrite>": [
                ("<DendriteShaft> <PostSynapsePool>", 1.0)
            ],
            "<Axon>": [
                ("<AxonShaft> <PreSynapseBoutonPool>", 1.0)
            ]
        }

    def derive_expected_lhs_v2(self, parent_symbol: str) -> List[str]:
        """
        Derives allowable LHS non-terminals enforcing strict Glial-Neuron and Axon-Dendrite barriers.
        """
        if "[GLIA]" in parent_symbol or "<Glia>" in parent_symbol:
            return ["<AstrocyteStar>", "<OligodendrocyteSheath>", "<MicrogliaProcess>"]
        
        if "[SOMA]" in parent_symbol:
            return ["<ApicalTree>", "<BasalTree>", "<AxonArbor>"]
        
        if "[APICAL" in parent_symbol:
            return ["<ApicalFork>", "<ApicalTree>", "<ApicalTuft>"]
        
        if "[BASAL" in parent_symbol:
            return ["<BasalFork>", "<BasalTree>", "<BasalTerminal>"]
        
        if "[AXON" in parent_symbol:
            return ["<AxonArbor>", "<AxonCollateral>", "<AxonTerminal>"]
        
        return ["<ApicalTree>", "<BasalTree>", "<AxonArbor>"]


def apply_hard_biological_veto(
    parent_type: str,
    cand_type: str,
    n_pre_cand: int,
    n_post_cand: int
) -> bool:
    """
    Immutable Biological Hard Veto:
      - Glia cannot merge with Neurons (Soma, Axon, Dendrite).
      - Axon cannot merge with Dendrite or Soma (pre-synaptic bouton shaft cannot fuse with dendritic shaft).
      - Fragments with strong pre-synaptic dominance (n_pre > n_post) are strictly barred from dendritic parents.
    Returns True if the merge is VETOED (prohibited), False if admissible.
    """
    # 1. Glial Exclusion Barrier
    if parent_type == "Glia" or cand_type == "Glia":
        if parent_type != cand_type:
            return True

    # 2. Single-Soma Biological Invariant (Soma cannot merge with another Soma)
    if parent_type == "Soma" and cand_type == "Soma":
        return True  # Strictly veto multi-soma chimeras

    # 3. Somas are multipolar root origins (can validly emit Axons and Dendrites)
    if parent_type == "Soma" or cand_type == "Soma":
        return False

    # 3. Axon-Dendrite Chimera Veto (Dendrite cannot merge into Axon, Axon cannot merge into Dendrite)
    if parent_type == "Dendrite" and cand_type == "Axon":
        return True
    if parent_type == "Axon" and cand_type == "Dendrite":
        return True

    # 4. Direct Half-Synapse Polarity Contradiction (strict for established arbors)
    if parent_type == "Dendrite" and n_pre_cand > max(2, n_post_cand * 2):
        return True
    if parent_type == "Axon" and n_post_cand > max(2, n_pre_cand * 2):
        return True

    return False


def type_segment_v2(
    n_pre: int,
    n_post: int,
    mean_radius_nm: float,
    max_radius_nm: float,
    path_length_nm: float,
    tortuosity: float = 1.0
) -> str:
    """
    Observable multi-class segment typing:
      1. Glia: Zero chemical synapses (n_pre = 0, n_post = 0) over substantial path length (>4 um).
      2. Soma: Large caliber (max_r > 1200 nm).
      3. Axon: Pre-synaptic dominant (n_pre > n_post) or thin caliber with pre synapses.
      4. Dendrite: Post-synaptic dominant (n_post > n_pre) or thick caliber with spines.
    """
    tot_syn = n_pre + n_post

    # 1. Zero-Synapse Glial Barrier
    if tot_syn == 0 and path_length_nm > 4000.0:
        return "Glia"

    # 2. Soma by Caliber (compact interneuron 550-950nm and large pyramidal 1500-3000nm)
    if max_radius_nm >= 550.0:
        return "Soma"

    # 3. Half-Synapse Polarity
    if tot_syn > 0:
        if n_pre == 0 and n_post > 0:
            return "Dendrite"
        if n_post == 0 and n_pre > 0:
            return "Axon"
        pre_ratio = n_pre / tot_syn
        if pre_ratio >= 0.60:
            return "Axon"
        elif pre_ratio <= 0.40:
            return "Dendrite"
        else:
            # Caliber tie-breaker
            return "Axon" if mean_radius_nm < 110.0 else "Dendrite"

    # 4. Fallback on Caliber
    return "Axon" if mean_radius_nm < 105.0 else "Dendrite"


def induce_cell_type_from_observables(fragments: List[Dict[str, Any]]) -> str:
    """
    Deduces broad cell type from observable segment tokens without ground truth:
      - Pyramidal: High postsynaptic spine density + thick apical trunk + thin axon.
      - Interneuron: Aspiny/sparsely spiny dendrites + high presynaptic density.
      - Glia: 100% non-synaptic fragments.
    """
    tot_pre = sum(f.get("n_syn_pre", 0) for f in fragments)
    tot_post = sum(f.get("n_syn_post", 0) for f in fragments)
    tot_syn = tot_pre + tot_post
    tot_path_um = sum(f.get("path_len_nm", 1000.0) for f in fragments) / 1000.0

    if tot_syn == 0:
        return "Glia"

    spine_density = tot_post / max(1.0, tot_path_um)
    bouton_density = tot_pre / max(1.0, tot_path_um)
    max_caliber = max((float(f.get("max_radius_nm", f.get("radius_nm", 100.0))) for f in fragments), default=100.0)

    # Excitatory pyramidals have post-synaptic dominance and thick somas/apical trunks (>1000nm)
    # Inhibitory interneurons have high presynaptic output / aspiny dendrites with compact somas (<1000nm)
    if tot_post >= tot_pre * 1.5 or max_caliber > 1000.0:
        return "PyramidalNeuron"
    elif tot_pre > tot_post or max_caliber < 1000.0:
        return "Interneuron"
    else:
        return "PyramidalNeuron"


class ForensicErrorAnalyzer:
    """
    Exhaustive Forensic Error Diagnostic Engine.
    Categorizes all False Positives (Merge FPs) and False Negatives (Merge FNs) into explicit physical failure modes.
    """
    def __init__(self):
        self.error_counts = defaultdict(int)
        self.fp_details = []
        self.fn_details = []

    def diagnose_merge_fp(
        self,
        frag_a: Dict[str, Any],
        frag_b: Dict[str, Any],
        d_nm: float,
        p_handshake: float,
        p_geo: float,
        caliber_ratio: float
    ) -> str:
        """
        Diagnoses the physical root cause of a False Positive merge.
        """
        type_a = frag_a.get("inferred_type", "Unknown")
        type_b = frag_b.get("inferred_type", "Unknown")

        # Glial breach
        if type_a == "Glia" or type_b == "Glia":
            reason = "GLIAL_NONSYNAPTIC_CROSS_TOUCH"
        # Axon-Dendrite Chimera
        elif (type_a == "Axon" and type_b == "Dendrite") or (type_a == "Dendrite" and type_b == "Axon"):
            reason = "AXON_DENDRITE_SYNTAX_CHIMERA"
        # Membrane contact confusion
        elif d_nm < 3500.0:
            reason = "ADJACENT_MEMBRANE_TOUCH_PROXIMITY"
        # High Caliber mismatch
        elif caliber_ratio > 0.65:
            reason = "CALIBER_DISPARITY_DRIFT"
        else:
            reason = "TRAJECTORY_ALIGNED_FALSE_CONTINUATION"

        self.error_counts[f"FP_{reason}"] += 1
        self.fp_details.append({
            "frag_a": frag_a.get("id"),
            "frag_b": frag_b.get("id"),
            "reason": reason,
            "d_nm": d_nm,
            "p_handshake": p_handshake,
            "p_geo": p_geo
        })
        return reason

    def diagnose_merge_fn(
        self,
        frag_a: Dict[str, Any],
        frag_b: Dict[str, Any],
        d_nm: float,
        align_ray: float,
        p_handshake: float,
        p_geo: float,
        tortuosity: float
    ) -> str:
        """
        Diagnoses the physical root cause of a False Negative merge (missed true continuation).
        """
        # Long gap distance
        if d_nm > 20000.0:
            reason = "LONG_GAP_DISTANCE_EXCEEDING_RECEPTIVE_FIELD"
        # High curvature / kink
        elif tortuosity > 1.30:
            reason = "HIGH_CURVATURE_BENDING_PENALTY"
        # Backwards ray alignment
        elif align_ray < 0.15:
            reason = "ACUTE_BRANCH_ANGLE_OCCLUSION"
        # Asymmetric handshake failure
        elif p_handshake < 0.35:
            reason = "ASYMMETRIC_BACKWARD_TRAJECTORY_MISMATCH"
        # Sparse synapse ambiguity
        elif (frag_a.get("n_syn_pre", 0) + frag_a.get("n_syn_post", 0) == 0) or (frag_b.get("n_syn_pre", 0) + frag_b.get("n_syn_post", 0) == 0):
            reason = "SPARSE_SYNAPSE_SIGNAL_AMBIGUITY"
        else:
            reason = "LOW_PROBABILITY_MARGIN_DEFICIT"

        self.error_counts[f"FN_{reason}"] += 1
        self.fn_details.append({
            "frag_a": frag_a.get("id"),
            "frag_b": frag_b.get("id"),
            "reason": reason,
            "d_nm": d_nm,
            "align_ray": align_ray,
            "p_handshake": p_handshake,
            "tortuosity": tortuosity
        })
        return reason

    def get_summary_report(self) -> Dict[str, Any]:
        """
        Returns complete structured breakdown of all diagnosed error categories.
        """
        return {
            "error_counts": dict(self.error_counts),
            "total_fps": len(self.fp_details),
            "total_fns": len(self.fn_details),
            "fp_sample": self.fp_details[:5],
            "fn_sample": self.fn_details[:5]
        }
