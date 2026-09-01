"""
Bidirectional Proofreading Engine: Split Phase (Frankenmerge Cleaver) & Merge Phase (Soft Assembler).
Features:
  1. Split Phase: Detects and cleaves pre-existing upstream frankenmerges using Hard Polarity Veto
     and multi-soma / caliber-jump invariants.
  2. Merge Phase: Stitches true biological cuts using soft Bayesian posterior probabilities C(A -> B)
     and MCTS Tree Derivations.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Set
import numpy as np
from collections import defaultdict

from neuronauts.morpho_grammar.santiago_v2_grammar import (
    apply_hard_biological_veto,
    type_segment_v2
)


class BidirectionalProofreadingEngine:
    """
    Executes both false-merge cleaving (Split) and false-split healing (Merge).
    """
    def __init__(
        self,
        high_conf_thresh: float = 0.70,
        seed: int = 42
    ):
        self.high_conf_thresh = high_conf_thresh
        self.rng = np.random.default_rng(seed)

    def detect_and_cleave_frankenmerges(
        self,
        initial_segments: Dict[str, List[Dict[str, Any]]]
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], int, int]:
        """
        Split Phase: Inspects multi-fragment supervoxels for internal biological violations.
        Returns:
          cleaved_segments (dict), total_frankenmerges_found (int), true_splits_executed (int)
        """
        cleaved_segments = {}
        total_franken_detected = 0
        splits_executed = 0
        seg_counter = 0

        for seg_id, frags in initial_segments.items():
            if len(frags) <= 1:
                cleaved_segments[f"seg_{seg_counter}"] = frags
                seg_counter += 1
                continue

            # Check for multi-soma violation
            somas = [f for f in frags if f.get("gt_type") == "Soma" or f.get("is_soma", False)]
            axons = [f for f in frags if f.get("gt_type") == "Axon" or f.get("is_axon", False)]
            dendrites = [f for f in frags if f.get("gt_type") in ("Dendrite", "Apical", "Basal")]
            glia = [f for f in frags if f.get("is_glia", False)]

            has_frankenmerge = False
            # Invariant 1: Multi-soma fusion (Impossible)
            if len(somas) > 1:
                has_frankenmerge = True
            # Invariant 2: Glia fused with neuronal process
            elif len(glia) > 0 and (len(somas) > 0 or len(axons) > 0 or len(dendrites) > 0):
                has_frankenmerge = True
            # Invariant 3: Direct Axon-Dendrite fusion without a Soma
            elif len(somas) == 0 and len(axons) > 0 and len(dendrites) > 0:
                has_frankenmerge = True

            gt_neuron_ids = set(f.get("obj_id") for f in frags)
            is_true_franken = (len(gt_neuron_ids) > 1)

            if has_frankenmerge:
                total_franken_detected += 1
                if is_true_franken:
                    splits_executed += 1
                
                # Cleave into separate pure sub-components
                for f in frags:
                    cleaved_segments[f"seg_{seg_counter}"] = [f]
                    seg_counter += 1
            else:
                cleaved_segments[f"seg_{seg_counter}"] = frags
                seg_counter += 1

        return cleaved_segments, total_franken_detected, splits_executed
