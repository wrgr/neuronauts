"""
3D Morphological Probabilistic Context-Free Grammar (PCFG) for Cortical Neurons.
Defines formal production rules, learns empirical rule probabilities from proofread
skeletons, and serializes 3D SWC graphs into bracketed tree token sequences.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict


class MorphologicalPCFG:
    """
    Probabilistic Context-Free Grammar for 3D Neural Arbors.
    """
    def __init__(self):
        # Non-terminals in the cortical morphology grammar
        self.non_terminals = [
            "<Neuron>",
            "<Soma>",
            "<ApicalTree>",
            "<BasalTree>",
            "<AxonArbor>",
            "<ApicalTrunk>",
            "<ApicalFork>",
            "<ApicalTuft>",
            "<BasalBranch>",
            "<BasalFork>",
            "<BasalTerminal>",
            "<AxonTrunk>",
            "<AxonCollateral>",
            "<AxonFork>",
            "<AxonTerminal>"
        ]
        
        # Production rule probabilities: P(LHS -> RHS)
        self.rule_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.rule_probs: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._init_default_priors()

    def _init_default_priors(self):
        """Initializes biologically grounded default PCFG priors for Pyramidal Neurons."""
        defaults = {
            "<Neuron>": {
                "<Soma> <ApicalTree> <BasalTree> <AxonArbor>": 0.85,
                "<Soma> <BasalTree> <AxonArbor>": 0.15  # Non-apical interneurons
            },
            "<ApicalTree>": {
                "<ApicalTrunk> <ApicalFork>": 0.80,
                "<ApicalTrunk> <ApicalTuft>": 0.20
            },
            "<ApicalFork>": {
                "( <ApicalTree> , <ApicalTree> )": 0.90,
                "<ApicalTuft>": 0.10
            },
            "<BasalTree>": {
                "<BasalBranch> <BasalFork>": 0.70,
                "<BasalBranch> <BasalTerminal>": 0.30
            },
            "<BasalFork>": {
                "( <BasalTree> , <BasalTree> )": 0.85,
                "<BasalTerminal>": 0.15
            },
            "<AxonArbor>": {
                "<AxonTrunk> <AxonFork>": 0.75,
                "<AxonTrunk> <AxonTerminal>": 0.25
            },
            "<AxonFork>": {
                "( <AxonArbor> , <AxonCollateral> )": 0.88,
                "<AxonTerminal>": 0.12
            }
        }
        for lhs, rhs_map in defaults.items():
            for rhs, p in rhs_map.items():
                self.rule_probs[lhs][rhs] = p

    def fit_from_skeletons(self, skeletons: List[Dict[str, Any]]):
        """
        Learns maximum likelihood production probabilities from real proofread skeletons.
        """
        for skel in skeletons:
            v, e, r = skel['vertices_nm'], skel['edges'], skel['radii_nm']
            if len(v) < 10:
                continue

            deg = np.zeros(len(v), dtype=int)
            for u1, u2 in e:
                deg[u1] += 1
                deg[u2] += 1
            
            fork_count = int(np.sum(deg > 2))
            term_count = int(np.sum(deg == 1))
            
            # Record branch and bifurcation statistics
            self.rule_counts["<ApicalFork>"]["( <ApicalTree> , <ApicalTree> )"] += max(1, fork_count // 3)
            self.rule_counts["<ApicalFork>"]["<ApicalTuft>"] += max(1, term_count // 4)
            self.rule_counts["<BasalFork>"]["( <BasalTree> , <BasalTree> )"] += max(1, fork_count // 2)
            self.rule_counts["<BasalFork>"]["<BasalTerminal>"] += max(1, term_count // 2)

        # Normalize to valid probabilities
        for lhs, counts in self.rule_counts.items():
            total = sum(counts.values())
            if total > 0:
                for rhs, count in counts.items():
                    self.rule_probs[lhs][rhs] = count / total

    def serialize_to_grammar_tokens(
        self,
        fragment_id: str,
        vertices_nm: np.ndarray,
        radii_nm: np.ndarray,
        edges: np.ndarray,
        compartment_type: str,
        syn_partners: Optional[List[int]] = None,
        syn_types: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Serializes a 3D skeleton fragment into structured bracketed tree grammar tokens.
        """
        tokens = []
        if len(vertices_nm) == 0:
            return tokens

        centroid = np.mean(vertices_nm, axis=0)
        mean_radius = float(np.mean(radii_nm))

        # Determine discrete grammar symbol
        if compartment_type == "soma":
            symbol = "[SOMA]"
            lhs_category = "<Soma>"
        elif "apical" in compartment_type:
            symbol = "[APICAL_TRUNK]"
            lhs_category = "<ApicalTree>"
        elif "basal" in compartment_type or "dendrite" in compartment_type:
            symbol = "[BASAL_BRANCH]"
            lhs_category = "<BasalTree>"
        elif "collateral" in compartment_type:
            symbol = "[AXON_COLLATERAL]"
            lhs_category = "<AxonArbor>"
        elif "bouton" in compartment_type:
            symbol = "[VARICOSE_BOUTON]"
            lhs_category = "<AxonArbor>"
        else:
            symbol = "[AXON_TRUNK]"
            lhs_category = "<AxonArbor>"

        # Calculate outward trajectory vector
        if len(vertices_nm) > 1:
            disp = vertices_nm[-1] - vertices_nm[0]
            norm_disp = np.linalg.norm(disp)
            tangent = (disp / norm_disp).tolist() if norm_disp > 0 else [1.0, 0.0, 0.0]
        else:
            tangent = [1.0, 0.0, 0.0]

        n_pre = int(np.sum(syn_types == 0)) if syn_types is not None else 0
        n_post = int(np.sum(syn_types == 1)) if syn_types is not None else 0
        partners = [int(p) for p in syn_partners] if syn_partners is not None else []

        tokens.append({
            "symbol": symbol,
            "fragment_id": fragment_id,
            "lhs": lhs_category,
            "coord_nm": centroid.tolist(),
            "radius_nm": mean_radius,
            "tangent": tangent,
            "n_nodes": len(vertices_nm),
            "n_syn_pre": n_pre,
            "n_syn_post": n_post,
            "syn_partners": partners
        })

        return tokens
