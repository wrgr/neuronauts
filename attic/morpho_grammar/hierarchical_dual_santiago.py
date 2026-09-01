"""
Hierarchical Dual-Scale SANTIAGO Engine (EXP-046).
Combines:
  1. Stage 1: Local Greedy Proposal Engine (Fast Marching + PCFG + Bidirectional Handshake).
  2. Stage 2: Global Whole-Cell Energy Arbitration (Murray Caliber Taper + Cajal Delay + Synaptic Consistency).
  3. Stage 3: Active Grammar-Guided Oracle Infilling for Long-Range Tears (> 20 um).
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from collections import defaultdict

from neuronauts.morpho_grammar.mcts_handshake_engine import TreeBeamMCTSAssembler
from neuronauts.morpho_grammar.global_hypothesis_search import GlobalMultiHypothesisTreeSearch
from neuronauts.morpho_grammar.active_gap_oracle import ActiveGapOracleEngine


class HierarchicalDualScaleSANTIAGO:
    """
    Unified Hierarchical Connectome Assembler combining Local High-Recall Proposals with Global Whole-Cell Arbitration.
    """
    def __init__(
        self,
        high_conf_thresh: float = 0.70,
        margin_thresh: float = 0.20,
        k_hypotheses: int = 4,
        enable_active_oracle: bool = True,
        max_oracle_queries_per_neuron: int = 4,
        seed: int = 42
    ):
        self.local_engine = TreeBeamMCTSAssembler(
            emb_dim=64,
            beam_width=5,
            geo_weight=2.5,
            cajal_weight=1.5,
            handshake_weight=1.6,
            synaptic_weight=1.2,
            acceptance_threshold=-1.0,
            seed=seed
        )
        self.global_arbitrator = GlobalMultiHypothesisTreeSearch(
            high_conf_thresh=high_conf_thresh,
            margin_thresh=margin_thresh,
            k_hypotheses=k_hypotheses,
            w_cajal=1.2,
            w_murray=1.5,
            w_syn=1.0,
            seed=seed
        )
        self.active_oracle = ActiveGapOracleEngine(
            max_queries_per_neuron=max_oracle_queries_per_neuron,
            seed=seed
        )
        self.enable_active_oracle = enable_active_oracle
        self.rng = np.random.default_rng(seed)

    def assemble_hierarchical_connectome(
        self,
        test_tokens: List[Dict[str, Any]],
        test_pieces_dict: Dict[str, Dict[str, Any]],
        gt_map: Optional[Dict[str, str]] = None
    ) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
        """
        Executes 3-Stage Hierarchical Assembly:
          1. Local Proposals
          2. Global Multi-Hypothesis Energy Arbitration
          3. Active Oracle Infilling (> 20 um)
        """
        tokens_dict = {t["fragment_id"]: t for t in test_tokens}
        candidate_pool = [{"token": t, "fragment_id": t["fragment_id"]} for t in test_tokens]

        test_cells = defaultdict(list)
        for t in test_tokens:
            if not t.get("is_glia", False):
                obj_id = int(t['fragment_id'].split('_')[1])
                test_cells[obj_id].append(t)

        # STAGE 1: Generate Local Proposals
        decision_points = []
        unresolved_cuts = []

        for obj_id, toks in test_cells.items():
            if len(toks) < 2:
                continue
            soma_tok = [t for t in toks if t['symbol'] == '[SOMA]']
            if len(soma_tok) == 0:
                soma_tok = [toks[0]]
            soma_tok = soma_tok[0]

            parent_piece = test_pieces_dict[soma_tok['fragment_id']]
            p_verts = parent_piece['verts']
            p_radii = parent_piece['radii']
            p_edges = parent_piece['edges']

            deg = np.zeros(len(p_verts), dtype=np.int64)
            if len(p_edges) > 0:
                np.add.at(deg, p_edges[:, 0], 1)
                np.add.at(deg, p_edges[:, 1], 1)
            leaf_indices = np.where(deg <= 1)[0]
            if len(leaf_indices) == 0:
                leaf_indices = [len(p_verts) - 1]

            for l_idx in leaf_indices:
                if l_idx == 0 and len(p_verts) > 1:
                    continue

                cut_pos = p_verts[l_idx]
                cut_r = float(p_radii[l_idx])
                conn_e = p_edges[(p_edges[:, 0] == l_idx) | (p_edges[:, 1] == l_idx)]
                if len(conn_e) > 0:
                    neighbor = conn_e[0, 1] if conn_e[0, 0] == l_idx else conn_e[0, 0]
                    disp = p_verts[l_idx] - p_verts[neighbor]
                    norm_d = np.linalg.norm(disp)
                    t_exit = (disp / norm_d).tolist() if norm_d > 0 else [1.0, 0.0, 0.0]
                else:
                    t_exit = [0.0, -1.0, 0.0]

                mask_tok = {
                    "symbol": "[MASK_FRAGMENT]",
                    "coord_nm": cut_pos.tolist(),
                    "radius_nm": cut_r,
                    "tangent": t_exit,
                    "fragment_id": f"mask_{obj_id}_{l_idx}",
                    "syn_partners": soma_tok.get("syn_partners", []),
                    "n_syn_pre": soma_tok.get("n_syn_pre", 0),
                    "n_syn_post": soma_tok.get("n_syn_post", 0)
                }

                res = self.local_engine.run_tree_beam_mcts(
                    parent_token=soma_tok,
                    mask_token=mask_tok,
                    candidate_pool=candidate_pool
                )

                candidates = res.get("beam_candidates", [])
                if len(candidates) > 0:
                    decision_points.append({
                        "parent_id": soma_tok["fragment_id"],
                        "candidates": candidates
                    })
                else:
                    unresolved_cuts.append((soma_tok, mask_tok, obj_id))

        # STAGE 2: Global Whole-Cell Energy Arbitration
        arbitrated_links = self.global_arbitrator.assemble_global_optimal_tree(
            decision_points=decision_points,
            tokens_dict=tokens_dict
        )

        # STAGE 3: Active Large-Gap Infilling (> 20 um)
        oracle_queries_used = 0
        active_links = []

        if self.enable_active_oracle and gt_map is not None:
            for soma_tok, mask_tok, obj_id in unresolved_cuts:
                if oracle_queries_used >= (len(test_cells) * 4):
                    break

                target_candidates = [t for t in test_tokens if not t.get("is_glia", False)]
                true_cell_frags = [t['fragment_id'] for t in test_tokens if gt_map.get(t['fragment_id']) == f"neuron_{obj_id}"]
                gt_target = true_cell_frags[1] if len(true_cell_frags) > 1 else None

                chosen_id, conf, q_used = self.active_oracle.query_long_range_gap(
                    cut_token=mask_tok,
                    candidate_tokens=target_candidates,
                    gt_target_id=gt_target
                )

                oracle_queries_used += q_used
                if chosen_id is not None:
                    active_links.append((soma_tok["fragment_id"], chosen_id))

        final_links = list(arbitrated_links) + active_links

        meta = {
            "local_proposals_count": len(decision_points),
            "arbitrated_joins_count": len(arbitrated_links),
            "oracle_queries_used": oracle_queries_used,
            "active_joins_count": len(active_links),
            "total_final_joins": len(final_links)
        }

        return final_links, meta
