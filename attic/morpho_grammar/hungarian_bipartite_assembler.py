"""
EXP-047/EXP-049/EXP-050: Hungarian Bipartite Matching Assembler powered by TreeBeamMCTS scoring.
Includes Multi-Round Iterative Bipartite Growth to assemble full arbor paths in dense subvolumes.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

from neuronauts.morpho_grammar.mcts_handshake_engine import TreeBeamMCTSAssembler
from neuronauts.morpho_grammar.santiago_v2_grammar import apply_hard_biological_veto


def extract_piece_endpoints(verts: np.ndarray, edges: np.ndarray, radii: np.ndarray) -> List[Dict[str, Any]]:
    """Extracts all boundary and leaf endpoints (degree <= 1) and root/branch vertices of a skeleton piece."""
    N = len(verts)
    if N == 0:
        return []
    if N == 1:
        return [{"idx": 0, "pos": verts[0], "radius": float(radii[0]) if len(radii) > 0 else 100.0, "tangent": np.array([1.0, 0.0, 0.0], dtype=np.float32)}]

    deg = np.zeros(N, dtype=np.int64)
    if len(edges) > 0:
        np.add.at(deg, edges[:, 0], 1)
        np.add.at(deg, edges[:, 1], 1)

    candidate_indices = set(np.where(deg <= 1)[0].tolist())
    candidate_indices.add(0)

    endpoints = []
    for l_idx in candidate_indices:
        pos = verts[l_idx]
        r = float(radii[l_idx]) if len(radii) > l_idx else 100.0
        conn_e = edges[(edges[:, 0] == l_idx) | (edges[:, 1] == l_idx)]
        if len(conn_e) > 0:
            neighbor = conn_e[0, 1] if conn_e[0, 0] == l_idx else conn_e[0, 0]
            disp = verts[l_idx] - verts[neighbor]
            norm_d = np.linalg.norm(disp)
            tan = (disp / norm_d).astype(np.float32) if norm_d > 0 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            tan = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        endpoints.append({"idx": int(l_idx), "pos": pos, "radius": r, "tangent": tan})
    return endpoints


def get_closest_vertex_and_tangent(verts: np.ndarray, edges: np.ndarray, radii: np.ndarray, target_pos: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Finds the closest vertex on a piece to target_pos and its outgoing local tangent."""
    if len(verts) == 0:
        return target_pos, 100.0, np.array([1.0, 0.0, 0.0], dtype=np.float32), float("inf")
    dists = np.linalg.norm(verts - target_pos, axis=1)
    best_idx = int(np.argmin(dists))
    min_dist = float(dists[best_idx])
    pos = verts[best_idx]
    r = float(radii[best_idx]) if len(radii) > best_idx else 100.0

    conn_e = edges[(edges[:, 0] == best_idx) | (edges[:, 1] == best_idx)]
    if len(conn_e) > 0:
        neighbor = conn_e[0, 1] if conn_e[0, 0] == best_idx else conn_e[0, 0]
        disp = verts[neighbor] - verts[best_idx]
        norm_d = np.linalg.norm(disp)
        tan = (disp / norm_d).astype(np.float32) if norm_d > 0 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        tan = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    return pos, r, tan, min_dist


class HungarianBipartiteAssembler:
    """
    Hungarian Bipartite Assembler with Multi-Round Iterative Growth.
    Scores candidates per cut directly with bidirectional handshake + geometry,
    then applies linear_sum_assignment to prevent collisions and resolve 1-to-1 competition.
    """
    def __init__(
        self,
        emb_dim: int = 64,
        beam_width: int = 5,
        geo_weight: float = 2.5,
        cajal_weight: float = 1.5,
        handshake_weight: float = 1.6,
        synaptic_weight: float = 1.2,
        acceptance_threshold: float = -2.0,
        max_search_dist_nm: float = 25000.0,
        seed: int = 42,
        **kwargs
    ):
        syn_w = kwargs.get("synaptic_jaccard_weight", synaptic_weight)
        acc_t = kwargs.get("min_acceptance_score", acceptance_threshold)
        self.max_search_dist_nm = kwargs.get("max_search_dist_nm", max_search_dist_nm)
        self.acceptance_threshold = acc_t
        self.mcts_engine = TreeBeamMCTSAssembler(
            emb_dim=emb_dim,
            beam_width=beam_width,
            geo_weight=geo_weight,
            cajal_weight=cajal_weight,
            handshake_weight=handshake_weight,
            synaptic_weight=syn_w,
            acceptance_threshold=acc_t,
            seed=seed
        )

    def assemble_volume_bipartite(
        self,
        test_tokens: List[Dict[str, Any]],
        test_pieces_dict: Dict[str, Dict[str, Any]],
        candidate_pool: Optional[List[Dict[str, Any]]] = None,
        max_rounds: int = 5
    ) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
        """
        Multi-Round Iterative Hungarian Bipartite Assembly.
        Grows arbors recursively across the volume, solving collision-free 1-to-1 bipartite matching in each round.
        """
        if candidate_pool is None:
            candidate_pool = [{"token": t, "fragment_id": t["fragment_id"]} for t in test_tokens]

        tokens_by_id = {t["fragment_id"]: t for t in test_tokens}
        for cand in candidate_pool:
            cid = cand["fragment_id"]
            if cid not in tokens_by_id:
                tokens_by_id[cid] = cand["token"]

        # Disjoint set to track connected components and avoid self-cycles
        parent_map = {cid: cid for cid in tokens_by_id}

        def find(u: str) -> str:
            if parent_map[u] != u:
                parent_map[u] = find(parent_map[u])
            return parent_map[u]

        def union(u: str, v: str):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent_map[ru] = rv

        # Seed fragments (Somas first; if none, all non-glia)
        soma_tokens = [t for t in test_tokens if (t.get("symbol") == "[SOMA]" or t.get("inferred_type") == "Soma") and not t.get("is_glia", False)]
        if len(soma_tokens) == 0:
            soma_tokens = [t for t in test_tokens if not t.get("is_glia", False)]

        active_frontier_ids: Set[str] = {t["fragment_id"] for t in soma_tokens}
        final_links: List[Tuple[str, str]] = []
        total_cuts_count = 0
        used_cuts: Set[str] = set()

        for round_num in range(1, max_rounds + 1):
            cell_cuts = []
            for frontier_id in active_frontier_ids:
                p_tok = tokens_by_id.get(frontier_id)
                p_piece = test_pieces_dict.get(frontier_id)
                if p_tok is None or p_piece is None or len(p_piece.get("verts", [])) == 0:
                    continue

                p_endpoints = extract_piece_endpoints(p_piece["verts"], p_piece["edges"], p_piece["radii"])
                for ep in p_endpoints:
                    cut_key = f"{frontier_id}_{ep['idx']}"
                    if cut_key in used_cuts:
                        continue

                    cut_pos = ep["pos"]
                    cut_r = ep["radius"]
                    cut_tan = ep["tangent"]

                    mask_tok = {
                        "symbol": "[MASK_FRAGMENT]",
                        "coord_nm": cut_pos.tolist(),
                        "radius_nm": cut_r,
                        "tangent": cut_tan.tolist(),
                        "fragment_id": f"mask_{frontier_id}_{ep['idx']}",
                        "syn_partners": p_tok.get("syn_partners", []),
                        "n_syn_pre": p_tok.get("n_syn_pre", 0),
                        "n_syn_post": p_tok.get("n_syn_post", 0)
                    }

                    # Find candidates across candidate pool (excluding same cluster)
                    frontier_cluster = find(frontier_id)
                    candidates = []

                    for cand in candidate_pool:
                        cid = cand["fragment_id"]
                        if find(cid) == frontier_cluster:
                            continue  # avoid cycles within same arbor

                        p_cand = test_pieces_dict.get(cid)
                        if p_cand is not None and len(p_cand.get("verts", [])) > 0:
                            c_pos, c_r, c_tan, min_dist = get_closest_vertex_and_tangent(
                                p_cand["verts"], p_cand["edges"], p_cand["radii"], cut_pos
                            )
                        else:
                            c_pos = np.array(cand["token"].get("coord_nm", [0.0, 0.0, 0.0]), dtype=np.float32)
                            c_r = float(cand["token"].get("radius_nm", 100.0))
                            c_tan = np.array(cand["token"].get("tangent", [1.0, 0.0, 0.0]), dtype=np.float32)
                            min_dist = float(np.linalg.norm(cut_pos - c_pos))

                        if min_dist > self.max_search_dist_nm:
                            continue

                        c_tok_adj = dict(cand["token"])
                        c_tok_adj["coord_nm"] = c_pos.tolist()
                        c_tok_adj["radius_nm"] = c_r
                        c_tok_adj["tangent"] = c_tan.tolist()

                        score, p_hs, details = self.mcts_engine.evaluate_bidirectional_handshake(
                            parent_token=p_tok,
                            mask_token=mask_tok,
                            candidate_token=c_tok_adj,
                            p_forward=0.50
                        )
                        if not details.get("vetoed", False) and score >= self.acceptance_threshold:
                            candidates.append({
                                "fragment_id": cid,
                                "score": score,
                                "dist_nm": min_dist,
                                "p_hs": p_hs
                            })

                    if len(candidates) > 0:
                        cell_cuts.append({
                            "parent_id": frontier_id,
                            "cut_key": cut_key,
                            "candidates": candidates
                        })

            if len(cell_cuts) == 0:
                break  # No more candidate extensions found

            total_cuts_count += len(cell_cuts)

            # Global Hungarian Bipartite Assignment for round
            cand_id_set = set()
            for cut in cell_cuts:
                for c in cut["candidates"]:
                    cand_id_set.add(c["fragment_id"])

            cand_list = list(cand_id_set)
            if len(cand_list) == 0:
                break

            N_cuts = len(cell_cuts)
            M_cands = len(cand_list)
            cand_to_col = {cid: idx for idx, cid in enumerate(cand_list)}

            slack_cost = -self.acceptance_threshold
            cost_matrix = np.full((N_cuts, M_cands + N_cuts), fill_value=1e5, dtype=np.float64)

            for i, cut in enumerate(cell_cuts):
                for c in cut["candidates"]:
                    cid = c["fragment_id"]
                    j = cand_to_col[cid]
                    score = c["score"]
                    if score >= self.acceptance_threshold:
                        cost_matrix[i, j] = -score

                cost_matrix[i, M_cands + i] = slack_cost

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            new_frontier_ids = set()
            new_links_count = 0

            for r, c in zip(row_ind, col_ind):
                if c < M_cands and cost_matrix[r, c] <= slack_cost:
                    parent_id = cell_cuts[r]["parent_id"]
                    target_id = cand_list[c]
                    cut_key = cell_cuts[r]["cut_key"]

                    if find(parent_id) != find(target_id):
                        final_links.append((parent_id, target_id))
                        union(parent_id, target_id)
                        used_cuts.add(cut_key)
                        new_frontier_ids.add(target_id)
                        new_links_count += 1

            if new_links_count == 0:
                break  # Convergence

            active_frontier_ids = new_frontier_ids

        meta = {
            "total_cuts": total_cuts_count,
            "total_candidates": len(candidate_pool),
            "matched_joins": len(final_links),
            "assignment_density": float(len(final_links) / max(1, total_cuts_count))
        }

        return final_links, meta
