#!/usr/bin/env python3
"""EXP-047 corrected proxy baselines and open-world Hungarian benchmark.

This remains a synthetic smoke benchmark.  Published rows carry a dagger and
are displayed only as context; they are not treated as head-to-head results.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuronauts.assemble import HungarianBipartiteAssembler
from neuronauts.global_merge.eval.benchmark import compute_pairwise_partition_metrics
from neuronauts.morpho_grammar.santiago_v2_grammar import type_segment_v2


def _tok(item: Mapping) -> Mapping:
    return item.get("token", item)


def _fid(item: Mapping) -> str:
    token = _tok(item)
    return str(item.get("fragment_id", token["fragment_id"]))


def _point(item: Mapping, end: bool = False) -> np.ndarray:
    token = _tok(item)
    for key in (("end_nm", "coord_nm", "centroid_nm") if end else ("start_nm", "coord_nm", "centroid_nm")):
        if key in token:
            return np.asarray(token[key], dtype=float)
    return np.zeros(3)


def run_proxy_autoproof(
    test_tokens: Sequence[Mapping], *, max_dist_nm: float = 2_000.0
) -> list[tuple[str, str]]:
    """Compute same-polarity proximity joins on the supplied test set."""
    links = []
    for i, left in enumerate(test_tokens):
        for right in test_tokens[i + 1 :]:
            if type_segment_v2(_tok(left)) != type_segment_v2(_tok(right)):
                continue
            if np.linalg.norm(_point(left, True) - _point(right)) <= max_dist_nm:
                links.append((_fid(left), _fid(right)))
    return links


def run_proxy_neurd(
    test_tokens: Sequence[Mapping],
    test_pieces_dict: Mapping | None = None,
    *,
    max_dist_nm: float = 8_000.0,
) -> list[tuple[str, str]]:
    """Greedily join each axon to its nearest unused axon candidate."""
    del test_pieces_dict
    axons = [item for item in test_tokens if type_segment_v2(_tok(item)) == "Axon"]
    edges = []
    candidates = []
    for i, left in enumerate(axons):
        for right in axons[i + 1 :]:
            d = float(np.linalg.norm(_point(left, True) - _point(right)))
            if d <= max_dist_nm:
                candidates.append((d, _fid(left), _fid(right)))
    used = set()
    for _, left, right in sorted(candidates):
        if left not in used and right not in used:
            edges.append((left, right))
            used.update((left, right))
    return edges


def links_to_pred_map(fragment_ids: Sequence[str], links: Sequence[tuple[str, str]]) -> dict[str, str]:
    parent = {str(x): str(x) for x in fragment_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links:
        a, b = str(a), str(b)
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    return {x: find(x) for x in parent}


class _Handshake:
    def evaluate_bidirectional_handshake(self, left: Mapping, right: Mapping) -> float:
        distance = np.linalg.norm(_point(left, True) - _point(right))
        polarity = type_segment_v2(left) == type_segment_v2(right)
        return float(np.exp(-distance / 1_500.0) * (1.0 if polarity else 0.1))


def _world(seed: int = 42) -> tuple[list[dict], dict[str, str], set[str]]:
    rng = np.random.default_rng(seed)
    tokens, gt, test_ids = [], {}, set()
    # All 150 cells enter inference; only the spatially separated last 30 are scored.
    for cell in range(150):
        base = rng.uniform(0, 100_000, 3)
        for piece in range(3):
            fid = f"n{cell}-p{piece}"
            point = base + np.array([piece * 600.0, 0, 0]) + rng.normal(0, 30, 3)
            tokens.append({"fragment_id": fid, "start_nm": point, "end_nm": point + [300, 0, 0],
                           "mean_radius_nm": 90, "n_pre": 8, "n_post": 1})
            if cell >= 120:
                gt[fid] = f"n{cell}"
                test_ids.add(fid)
    for glia in range(15):
        point = rng.uniform(0, 100_000, 3)
        tokens.append({"fragment_id": f"g{glia}", "coord_nm": point,
                       "mean_radius_nm": 250, "n_pre": 0, "n_post": 0, "is_glia": True})
    return tokens, gt, test_ids


def main() -> None:
    all_tokens, gt_map, test_ids = _world()
    test_tokens = [token for token in all_tokens if token["fragment_id"] in test_ids]
    methods = {
        "AutoProof-proxy": run_proxy_autoproof(test_tokens),
        "NEURD-proxy": run_proxy_neurd(test_tokens),
    }
    assembler = HungarianBipartiteAssembler(_Handshake(), max_search_dist_nm=8_000, verbose=True)
    # Cuts are test-only, while every fragment in the volume remains a confuser.
    methods["SANTIAGO-Hungarian"] = assembler.assemble_volume_bipartite(
        test_tokens, candidate_pool=all_tokens
    )[0]
    print(f"Candidate pool: {len(all_tokens)} fragments; evaluated: {len(test_ids)}")
    for name, links in methods.items():
        pred = links_to_pred_map(list(gt_map), links)
        m = compute_pairwise_partition_metrics(pred, gt_map)
        print(f"{name:22s} joins={len(links):4d} ARI={m['ari']:.3f} P={m['merge_P']:.3f} R={m['merge_R']:.3f}")
    print("† Lifted Multicut / SegCLR / RoboEM: published on a different neuron set; not re-evaluated or directly comparable.")


if __name__ == "__main__":
    main()
