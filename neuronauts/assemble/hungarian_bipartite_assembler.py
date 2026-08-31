"""Collision-aware volume assembly using exhaustive local handshake scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def _token(item: Mapping[str, Any]) -> Mapping[str, Any]:
    return item.get("token", item)


def _fragment_id(item: Mapping[str, Any]) -> str:
    tok = _token(item)
    return str(item.get("fragment_id", tok.get("fragment_id")))


def _point(tok: Mapping[str, Any], *, end: bool) -> np.ndarray:
    names = ("end_nm", "endpoint_nm", "coord_nm", "coords_nm", "centroid_nm") if end else (
        "start_nm", "endpoint_nm", "coord_nm", "coords_nm", "centroid_nm"
    )
    for name in names:
        if name in tok:
            value = np.asarray(tok[name], dtype=float)
            if value.shape == (3,):
                return value
            if value.ndim == 2 and value.shape[1] == 3 and len(value):
                return value[-1 if end else 0]
    return np.zeros(3, dtype=float)


class HungarianBipartiteAssembler:
    """Assign cut fragments to candidates, allowing each target at most once.

    Unlike the former grammar proposal beam, every candidate inside the spatial
    gate is evaluated by the bidirectional handshake scorer.  The scorer may be
    an MCTS engine exposing ``evaluate_bidirectional_handshake`` or any callable.
    """

    def __init__(
        self,
        mcts_engine: Any,
        *,
        max_search_dist_nm: float = 10_000.0,
        max_cands: int = 50,
        acceptance_threshold: float = 0.5,
        verbose: bool = True,
    ) -> None:
        self.mcts_engine = mcts_engine
        self.max_search_dist_nm = float(max_search_dist_nm)
        self.max_cands = int(max_cands)
        self.acceptance_threshold = float(acceptance_threshold)
        self.verbose = bool(verbose)

    def _score(self, left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
        scorer = getattr(self.mcts_engine, "evaluate_bidirectional_handshake", self.mcts_engine)
        result = scorer(left, right)
        if isinstance(result, Mapping):
            for key in ("score", "handshake_score", "probability", "confidence"):
                if key in result:
                    return float(result[key])
            raise ValueError("handshake result contains no recognized score")
        return float(result)

    def assemble_volume_bipartite(
        self,
        test_tokens: Sequence[Mapping[str, Any]],
        test_pieces_dict: Mapping[str, Any] | None = None,
        *,
        candidate_pool: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[list[tuple[str, str]], dict[str, Any]]:
        del test_pieces_dict  # retained for compatibility with benchmark callers
        cuts = list(test_tokens)
        pool = list(candidate_pool if candidate_pool is not None else test_tokens)
        candidate_ids = sorted({_fragment_id(x) for x in pool})
        col = {fid: i for i, fid in enumerate(candidate_ids)}
        scores = np.full((len(cuts), len(candidate_ids)), -np.inf, dtype=float)
        n_scored = 0

        for i, cut_item in enumerate(cuts):
            cut, cut_id = _token(cut_item), _fragment_id(cut_item)
            ranked: list[tuple[float, Mapping[str, Any]]] = []
            cut_point = _point(cut, end=True)
            for cand_item in pool:
                cand_id = _fragment_id(cand_item)
                if cand_id == cut_id:
                    continue
                distance = float(np.linalg.norm(cut_point - _point(_token(cand_item), end=False)))
                if distance <= self.max_search_dist_nm:
                    ranked.append((distance, cand_item))
            ranked.sort(key=lambda pair: (pair[0], _fragment_id(pair[1])))
            for _, cand_item in ranked[: self.max_cands]:
                score = self._score(cut, _token(cand_item))
                scores[i, col[_fragment_id(cand_item)]] = score
                n_scored += 1

        links: list[tuple[str, str]] = []
        if scores.size and np.isfinite(scores).any():
            cost = np.where(np.isfinite(scores), -scores, 1e9)
            rows, cols = linear_sum_assignment(cost)
            for row, column in zip(rows.tolist(), cols.tolist()):
                if np.isfinite(scores[row, column]) and scores[row, column] >= self.acceptance_threshold:
                    links.append((_fragment_id(cuts[row]), candidate_ids[column]))

        meta = {
            "n_cuts": len(cuts),
            "n_candidates": len(candidate_ids),
            "n_cands_scored": n_scored,
            "n_accepted": len(links),
            "candidate_pool_size": len(pool),
        }
        if self.verbose:
            print("Hungarian diagnostics: " + ", ".join(f"{k}={v}" for k, v in meta.items()))
        return links, meta


__all__ = ["HungarianBipartiteAssembler"]
