"""Easy / medium / hard difficulty and tube radii."""

from __future__ import annotations

from typing import Literal

Difficulty = Literal["easy", "medium", "hard"]


def tube_radius_um_for_difficulty(d: Difficulty) -> float:
    """Default axis-aligned tube **half-width** scale in xy (µm) by tier.

    These are starting points; tune per experiment. Larger radius = more synapses
    and more tube overlap (use ``dedup.tube_overlap_weights``).
    """
    return {"easy": 8.0, "medium": 15.0, "hard": 25.0}[d]


def difficulty_from_proofread_row(
    *,
    status_dendrite: object | None,
    status_axon: object | None,
    strategy_dendrite: str | None,
    strategy_axon: str | None,
) -> Difficulty:
    """Map proofreading table columns to a coarse difficulty label.

    Heuristic: easy = strong dendrite proofreading; hard = weak or unproofread axon.
    """
    def _truthy(x: object | None) -> bool:
        if x is None:
            return False
        if isinstance(x, bool):
            return x
        s = str(x).strip().lower()
        return s in ("t", "true", "1", "yes")

    d_ok = _truthy(status_dendrite) and (strategy_dendrite or "") not in ("", "none", None)
    a_ok = _truthy(status_axon) and (strategy_axon or "") not in ("", "none", None)

    if d_ok and a_ok:
        return "easy"
    if d_ok:
        return "medium"
    return "hard"
