"""Small box-scale assembly search utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateMerge:
    left_agent: int
    right_agent: int
    score: float


@dataclass(frozen=True)
class BeamState:
    groups: tuple[frozenset[int], ...]
    score: float


def _canonical_groups(groups: tuple[frozenset[int], ...]) -> tuple[frozenset[int], ...]:
    return tuple(sorted(groups, key=lambda group: (min(group), len(group), tuple(sorted(group)))))


def _merge_groups(groups: tuple[frozenset[int], ...], left_agent: int, right_agent: int) -> tuple[frozenset[int], ...]:
    left_idx = None
    right_idx = None
    for idx, group in enumerate(groups):
        if left_agent in group:
            left_idx = idx
        if right_agent in group:
            right_idx = idx
    if left_idx is None or right_idx is None or left_idx == right_idx:
        return groups

    merged = frozenset(set(groups[left_idx]) | set(groups[right_idx]))
    new_groups = [group for idx, group in enumerate(groups) if idx not in {left_idx, right_idx}]
    new_groups.append(merged)
    return _canonical_groups(tuple(new_groups))


def beam_search_merge_groups(
    agent_ids: list[int],
    candidates: list[CandidateMerge],
    *,
    beam_width: int = 4,
    atomicity_score_fn=None,
    atomicity_weight: float = 0.25,
) -> tuple[frozenset[int], ...]:
    """Explore a small beam of accept/reject decisions over candidate merges."""
    initial = BeamState(groups=_canonical_groups(tuple(frozenset({agent_id}) for agent_id in agent_ids)), score=0.0)
    beam = [initial]

    for candidate in candidates:
        expanded: list[BeamState] = []
        for state in beam:
            expanded.append(state)
            merged_groups = _merge_groups(state.groups, candidate.left_agent, candidate.right_agent)
            if merged_groups == state.groups:
                continue

            accept_score = state.score + float(candidate.score)
            if atomicity_score_fn is not None:
                target_group = next(group for group in merged_groups if candidate.left_agent in group and candidate.right_agent in group)
                accept_score += float(atomicity_weight) * float(atomicity_score_fn(tuple(sorted(target_group))))
            expanded.append(BeamState(groups=merged_groups, score=accept_score))

        dedup: dict[tuple[frozenset[int], ...], BeamState] = {}
        for state in expanded:
            prev = dedup.get(state.groups)
            if prev is None or state.score > prev.score:
                dedup[state.groups] = state
        beam = sorted(dedup.values(), key=lambda state: state.score, reverse=True)[: max(1, beam_width)]

    return beam[0].groups if beam else initial.groups
