"""Ground-truth overlay: atom purity, tiers, and pair labels.

The pair rules decide what the paper is allowed to claim, so they are checked
case by case rather than in aggregate.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuronauts.harness.labels import (
    LABEL_NEG, LABEL_POS, LABEL_UNKNOWN, TIER_GOLD, TIER_NONE, TIER_SILVER,
    lookup_index, pair_labels, summarize, tally_atom_targets,
)


def tally(sides, tiers=None, **kw):
    a = np.asarray([s[0] for s in sides], np.uint64)
    t = np.asarray([s[1] for s in sides], np.uint64)
    return tally_atom_targets(a, t, tiers=tiers, **kw)


def test_pure_atom_has_one_owner():
    lab = tally([(1, 100)] * 5)
    assert lab.atom_id.tolist() == [1]
    assert int(lab.owner[0]) == 100
    assert lab.owner_frac[0] == pytest.approx(1.0)
    assert lab.n_roots[0] == 1
    assert bool(lab.pure[0]) and not bool(lab.mixed[0])


def test_single_stray_side_does_not_make_a_frankenmerge():
    lab = tally([(1, 100)] * 40 + [(1, 200)])
    assert lab.n_roots_raw[0] == 2
    assert lab.n_roots[0] == 1          # the stray fails both robustness tests
    assert bool(lab.pure[0])


def test_two_real_targets_make_a_mixed_atom():
    lab = tally([(1, 100)] * 20 + [(1, 200)] * 20)
    assert lab.n_roots[0] == 2
    assert bool(lab.mixed[0]) and not bool(lab.pure[0])


def test_owner_is_the_dominant_target():
    lab = tally([(1, 100)] * 20 + [(1, 200)] * 7)
    assert int(lab.owner[0]) == 100
    assert lab.owner_frac[0] == pytest.approx(20 / 27)


def test_zero_atom_or_target_sides_are_dropped():
    lab = tally([(0, 100), (1, 0), (1, 100), (1, 100)])
    assert lab.atom_id.tolist() == [1]
    assert lab.n_sides[0] == 2


def test_tier_of_owner_is_carried():
    lab = tally([(1, 100)] * 4 + [(2, 200)] * 4 + [(3, 300)] * 4,
                tiers={100: TIER_GOLD, 200: TIER_SILVER})
    order = np.argsort(lab.atom_id)
    assert lab.owner_tier[order].tolist() == [TIER_GOLD, TIER_SILVER, TIER_NONE]


def test_mixed_proofread_counts_only_verified_targets():
    lab = tally([(1, 100)] * 10 + [(1, 200)] * 10, tiers={100: TIER_GOLD})
    assert lab.n_roots[0] == 2
    assert lab.n_roots_proofread[0] == 1
    assert not bool(lab.mixed_proofread[0])


def test_lookup_index_handles_missing():
    keys = np.asarray([7, 3, 9], np.uint64)
    got = lookup_index(keys, np.asarray([9, 4, 7], np.uint64))
    assert got.tolist() == [2, -1, 0]


# ---------------------------------------------------------------------------
# pair rules
# ---------------------------------------------------------------------------

def two_atom_labels(tier_a, tier_b, same_owner):
    owner_b = 100 if same_owner else 200
    tiers = {100: tier_a, owner_b: tier_b}
    return tally([(1, 100)] * 6 + [(2, owner_b)] * 6, tiers=tiers)


@pytest.mark.parametrize("tier,expected", [
    (TIER_GOLD, LABEL_POS), (TIER_SILVER, LABEL_POS), (TIER_NONE, LABEL_UNKNOWN)])
def test_same_owner_positive_only_when_proofread(tier, expected):
    lab = two_atom_labels(tier, tier, same_owner=True)
    got = pair_labels(lab, np.asarray([1], np.uint64), np.asarray([2], np.uint64))
    assert int(got[0]) == expected


@pytest.mark.parametrize("ta,tb,expected", [
    (TIER_GOLD, TIER_NONE, LABEL_NEG),      # gold is complete: outside means outside
    (TIER_NONE, TIER_GOLD, LABEL_NEG),
    (TIER_GOLD, TIER_GOLD, LABEL_NEG),
    (TIER_SILVER, TIER_SILVER, LABEL_NEG),  # two verified cells are two cells
    (TIER_SILVER, TIER_NONE, LABEL_UNKNOWN),  # silver axon may be unfinished
    (TIER_NONE, TIER_NONE, LABEL_UNKNOWN),
])
def test_different_owner_negative_rules(ta, tb, expected):
    lab = two_atom_labels(ta, tb, same_owner=False)
    got = pair_labels(lab, np.asarray([1], np.uint64), np.asarray([2], np.uint64))
    assert int(got[0]) == expected


def test_mixed_atom_pairs_are_unknown():
    lab = tally([(1, 100)] * 10 + [(1, 200)] * 10 + [(2, 100)] * 6,
                tiers={100: TIER_GOLD, 200: TIER_GOLD})
    got = pair_labels(lab, np.asarray([1], np.uint64), np.asarray([2], np.uint64))
    assert int(got[0]) == LABEL_UNKNOWN


def test_lenient_mode_ignores_proofread_status():
    lab = two_atom_labels(TIER_NONE, TIER_NONE, same_owner=True)
    strict = pair_labels(lab, np.asarray([1], np.uint64), np.asarray([2], np.uint64))
    lenient = pair_labels(lab, np.asarray([1], np.uint64),
                          np.asarray([2], np.uint64), mode="lenient")
    assert int(strict[0]) == LABEL_UNKNOWN
    assert int(lenient[0]) == LABEL_POS


def test_unknown_atom_ids_are_unknown():
    lab = two_atom_labels(TIER_GOLD, TIER_GOLD, same_owner=True)
    got = pair_labels(lab, np.asarray([1, 77], np.uint64),
                      np.asarray([2, 88], np.uint64))
    assert int(got[1]) == LABEL_UNKNOWN


def test_summarize_counts_tiers():
    lab = tally([(1, 100)] * 6 + [(2, 200)] * 6 + [(3, 300)] * 6
                + [(4, 100)] * 4 + [(4, 400)] * 4,
                tiers={100: TIER_GOLD, 200: TIER_SILVER})
    s = summarize(lab)
    assert s["n_atoms"] == 4
    assert s["n_pure"] == 3 and s["n_mixed"] == 1
    assert s["n_pure_gold"] == 1 and s["n_pure_silver"] == 1
    assert s["n_pure_unproofread"] == 1
    assert s["n_owner_roots_proofread"] == 2
