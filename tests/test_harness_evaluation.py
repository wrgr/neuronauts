"""Ranking and assembly metrics.

The pair counters are written in closed form over group counts because the
panel has millions of pairs; every one of them is checked here against a
brute-force enumeration on random small inputs, which is the only way to trust
an O(1)-per-group formula.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from neuronauts.harness.evaluation import (
    adjusted_rand_index, assemble_at_threshold, assembly_metrics,
    average_precision, precision_recall_curve, rank_metrics, roc_auc,
    strict_pair_counts, union_find_components,
)
from neuronauts.harness.labels import (
    LABEL_NEG, LABEL_POS, TIER_GOLD, TIER_NONE, TIER_SILVER, AtomLabels,
)


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------

def test_auc_perfect_and_inverted():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)


def test_auc_all_ties_is_half():
    y = np.array([0, 1, 0, 1])
    assert roc_auc(y, np.ones(4)) == pytest.approx(0.5)


def test_auc_matches_brute_force_on_random_data():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, 40)
        s = rng.integers(0, 6, 40).astype(float)   # ties on purpose
        pos, neg = s[y == 1], s[y == 0]
        if not len(pos) or not len(neg):
            continue
        brute = np.mean([(1.0 if p > n else 0.5 if p == n else 0.0)
                         for p in pos for n in neg])
        assert roc_auc(y, s) == pytest.approx(brute)


def test_auc_undefined_without_both_classes():
    assert np.isnan(roc_auc(np.zeros(5), np.arange(5.0)))


def test_average_precision_perfect_ranking():
    y = np.array([1, 1, 0, 0])
    assert average_precision(y, np.array([4.0, 3, 2, 1])) == pytest.approx(1.0)


def test_precision_recall_curve_endpoints():
    y = np.array([1, 0, 1, 0])
    p, r, thr = precision_recall_curve(y, np.array([4.0, 3, 2, 1]))
    assert r[-1] == pytest.approx(1.0)
    assert p[0] == pytest.approx(1.0)
    assert len(p) == len(r) == len(thr)


def test_rank_metrics_operating_points():
    y = np.array([1, 1, 1, 0, 0, 0])
    m = rank_metrics(y, np.array([6.0, 5, 4, 3, 2, 1]))
    assert m["auc"] == pytest.approx(1.0)
    assert m["precision_at_recall_0.9"] == pytest.approx(1.0)
    assert m["recall_at_precision_0.99"] == pytest.approx(1.0)
    assert m["n_pos"] == 3 and m["n_neg"] == 3


def test_rank_metrics_reports_zero_when_target_unreachable():
    y = np.array([0, 0, 1, 1])
    m = rank_metrics(y, np.array([4.0, 3, 2, 1]))          # inverted
    assert m["recall_at_precision_0.99"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# clustering primitives
# ---------------------------------------------------------------------------

def test_union_find_chains_transitively():
    comp = union_find_components(5, np.array([0, 1, 3]), np.array([1, 2, 4]))
    assert comp[0] == comp[1] == comp[2]
    assert comp[3] == comp[4]
    assert comp[0] != comp[3]


def test_union_find_no_edges_is_all_singletons():
    comp = union_find_components(4, np.zeros(0, int), np.zeros(0, int))
    assert len(np.unique(comp)) == 4


def test_ari_identical_and_random():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert adjusted_rand_index(a, a) == pytest.approx(1.0)
    allone = np.zeros(6, int)
    assert abs(adjusted_rand_index(allone, a)) < 1e-9


def test_ari_matches_pair_counting_definition():
    rng = np.random.default_rng(3)
    for _ in range(10):
        pred = rng.integers(0, 4, 30)
        true = rng.integers(0, 3, 30)
        n = len(pred)
        same_p = np.array([pred[i] == pred[j] for i, j in combinations(range(n), 2)])
        same_t = np.array([true[i] == true[j] for i, j in combinations(range(n), 2)])
        a = float((same_p & same_t).sum())
        b = float((same_p & ~same_t).sum())
        c = float((~same_p & same_t).sum())
        d = float((~same_p & ~same_t).sum())
        exp = (a + b) * (a + c) / (a + b + c + d)
        maxi = ((a + b) + (a + c)) / 2.0
        brute = (a - exp) / (maxi - exp)
        assert adjusted_rand_index(pred, true) == pytest.approx(brute)


# ---------------------------------------------------------------------------
# strict pair counting vs brute force
# ---------------------------------------------------------------------------

def brute_pair_counts(cluster, owner, tier, pure):
    tp = fp = pos_total = neg_total = 0
    n = len(cluster)
    for i, j in combinations(range(n), 2):
        if not (pure[i] and pure[j]):
            continue
        same = owner[i] == owner[j]
        ta, tb = tier[i], tier[j]
        if same and ta > TIER_NONE:
            pos_total += 1
            if cluster[i] == cluster[j]:
                tp += 1
        elif not same and (ta == TIER_GOLD or tb == TIER_GOLD
                           or (ta > TIER_NONE and tb > TIER_NONE)):
            neg_total += 1
            if cluster[i] == cluster[j]:
                fp += 1
    return {"tp": tp, "fp": fp, "fn": pos_total - tp,
            "pos_total": pos_total, "neg_total": neg_total}


@pytest.mark.parametrize("seed", range(25))
def test_strict_pair_counts_match_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 40))
    owners = rng.integers(100, 100 + int(rng.integers(2, 6)), n).astype(np.uint64)
    # tier is a property of the owner root, as in the real table
    tier_of_owner = {int(o): int(rng.choice([TIER_NONE, TIER_SILVER, TIER_GOLD]))
                     for o in np.unique(owners)}
    tier = np.array([tier_of_owner[int(o)] for o in owners], np.int8)
    cluster = rng.integers(0, int(rng.integers(1, 5)) + 1, n)
    pure = rng.random(n) > 0.25
    got = strict_pair_counts(cluster, owners, tier, pure)
    want = brute_pair_counts(cluster, owners, tier, pure)
    for k in want:
        assert got[k] == pytest.approx(want[k]), k


def test_strict_pair_counts_handles_no_pure_atoms():
    got = strict_pair_counts(np.array([0, 0]), np.array([1, 2], np.uint64),
                             np.array([TIER_GOLD, TIER_GOLD], np.int8),
                             np.array([False, False]))
    assert got["tp"] == 0 and got["fp"] == 0 and got["pos_total"] == 0


# ---------------------------------------------------------------------------
# assembly metrics
# ---------------------------------------------------------------------------

def make_labels(owner, tier, pure_frac=1.0):
    owner = np.asarray(owner, np.uint64)
    n = len(owner)
    return AtomLabels(
        atom_id=np.arange(1, n + 1, dtype=np.uint64), owner=owner,
        owner_frac=np.full(n, pure_frac, np.float32),
        owner_tier=np.asarray(tier, np.int8), n_sides=np.full(n, 10, np.int32),
        n_roots_raw=np.ones(n, np.int32), n_roots=np.ones(n, np.int32),
        n_roots_proofread=np.ones(n, np.int32),
        meta={"pure_min_owner_frac": 0.9})


def test_perfect_assembly_scores_one():
    labels = make_labels([10, 10, 20, 20], [TIER_GOLD] * 4)
    m = assembly_metrics(np.array([0, 0, 1, 1]), np.arange(4), labels)
    assert m["ari_labelled"] == pytest.approx(1.0)
    assert m["merge_precision"] == pytest.approx(1.0)
    assert m["merge_recall"] == pytest.approx(1.0)
    assert m["n_contaminated_clusters"] == 0


def test_collapsed_assembly_is_caught():
    labels = make_labels([10, 10, 20, 20], [TIER_GOLD] * 4)
    m = assembly_metrics(np.zeros(4, int), np.arange(4), labels)
    assert m["merge_recall"] == pytest.approx(1.0)
    assert m["merge_precision"] < 0.6
    assert m["largest_cluster"] == 4
    assert m["n_contaminated_clusters"] == 1


def test_do_nothing_has_perfect_precision_and_zero_recall():
    labels = make_labels([10, 10, 20, 20], [TIER_GOLD] * 4)
    m = assembly_metrics(np.arange(4), np.arange(4), labels)
    assert m["merge_recall"] == pytest.approx(0.0)
    assert np.isnan(m["merge_precision"])       # no merge proposed at all
    assert m["merge_fp_pairs"] == 0


def test_unlabelled_atoms_do_not_enter_the_metrics():
    labels = make_labels([10, 10, 20], [TIER_GOLD, TIER_GOLD, TIER_NONE])
    idx = np.array([0, 1, 2, -1])                       # last atom has no row
    m = assembly_metrics(np.array([0, 0, 0, 0]), idx, labels)
    assert m["n_labelled_atoms"] == 2
    assert m["merge_tp_pairs"] == 1
    # the unproofread atom pairs with two gold atoms -> two false merges
    assert m["merge_fp_pairs"] == 2


def test_cable_weighted_purity_and_completeness():
    labels = make_labels([10, 10, 20, 20], [TIER_GOLD] * 4)
    cable = np.array([100.0, 100.0, 100.0, 100.0])
    perfect = assembly_metrics(np.array([0, 0, 1, 1]), np.arange(4), labels,
                               cable_nm=cable)
    assert perfect["cable_purity"] == pytest.approx(1.0)
    assert perfect["cable_completeness"] == pytest.approx(1.0)
    collapsed = assembly_metrics(np.zeros(4, int), np.arange(4), labels,
                                 cable_nm=cable)
    assert collapsed["cable_purity"] == pytest.approx(0.5)
    assert collapsed["cable_completeness"] == pytest.approx(1.0)
    split = assembly_metrics(np.arange(4), np.arange(4), labels, cable_nm=cable)
    assert split["cable_purity"] == pytest.approx(1.0)
    assert split["cable_completeness"] == pytest.approx(0.5)


def test_assemble_at_threshold_respects_score():
    comp = assemble_at_threshold(3, np.array([0, 1]), np.array([1, 2]),
                                 np.array([0.9, 0.1]), 0.5)
    assert comp[0] == comp[1] != comp[2]
