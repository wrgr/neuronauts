"""Partition metrics: ARI, pair P/R, entropy metrics, VI, purity, ERL.

Where a reference exists (scikit-learn, or the pre-consolidation
implementations in the repo), the new code is checked against it rather than
against hand-computed numbers alone.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from neuronauts.metrics import (
    adjusted_rand_index,
    cluster_purity,
    contingency,
    expected_run_length,
    partition_metrics,
    rand_disagreement,
    variation_of_information,
)


# ---------------------------------------------------------------------------
# ARI
# ---------------------------------------------------------------------------

def test_ari_identical_partitions_is_one():
    labels = np.array([1, 1, 2, 2, 3])
    assert adjusted_rand_index(labels, labels.copy()) == pytest.approx(1.0)


def test_ari_relabelling_does_not_matter():
    true = np.array([1, 1, 2, 2])
    pred = np.array([9, 9, 4, 4])
    assert adjusted_rand_index(true, pred) == pytest.approx(1.0)


def test_ari_all_one_cluster_is_zero():
    true = np.array([1, 1, 2, 2])
    pred = np.zeros(4, dtype=int)
    assert adjusted_rand_index(true, pred) == pytest.approx(0.0)


def test_ari_all_singletons_is_zero():
    true = np.array([1, 1, 2, 2])
    assert adjusted_rand_index(true, np.arange(4)) == pytest.approx(0.0)


def test_ari_worse_than_chance_is_negative():
    true = np.array([0, 0, 0, 1, 1, 1])
    pred = np.array([0, 1, 0, 1, 0, 1])
    assert adjusted_rand_index(true, pred) < 0.0


def test_ari_matches_sklearn_on_random_partitions():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(11)
    for _ in range(20):
        true = rng.integers(0, 6, size=120)
        pred = rng.integers(0, 5, size=120)
        assert adjusted_rand_index(true, pred) == pytest.approx(
            sk.adjusted_rand_score(true, pred))


def test_ari_matches_legacy_numpy_implementation():
    from neuronauts.assemble.partition_gnn import _adjusted_rand_score_np
    rng = np.random.default_rng(5)
    true = rng.integers(1, 7, size=200)
    pred = rng.integers(1, 5, size=200)
    assert adjusted_rand_index(true, pred) == pytest.approx(
        _adjusted_rand_score_np(true, pred))


def test_ari_single_item_is_one():
    assert adjusted_rand_index([1], [1]) == 1.0


# ---------------------------------------------------------------------------
# pair precision / recall
# ---------------------------------------------------------------------------

def test_pair_metrics_known_case():
    # pred {0,1}{2,3} ; true {0,1,2}{3}
    m = partition_metrics(np.array([0, 0, 1, 1]), np.array([5, 5, 5, 6]))
    assert m["pair_precision"] == pytest.approx(0.5)
    assert m["pair_recall"] == pytest.approx(1 / 3)


def test_pair_metrics_ignore_label_excludes_unknown_truth():
    m = partition_metrics(np.array([0, 0, 0, 1, 1]), np.array([0, 1, 1, 2, 2]))
    assert m["n_items"] == 4
    assert m["pair_f1"] == pytest.approx(1.0)


def test_pair_metrics_abstained_items_are_singletons():
    merged = partition_metrics(np.array([-1, -1]), np.array([5, 6]), pred_ignore=None)
    assert merged["pair_fp"] == 1
    split = partition_metrics(np.array([-1, -1]), np.array([5, 6]), pred_ignore=-1)
    assert split["pair_fp"] == 0


def test_pair_metrics_handle_real_root_id_magnitudes():
    true = np.array([864691135000000001, 864691135000000001,
                     864691135000000002, 864691135000000002], dtype=np.int64)
    m = partition_metrics(np.array([0, 0, 1, 1]), true)
    assert m["pair_precision"] == pytest.approx(1.0)
    assert m["pair_recall"] == pytest.approx(1.0)


def test_pair_metrics_undefined_when_nothing_merged_anywhere():
    m = partition_metrics(np.arange(4), np.array([1, 2, 3, 4]))
    assert math.isnan(m["pair_precision"])
    assert math.isnan(m["pair_recall"])


def test_pair_metrics_undefined_can_be_overridden():
    m = partition_metrics(np.arange(4), np.array([1, 2, 3, 4]), undefined=1.0)
    assert m["pair_precision"] == 1.0


def test_empty_input_returns_documented_keys_not_an_exception():
    m = partition_metrics(np.array([], dtype=int), np.array([], dtype=int))
    assert m["n_items"] == 0
    assert math.isnan(m["ari"])


def test_rand_disagreement_counts_fixable_pairs():
    # pred merges everything; truth is two pairs -> 4 cross pairs wrongly merged
    assert rand_disagreement(np.array([1, 1, 2, 2]), np.zeros(4, int)) == 4


def test_rand_disagreement_matches_legacy_counter_form():
    from collections import Counter
    rng = np.random.default_rng(2)
    truth = rng.integers(0, 4, size=80)
    pred = rng.integers(0, 3, size=80)

    def c2(n):
        return n * (n - 1) // 2

    legacy = (sum(c2(c) for c in Counter(truth.tolist()).values())
              + sum(c2(c) for c in Counter(pred.tolist()).values())
              - 2 * sum(c2(c) for c in Counter(zip(truth.tolist(), pred.tolist())).values()))
    assert rand_disagreement(truth, pred) == legacy


# ---------------------------------------------------------------------------
# entropy-based metrics
# ---------------------------------------------------------------------------

def test_homogeneity_completeness_match_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(19)
    true = rng.integers(0, 5, size=150)
    pred = rng.integers(0, 4, size=150)
    m = partition_metrics(pred, true, ignore=None)
    h, c, v = sk.homogeneity_completeness_v_measure(true, pred)
    assert m["homogeneity"] == pytest.approx(h)
    assert m["completeness"] == pytest.approx(c)
    assert m["v_measure"] == pytest.approx(v)


def test_homogeneity_completeness_match_legacy_numpy_implementation():
    from neuronauts.assemble.partition_gnn import _homogeneity_completeness_np
    rng = np.random.default_rng(23)
    true = rng.integers(1, 6, size=120)
    pred = rng.integers(1, 4, size=120)
    m = partition_metrics(pred, true, ignore=None)
    h, c, v = _homogeneity_completeness_np(true, pred)
    assert (m["homogeneity"], m["completeness"], m["v_measure"]) == pytest.approx((h, c, v))


def test_perfect_partition_has_zero_variation_of_information():
    labels = np.array([1, 1, 2, 2, 3, 3])
    vi, split, merge = variation_of_information(contingency(labels, labels))
    assert (vi, split, merge) == pytest.approx((0.0, 0.0, 0.0))


def test_over_segmentation_shows_up_only_in_vi_split():
    true = np.array([1, 1, 1, 1])
    pred = np.array([0, 1, 2, 3])
    _, split, merge = variation_of_information(contingency(true, pred))
    assert split == pytest.approx(2.0)   # 4 equal pieces = 2 bits
    assert merge == pytest.approx(0.0)


def test_false_merge_shows_up_only_in_vi_merge():
    true = np.array([1, 1, 2, 2])
    pred = np.zeros(4, dtype=int)
    _, split, merge = variation_of_information(contingency(true, pred))
    assert split == pytest.approx(0.0)
    assert merge == pytest.approx(1.0)   # two equal labels in one cluster = 1 bit


# ---------------------------------------------------------------------------
# purity
# ---------------------------------------------------------------------------

def test_purity_is_one_for_pure_clusters():
    labels = np.array([1, 1, 2, 2])
    mass, mean, frac = cluster_purity(contingency(labels, labels))
    assert (mass, mean, frac) == pytest.approx((1.0, 1.0, 1.0))


def test_purity_mass_counts_majority_items():
    true = np.array([1, 1, 1, 2])
    pred = np.zeros(4, dtype=int)
    mass, mean, frac = cluster_purity(contingency(true, pred))
    assert mass == pytest.approx(0.75)
    assert mean == pytest.approx(0.75)
    assert frac == pytest.approx(0.0)


def test_purity_mean_is_not_dominated_by_a_large_impure_cluster():
    # one impure cluster of 4, four pure singletons
    true = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    pred = np.array([0, 0, 0, 0, 1, 2, 3, 4])
    mass, mean, _ = cluster_purity(contingency(true, pred))
    assert mass == pytest.approx((1 + 4) / 8)
    assert mean == pytest.approx((0.25 + 1 + 1 + 1 + 1) / 5)


# ---------------------------------------------------------------------------
# weighted pairs and ERL
# ---------------------------------------------------------------------------

def test_weighted_pair_metrics_equal_unweighted_when_weights_are_equal():
    rng = np.random.default_rng(31)
    true = rng.integers(0, 4, size=50)
    pred = rng.integers(0, 3, size=50)
    m = partition_metrics(pred, true, ignore=None, weights=np.ones(50))
    assert m["wpair_precision"] == pytest.approx(m["pair_precision"])
    assert m["wpair_recall"] == pytest.approx(m["pair_recall"])


def test_long_cable_dominates_the_weighted_metric():
    # one false merge between two long fragments, one between two short ones
    true = np.array([1, 2, 3, 4])
    pred = np.array([0, 0, 1, 1])
    weights = np.array([100.0, 100.0, 1.0, 1.0])
    m = partition_metrics(pred, true, ignore=None, weights=weights)
    assert m["pair_precision"] == pytest.approx(0.0)
    assert m["wpair_fp"] == pytest.approx(100 * 100 + 1 * 1)


def test_erl_of_a_perfect_reconstruction_is_total_weight_per_neuron():
    true = np.array([1, 1, 1])
    pred = np.array([0, 0, 0])
    ct = contingency(true, pred, np.array([10.0, 20.0, 30.0]))
    assert expected_run_length(ct) == pytest.approx(60.0)


def test_erl_halves_when_a_neuron_is_split_in_two():
    true = np.array([1, 1])
    pred = np.array([0, 1])
    ct = contingency(true, pred, np.array([50.0, 50.0]))
    assert expected_run_length(ct) == pytest.approx(50.0)


def test_erl_is_nan_without_weights():
    assert math.isnan(expected_run_length(contingency([1, 1], [0, 0])))


def test_erl_penalises_a_false_merge_by_splitting_the_gt_neuron():
    # two GT neurons merged into one cluster: each contributes its own piece,
    # so ERL reflects piece size, not the inflated cluster
    true = np.array([1, 1, 2, 2])
    pred = np.zeros(4, dtype=int)
    ct = contingency(true, pred, np.full(4, 25.0))
    assert expected_run_length(ct) == pytest.approx(50.0)


def test_empty_input_honours_undefined_override():
    m = partition_metrics(np.array([], dtype=int), np.array([], dtype=int), undefined=0.0)
    assert m["pair_precision"] == 0.0
    assert m["pair_recall"] == 0.0
    assert m["pair_f1"] == 0.0


def test_single_item_honours_undefined_override():
    m = partition_metrics(np.array([1]), np.array([5]), ignore=None, undefined=0.0)
    assert m["pair_precision"] == 0.0
    assert m["pair_recall"] == 0.0
    assert m["pair_f1"] == 0.0


def test_empty_weighted_input_honours_undefined_override():
    m = partition_metrics(np.array([], dtype=int), np.array([], dtype=int),
                          weights=np.array([]), undefined=0.0)
    assert m["wpair_precision"] == 0.0
    assert m["wpair_recall"] == 0.0
    assert m["wpair_f1"] == 0.0
