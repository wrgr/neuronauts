import numpy as np

from neuronauts.atomization import (
    cut_components, euclidean_mst, metrics_from_counts, pair_counts,
)


def test_long_bridge_cut_recovers_two_lineages():
    points = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0],
                         [20, 0, 0], [21, 0, 0], [22, 0, 0]], dtype=float)
    truth = np.asarray([0, 0, 0, 1, 1, 1])
    edges, lengths = euclidean_mst(points)
    prediction = cut_components(len(points), edges, lengths, cutoff_nm=5)
    metrics = metrics_from_counts(pair_counts(truth, prediction))
    assert metrics == {"pair_precision": 1.0, "pair_recall": 1.0,
                       "pair_f1": 1.0, "cross_lineage_split_recall": 1.0}


def test_atomic_root_has_full_recall_but_cross_lineage_failure():
    truth = np.asarray([0, 0, 1, 1])
    metrics = metrics_from_counts(pair_counts(truth, np.zeros(4, dtype=int)))
    assert metrics["pair_recall"] == 1.0
    assert metrics["cross_lineage_split_recall"] == 0.0
