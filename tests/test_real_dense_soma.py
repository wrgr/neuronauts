import numpy as np
import pytest

from neuronauts.real_dense_soma import (
    CandidateEdge,
    Fragment,
    assert_real_root_ids,
    partition_metrics,
    single_soma_compliance,
    soma_seeded_assemble,
)


def fragment(root, soma=0, truth=0):
    return Fragment(
        root,
        np.asarray([[0, 0, 0], [1, 0, 0]], np.float32),
        np.asarray([[0, 1]], np.int64),
        soma,
        truth,
        1.0,
    )


def edge(left, right, score):
    return CandidateEdge(left, right, 10.0, score, 1.0, score)


def test_synthetic_counter_ids_fail_closed():
    with pytest.raises(ValueError, match="CAVE"):
        assert_real_root_ids([1, 2, 3])


def test_soma_growth_competes_and_never_joins_somas():
    base = 1 << 50
    fragments = [fragment(base + 1, 1), fragment(base + 2), fragment(base + 3, 1)]
    prediction = soma_seeded_assemble(
        fragments,
        [edge(base + 1, base + 2, 2), edge(base + 2, base + 3, 3)],
    )
    assert prediction[base + 2] in {prediction[base + 1], prediction[base + 3]}
    assert prediction[base + 1] != prediction[base + 3]
    assert single_soma_compliance(fragments, prediction)["multi_soma_clusters"] == 0


def test_unknown_truth_is_retained_but_excluded_from_metrics():
    base = 1 << 50
    fragments = [
        fragment(base + 1, truth=11),
        fragment(base + 2, truth=11),
        fragment(base + 3, truth=0),
    ]
    prediction = {base + 1: 0, base + 2: 0, base + 3: 0}
    metrics = partition_metrics(fragments, prediction)
    assert metrics["n_labeled_fragments"] == 2
    assert metrics["ari"] == pytest.approx(1.0)
