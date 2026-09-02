"""The numpy port of global_shape_merge.global_features must agree with the
scikit-learn original it replaces, or EXP-063's baseline rung is not the
baseline it claims to be.

scikit-learn is not installed in this environment, so the reference vectors
were produced once, from the original function extracted verbatim via ``ast``,
and stored in ``tests/fixtures/global_shape_sklearn_ref.npz`` alongside the
exact input clouds. Seven of the ten columns are deterministic functions of the
cloud and must match to floating-point precision. The three that derive from
2-means (``bimod``, ``center_gap``, ``balance``) depend on which local optimum
the restarts find; the rule for those is that the port's optimum is at least as
good as scikit-learn's (a higher ``bimod`` means a lower 2-means inertia), and
when the optimum is the same the other two columns match too.
"""

from pathlib import Path

import numpy as np
import pytest

from neuronauts.harness.atom_features import (
    GLOBAL_SHAPE_COLS, global_shape_features, polarity_features,
    precision_at_top, _dbscan_labels, _kmeans2,
)

FIXTURE = Path(__file__).parent / "fixtures" / "global_shape_sklearn_ref.npz"
DETERMINISTIC = ["log_n", "log_extent", "pc1_rel", "pc2_rel", "pc3_rel",
                 "log_n_blobs", "largest_blob_frac"]
KMEANS = ["bimod", "center_gap", "balance"]


def _cases():
    z = np.load(FIXTURE)
    return [(str(n), z[f"pts_{n}"], z[f"ref_{n}"]) for n in z["names"]]


@pytest.mark.parametrize("name,pts,ref", _cases(), ids=lambda c: c if isinstance(c, str) else "")
def test_deterministic_columns_match_sklearn_exactly(name, pts, ref):
    got = global_shape_features(pts)
    for col in DETERMINISTIC:
        k = GLOBAL_SHAPE_COLS.index(col)
        assert got[k] == pytest.approx(ref[k], abs=1e-9), (name, col, got[k], ref[k])


@pytest.mark.parametrize("name,pts,ref", _cases(), ids=lambda c: c if isinstance(c, str) else "")
def test_kmeans_columns_reach_sklearn_optimum_or_better(name, pts, ref):
    got = global_shape_features(pts)
    kb = GLOBAL_SHAPE_COLS.index("bimod")
    # equal-or-better optimum: never a worse inertia than the reference found
    assert got[kb] >= ref[kb] - 1e-9, (name, got[kb], ref[kb])
    if abs(got[kb] - ref[kb]) <= 1e-9:
        # same optimum -> the split is the same -> the other two agree
        for col in ("center_gap", "balance"):
            k = GLOBAL_SHAPE_COLS.index(col)
            assert got[k] == pytest.approx(ref[k], abs=1e-6), (name, col)


def test_dbscan_semantics_on_a_hand_built_case():
    # two tight triples 100 nm apart within each, 20 um between them, plus one
    # isolated point: eps=5um, min_samples=3 -> two clusters and one noise point
    a = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0]], float)
    b = a + [20_000, 0, 0]
    lone = np.array([[50_000, 50_000, 50_000]], float)
    pts = np.vstack([a, b, lone])
    lab = _dbscan_labels(pts, eps=5000.0, min_samples=3)
    assert lab[6] == -1
    assert len(set(lab[:3])) == 1 and len(set(lab[3:6])) == 1
    assert lab[0] != lab[3]


def test_dbscan_border_point_joins_a_core_cluster():
    core = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0]], float)
    border = np.array([[4_000, 0, 0]], float)      # within eps of core, but has
    pts = np.vstack([core, border])                # only 1 neighbour itself... no:
    # border sees core[0] (4000) within 5000 -> count 2 < 3, so not core; it
    # attaches to core[0]'s cluster rather than being noise.
    lab = _dbscan_labels(pts, eps=5000.0, min_samples=3)
    assert lab[3] == lab[0] and lab[3] >= 0


def test_kmeans2_finds_the_obvious_split():
    rng = np.random.default_rng(0)
    pts = np.vstack([rng.normal(0, 100, (50, 3)),
                     rng.normal([10_000, 0, 0], 100, (50, 3))])
    inertia, centres, lab = _kmeans2(pts, seed=0)
    assert len(set(lab[:50])) == 1 and len(set(lab[50:])) == 1
    assert lab[0] != lab[50]
    assert abs(np.linalg.norm(centres[0] - centres[1]) - 10_000) < 100


def test_polarity_minority_fraction():
    f = polarity_features(np.array([10, 0, 5, 3]), np.array([0, 10, 5, 7]))
    np.testing.assert_allclose(f[:, 1], [0.0, 0.0, 0.5, 0.3])
    np.testing.assert_allclose(f[:, 0], [1.0, 0.0, 0.5, 0.3])


def test_precision_at_top_flags_the_top_fraction():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], bool)
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    r = precision_at_top(y, s, 0.2)
    assert r["n_flagged"] == 2 and r["precision"] == 1.0 and r["recall"] == 1.0
    r = precision_at_top(y, s, 0.3)
    assert r["n_flagged"] == 3 and r["precision"] == pytest.approx(2 / 3)


def test_empty_cloud_is_finite():
    f = global_shape_features(np.empty((0, 3)))
    assert np.isfinite(f).all() and len(f) == len(GLOBAL_SHAPE_COLS)
