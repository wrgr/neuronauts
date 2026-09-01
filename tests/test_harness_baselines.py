"""Baseline scorers: feature construction, the unlearned ladder, and the two
learned models fit purely in numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuronauts.harness.baselines import (
    AtomContext, FEATURE_COLS, GradientBoostedStumps, LogisticRegression,
    all_baselines, pair_features, score_directed, score_facing, score_gap,
    score_random,
)
from neuronauts.harness.candidates import build_candidate_panel


def _panel_and_ctx():
    ep_atom = np.array([1, 1, 2, 2, 3, 3], np.uint64)
    ep_pos = np.array([[0, 0, 0], [0, 0, 20000],
                       [1000, 0, 0], [1000, 0, 30000],
                       [500000, 0, 0], [500000, 0, 10000]], np.float32)
    ep_tan = np.array([[1, 0, 0], [0, 0, 1],
                       [-1, 0, 0], [0, 0, 1],
                       [1, 0, 0], [0, 0, 1]], np.float32)
    ep_leaf = np.full(6, 2000.0, np.float32)
    ep_cal = np.full(6, 50.0, np.float32)
    panel = build_candidate_panel(ep_atom, ep_pos, ep_tan, ep_leaf, ep_cal,
                                  min_leaf_nm=100, min_caliber_nm=10,
                                  radius_nm=600000, k=8)
    ctx = AtomContext(atom_id=np.array([1, 2, 3], np.uint64),
                      n_syn=np.array([10, 12, 5], np.float32),
                      cable_nm=np.array([50000, 60000, 20000], np.float32),
                      n_end=np.array([4, 6, 3], np.float32),
                      n_comp=np.array([1, 1, 2], np.float32),
                      n_pre=np.array([8, 2, 1], np.float32),
                      n_post=np.array([2, 10, 4], np.float32))
    return panel, ctx


def test_pair_features_shape_and_finiteness():
    panel, ctx = _panel_and_ctx()
    x = pair_features(panel, ctx)
    assert x.shape == (len(panel), len(FEATURE_COLS))
    assert np.isfinite(x).all()


def test_pair_features_missing_atom_context_does_not_crash():
    panel, ctx = _panel_and_ctx()
    ctx2 = AtomContext(atom_id=np.array([1, 2], np.uint64),   # atom 3 missing
                       n_syn=np.array([10, 12], np.float32),
                       cable_nm=np.array([50000, 60000], np.float32),
                       n_end=np.array([4, 6], np.float32),
                       n_comp=np.array([1, 1], np.float32),
                       n_pre=np.array([8, 2], np.float32),
                       n_post=np.array([2, 10], np.float32))
    x = pair_features(panel, ctx2)
    assert np.isfinite(x).all()


def test_atom_context_rows_lookup():
    ctx = AtomContext(atom_id=np.array([5, 1, 3], np.uint64),
                      n_syn=np.zeros(3, np.float32), cable_nm=np.zeros(3, np.float32),
                      n_end=np.zeros(3, np.float32), n_comp=np.zeros(3, np.float32),
                      n_pre=np.zeros(3, np.float32), n_post=np.zeros(3, np.float32))
    rows = ctx.rows(np.array([3, 5, 9], np.uint64))
    assert rows.tolist() == [2, 0, -1]


# ---------------------------------------------------------------------------
# unlearned scorers
# ---------------------------------------------------------------------------

def test_score_random_is_seed_deterministic():
    x = np.zeros((10, 3))
    a = score_random(x, seed=1)
    b = score_random(x, seed=1)
    c = score_random(x, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_score_gap_prefers_closer_pairs():
    x = np.zeros((2, len(FEATURE_COLS)))
    x[:, FEATURE_COLS.index("gap_um")] = [1.0, 5.0]
    s = score_gap(x)
    assert s[0] > s[1]


def test_score_facing_matches_column():
    x = np.zeros((2, len(FEATURE_COLS)))
    x[:, FEATURE_COLS.index("facing")] = [0.9, -0.9]
    s = score_facing(x)
    assert s.tolist() == [0.9, -0.9]


def test_score_directed_combines_evidence_directionally():
    x = np.zeros((2, len(FEATURE_COLS)))
    # row 0: close, facing, aligned, similar caliber -> should score higher
    x[0, [FEATURE_COLS.index(c) for c in
          ("gap_um", "facing", "align_min", "caliber_ratio")]] = [1.0, 1.0, 1.0, 1.0]
    # row 1: far, averted, misaligned, dissimilar caliber
    x[1, [FEATURE_COLS.index(c) for c in
          ("gap_um", "facing", "align_min", "caliber_ratio")]] = [20.0, -1.0, -1.0, 0.1]
    s = score_directed(x)
    assert s[0] > s[1]


def test_all_baselines_registry_has_expected_kinds():
    names = {b.name: b.kind for b in all_baselines()}
    assert names["random"] == "unlearned"
    assert names["gap"] == "unlearned"
    assert names["directed"] == "unlearned"
    assert names["logistic"] == "learned"
    assert names["gbdt"] == "learned"


# ---------------------------------------------------------------------------
# learned scorers: recover a known-separable signal
# ---------------------------------------------------------------------------

def _separable_dataset(seed=0, n=400):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    x = np.zeros((n, 4))
    # feature 0 is fully separating; the rest are noise
    x[:, 0] = np.where(y == 1, rng.normal(3, 1, n), rng.normal(-3, 1, n))
    x[:, 1:] = rng.normal(0, 1, (n, 3))
    return x, y


def test_logistic_regression_recovers_separable_signal():
    x, y = _separable_dataset()
    model = LogisticRegression.fit(x, y, n_iter=300)
    p = model(x)
    from neuronauts.harness.evaluation import roc_auc
    assert roc_auc(y, p) > 0.95


def test_logistic_regression_balances_rare_positive_class():
    rng = np.random.default_rng(1)
    n = 500
    y = np.zeros(n, int)
    y[:20] = 1                                    # 4% positive
    x = np.zeros((n, 2))
    x[:, 0] = np.where(y == 1, rng.normal(3, 1, n), rng.normal(-3, 1, n))
    model = LogisticRegression.fit(x, y, n_iter=300, balance=True)
    p = model(x)
    # a model that collapsed to "always negative" would put every prediction
    # below 0.5; a balanced fit should place most positives above it
    assert (p[y == 1] > 0.5).mean() > 0.7


def test_gradient_boosted_stumps_recovers_separable_signal():
    x, y = _separable_dataset(seed=2)
    model = GradientBoostedStumps.fit(x, y, n_rounds=60, seed=0)
    p = model(x)
    from neuronauts.harness.evaluation import roc_auc
    assert roc_auc(y, p) > 0.95


def test_gradient_boosted_stumps_is_seed_reproducible():
    x, y = _separable_dataset(seed=3)
    m1 = GradientBoostedStumps.fit(x, y, n_rounds=30, seed=0)
    m2 = GradientBoostedStumps.fit(x, y, n_rounds=30, seed=0)
    assert np.allclose(m1.decision(x), m2.decision(x))


def test_gradient_boosted_stumps_handles_constant_feature():
    rng = np.random.default_rng(4)
    n = 100
    y = rng.integers(0, 2, n)
    x = np.zeros((n, 3))
    x[:, 0] = 5.0                                  # constant, no split possible
    x[:, 1] = rng.normal(0, 1, n)
    model = GradientBoostedStumps.fit(x, y, n_rounds=20)
    p = model(x)
    assert np.isfinite(p).all()
