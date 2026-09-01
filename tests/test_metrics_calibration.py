"""Probability calibration: neuronauts.metrics.calibration."""

from __future__ import annotations

import math

import numpy as np
import pytest

from neuronauts.metrics import (
    brier_score,
    calibration_metrics,
    expected_calibration_error,
    reliability_bins,
)


def test_perfectly_calibrated_probabilities_have_zero_ece():
    rng = np.random.default_rng(0)
    n = 20000
    probs = rng.uniform(0, 1, size=n)
    y = (rng.uniform(0, 1, size=n) < probs).astype(float)
    bins = reliability_bins(probs, y, n_bins=10)
    assert expected_calibration_error(bins) < 0.03


def test_overconfident_probabilities_have_nonzero_ece():
    # true rate is always 0.5 but the model reports near-certainty
    rng = np.random.default_rng(1)
    n = 2000
    probs = np.where(rng.random(n) < 0.5, 0.95, 0.05)
    y = (rng.random(n) < 0.5).astype(float)
    bins = reliability_bins(probs, y, n_bins=10)
    assert expected_calibration_error(bins) > 0.3


def test_reliability_bins_empty_bin_is_nan_not_zero():
    probs = np.array([0.05, 0.95])
    y = np.array([0.0, 1.0])
    bins = reliability_bins(probs, y, n_bins=10)
    # bins around 0.5 have no data
    middle = bins["frac_pos"][4]
    assert math.isnan(middle)
    assert bins["counts"][4] == 0


def test_ece_ignores_empty_bins_in_the_weighted_average():
    probs = np.array([0.1, 0.1, 0.9])
    y = np.array([0.0, 0.0, 1.0])
    bins = reliability_bins(probs, y, n_bins=10)
    ece = expected_calibration_error(bins)
    assert not math.isnan(ece)


def test_ece_of_empty_input_is_nan():
    bins = reliability_bins(np.array([]), np.array([]))
    assert math.isnan(expected_calibration_error(bins))


def test_brier_score_perfect_predictions_is_zero():
    assert brier_score(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(0.0)


def test_brier_score_worst_case_is_one():
    assert brier_score(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


def test_calibration_metrics_bundles_ece_and_brier():
    probs = np.array([0.2, 0.8])
    y = np.array([0.0, 1.0])
    m = calibration_metrics(probs, y, n_bins=5)
    assert "ece" in m and "brier" in m and "bins" in m
    assert m["n_scored"] == 2


def test_matches_legacy_treestitch_calibration_reliability_diagram_shape():
    """The bin math should agree with treestitch.calibration's independent
    implementation (same binning rule, same weighted-mean ECE)."""
    from treestitch.calibration import expected_calibration_error as legacy_ece

    rng = np.random.default_rng(3)
    probs = rng.uniform(0, 1, size=500)
    y = (rng.uniform(0, 1, size=500) < probs).astype(np.float32)

    new_bins = reliability_bins(probs, y, n_bins=10)
    legacy_diag = {
        "bin_centers": new_bins["bin_centers"],
        "mean_conf": new_bins["mean_conf"],
        "frac_pos": new_bins["frac_pos"],
        "counts": new_bins["counts"],
        "T": 1.0,
    }
    assert expected_calibration_error(new_bins) == pytest.approx(legacy_ece(legacy_diag))
