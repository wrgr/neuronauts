"""Offline tests for the seam-localized two-cue test helpers (no network)."""
import numpy as np

from experiments.proofread.seam_test import (
    _geodesic_pair, _closest_approach, summarize, SeamRow)


def _line(n=10, step=500.0):
    v = np.array([[i * step, 0.0, 0.0] for i in range(n)])
    e = np.array([[i, i + 1] for i in range(n - 1)])
    return v, e


def test_geodesic_pair_respects_gap():
    v, e = _line()
    i, j, glen = _geodesic_pair(v, e, 2000.0, np.random.default_rng(0))
    assert glen >= 2000.0
    assert i != j


def test_closest_approach_picks_nearest():
    ia, jb, d = _closest_approach(np.array([[0.0, 0, 0]]),
                                  np.array([[100.0, 0, 0], [3.0, 0, 0]]))
    assert jb == 1
    assert abs(d - 3.0) < 1e-6


def test_summarize_separation_and_grammar():
    rows = [
        SeamRow("seam", 1, 0.20, 0.30, 1, 1500, -1.0, (1, 2)),
        SeamRow("seam", 1, 0.30, 0.20, 1, 1200, -1.0, (3, 4)),
        SeamRow("continuation", 0, 0.80, 0.02, 1, 2000, float("nan"), (5, 5)),
        SeamRow("continuation", 0, 0.70, 0.05, 1, 2100, float("nan"), (6, 6)),
    ]
    s = summarize(rows)
    assert s["cutface_sim_seam_mean"] < s["cutface_sim_cont_mean"]
    assert s["barrier_seam_mean"] > s["barrier_cont_mean"]
    assert s["auc_cutface"] == 1.0           # perfectly separable here
    assert s["grammar_seam_reject_frac"] == 1.0
