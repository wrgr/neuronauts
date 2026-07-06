"""Offline tests for the v117-object joiner geometry/candidate/eval logic (no network)."""
import numpy as np

from experiments.proofread.v117_proofread import build_candidates, evaluate, _tangent


def _line(start, direction, n=6, step=400.0):
    d = np.asarray(direction, float); d /= np.linalg.norm(d)
    return (np.asarray(start, float) + np.outer(np.arange(n), d) * step).astype(np.float32)


def test_candidate_prefers_colinear_same_direction():
    # obj A along +x; obj B continues along +x just past A's tip -> high score, correct
    A = _line([0, 0, 0], [1, 0, 0])
    B = _line([2400, 0, 0], [1, 0, 0])
    objs = {1: dict(truth=1, region="t", pts=A, caliber=100, n_out=3, n_in=0),
            2: dict(truth=1, region="t", pts=B, caliber=100, n_out=3, n_in=0)}
    edges = build_candidates(objs, max_gap=2000.0)
    assert len(edges) == 1
    score, a, b, correct = edges[0]
    assert correct == 1 and score > 0.5


def test_candidate_labels_wrong_neuron():
    A = _line([0, 0, 0], [1, 0, 0])
    B = _line([2400, 0, 0], [1, 0, 0])
    objs = {1: dict(truth=1, region="t", pts=A, caliber=100, n_out=3, n_in=0),
            2: dict(truth=2, region="t", pts=B, caliber=100, n_out=3, n_in=0)}  # different truth
    edges = build_candidates(objs, max_gap=2000.0)
    assert edges and edges[0][3] == 0


def test_evaluate_oracle_recovers_split_axon():
    # one neuron: a "soma/dendrite" object (many inputs) + two axon-output fragments
    # placed colinearly so oracle joins them all -> axon recall rises to 1.0
    soma = _line([0, 0, 0], [1, 0, 0], n=8)
    ax1 = _line([3200, 0, 0], [1, 0, 0])
    ax2 = _line([5600, 0, 0], [1, 0, 0])
    objs = {
        10: dict(truth=1, region="t", pts=soma, caliber=500, n_out=0, n_in=20),
        11: dict(truth=1, region="t", pts=ax1, caliber=120, n_out=5, n_in=0),
        12: dict(truth=1, region="t", pts=ax2, caliber=120, n_out=5, n_in=0),
    }
    halves = [dict(
        obj=np.array([10] * 20 + [11] * 5 + [12] * 5, np.int64),
        side=np.array([1] * 20 + [0] * 10, np.int8), root=1)]
    edges = build_candidates(objs, max_gap=3000.0)
    res = evaluate(objs, halves, edges)
    base_axon = res["base"]["perside"]["pre_axon_out"]["micro_recall"]
    orc_axon = res["oracle"]["perside"]["pre_axon_out"]["micro_recall"]
    assert base_axon < 0.6           # axon outputs split away from the matched (soma) object
    assert orc_axon > base_axon      # oracle joins recover them


def test_tangent_direction():
    pts = _line([0, 0, 0], [0, 1, 0], n=10)
    t = _tangent(pts, pts[5])
    assert abs(abs(t[1]) - 1.0) < 1e-3   # points along +y
