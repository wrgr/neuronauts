"""Offline tests for the matched v117-vs-truth confusion metric (no network)."""
import numpy as np

from experiments.proofread.v117_baseline import matched_confusion


def test_perfect_partition():
    # each true neuron entirely in its own object -> P=R=1
    pred = np.array([10, 10, 20, 20, 20])
    truth = np.array([1, 1, 2, 2, 2])
    r = matched_confusion(pred, truth)
    assert r["P"] == 1.0 and r["R"] == 1.0 and r["F1"] == 1.0
    assert r["FP"] == 0 and r["FN"] == 0 and r["catastrophic"] == 0


def test_pure_split_keeps_precision():
    # neuron 1 shattered across 3 objects (2,1,1) -> matched obj holds 2/4, FN=2, FP=0
    pred = np.array([10, 10, 11, 12])
    truth = np.array([1, 1, 1, 1])
    r = matched_confusion(pred, truth)
    assert r["FP"] == 0 and r["P"] == 1.0
    assert r["TP"] == 2 and r["FN"] == 2
    assert abs(r["R"] - 0.5) < 1e-9


def test_merge_creates_fp():
    # object 10 holds neuron 1 (3 halves) and neuron 2 (1 half). Neuron 1 matches 10.
    # neuron 2 also best-matches 10 -> catastrophic; FP appears.
    pred = np.array([10, 10, 10, 10])
    truth = np.array([1, 1, 1, 2])
    r = matched_confusion(pred, truth)
    assert r["FP"] > 0 and r["P"] < 1.0
    assert r["catastrophic"] == 1


def test_per_side_breakdown():
    # neuron 1: 2 inputs both in obj 10 (recall 1.0); 2 outputs split 10/11 (recall 0.5)
    pred = np.array([10, 10, 10, 11])
    truth = np.array([1, 1, 1, 1])
    side = np.array([1, 1, 0, 0], np.int8)   # 1=post/input, 0=pre/output
    r = matched_confusion(pred, truth, side)
    ps = r["perside"]
    assert abs(ps["post_dend_in"]["micro_recall"] - 1.0) < 1e-9
    assert abs(ps["pre_axon_out"]["micro_recall"] - 0.5) < 1e-9


def test_unmapped_halves_ignored():
    pred = np.array([0, 10, 10])
    truth = np.array([1, 1, 1])
    r = matched_confusion(pred, truth)
    assert r["halves"] == 2   # the 0-pred half dropped
