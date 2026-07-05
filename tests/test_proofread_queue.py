"""Offline tests for the complementarity candidate builder and ranked queue."""
import numpy as np

from experiments.pcfg.synapse_correction import SideTable
from experiments.proofread.complementarity import build_pair_candidates, JoinCandidate
from experiments.proofread.queue import build_queue, topk_precision, queue_summary


def _synthetic_table(seed=0):
    """One false-merge (v117 root 101 spans later 900 & 901) + a clean root."""
    rng = np.random.default_rng(seed)
    # root 101: two lobes -> later 900 (lobe A) and 901 (lobe B): a false MERGE
    A = np.array([0.0, 0, 0]) + rng.normal(0, 300, (10, 3))
    B = np.array([4000.0, 0, 0]) + rng.normal(0, 300, (10, 3))
    # root 202: a clean cell, all later 902
    C = np.array([0.0, 8000, 0]) + rng.normal(0, 300, (8, 3))
    pt = np.vstack([A, B, C])
    side = np.ones(len(pt), np.int8)
    rv = np.array([101] * 20 + [202] * 8, np.int64)
    rl = np.array([900] * 10 + [901] * 10 + [902] * 8, np.int64)
    sid = np.arange(len(pt), dtype=np.int64)
    return SideTable(sid, side, pt, rv, rl)


def test_candidate_builder_finds_false_merge():
    tab = _synthetic_table()
    cands = build_pair_candidates(tab, rng=np.random.default_rng(0))
    # within-root 101 pairs across the two lobes are y=0 (false merge / cut)
    cut_errs = [c for c in cands if c.stratum == 0 and c.label == 0]
    assert len(cut_errs) > 0
    assert all(c.rv_a == 101 and c.rv_b == 101 for c in cut_errs)


def test_max_pair_nm_filters_far_pairs():
    tab = _synthetic_table()
    near = build_pair_candidates(tab, max_pair_nm=1500.0, rng=np.random.default_rng(0))
    far = build_pair_candidates(tab, max_pair_nm=None, rng=np.random.default_rng(0))
    assert len(near) < len(far)
    assert all(np.linalg.norm(c.pos_a - c.pos_b) <= 1500.0 for c in near)


def _fake_res():
    # 3 candidates: a confident cut (correct), a confident join (correct), an abstain
    cands = [
        JoinCandidate(np.zeros(3), np.array([1000.0, 0, 0]), label=0, group=0,
                      shape_feat=np.zeros(4), side=1, rv_a=1, rv_b=1, stratum=0),
        JoinCandidate(np.zeros(3), np.array([1000.0, 0, 0]), label=1, group=1,
                      shape_feat=np.zeros(4), side=1, rv_a=2, rv_b=3, stratum=1),
        JoinCandidate(np.zeros(3), np.array([1000.0, 0, 0]), label=0, group=2,
                      shape_feat=np.zeros(4), side=1, rv_a=4, rv_b=4, stratum=0),
    ]
    return {"cands": cands, "y": np.array([0, 1, 0]),
            "p_joint": np.array([0.1, 0.9, 0.5]),
            "local": np.array([[0.2, 0.4], [0.9, 0.0], [0.5, 0.1]])}


def test_build_queue_actions_and_precision():
    items = build_queue(_fake_res(), with_urls=False)
    kinds = {it.rank: it.kind for it in items}
    assert "CUT" in [it.kind for it in items]
    assert "JOIN" in [it.kind for it in items]
    assert "ABSTAIN" in [it.kind for it in items]
    # both confident edits are correct -> top-k precision 1.0
    r = topk_precision(items, 5)
    assert r["n"] == 2 and r["precision"] == 1.0
    assert isinstance(queue_summary(items), str)
