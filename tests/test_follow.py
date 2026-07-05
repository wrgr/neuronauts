"""Offline tests for the trajectory-follow test (synthetic skeletons, no network)."""
import numpy as np

from experiments.proofread.follow_test import Neuron, evaluate, build_instances


def _straight(y=0.0, z=0.0, n=12, step=500.0, r=150.0):
    v = np.array([[i * step, y, z] for i in range(n)])
    e = np.array([[i, i + 1] for i in range(n - 1)])
    adj = [[] for _ in range(n)]
    for a, b in e:
        adj[a].append(b); adj[b].append(a)
    return Neuron(v, e, np.full(n, r), adj)


def test_trajectory_beats_nearest_with_crossing_distractor():
    # neuron A runs straight along x; distractor neurons cross near A's midpoint,
    # closer than A's own far side -> proximity is misled, trajectory should win.
    A = _straight(n=16, step=500.0)
    # crossing processes: short cables passing near x=3750 at an angle, very close
    crossers = []
    for k, y0 in enumerate((300.0, -300.0, 350.0)):
        cy = np.array([[3750.0 + j * 120.0, y0 + j * 400.0, 0.0] for j in range(6)])
        ce = np.array([[j, j + 1] for j in range(5)])
        adj = [[] for _ in range(6)]
        for a, b in ce:
            adj[a].append(b); adj[b].append(a)
        crossers.append(Neuron(cy, ce, np.full(6, 150.0), adj))
    res = evaluate([A] + crossers, gap_nm=1200, search_radius=4000,
                   per_neuron=20, seed=0, verbose=False)
    assert "error" not in res
    # direction alignment should follow the straight cable better than raw proximity
    assert res["align"]["top1"] >= res["nearest"]["top1"]


def test_instances_have_true_and_distractors():
    A = _straight(n=20, step=400.0)
    B = _straight(y=250.0, n=20, step=400.0)   # a parallel neighbour
    inst = build_instances([A, B], gap_nm=1000, search_radius=3000, per_neuron=10)
    assert len(inst) > 0
    # every kept instance has at least one true and the pool is non-trivial
    assert all(it["is_true"].any() for it in inst)
    assert any((~it["is_true"]).any() for it in inst)  # some distractors exist
