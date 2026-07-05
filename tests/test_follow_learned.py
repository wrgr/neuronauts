"""Offline test for the learned-from-raw-geometry follow model (synthetic, no net train)."""
import numpy as np

from experiments.proofread.follow_test import Neuron, build_instances
from experiments.proofread.follow_learned import _frame, _raw_features


def _straight(y=0.0, n=16, step=500.0, r=150.0):
    v = np.array([[i * step, y, 0.0] for i in range(n)])
    e = np.array([[i, i + 1] for i in range(n - 1)])
    adj = [[] for _ in range(n)]
    for a, b in e:
        adj[a].append(b); adj[b].append(a)
    return Neuron(v, e, np.full(n, r), adj)


def test_frame_is_orthonormal_with_d_as_x():
    d = np.array([1.0, 2.0, -1.0])
    R = _frame(d)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)   # orthonormal
    assert np.allclose(R[0], d / np.linalg.norm(d))     # first row = unit d


def test_raw_features_are_canonical_coords():
    A = _straight(n=16); B = _straight(y=250.0, n=16)   # parallel neighbour
    inst = build_instances([A, B], gap_nm=1000, search_radius=3000, per_neuron=10)
    assert inst, "expected instances"
    F = _raw_features(inst[0], [A, B], n_nbr=4)
    # 3 (candidate pos) + 4*3 (neighbour offsets) + 2 (radii) = 17 raw coords
    assert F.shape[1] == 17
    assert np.isfinite(F).all()
