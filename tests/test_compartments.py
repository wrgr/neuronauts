"""Offline tests for soma clustering + compartment labeling (no network)."""
from types import SimpleNamespace

import numpy as np

from neuronauts.soma_clusters import soma_clusters, n_soma
from experiments.pcfg.compartments import label_compartments, SOMA, AXON, DEND


def _line_skeleton(n=11, step=1000.0):
    V = np.array([[i * step, 0.0, 0.0] for i in range(n)])
    edges = np.array([[i, i + 1] for i in range(n - 1)])
    radius = np.full(n, 200.0)
    return V, edges, radius


def test_soma_clusters_single():
    V, edges, radius = _line_skeleton()
    radius[0:3] = 5000.0  # one soma-caliber blob
    clusters = soma_clusters(V, radius)
    assert len(clusters) == 1
    assert set(clusters[0].tolist()) == {0, 1, 2}
    assert n_soma(V, radius) == 1


def test_soma_clusters_two_far_apart():
    # two big-radius blobs separated well beyond link_nm -> two somas (a merge)
    V = np.array([[0, 0, 0], [1000, 0, 0], [50_000, 0, 0], [51_000, 0, 0]], float)
    radius = np.array([5000.0, 5000.0, 5000.0, 5000.0])
    assert n_soma(V, radius) == 2


def test_soma_clusters_none_radius():
    V, edges, radius = _line_skeleton()
    assert soma_clusters(V, None) == []
    assert n_soma(V, np.full(len(V), 100.0)) == 0


def test_label_compartments_polarity_and_soma():
    V, edges, radius = _line_skeleton()
    radius[0:3] = 5000.0  # soma at the origin end
    sk = SimpleNamespace(root_id=1, vertices=V, edges=edges, radius=radius)

    vox = np.array([32.0, 32.0, 40.0])
    # PRE (axonal) synapses at the far end; POST (dendritic) in the middle
    pre_nm = np.array([[8000, 0, 0], [9000, 0, 0], [10000, 0, 0]], float)
    post_nm = np.array([[3000, 0, 0], [4000, 0, 0], [5000, 0, 0]], float)
    syn = SimpleNamespace(
        n_synapses=6,
        pre_pt=np.vstack([pre_nm / vox, np.zeros((3, 3))]),
        post_pt=np.vstack([np.zeros((3, 3)), post_nm / vox]),
        pre_root_id=np.array([1, 1, 1, 2, 2, 2]),
        post_root_id=np.array([2, 2, 2, 1, 1, 1]),
    )
    lab = label_compartments(sk, syn, root_id=1, mip=2)
    assert lab.n_soma == 1
    # far end axonal, middle dendritic, soma at origin
    assert lab.label[10] == AXON
    assert lab.label[4] == DEND
    assert lab.label[0] == SOMA
    s = lab.summary()
    assert s["n_soma_clusters"] == 1
    assert s["n_axon_verts"] > 0 and s["n_dend_verts"] > 0


def test_label_compartments_no_synapses_all_unknown_or_soma():
    V, edges, radius = _line_skeleton()
    sk = SimpleNamespace(root_id=1, vertices=V, edges=edges, radius=radius)
    syn = SimpleNamespace(n_synapses=0, pre_pt=np.zeros((0, 3)), post_pt=np.zeros((0, 3)),
                          pre_root_id=np.array([]), post_root_id=np.array([]))
    lab = label_compartments(sk, syn, root_id=1, mip=2)
    # no polarity signal and no soma -> everything UNKNOWN
    assert (lab.axon_mass == 0).all() and (lab.dend_mass == 0).all()
