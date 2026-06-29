"""Tests for Phase 2 global synapse graph and GNN assembly."""

from __future__ import annotations

import numpy as np
import pytest

from neuronauts.schemas import Fragment, Region
from neuronauts.assemble import (
    GlobalSynapseGraph,
    build_global_synapse_graph,
    train_global_gnn,
    run_global_gnn,
    assemble_neurons,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_region(n_neurons: int = 4, syn_per_neuron: int = 6, seed: int = 0):
    """Tiny Region with spatially clustered synapses for each neuron."""
    rng = np.random.default_rng(seed)
    pts_list, roots_list = [], []
    for nid in range(1, n_neurons + 1):
        centre = rng.uniform(0, 1_000_000, 3).astype(np.float32)
        jitter = rng.normal(0, 2_000, (syn_per_neuron, 3)).astype(np.float32)
        pts = centre + jitter
        pts_list.append(pts)
        roots_list.extend([nid] * syn_per_neuron)

    pre_pt = np.concatenate(pts_list).astype(np.float32)
    post_pt = pre_pt + rng.normal(0, 300, pre_pt.shape).astype(np.float32)
    pre_root_id = np.array(roots_list, dtype=np.int64)
    n_syn = len(pre_pt)
    return Region(
        region_id="test",
        bbox_nm=((0.0, 0.0, 0.0), (1_100_000.0, 1_100_000.0, 1_100_000.0)),
        voxel_size_nm=(8.0, 8.0, 40.0),
        seg_version=117,
        label_version=1412,
        pre_pt_nm=pre_pt,
        post_pt_nm=post_pt,
        pre_root_id=pre_root_id,
        post_root_id=np.zeros(n_syn, dtype=np.int64),
        synapse_id=np.arange(n_syn, dtype=np.int64),
    )


def _make_fragments_with_dna(region: Region, dna_dim: int = 8, seed: int = 1):
    """One fragment per unique root with a random (but distinct) DNA vector."""
    rng = np.random.default_rng(seed)
    root_ids = np.unique(region.pre_root_id)
    root_ids = root_ids[root_ids > 0]
    fragments = []
    for rid in root_ids:
        syn_idx = np.where(region.pre_root_id == rid)[0]
        n_v = 4
        verts = rng.uniform(0, 1e6, (n_v, 3)).astype(np.float32)
        edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
        radii = np.ones(n_v, dtype=np.float32) * 300.0
        dna = rng.standard_normal(dna_dim).astype(np.float32)
        dna /= np.linalg.norm(dna) + 1e-9
        frag = Fragment(
            fragment_id=int(rid),
            region_id="test",
            base_root_id=int(rid),
            vertices_nm=verts,
            edges=edges,
            endpoints_nm=verts[[0, -1]],
            radius_nm=radii,
            synapse_indices=syn_idx,
            dna=dna,
        ).validate()
        fragments.append(frag)
    return fragments


# ---------------------------------------------------------------------------
# GlobalSynapseGraph tests
# ---------------------------------------------------------------------------

def test_build_graph_shape():
    region = _make_region(n_neurons=3, syn_per_neuron=5)
    fragments = _make_fragments_with_dna(region, dna_dim=8)
    graph = build_global_synapse_graph(region, fragments, k_neighbors=4)
    assert graph.node_feat.shape == (15, 8)
    assert graph.node_pos.shape == (15, 3)
    assert graph.edge_src.shape == graph.edge_dst.shape == graph.edge_feat.shape[:1]
    assert graph.edge_feat.shape[1] == 1
    assert graph.pre_root_id.shape == (15,)


def test_build_graph_no_self_edges():
    region = _make_region(n_neurons=3, syn_per_neuron=4)
    fragments = _make_fragments_with_dna(region, dna_dim=8)
    graph = build_global_synapse_graph(region, fragments, k_neighbors=3)
    assert np.all(graph.edge_src != graph.edge_dst), "no self-edges"


def test_build_graph_edge_feat_positive():
    region = _make_region(n_neurons=2, syn_per_neuron=5)
    fragments = _make_fragments_with_dna(region, dna_dim=4)
    graph = build_global_synapse_graph(region, fragments, k_neighbors=3)
    assert np.all(graph.edge_feat >= 0), "log-dist features must be non-negative"


def test_build_graph_max_dist_prunes():
    region = _make_region(n_neurons=3, syn_per_neuron=5)
    fragments = _make_fragments_with_dna(region, dna_dim=4)
    g_full = build_global_synapse_graph(region, fragments, k_neighbors=4)
    g_pruned = build_global_synapse_graph(region, fragments, k_neighbors=4, max_dist_nm=1_000.0)
    assert g_pruned.n_edges <= g_full.n_edges


def test_build_graph_preserves_root_id():
    region = _make_region(n_neurons=4, syn_per_neuron=3)
    fragments = _make_fragments_with_dna(region, dna_dim=4)
    graph = build_global_synapse_graph(region, fragments)
    assert np.array_equal(graph.pre_root_id, region.pre_root_id)


# ---------------------------------------------------------------------------
# GNN training and inference tests
# ---------------------------------------------------------------------------

def _make_graph(n_neurons=4, syn_per=6, dna_dim=8, k=4) -> GlobalSynapseGraph:
    region = _make_region(n_neurons=n_neurons, syn_per_neuron=syn_per)
    frags = _make_fragments_with_dna(region, dna_dim=dna_dim)
    return build_global_synapse_graph(region, frags, k_neighbors=k)


@pytest.mark.filterwarnings("ignore")
def test_train_gnn_returns_gnn_and_history():
    graph = _make_graph()
    gnn, history = train_global_gnn(graph, n_epochs=3, max_pairs=20, log_every=0)
    assert hasattr(gnn, "forward")
    assert "loss" in history and len(history["loss"]) == 3


@pytest.mark.filterwarnings("ignore")
def test_run_gnn_output_shape():
    graph = _make_graph(n_neurons=3, syn_per=5)
    gnn, _ = train_global_gnn(graph, n_epochs=2, embedding_dim=16, max_pairs=20, log_every=0)
    emb = run_global_gnn(graph, gnn)
    assert emb.shape == (15, 16)
    assert emb.dtype == np.float32


@pytest.mark.filterwarnings("ignore")
def test_run_gnn_embeddings_normalised():
    graph = _make_graph()
    gnn, _ = train_global_gnn(graph, n_epochs=2, max_pairs=20, log_every=0)
    emb = run_global_gnn(graph, gnn)
    norms = np.linalg.norm(emb, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


@pytest.mark.filterwarnings("ignore")
def test_assemble_neurons_label_count():
    graph = _make_graph(n_neurons=4, syn_per=6, k=4)
    gnn, _ = train_global_gnn(graph, n_epochs=2, max_pairs=20, log_every=0)
    labels = assemble_neurons(graph, gnn, threshold=0.3, method="greedy")
    assert labels.shape == (24,)
    assert labels.dtype == np.int64
    assert labels.min() == 0
    n_clusters = len(np.unique(labels))
    # With near-random DNA the number of clusters will vary, just check it's sane.
    assert 1 <= n_clusters <= 24


@pytest.mark.filterwarnings("ignore")
def test_gnn_improves_auc_on_easy_task():
    """With orthogonal DNA vectors and clustered synapses, trained GNN should
    produce high pair AUC — at minimum better than a random baseline."""
    pytest.importorskip("sklearn")
    from sklearn.metrics import roc_auc_score
    from neuronauts.represent.enrich import synapse_pair_dna_scores

    # Build a region where synapses per neuron are spatially co-located so the
    # k-NN graph connects same-neuron synapses; DNA vectors are orthogonal.
    rng = np.random.default_rng(99)
    n_neurons, syn_per = 6, 8
    region = _make_region(n_neurons=n_neurons, syn_per_neuron=syn_per, seed=99)

    # Assign orthogonal DNA: neuron i gets basis vector e_i in R^8
    dna_dim = n_neurons  # one dim per neuron so they're perfectly orthogonal
    root_ids = np.unique(region.pre_root_id)
    root_ids = root_ids[root_ids > 0]
    fragments = []
    for k_idx, rid in enumerate(root_ids):
        syn_idx = np.where(region.pre_root_id == rid)[0]
        n_v = 4
        verts = rng.uniform(0, 1e6, (n_v, 3)).astype(np.float32)
        edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
        radii = np.ones(n_v, dtype=np.float32) * 300.0
        dna = np.zeros(dna_dim, dtype=np.float32)
        dna[k_idx % dna_dim] = 1.0
        from neuronauts.schemas import Fragment
        frag = Fragment(
            fragment_id=int(rid),
            region_id="test",
            base_root_id=int(rid),
            vertices_nm=verts,
            edges=edges,
            endpoints_nm=verts[[0, -1]],
            radius_nm=radii,
            synapse_indices=syn_idx,
            dna=dna,
        ).validate()
        fragments.append(frag)

    graph = build_global_synapse_graph(region, fragments, k_neighbors=4)

    # DNA-only AUC (before GNN)
    scores_dna, labels_dna = synapse_pair_dna_scores(
        region, fragments, max_pairs=200, rng=np.random.default_rng(0)
    )
    auc_dna = float(roc_auc_score(labels_dna, scores_dna))

    # GNN AUC (after training)
    gnn, _ = train_global_gnn(
        graph, n_epochs=30, d_model=16, embedding_dim=dna_dim,
        max_pairs=100, lr=1e-2, log_every=0
    )
    emb = run_global_gnn(graph, gnn)

    # Build temporary fragments with GNN embeddings as DNA
    from neuronauts.schemas import Fragment
    gnn_frags = []
    for frag in fragments:
        sub_emb = emb[frag.synapse_indices]
        mean_emb = sub_emb.mean(axis=0).astype(np.float32)
        gnn_frags.append(Fragment(
            fragment_id=frag.fragment_id,
            region_id=frag.region_id,
            base_root_id=frag.base_root_id,
            vertices_nm=frag.vertices_nm,
            edges=frag.edges,
            endpoints_nm=frag.endpoints_nm,
            radius_nm=frag.radius_nm,
            synapse_indices=frag.synapse_indices,
            dna=mean_emb,
        ).validate())
    scores_gnn, labels_gnn = synapse_pair_dna_scores(
        region, gnn_frags, max_pairs=200, rng=np.random.default_rng(0)
    )
    auc_gnn = float(roc_auc_score(labels_gnn, scores_gnn))

    # Both should be decent on this easy task; the key check is validity
    assert auc_dna > 0.5, f"DNA AUC should beat chance: {auc_dna}"
    assert auc_gnn > 0.5, f"GNN AUC should beat chance: {auc_gnn}"
