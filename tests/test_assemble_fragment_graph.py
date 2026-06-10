"""Tests for neuronauts/assemble/fragment_graph.py (Phase 2)."""
from __future__ import annotations

import numpy as np
import pytest

from neuronauts.schemas import Fragment, NeuronHypothesis
from neuronauts.assemble.fragment_graph import (
    assemble_fragments,
    build_fragment_graph,
    score_edge,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chain_fragment(
    fragment_id: int,
    start: np.ndarray,
    end: np.ndarray,
    n_verts: int = 8,
    *,
    region_id: str = "test",
    dna: np.ndarray | None = None,
) -> Fragment:
    """Build a simple linear-chain fragment between two 3-D points."""
    t = np.linspace(0, 1, n_verts)[:, None]
    verts = (start * (1 - t) + end * t).astype(np.float32)
    edges = np.stack([np.arange(n_verts - 1), np.arange(1, n_verts)], axis=1).astype(np.int64)
    endpoints = verts[[0, -1]]  # leaf vertices
    radii = np.ones(n_verts, dtype=np.float32) * 300.0
    return Fragment(
        fragment_id=fragment_id,
        region_id=region_id,
        base_root_id=fragment_id,
        vertices_nm=verts,
        edges=edges,
        endpoints_nm=endpoints,
        radius_nm=radii,
        synapse_indices=np.arange(fragment_id * 3, fragment_id * 3 + 3, dtype=np.int64),
        dna=dna,
    ).validate()


# ---------------------------------------------------------------------------
# score_edge
# ---------------------------------------------------------------------------

class TestScoreEdge:
    def test_touching_no_dna(self):
        """Fragments whose endpoints touch get proximity ≈ 1, score > 0."""
        a = _chain_fragment(1, np.array([0, 0, 0], dtype=np.float32),
                            np.array([1000, 0, 0], dtype=np.float32))
        b = _chain_fragment(2, np.array([1000, 0, 0], dtype=np.float32),
                            np.array([2000, 0, 0], dtype=np.float32))
        s = score_edge(a, b, endpoint_radius_nm=5_000.0)
        assert s > 0.9, f"expected > 0.9 for touching ends, got {s}"

    def test_far_apart_returns_zero(self):
        """Fragments outside endpoint_radius_nm get score == 0."""
        a = _chain_fragment(1, np.array([0, 0, 0], dtype=np.float32),
                            np.array([1000, 0, 0], dtype=np.float32))
        b = _chain_fragment(2, np.array([1e7, 0, 0], dtype=np.float32),
                            np.array([1.1e7, 0, 0], dtype=np.float32))
        s = score_edge(a, b, endpoint_radius_nm=5_000.0)
        assert s == 0.0

    def test_dna_compatible_boosts_score(self):
        """Identical DNA embeddings should produce score == proximity × 1."""
        dna = np.ones(16, dtype=np.float32)
        a = _chain_fragment(1, np.array([0, 0, 0], dtype=np.float32),
                            np.array([1000, 0, 0], dtype=np.float32), dna=dna)
        b = _chain_fragment(2, np.array([1000, 0, 0], dtype=np.float32),
                            np.array([2000, 0, 0], dtype=np.float32), dna=dna)
        s_with_dna = score_edge(a, b, endpoint_radius_nm=5_000.0)
        # Without DNA
        a2 = _chain_fragment(1, np.array([0, 0, 0], dtype=np.float32),
                             np.array([1000, 0, 0], dtype=np.float32))
        b2 = _chain_fragment(2, np.array([1000, 0, 0], dtype=np.float32),
                             np.array([2000, 0, 0], dtype=np.float32))
        s_no_dna = score_edge(a2, b2, endpoint_radius_nm=5_000.0)
        # Identical DNA: dna_compat = (1+1)/2 = 1.0, so score should equal no-DNA
        assert abs(s_with_dna - s_no_dna) < 1e-4

    def test_dna_orthogonal_halves_score(self):
        """Orthogonal DNA embeddings reduce score (dna_compat = 0.5)."""
        dna_a = np.array([1, 0] * 8, dtype=np.float32)
        dna_b = np.array([0, 1] * 8, dtype=np.float32)
        a = _chain_fragment(1, np.array([0, 0, 0], dtype=np.float32),
                            np.array([1000, 0, 0], dtype=np.float32), dna=dna_a)
        b = _chain_fragment(2, np.array([1000, 0, 0], dtype=np.float32),
                            np.array([2000, 0, 0], dtype=np.float32), dna=dna_b)
        s = score_edge(a, b, endpoint_radius_nm=5_000.0)
        proximity = 1.0 - 0.0 / 5_000.0  # gap = 0 nm (touching)
        expected = proximity * 0.5  # cosine = 0 → compat = 0.5
        assert abs(s - expected) < 0.01, f"expected {expected:.3f}, got {s:.3f}"


# ---------------------------------------------------------------------------
# build_fragment_graph
# ---------------------------------------------------------------------------

class TestBuildFragmentGraph:
    def test_empty_returns_empty(self):
        src, dst, sc = build_fragment_graph([])
        assert len(src) == 0 and len(dst) == 0 and len(sc) == 0

    def test_single_fragment(self):
        f = _chain_fragment(0, np.zeros(3, np.float32), np.array([1000, 0, 0], np.float32))
        src, dst, sc = build_fragment_graph([f], endpoint_radius_nm=5_000.0)
        assert len(src) == 0

    def test_two_touching_fragments_connected(self):
        """Two fragments that share an endpoint should be connected."""
        a = _chain_fragment(0, np.zeros(3, np.float32), np.array([1000, 0, 0], np.float32))
        b = _chain_fragment(1, np.array([1000, 0, 0], np.float32),
                            np.array([2000, 0, 0], np.float32))
        src, dst, sc = build_fragment_graph([a, b], endpoint_radius_nm=5_000.0)
        assert len(src) == 1
        assert {int(src[0]), int(dst[0])} == {0, 1}
        assert float(sc[0]) > 0.8

    def test_two_far_fragments_not_connected(self):
        a = _chain_fragment(0, np.zeros(3, np.float32), np.array([1000, 0, 0], np.float32))
        b = _chain_fragment(1, np.array([1e7, 0, 0], np.float32),
                            np.array([1.1e7, 0, 0], np.float32))
        src, dst, sc = build_fragment_graph([a, b], endpoint_radius_nm=5_000.0)
        assert len(src) == 0

    def test_three_chain_two_edges(self):
        """Three sequential fragments spaced 10 µm apart form exactly 2 edges
        with a 12 µm radius (adjacent pairs connect; 0↔2 gap is 20 µm)."""
        frags = [
            _chain_fragment(i, np.array([i * 10_000, 0, 0], np.float32),
                            np.array([(i + 1) * 10_000, 0, 0], np.float32))
            for i in range(3)
        ]
        # radius = 12_000 nm: gap between frags 0 and 1 is 0 nm (touching),
        # gap between frags 1 and 2 is 0 nm (touching),
        # gap between frags 0 and 2 is 10_000 nm < 12_000 → all 3 connect.
        # Use tighter radius so only immediate neighbours connect.
        src, dst, sc = build_fragment_graph(frags, endpoint_radius_nm=5_000.0)
        # With radius 5000 and spacing 10000: only touching ends (gap=0) connect.
        assert len(src) == 2

    def test_edge_indices_in_range(self):
        frags = [
            _chain_fragment(i, np.array([i * 1000, 0, 0], np.float32),
                            np.array([(i + 1) * 1000, 0, 0], np.float32))
            for i in range(5)
        ]
        src, dst, sc = build_fragment_graph(frags, endpoint_radius_nm=5_000.0)
        assert src.min() >= 0 and src.max() < 5
        assert dst.min() >= 0 and dst.max() < 5

    def test_src_less_than_dst(self):
        """Edges are deduplicated with src < dst."""
        frags = [
            _chain_fragment(i, np.array([i * 1000, 0, 0], np.float32),
                            np.array([(i + 1) * 1000, 0, 0], np.float32))
            for i in range(3)
        ]
        src, dst, sc = build_fragment_graph(frags, endpoint_radius_nm=5_000.0)
        assert all(int(s) < int(d) for s, d in zip(src, dst))

    def test_scores_in_range(self):
        frags = [
            _chain_fragment(i, np.array([i * 1000, 0, 0], np.float32),
                            np.array([(i + 1) * 1000, 0, 0], np.float32))
            for i in range(4)
        ]
        src, dst, sc = build_fragment_graph(frags, endpoint_radius_nm=5_000.0)
        assert (sc >= 0.0).all() and (sc <= 1.0).all()

    def test_degree_cap_respected(self):
        """max_edges_per_fragment caps neighbourhood size."""
        # Hub fragment 0 touching 6 satellites; cap at 2.
        hub_start = np.zeros(3, np.float32)
        hub_end = np.array([500, 0, 0], np.float32)
        frags = [_chain_fragment(0, hub_start, hub_end)]
        for i in range(1, 7):
            frags.append(_chain_fragment(
                i, hub_end, hub_end + np.array([i * 100, i * 100, 0], np.float32)
            ))
        src, dst, sc = build_fragment_graph(
            frags, endpoint_radius_nm=5_000.0, max_edges_per_fragment=2
        )
        # Hub (index 0) appears in at most 2 edges.
        hub_degree = int((src == 0).sum()) + int((dst == 0).sum())
        assert hub_degree <= 2


# ---------------------------------------------------------------------------
# assemble_neurons
# ---------------------------------------------------------------------------

class TestAssembleNeurons:
    def _three_chain(self) -> list[Fragment]:
        return [
            _chain_fragment(i, np.array([i * 1000, 0, 0], np.float32),
                            np.array([(i + 1) * 1000, 0, 0], np.float32))
            for i in range(3)
        ]

    def test_all_connected_gives_one_hypothesis(self):
        frags = self._three_chain()
        src, dst, sc = build_fragment_graph(frags, endpoint_radius_nm=5_000.0)
        hyps = assemble_fragments(frags, src, dst, sc, score_threshold=0.0)
        assert len(hyps) == 1
        assert set(hyps[0].fragment_ids) == {0, 1, 2}

    def test_high_threshold_gives_isolated_hypotheses(self):
        frags = self._three_chain()
        src, dst, sc = build_fragment_graph(frags, endpoint_radius_nm=5_000.0)
        hyps = assemble_fragments(frags, src, dst, sc, score_threshold=2.0)
        assert len(hyps) == 3

    def test_hypothesis_spans_regions_when_cross_region(self):
        a = _chain_fragment(0, np.zeros(3, np.float32),
                            np.array([1000, 0, 0], np.float32), region_id="region_A")
        b = _chain_fragment(1, np.array([1000, 0, 0], np.float32),
                            np.array([2000, 0, 0], np.float32), region_id="region_B")
        src, dst, sc = build_fragment_graph([a, b], endpoint_radius_nm=5_000.0)
        hyps = assemble_fragments([a, b], src, dst, sc, score_threshold=0.0)
        assert len(hyps) == 1
        assert set(hyps[0].spans_regions) == {"region_A", "region_B"}

    def test_hypothesis_single_region_when_same_region(self):
        frags = self._three_chain()
        src, dst, sc = build_fragment_graph(frags, endpoint_radius_nm=5_000.0)
        hyps = assemble_fragments(frags, src, dst, sc, score_threshold=0.0)
        assert hyps[0].spans_regions == ["test"]

    def test_pooled_dna_is_mean(self):
        dna_a = np.array([1.0, 0.0], dtype=np.float32)
        dna_b = np.array([0.0, 1.0], dtype=np.float32)
        a = _chain_fragment(0, np.zeros(3, np.float32),
                            np.array([1000, 0, 0], np.float32), dna=dna_a)
        b = _chain_fragment(1, np.array([1000, 0, 0], np.float32),
                            np.array([2000, 0, 0], np.float32), dna=dna_b)
        src, dst, sc = build_fragment_graph([a, b], endpoint_radius_nm=5_000.0)
        hyps = assemble_fragments([a, b], src, dst, sc, score_threshold=0.0)
        expected = np.array([0.5, 0.5], dtype=np.float32)
        np.testing.assert_allclose(hyps[0].pooled_dna, expected, atol=1e-6)

    def test_synapse_indices_union(self):
        frags = self._three_chain()
        src, dst, sc = build_fragment_graph(frags, endpoint_radius_nm=5_000.0)
        hyps = assemble_fragments(frags, src, dst, sc, score_threshold=0.0)
        # Fragment synapse_indices: 0→[0,1,2], 1→[3,4,5], 2→[6,7,8]
        expected_syns = set(range(9))
        assert set(hyps[0].synapse_indices.tolist()) == expected_syns

    def test_sorted_by_descending_fragment_count(self):
        """Largest hypothesis first."""
        # Two chains of length 3 and 2, far apart.
        chain3 = [
            _chain_fragment(i, np.array([i * 1000, 0, 0], np.float32),
                            np.array([(i + 1) * 1000, 0, 0], np.float32))
            for i in range(3)
        ]
        chain2 = [
            _chain_fragment(10 + i, np.array([1e7 + i * 1000, 0, 0], np.float32),
                            np.array([1e7 + (i + 1) * 1000, 0, 0], np.float32))
            for i in range(2)
        ]
        frags = chain3 + chain2
        src, dst, sc = build_fragment_graph(frags, endpoint_radius_nm=5_000.0)
        hyps = assemble_fragments(frags, src, dst, sc, score_threshold=0.0)
        assert len(hyps) == 2
        assert len(hyps[0].fragment_ids) >= len(hyps[1].fragment_ids)

    def test_no_edges_all_singletons(self):
        """With no edges, every fragment becomes its own hypothesis."""
        frags = [
            _chain_fragment(i, np.array([i * 1e6, 0, 0], np.float32),
                            np.array([i * 1e6 + 1000, 0, 0], np.float32))
            for i in range(4)
        ]
        src, dst, sc = build_fragment_graph(frags, endpoint_radius_nm=5_000.0)
        hyps = assemble_fragments(frags, src, dst, sc, score_threshold=0.0)
        assert len(hyps) == 4
        for h in hyps:
            assert len(h.fragment_ids) == 1

    def test_hypothesis_validates_without_error(self):
        frags = self._three_chain()
        src, dst, sc = build_fragment_graph(frags, endpoint_radius_nm=5_000.0)
        hyps = assemble_fragments(frags, src, dst, sc, score_threshold=0.0)
        # validate() should not raise
        for h in hyps:
            h.validate()
