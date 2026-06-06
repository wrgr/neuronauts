"""Tests for PR 5: HeuristicConfig and learned-mode decommissioning."""

import unittest

import numpy as np

from neuronauts.legacy.run import (
    HeuristicConfig,
    MERGE_RADIUS,
    MERGE_OVERLAP_THRESHOLD,
    POLARITY_CAPTURE_R,
    MAX_SYNAPSES_PER_NEURON,
    ROLE_MERGE_MIN_SHARED_HITS,
    _merge_role_groups,
    _build_graph,
    _nearest_owner,
)
from neuronauts._scipy_compat import cKDTree


# ---------------------------------------------------------------------------
# HeuristicConfig unit tests
# ---------------------------------------------------------------------------

class HeuristicConfigDefaultsTest(unittest.TestCase):
    def test_legacy_matches_module_constants(self):
        cfg = HeuristicConfig.legacy()
        self.assertEqual(cfg.merge_radius, MERGE_RADIUS)
        self.assertEqual(cfg.merge_overlap_threshold, MERGE_OVERLAP_THRESHOLD)
        self.assertEqual(cfg.polarity_capture_r, POLARITY_CAPTURE_R)
        self.assertEqual(cfg.max_synapses_per_neuron, MAX_SYNAPSES_PER_NEURON)
        self.assertEqual(cfg.role_merge_min_shared_hits, ROLE_MERGE_MIN_SHARED_HITS)
        self.assertFalse(cfg.use_learned_decisions)

    def test_learned_mode_flags(self):
        cfg = HeuristicConfig.learned()
        self.assertTrue(cfg.use_learned_decisions)
        self.assertEqual(cfg.merge_overlap_threshold, 0.0)
        self.assertEqual(cfg.polarity_capture_r, float("inf"))
        self.assertGreater(cfg.max_synapses_per_neuron, MAX_SYNAPSES_PER_NEURON)

    def test_learned_keeps_merge_radius(self):
        # merge_radius is still a valid spatial pre-filter in learned mode.
        cfg = HeuristicConfig.learned()
        self.assertEqual(cfg.merge_radius, MERGE_RADIUS)

    def test_learned_keeps_min_shared_hits(self):
        # Data-quality guard retained in both modes.
        cfg = HeuristicConfig.learned()
        self.assertEqual(cfg.role_merge_min_shared_hits, ROLE_MERGE_MIN_SHARED_HITS)

    def test_frozen_immutable(self):
        cfg = HeuristicConfig()
        with self.assertRaises((AttributeError, TypeError)):
            cfg.merge_radius = 99.0  # type: ignore[misc]

    def test_custom_values(self):
        cfg = HeuristicConfig(merge_radius=5.0, polarity_capture_r=10.0)
        self.assertEqual(cfg.merge_radius, 5.0)
        self.assertEqual(cfg.polarity_capture_r, 10.0)


# ---------------------------------------------------------------------------
# _merge_role_groups: legacy vs learned behaviour
# ---------------------------------------------------------------------------

def _make_overlapping_hits(n_agents=4, n_syn=4):
    """Two natural groups: agents 0,1 share synapses 0,1; agents 2,3 share 2,3."""
    role_hits = np.zeros((n_agents, n_syn), dtype=bool)
    role_hits[0, 0] = role_hits[0, 1] = True
    role_hits[1, 0] = role_hits[1, 1] = True
    role_hits[2, 2] = role_hits[2, 3] = True
    role_hits[3, 2] = role_hits[3, 3] = True
    return role_hits


def _make_path_arr(n_agents=4, n_steps=10, spread=1.0, seed=0):
    rng = np.random.default_rng(seed)
    # Place agents 0,1 near origin; 2,3 far away so they don't cross-merge.
    pts = np.zeros((n_agents, n_steps, 3), dtype=np.float32)
    pts[0] = rng.random((n_steps, 3)) * spread
    pts[1] = rng.random((n_steps, 3)) * spread
    pts[2] = rng.random((n_steps, 3)) * spread + 50.0
    pts[3] = rng.random((n_steps, 3)) * spread + 50.0
    return pts


class MergeRoleGroupsHeuristicTest(unittest.TestCase):
    def _run(self, hcfg, merge_score_fn=None):
        path_arr = _make_path_arr()
        role_hits = _make_overlapping_hits()
        return _merge_role_groups(
            path_arr, role_hits, "pre", 0,
            heuristic_config=hcfg,
            learned_merge_score_fn=merge_score_fn,
        )

    def test_legacy_mode_merges_by_overlap(self):
        # Wide overlap threshold — should still merge the obvious pairs.
        cfg = HeuristicConfig(
            merge_overlap_threshold=0.0, use_learned_decisions=False,
            merge_radius=100.0,  # wide so all agents are candidates
        )
        neurons, _, _, _ = self._run(cfg)
        # Should produce 2 neurons (one per natural group).
        self.assertEqual(len(neurons), 2)

    def test_legacy_high_threshold_prevents_merge(self):
        # Overlap threshold of 1.0 means agents need 100% hit overlap to merge.
        cfg = HeuristicConfig(
            merge_overlap_threshold=1.01,  # impossible
            use_learned_decisions=False,
            merge_radius=100.0,
        )
        neurons, _, _, _ = self._run(cfg)
        # No merges → 4 separate neurons.
        self.assertEqual(len(neurons), 4)

    def test_learned_mode_no_scorer_unions_optimistically(self):
        # In learned mode with no scorer, pairs within merge_radius
        # are unioned immediately (deferred to GAT). With a huge merge_radius
        # all 4 agents merge together.
        cfg = HeuristicConfig(
            merge_radius=1000.0,
            use_learned_decisions=True,
        )
        neurons, _, _, _ = self._run(cfg)
        # All agents collapse into ≤ 2 neurons.
        self.assertLessEqual(len(neurons), 2)

    def test_min_shared_hits_guard_kept_in_learned_mode(self):
        # Two agents sharing 0 hits should never merge even in learned mode.
        path_arr = _make_path_arr()
        # Disjoint hits — no shared synapses anywhere.
        role_hits = np.zeros((4, 4), dtype=bool)
        role_hits[0, 0] = True
        role_hits[1, 1] = True
        role_hits[2, 2] = True
        role_hits[3, 3] = True
        cfg = HeuristicConfig(
            merge_radius=1000.0,
            role_merge_min_shared_hits=2,  # need 2 shared hits
            use_learned_decisions=True,
        )
        neurons, _, _, _ = _merge_role_groups(
            path_arr, role_hits, "pre", 0, heuristic_config=cfg
        )
        # No pairs share ≥ 2 hits → no merges.
        self.assertEqual(len(neurons), 4)

    def test_learned_mode_with_always_merge_scorer(self):
        # When a scorer always returns a high score, all candidates should merge.
        cfg = HeuristicConfig(merge_radius=1000.0, use_learned_decisions=True)
        neurons, _, _, _ = self._run(cfg, merge_score_fn=lambda a, b: 10.0)
        self.assertLessEqual(len(neurons), 2)

    def test_learned_mode_with_always_reject_scorer(self):
        # When scorer returns a very negative score (below threshold 0.0),
        # no merges happen.
        cfg = HeuristicConfig(merge_radius=1000.0, use_learned_decisions=True)
        neurons, _, _, _ = self._run(cfg, merge_score_fn=lambda a, b: -100.0)
        # 4 agents → 4 neurons (no merges accepted).
        self.assertEqual(len(neurons), 4)


# ---------------------------------------------------------------------------
# _build_graph: polarity_capture_r and max_synapses_per_neuron
# ---------------------------------------------------------------------------

class BuildGraphHeuristicTest(unittest.TestCase):
    def _minimal_graph(self, hcfg):
        """Build a minimal but valid graph fixture.

        Agents need non-zero entries past MIN_PATH_LENGTH (=5) for
        _valid_agent_indices to keep them.  We place 10 steps, all non-zero.
        """
        n_steps = 10
        path_arr = np.zeros((4, n_steps, 3), dtype=np.float32)
        rng = np.random.default_rng(0)
        # Pre-agents near (1, 1, 1)
        path_arr[0] = rng.random((n_steps, 3)) * 0.2 + 1.0
        path_arr[1] = rng.random((n_steps, 3)) * 0.2 + 1.0
        # Post-agents near (5, 5, 5)
        path_arr[2] = rng.random((n_steps, 3)) * 0.2 + 5.0
        path_arr[3] = rng.random((n_steps, 3)) * 0.2 + 5.0
        path_lengths = np.ones(4, dtype=np.int32) * n_steps

        pre_pts = np.array([[1.1, 1.1, 1.1]], dtype=np.float32)
        post_pts = np.array([[5.1, 5.1, 5.1]], dtype=np.float32)

        # hits: agent 0 hits pre, agent 2 hits post
        synapse_hits = np.zeros((4, 2), dtype=bool)
        synapse_hits[0, 0] = True  # pre hit
        synapse_hits[2, 1] = True  # post hit
        synapse_hits[1, 0] = True  # pre hit (second agent)
        synapse_hits[3, 1] = True  # post hit (second agent)

        return _build_graph(
            path_arr, path_lengths, synapse_hits, pre_pts, post_pts,
            heuristic_config=hcfg,
        )

    def test_legacy_tight_capture_radius_blocks_assignment(self):
        # With polarity_capture_r=0.0 nothing can be assigned.
        cfg = HeuristicConfig(polarity_capture_r=0.0, merge_radius=100.0)
        graph = self._minimal_graph(cfg)
        self.assertEqual(len(graph.edges), 0)
        self.assertIn(0, graph.unresolved_synapse_indices)

    def test_legacy_wide_capture_radius_allows_assignment(self):
        cfg = HeuristicConfig(polarity_capture_r=100.0, merge_radius=100.0)
        graph = self._minimal_graph(cfg)
        self.assertGreater(len(graph.edges), 0)

    def test_learned_inf_capture_radius_allows_all_assignments(self):
        cfg = HeuristicConfig.learned()
        cfg = HeuristicConfig(
            polarity_capture_r=float("inf"),
            merge_radius=100.0,
            use_learned_decisions=True,
        )
        graph = self._minimal_graph(cfg)
        self.assertGreater(len(graph.edges), 0)

    def test_max_synapses_cap_respected(self):
        # With cap=0, no edges should be accepted.
        cfg = HeuristicConfig(
            polarity_capture_r=100.0,
            max_synapses_per_neuron=0,
            merge_radius=100.0,
        )
        graph = self._minimal_graph(cfg)
        self.assertEqual(len(graph.edges), 0)


# ---------------------------------------------------------------------------
# _nearest_owner: owner_margin parameter
# ---------------------------------------------------------------------------

class NearestOwnerTest(unittest.TestCase):
    def _make_trees(self):
        pts_a = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        pts_b = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        trees = {0: cKDTree(pts_a), 1: cKDTree(pts_b)}
        owners = {0: [0, 1]}
        return owners, trees

    def test_returns_closest_owner(self):
        owners, trees = self._make_trees()
        pt = np.array([0.1, 0, 0], dtype=np.float32)
        nid, dist = _nearest_owner(0, pt, owners, trees, owner_margin=0.0)
        self.assertEqual(nid, 0)
        self.assertLess(dist, 1.0)

    def test_ambiguous_owners_blocked_by_margin(self):
        owners, trees = self._make_trees()
        # Point exactly halfway between the two owners.
        pt = np.array([0.5, 0, 0], dtype=np.float32)
        # Large margin: |dist_1 - dist_0| = 0.0 < margin → ambiguous → None
        nid, dist = _nearest_owner(0, pt, owners, trees, owner_margin=1.0)
        self.assertIsNone(nid)

    def test_no_owners_returns_none(self):
        pt = np.array([0.0, 0, 0], dtype=np.float32)
        nid, dist = _nearest_owner(99, pt, {}, {})
        self.assertIsNone(nid)
        self.assertEqual(dist, float("inf"))


# ---------------------------------------------------------------------------
# run() integration: auto-mode selection
# ---------------------------------------------------------------------------

class RunAutoModeTest(unittest.TestCase):
    def test_legacy_mode_by_default(self):
        """Without any checkpoint, run() should use legacy heuristic mode."""
        from neuronauts.fetch import SyntheticBenchmarkConfig, make_test_volume
        from neuronauts.legacy.run import run

        cfg = SyntheticBenchmarkConfig(n_synapses=6, shape=(32, 32, 32))
        chunk, syn = make_test_volume(config=cfg, seed=42)
        metrics = run(
            chunk.data, syn.pre_pt, syn.post_pt,
            syn.pre_root_id, syn.post_root_id,
            verbose=False,
        )
        self.assertGreaterEqual(metrics.f1, 0.0)
        self.assertLessEqual(metrics.f1, 1.0)

    def test_learned_mode_with_gat_checkpoint(self):
        """Providing a GAT checkpoint should activate learned mode (inf polarity_capture_r)."""
        import tempfile
        from pathlib import Path
        import torch
        from neuronauts.fetch import SyntheticBenchmarkConfig, make_test_volume
        from neuronauts.legacy.run import run
        from neuronauts.shared_grammar_model import GlobalAssemblyGAT, save_global_assembly_gat

        gat = GlobalAssemblyGAT(node_dim=32, gat_dim=32, n_heads=2, n_layers=1, dropout=0.0)
        cfg = SyntheticBenchmarkConfig(n_synapses=6, shape=(32, 32, 32))
        chunk, syn = make_test_volume(config=cfg, seed=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "gat.pt"
            save_global_assembly_gat(ckpt, gat)
            metrics = run(
                chunk.data, syn.pre_pt, syn.post_pt,
                syn.pre_root_id, syn.post_root_id,
                verbose=False,
                gat_assembly_checkpoint=str(ckpt),
            )

        self.assertGreaterEqual(metrics.f1, 0.0)
        self.assertLessEqual(metrics.f1, 1.0)


if __name__ == "__main__":
    unittest.main()
