"""Coverage gap tests — medium priority items from coverage audit.

Covers:
- A1: label_graph_edges — mixed-role ('mixed') neuron path in assembly.py
- D1: examples_to_branch_sequence_arrays and examples_to_multi_branch_arrays
      with empty input list
"""

from __future__ import annotations

import unittest
from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# A1 — label_graph_edges with mixed-role neurons
# ---------------------------------------------------------------------------

class LabelGraphEdgesMixedRoleTest(unittest.TestCase):
    """A1 — MEDIUM: 'mixed' role branch in label_graph_edges (lines 242-248 in assembly.py).

    A neuron with role='mixed' should have its synapse indices split across
    both pre and post majority maps.  Wrong logic here gives wrong GAT
    supervision labels.
    """

    def _make_graph(self, n_synapses: int, role: str = "mixed"):
        """Build a minimal ConnectivityGraph with one neuron of the given role."""
        from neuronauts.merge import ConnectivityGraph, MergedNeuron

        rng = np.random.default_rng(0)
        pts = rng.random((4, 3), dtype=np.float32)
        neuron = MergedNeuron(
            neuron_id=0,
            agent_ids=[0],
            path_points=pts,
            synapse_indices=list(range(n_synapses)),
            role=role,
        )
        # Build edges: one per synapse, pre_neuron=0, post_neuron=0 (self-loop
        # as a degenerate test — enough to verify label computation).
        edges = [(0, 0, i) for i in range(n_synapses)]
        return ConnectivityGraph(
            neurons={0: neuron},
            edges=edges,
            unresolved_synapse_indices=[],
        )

    def test_mixed_role_neuron_labels_correct_edges_as_1(self):
        """When pre and post root IDs agree with synapse truth, label=1."""
        from neuronauts.assembly import label_graph_edges

        n = 4
        graph = self._make_graph(n, role="mixed")
        # All synapses from neuron 0 → pre_root_id=10, post_root_id=20
        pre_root_ids = np.full(n, 10, dtype=np.int64)
        post_root_ids = np.full(n, 20, dtype=np.int64)

        labels = label_graph_edges(graph, pre_root_ids, post_root_ids)
        self.assertEqual(labels.shape, (n,))
        self.assertTrue(np.all(labels == 1.0), f"expected all 1.0, got {labels}")

    def test_mixed_role_neuron_labels_wrong_edges_as_0(self):
        """When roots don't match the majority, label=0."""
        from neuronauts.assembly import label_graph_edges

        n = 4
        graph = self._make_graph(n, role="mixed")
        # Neuron 0 sees pre_root_id=10 majority, but each edge's synapse has pre=99 → mismatch.
        pre_root_ids = np.full(n, 99, dtype=np.int64)
        post_root_ids = np.full(n, 20, dtype=np.int64)

        # Majority pre = 99, edge[i] pre_root_ids[syn_idx] = 99 → pre_correct=True
        # But let's make the majority come from a different root:
        # Neuron has syn_indices=[0,1,2,3]; pre_root_ids = [10,10,10,99]
        # → majority pre = 10, but edge syn_idx=3 has pre=99 → wrong
        pre_root_ids = np.array([10, 10, 10, 99], dtype=np.int64)
        post_root_ids = np.full(n, 20, dtype=np.int64)

        labels = label_graph_edges(graph, pre_root_ids, post_root_ids)
        self.assertEqual(labels[3], 0.0, "edge with non-majority root should be 0")

    def test_mixed_role_output_shape_matches_edge_count(self):
        from neuronauts.assembly import label_graph_edges

        n = 6
        graph = self._make_graph(n, role="mixed")
        pre_root_ids = np.ones(n, dtype=np.int64)
        post_root_ids = np.ones(n, dtype=np.int64) * 2
        labels = label_graph_edges(graph, pre_root_ids, post_root_ids)
        self.assertEqual(labels.shape[0], n)

    def test_mixed_role_labels_float32(self):
        from neuronauts.assembly import label_graph_edges

        n = 3
        graph = self._make_graph(n, role="mixed")
        labels = label_graph_edges(
            graph,
            np.ones(n, dtype=np.int64),
            np.ones(n, dtype=np.int64) * 2,
        )
        self.assertEqual(labels.dtype, np.float32)

    def test_mixed_role_populates_both_pre_and_post_majority_maps(self):
        """A 'mixed' neuron contributes to BOTH the pre and post majority maps.

        This means an edge (n0, n0, syn_idx) is labelled 1 only when BOTH the
        pre-side root matches AND the post-side root matches.  A 'pre' neuron
        by contrast has no post majority → post_correct is always False → all
        edges label 0 even when pre matches.
        """
        from neuronauts.assembly import label_graph_edges
        from neuronauts.merge import ConnectivityGraph, MergedNeuron

        rng = np.random.default_rng(1)
        pts = rng.random((4, 3), dtype=np.float32)
        n = 4

        # All synapses share the same pre/post root → majority will match each edge.
        pre_root_ids = np.full(n, 10, dtype=np.int64)
        post_root_ids = np.full(n, 30, dtype=np.int64)

        def make(role):
            neuron = MergedNeuron(
                neuron_id=0,
                agent_ids=[0],
                path_points=pts,
                synapse_indices=list(range(n)),
                role=role,
            )
            edges = [(0, 0, i) for i in range(n)]
            return ConnectivityGraph(
                neurons={0: neuron},
                edges=edges,
                unresolved_synapse_indices=[],
            )

        labels_mixed = label_graph_edges(make("mixed"), pre_root_ids, post_root_ids)
        labels_pre = label_graph_edges(make("pre"), pre_root_ids, post_root_ids)

        # Mixed role: both pre and post majority maps populated → all edges = 1.
        np.testing.assert_array_equal(
            labels_mixed, np.ones(n, dtype=np.float32),
            err_msg="mixed role should label all correct edges as 1",
        )
        # Pre-only role: no post majority for the neuron → post_correct always False → all 0.
        np.testing.assert_array_equal(
            labels_pre, np.zeros(n, dtype=np.float32),
            err_msg="pure pre role has no post majority → all edges should be 0",
        )

    def test_no_synapse_indices_gives_all_zero_labels(self):
        """Neurons with empty synapse_indices have no majority → labels=0."""
        from neuronauts.assembly import label_graph_edges
        from neuronauts.merge import ConnectivityGraph, MergedNeuron

        rng = np.random.default_rng(2)
        pts = rng.random((4, 3), dtype=np.float32)
        n = 3
        neuron = MergedNeuron(
            neuron_id=0,
            agent_ids=[],
            path_points=pts,
            synapse_indices=[],  # empty
            role="mixed",
        )
        edges = [(0, 0, i) for i in range(n)]
        graph = ConnectivityGraph(
            neurons={0: neuron},
            edges=edges,
            unresolved_synapse_indices=[],
        )
        labels = label_graph_edges(
            graph,
            np.ones(n, dtype=np.int64),
            np.ones(n, dtype=np.int64) * 2,
        )
        np.testing.assert_array_equal(labels, np.zeros(n, dtype=np.float32))


# ---------------------------------------------------------------------------
# D1 — empty-input topology array builders
# ---------------------------------------------------------------------------

class EmptyTopologyArrayBuildersTest(unittest.TestCase):
    """D1 — MEDIUM: empty ClusterExample lists should not crash downstream.

    torch.from_numpy on a zero-row array is fine but wrong ndim silently
    breaks the training loop.
    """

    def test_examples_to_multi_branch_arrays_empty_returns_zero_rows(self):
        from neuronauts.topology_dataset import examples_to_multi_branch_arrays
        x, y, mask = examples_to_multi_branch_arrays([])
        self.assertEqual(x.shape[0], 0, "expected 0 rows")
        self.assertEqual(y.shape[0], 0)
        self.assertEqual(mask.shape[0], 0)

    def test_examples_to_multi_branch_arrays_empty_correct_ndim(self):
        from neuronauts.topology_dataset import examples_to_multi_branch_arrays
        x, y, mask = examples_to_multi_branch_arrays([])
        self.assertEqual(x.ndim, 3, "x must be 3-D for downstream nn.Linear")
        self.assertEqual(y.ndim, 1)
        self.assertEqual(mask.ndim, 2)

    def test_examples_to_multi_branch_arrays_empty_dtype(self):
        from neuronauts.topology_dataset import examples_to_multi_branch_arrays
        x, y, mask = examples_to_multi_branch_arrays([])
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(y.dtype, np.int64)

    def test_examples_to_branch_sequence_arrays_empty_returns_zero_rows(self):
        from neuronauts.topology_dataset import examples_to_branch_sequence_arrays
        bx, bsm, bm = examples_to_branch_sequence_arrays([])
        self.assertEqual(bx.shape[0], 0, "expected 0 rows")
        self.assertEqual(bsm.shape[0], 0)
        self.assertEqual(bm.shape[0], 0)

    def test_examples_to_branch_sequence_arrays_empty_4d(self):
        from neuronauts.topology_dataset import examples_to_branch_sequence_arrays
        bx, _, _ = examples_to_branch_sequence_arrays([])
        self.assertEqual(bx.ndim, 4, "branch_x must be 4-D for downstream transformer")

    def test_examples_to_branch_sequence_arrays_empty_float32(self):
        from neuronauts.topology_dataset import examples_to_branch_sequence_arrays
        bx, bsm, bm = examples_to_branch_sequence_arrays([])
        self.assertEqual(bx.dtype, np.float32)

    def test_both_builders_roundtrip_nonempty_then_empty(self):
        """Calling the builders after a real run shouldn't leave state that breaks empty."""
        from neuronauts.topology_dataset import (
            examples_to_branch_sequence_arrays,
            examples_to_multi_branch_arrays,
        )
        from neuronauts.fetch import SynapseTable
        from neuronauts.topology_dataset import build_cluster_examples

        rng = np.random.default_rng(3)
        n = 15
        syn = SynapseTable(
            pre_pt=rng.random((n, 3), dtype=np.float32) * 40,
            post_pt=rng.random((n, 3), dtype=np.float32) * 40,
            pre_root_id=rng.integers(1, 4, size=n, dtype=np.int64),
            post_root_id=rng.integers(11, 14, size=n, dtype=np.int64),
            synapse_id=np.arange(n, dtype=np.int64),
        )
        examples = build_cluster_examples(syn, membrane_field=None)
        if examples:
            _ = examples_to_branch_sequence_arrays(examples)
            _ = examples_to_multi_branch_arrays(examples)

        # Now call with empty — should still work.
        bx, bsm, bm = examples_to_branch_sequence_arrays([])
        x, y, mask = examples_to_multi_branch_arrays([])
        self.assertEqual(bx.shape[0], 0)
        self.assertEqual(x.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
