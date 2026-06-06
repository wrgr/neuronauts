"""Tests for PR 2: BridgeHead, bridge loss, and Dijkstra bridge proposals."""

import unittest

import numpy as np

from neuronauts.dijkstra import BridgeGraph, BridgePath

try:
    import torch
    from neuronauts.shared_grammar_model import (
        SharedGrammarModel,
        multitask_train_step,
    )
    from neuronauts.merge_dataset import build_merge_examples
    from neuronauts.merge_dataset import examples_to_arrays as merge_examples_to_arrays
    from neuronauts.topology_dataset import build_cluster_examples, examples_to_branch_sequence_arrays
    from neuronauts.fetch import SynapseTable
    _TORCH = True
except ImportError:
    _TORCH = False


# ---------------------------------------------------------------------------
# Dijkstra / BridgeGraph tests (no torch required)
# ---------------------------------------------------------------------------

class BridgeGraphTest(unittest.TestCase):
    def test_add_edge_rejects_negative_weight(self):
        g = BridgeGraph()
        with self.assertRaises(ValueError):
            g.add_edge(0, 1, -1.0)

    def test_add_edge_rejects_inf_weight(self):
        g = BridgeGraph()
        with self.assertRaises(ValueError):
            g.add_edge(0, 1, float("inf"))

    def test_add_edge_ignores_self_loops(self):
        g = BridgeGraph()
        g.add_edge(0, 0, 0.0)
        self.assertEqual(g._adj, {})

    def test_dijkstra_simple_path(self):
        g = BridgeGraph()
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 2.0)
        paths = g.dijkstra(sources=[0])
        self.assertAlmostEqual(paths[2].cost, 3.0)
        self.assertEqual(paths[2].nodes, (0, 1, 2))

    def test_dijkstra_chooses_cheaper_path(self):
        g = BridgeGraph()
        # direct: 0->2 costs 10; via 1: 0->1 + 1->2 = 1 + 1 = 2
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)
        g.add_edge(0, 2, 10.0)
        paths = g.dijkstra(sources=[0])
        self.assertAlmostEqual(paths[2].cost, 2.0)

    def test_dijkstra_max_cost_prunes(self):
        g = BridgeGraph()
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)
        paths = g.dijkstra(sources=[0], max_cost=1.5)
        self.assertIn(1, paths)
        self.assertNotIn(2, paths)

    def test_dijkstra_multi_source(self):
        g = BridgeGraph()
        g.add_edge(0, 2, 5.0)
        g.add_edge(1, 2, 1.0)
        paths = g.dijkstra(sources=[0, 1])
        self.assertAlmostEqual(paths[2].cost, 1.0)
        self.assertEqual(paths[2].nodes[0], 1)

    def test_best_bridge_finds_cheapest_target(self):
        g = BridgeGraph()
        g.add_edge(0, 3, 3.0)
        g.add_edge(0, 4, 1.0)
        result = g.best_bridge(sources=[0], targets=[3, 4])
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.cost, 1.0)
        self.assertIn(4, result.nodes)

    def test_best_bridge_returns_none_when_unreachable(self):
        g = BridgeGraph()
        g.add_edge(0, 1, 1.0)
        result = g.best_bridge(sources=[0], targets=[99])
        self.assertIsNone(result)

    def test_add_path_chains_edges(self):
        g = BridgeGraph()
        g.add_path([10, 11, 12, 13], weight=1.0)
        paths = g.dijkstra(sources=[10])
        self.assertAlmostEqual(paths[13].cost, 3.0)


# ---------------------------------------------------------------------------
# BridgeHead and bridge loss tests (requires torch)
# ---------------------------------------------------------------------------

def _make_synapses():
    return SynapseTable(
        pre_pt=np.array(
            [[1, 1, 1], [2, 1, 1], [10, 10, 10], [11, 10, 10]], dtype=np.float32
        ),
        post_pt=np.array(
            [[1, 5, 1], [2, 5, 1], [10, 15, 10], [11, 15, 10]], dtype=np.float32
        ),
        pre_root_id=np.array([101, 101, 202, 202], dtype=np.int64),
        post_root_id=np.array([301, 301, 402, 402], dtype=np.int64),
        synapse_id=np.arange(4, dtype=np.int64),
    )


@unittest.skipIf(not _TORCH, "torch not installed")
class BridgeHeadTest(unittest.TestCase):
    def test_predict_bridge_output_shape(self):
        model = SharedGrammarModel()
        model.eval()
        D = model._init_kwargs["input_dim"]
        B, T = 4, 6
        x = torch.randn(B, T, D)
        mask = torch.zeros(B, T, dtype=torch.bool)
        with torch.no_grad():
            out = model.predict_bridge(x, mask, x, mask)
        self.assertEqual(tuple(out.shape), (B, 6))

    def test_predict_bridge_midpoint_and_direction_are_finite(self):
        model = SharedGrammarModel()
        model.eval()
        D = model._init_kwargs["input_dim"]
        x = torch.randn(2, 5, D)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        with torch.no_grad():
            out = model.predict_bridge(x, mask, x, mask)
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_bridge_head_exists_in_state_dict(self):
        model = SharedGrammarModel()
        keys = list(model.state_dict().keys())
        bridge_keys = [k for k in keys if k.startswith("bridge_head")]
        self.assertGreater(len(bridge_keys), 0, "bridge_head weights missing from state_dict")

    def test_bridge_loss_updates_bridge_head_weights(self):
        model = SharedGrammarModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        synapses = _make_synapses()

        merge_examples = build_merge_examples(
            synapses, min_fragment_size=2, max_negative_pairs_per_role=4
        )
        from neuronauts.merge_dataset import examples_to_arrays as _mta
        left_x, left_mask, right_x, right_mask, merge_y = _mta(merge_examples)

        topology_examples = build_cluster_examples(
            synapses,
            membrane_field=np.zeros((20, 20, 20), dtype=np.float32),
            min_cluster_size=2,
            max_negative_pairs_per_role=4,
            max_branches=4,
            seed=7,
        )
        branch_x, branch_sequence_mask, branch_mask = examples_to_branch_sequence_arrays(
            topology_examples, max_branches=4
        )
        topology_y = np.array([e.label for e in topology_examples], dtype=np.float32)

        B = 4
        D = model._init_kwargs["input_dim"]
        bridge_batch = {
            "left_x": torch.randn(B, 5, D),
            "left_mask": torch.zeros(B, 5, dtype=torch.bool),
            "right_x": torch.randn(B, 5, D),
            "right_mask": torch.zeros(B, 5, dtype=torch.bool),
            "target_midpoint": torch.randn(B, 3),
            "target_direction": torch.nn.functional.normalize(torch.randn(B, 3), dim=-1),
        }

        before = [p.detach().clone() for p in model.bridge_head.parameters()]
        metrics = multitask_train_step(
            model,
            optimizer,
            merge_batch={
                "left_x": torch.from_numpy(left_x),
                "left_mask": torch.from_numpy(left_mask),
                "right_x": torch.from_numpy(right_x),
                "right_mask": torch.from_numpy(right_mask),
                "y": torch.from_numpy(merge_y.astype(np.float32)),
            },
            topology_batch={
                "branch_x": torch.from_numpy(branch_x),
                "branch_sequence_mask": torch.from_numpy(branch_sequence_mask),
                "branch_mask": torch.from_numpy(branch_mask),
                "y": torch.from_numpy(topology_y),
            },
            bridge_batch=bridge_batch,
        )
        after = list(model.bridge_head.parameters())

        self.assertIn("bridge_loss", metrics)
        self.assertGreater(metrics["bridge_loss"], 0.0)
        # At least one bridge_head parameter should have changed.
        any_changed = any(
            not torch.equal(b, a) for b, a in zip(before, after)
        )
        self.assertTrue(any_changed, "bridge_head weights did not update")

    def test_multitask_step_without_bridge_batch_still_works(self):
        model = SharedGrammarModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        synapses = _make_synapses()

        merge_examples = build_merge_examples(
            synapses, min_fragment_size=2, max_negative_pairs_per_role=4
        )
        from neuronauts.merge_dataset import examples_to_arrays as _mta
        left_x, left_mask, right_x, right_mask, merge_y = _mta(merge_examples)

        topology_examples = build_cluster_examples(
            synapses,
            membrane_field=np.zeros((20, 20, 20), dtype=np.float32),
            min_cluster_size=2,
            max_negative_pairs_per_role=4,
            max_branches=4,
            seed=7,
        )
        branch_x, branch_sequence_mask, branch_mask = examples_to_branch_sequence_arrays(
            topology_examples, max_branches=4
        )
        topology_y = np.array([e.label for e in topology_examples], dtype=np.float32)

        metrics = multitask_train_step(
            model,
            optimizer,
            merge_batch={
                "left_x": torch.from_numpy(left_x),
                "left_mask": torch.from_numpy(left_mask),
                "right_x": torch.from_numpy(right_x),
                "right_mask": torch.from_numpy(right_mask),
                "y": torch.from_numpy(merge_y.astype(np.float32)),
            },
            topology_batch={
                "branch_x": torch.from_numpy(branch_x),
                "branch_sequence_mask": torch.from_numpy(branch_sequence_mask),
                "branch_mask": torch.from_numpy(branch_mask),
                "y": torch.from_numpy(topology_y),
            },
        )
        self.assertIn("loss", metrics)
        self.assertEqual(metrics["bridge_loss"], 0.0)


# ---------------------------------------------------------------------------
# _build_bridge_graph / _propose_bridges integration tests
# ---------------------------------------------------------------------------

@unittest.skipIf(not _TORCH, "torch not installed")
class BridgeCandidateTest(unittest.TestCase):
    def _make_neurons(self):
        from neuronauts.merge import MergedNeuron
        neurons = {
            0: MergedNeuron(
                neuron_id=0,
                agent_ids=[0],
                path_points=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
                synapse_indices=[0],
                role="pre",
            ),
            1: MergedNeuron(
                neuron_id=1,
                agent_ids=[1],
                path_points=np.array([[5, 0, 0], [6, 0, 0], [7, 0, 0]], dtype=np.float32),
                synapse_indices=[1],
                role="pre",
            ),
            2: MergedNeuron(
                neuron_id=2,
                agent_ids=[2],
                path_points=np.array([[20, 20, 20], [21, 20, 20]], dtype=np.float32),
                synapse_indices=[2],
                role="pre",
            ),
        }
        return neurons

    def test_bridge_graph_has_correct_nodes(self):
        from neuronauts.legacy.run import _build_bridge_graph
        neurons = self._make_neurons()
        g = _build_bridge_graph(neurons)
        # Each neuron contributes 2 endpoint nodes + 1 intra-neuron edge.
        # All 6 nodes should be reachable.
        paths = g.dijkstra(sources=[0])
        self.assertIn(1, paths)  # other endpoint of neuron 0

    def test_propose_bridges_returns_sorted_proposals(self):
        from neuronauts.legacy.run import _build_bridge_graph, _propose_bridges
        neurons = self._make_neurons()
        g = _build_bridge_graph(neurons)
        proposals = _propose_bridges(neurons, g)
        # Should get proposals between the 3 neuron pairs.
        self.assertGreater(len(proposals), 0)
        # Costs should be non-decreasing.
        costs = [c for _, _, c in proposals]
        self.assertEqual(costs, sorted(costs))

    def test_propose_bridges_nearest_pair_is_0_and_1(self):
        from neuronauts.legacy.run import _build_bridge_graph, _propose_bridges
        neurons = self._make_neurons()
        g = _build_bridge_graph(neurons)
        proposals = _propose_bridges(neurons, g, top_k=1)
        nid_a, nid_b, _ = proposals[0]
        self.assertEqual({nid_a, nid_b}, {0, 1})

    def test_propose_bridges_max_cost_prunes_distant_neuron(self):
        from neuronauts.legacy.run import _build_bridge_graph, _propose_bridges
        neurons = self._make_neurons()
        # Neuron 2 is ~28 units away; prune anything > 10.
        g = _build_bridge_graph(neurons, max_bridge_cost=10.0)
        proposals = _propose_bridges(neurons, g, max_bridge_cost=10.0)
        neuron_ids_in_proposals = {nid for nid_a, nid_b, _ in proposals for nid in (nid_a, nid_b)}
        self.assertNotIn(2, neuron_ids_in_proposals)


if __name__ == "__main__":
    unittest.main()
