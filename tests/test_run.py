import tempfile
import unittest
from pathlib import Path

import numpy as np

from neuronauts.line_graph import build_estimated_line_graph
from neuronauts.legacy.run import _build_graph, _load_shared_merge_score_fn
from neuronauts.shared_grammar_model import SharedGrammarModel, save_shared_grammar_model

try:
    import torch
except ImportError:
    torch = None


class BuildGraphTest(unittest.TestCase):
    def test_build_graph_separates_pre_and_post_role_groups(self):
        path_arr = np.zeros((2, 6, 3), dtype=np.float32)
        shared_path = np.array(
            [
                [1, 1, 1],
                [2, 1, 1],
                [3, 1, 1],
                [4, 1, 1],
                [5, 1, 1],
                [6, 1, 1],
            ],
            dtype=np.float32,
        )
        path_arr[0] = shared_path
        path_arr[1] = shared_path

        pre_pts = np.array([[1, 1, 1], [6, 1, 1]], dtype=np.float32)
        post_pts = np.array([[6, 1, 1], [1, 1, 1]], dtype=np.float32)

        synapse_hits = np.zeros((2, 4), dtype=bool)
        synapse_hits[0, 0] = True
        synapse_hits[0, 3] = True
        synapse_hits[1, 1] = True
        synapse_hits[1, 2] = True

        graph = _build_graph(
            path_arr=path_arr,
            path_lengths=np.array([6, 6]),
            synapse_hits=synapse_hits,
            pre_pts=pre_pts,
            post_pts=post_pts,
        )

        estimated_edges = build_estimated_line_graph(graph, n_synapses=2)
        self.assertEqual(estimated_edges, set())
        self.assertEqual(len(graph.edges), 2)

    def test_learned_merge_scorer_can_veto_heuristic_merge(self):
        path_arr = np.zeros((2, 6, 3), dtype=np.float32)
        shared_path = np.array(
            [[1, 1, 1], [2, 1, 1], [3, 1, 1], [4, 1, 1], [5, 1, 1], [6, 1, 1]],
            dtype=np.float32,
        )
        path_arr[0] = shared_path
        path_arr[1] = shared_path
        pre_pts = np.array([[1, 1, 1], [6, 1, 1]], dtype=np.float32)
        post_pts = np.array([[1, 2, 1], [6, 2, 1]], dtype=np.float32)
        synapse_hits = np.zeros((2, 4), dtype=bool)
        synapse_hits[:, 0] = True
        synapse_hits[:, 1] = True

        graph = _build_graph(
            path_arr=path_arr,
            path_lengths=np.array([6, 6]),
            synapse_hits=synapse_hits,
            pre_pts=pre_pts,
            post_pts=post_pts,
            learned_merge_score_fn=lambda left, right: -1.0,
            learned_merge_score_threshold=0.0,
        )
        self.assertEqual(len(graph.neurons), 2)

    def test_learned_merge_scorer_can_override_overlap_threshold(self):
        path_arr = np.zeros((2, 6, 3), dtype=np.float32)
        shared_path = np.array(
            [[1, 1, 1], [2, 1, 1], [3, 1, 1], [4, 1, 1], [5, 1, 1], [6, 1, 1]],
            dtype=np.float32,
        )
        path_arr[0] = shared_path
        path_arr[1] = shared_path
        pre_pts = np.array([[1, 1, 1], [3, 1, 1], [6, 1, 1]], dtype=np.float32)
        post_pts = np.array([[1, 2, 1], [3, 2, 1], [6, 2, 1]], dtype=np.float32)
        synapse_hits = np.zeros((2, 6), dtype=bool)
        synapse_hits[0, 0] = True
        synapse_hits[0, 1] = True
        synapse_hits[1, 1] = True
        synapse_hits[1, 2] = True

        graph = _build_graph(
            path_arr=path_arr,
            path_lengths=np.array([6, 6]),
            synapse_hits=synapse_hits,
            pre_pts=pre_pts,
            post_pts=post_pts,
            learned_merge_score_fn=lambda left, right: 1.0,
            learned_merge_score_threshold=0.0,
        )
        self.assertEqual(len(graph.neurons), 1)

    def test_beam_search_atomicity_can_prevent_overmerge(self):
        path_arr = np.zeros((3, 6, 3), dtype=np.float32)
        shared_path = np.array(
            [[1, 1, 1], [2, 1, 1], [3, 1, 1], [4, 1, 1], [5, 1, 1], [6, 1, 1]],
            dtype=np.float32,
        )
        path_arr[0] = shared_path
        path_arr[1] = shared_path
        path_arr[2] = shared_path
        pre_pts = np.array([[1, 1, 1], [3, 1, 1], [6, 1, 1]], dtype=np.float32)
        post_pts = np.array([[1, 2, 1], [3, 2, 1], [6, 2, 1]], dtype=np.float32)
        synapse_hits = np.zeros((3, 6), dtype=bool)
        synapse_hits[0, 0] = True
        synapse_hits[0, 1] = True
        synapse_hits[1, 1] = True
        synapse_hits[1, 2] = True
        synapse_hits[2, 2] = True
        synapse_hits[2, 1] = True

        graph = _build_graph(
            path_arr=path_arr,
            path_lengths=np.array([6, 6, 6]),
            synapse_hits=synapse_hits,
            pre_pts=pre_pts,
            post_pts=post_pts,
            learned_merge_score_fn=lambda left, right: 1.0,
            learned_merge_score_threshold=0.0,
            atomicity_score_fn=lambda branches: -10.0 if len(branches) >= 3 else 0.0,
            beam_width=3,
            beam_max_candidates=3,
            atomicity_score_weight=1.0,
        )
        self.assertGreaterEqual(len(graph.neurons), 2)

    def test_build_graph_records_cell_quality_diagnostics(self):
        path_arr = np.zeros((2, 6, 3), dtype=np.float32)
        shared_path = np.array(
            [[1, 1, 1], [2, 1, 1], [3, 1, 1], [4, 1, 1], [5, 1, 1], [6, 1, 1]],
            dtype=np.float32,
        )
        path_arr[0] = shared_path
        path_arr[1] = shared_path
        pre_pts = np.array([[1, 1, 1], [6, 1, 1]], dtype=np.float32)
        post_pts = np.array([[1, 2, 1], [6, 2, 1]], dtype=np.float32)
        synapse_hits = np.zeros((2, 4), dtype=bool)
        synapse_hits[0, 0] = True
        synapse_hits[0, 1] = True
        synapse_hits[1, 2] = True
        synapse_hits[1, 3] = True

        graph = _build_graph(
            path_arr=path_arr,
            path_lengths=np.array([6, 6]),
            synapse_hits=synapse_hits,
            pre_pts=pre_pts,
            post_pts=post_pts,
            pre_root_ids=np.array([10, 10], dtype=np.int64),
            post_root_ids=np.array([20, 21], dtype=np.int64),
        )

        diagnostics = graph.metadata["cell_diagnostics"]
        self.assertEqual(len(diagnostics), 2)
        pre_diag = next(diag for diag in diagnostics.values() if diag["role"] == "pre")
        self.assertEqual(pre_diag["majority_root_id"], 10)
        self.assertEqual(pre_diag["purity"], 1.0)
        self.assertEqual(pre_diag["completeness"], 1.0)

    def test_low_atomicity_repartition_updates_graph_and_metadata(self):
        path_arr = np.zeros((6, 6, 3), dtype=np.float32)
        base = np.array(
            [[1, 1, 0], [2, 1, 0], [3, 1, 0], [4, 1, 0], [5, 1, 0], [6, 1, 0]],
            dtype=np.float32,
        )
        path_arr[0] = base
        path_arr[1] = base + np.array([0, 0, 0.1], dtype=np.float32)
        path_arr[2] = base + np.array([0, 0, 1.0], dtype=np.float32)
        path_arr[3] = base + np.array([0, 6, 0], dtype=np.float32)
        path_arr[4] = base + np.array([0, 12, 0], dtype=np.float32)
        path_arr[5] = base + np.array([0, 18, 0], dtype=np.float32)

        pre_pts = np.array([[1, 1, 0], [3, 1, 0], [6, 1, 0]], dtype=np.float32)
        post_pts = np.array([[1, 7, 0], [3, 13, 0], [6, 19, 0]], dtype=np.float32)
        synapse_hits = np.zeros((6, 6), dtype=bool)
        synapse_hits[0, 0] = True
        synapse_hits[0, 1] = True
        synapse_hits[1, 0] = True
        synapse_hits[1, 1] = True
        synapse_hits[1, 2] = True
        synapse_hits[2, 1] = True
        synapse_hits[2, 2] = True
        synapse_hits[3, 3] = True
        synapse_hits[4, 4] = True
        synapse_hits[5, 5] = True

        def merge_score(left, right):
            z_delta = abs(float(left[0, 2] - right[0, 2]))
            return 2.0 if z_delta < 0.5 else 0.1

        graph = _build_graph(
            path_arr=path_arr,
            path_lengths=np.array([6] * 6),
            synapse_hits=synapse_hits,
            pre_pts=pre_pts,
            post_pts=post_pts,
            pre_root_ids=np.array([10, 10, 20], dtype=np.int64),
            post_root_ids=np.array([30, 31, 32], dtype=np.int64),
            learned_merge_score_fn=merge_score,
            learned_merge_score_threshold=0.0,
            atomicity_score_fn=lambda branches: -2.0 if len(branches) >= 3 else 1.0,
            beam_width=1,
            beam_max_candidates=3,
            cell_split_atomicity_threshold=0.0,
            cell_split_min_group_size=3,
            cell_split_max_rounds=2,
        )

        pre_neurons = [neuron for neuron in graph.neurons.values() if neuron.role == "pre"]
        repartition = graph.metadata["repartition"]
        self.assertGreaterEqual(len(pre_neurons), 2)
        self.assertGreaterEqual(repartition["pre"]["cells_split"], 1)
        self.assertGreater(repartition["final_f1"], repartition["baseline_f1"])


@unittest.skipIf(torch is None, "torch not installed")
class LearnedMergeCheckpointTest(unittest.TestCase):
    def test_load_shared_merge_score_fn_from_checkpoint(self):
        model = SharedGrammarModel()
        D = model._init_kwargs["input_dim"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str((Path(tmpdir) / "shared_grammar.pt"))
            save_shared_grammar_model(path, model)
            score_fn = _load_shared_merge_score_fn(path)
            rng = np.random.default_rng(0)
            left = rng.random((2, D), dtype=np.float32)
            right = rng.random((2, D), dtype=np.float32)
            score = score_fn(left, right)
            self.assertTrue(np.isfinite(score))
