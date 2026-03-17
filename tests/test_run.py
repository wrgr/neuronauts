import tempfile
import unittest
from pathlib import Path

import numpy as np

from neuronauts.line_graph import build_estimated_line_graph
from neuronauts.run import _build_graph, _load_shared_merge_score_fn
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


@unittest.skipIf(torch is None, "torch not installed")
class LearnedMergeCheckpointTest(unittest.TestCase):
    def test_load_shared_merge_score_fn_from_checkpoint(self):
        model = SharedGrammarModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str((Path(tmpdir) / "shared_grammar.pt"))
            save_shared_grammar_model(path, model)
            score_fn = _load_shared_merge_score_fn(path)
            left = np.array([[1.0, 1.0, 0.0], [2.0, 1.0, 0.1]], dtype=np.float32)
            right = np.array([[1.5, 1.0, 0.0], [2.5, 1.0, 0.1]], dtype=np.float32)
            score = score_fn(left, right)
            self.assertTrue(np.isfinite(score))
