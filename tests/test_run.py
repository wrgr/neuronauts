import unittest

import numpy as np

from neuronauts.line_graph import build_estimated_line_graph
from neuronauts.run import _build_graph


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
