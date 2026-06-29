from __future__ import annotations

import unittest
from unittest import mock

import numpy as np


class SkeletonGraphConfigTest(unittest.TestCase):
    def test_allows_base_materialization_skeletons(self):
        from neuronauts.skeleton_graph import validate_skeleton_graph_config

        cfg = validate_skeleton_graph_config(
            base_version=117,
            target_version=1412,
            skeleton_version=117,
            graph_source="skeleton",
        )
        self.assertEqual(cfg.skeleton_version, 117)

    def test_rejects_target_materialization_skeletons(self):
        from neuronauts.skeleton_graph import validate_skeleton_graph_config

        with self.assertRaises(ValueError):
            validate_skeleton_graph_config(
                base_version=117,
                target_version=1412,
                skeleton_version=1412,
                graph_source="skeleton",
            )


class FetchRootSkeletonTest(unittest.TestCase):
    def test_fetch_root_skeleton_parses_flat_dict_payload(self):
        from neuronauts.fetch import fetch_root_skeleton

        fake_payload = {
            "vertices": np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32),
            "edges": np.array([[0, 1]], dtype=np.int64),
            "radius": np.array([10.0, 12.0], dtype=np.float32),
        }

        class FakeSkeletonService:
            def get_skeleton(self, root_id, datastack_name=None, skeleton_version=4, output_format="dict"):
                return fake_payload

        class FakeClient:
            def __init__(self, datastack, server_address=None, auth_token=None):
                self.version = None
                self.skeleton = FakeSkeletonService()

        with mock.patch.dict("sys.modules", {"caveclient": mock.Mock(CAVEclient=FakeClient)}):
            sk = fetch_root_skeleton(123, version=117, datastack="minnie65_public")

        self.assertEqual(sk.root_id, 123)
        self.assertEqual(sk.materialization_version, 117)
        self.assertEqual(sk.vertices.shape, (2, 3))
        self.assertEqual(sk.edges.shape, (1, 2))
        self.assertEqual(sk.radius.shape, (2,))

    def test_fetch_root_skeletons_falls_back_to_empty_on_service_failure(self):
        from neuronauts.fetch import fetch_root_skeletons

        class FakeSkeletonService:
            def get_skeleton(self, *args, **kwargs):
                raise RuntimeError("503 Service Temporarily Unavailable")

        class FakeClient:
            def __init__(self, datastack, server_address=None, auth_token=None):
                self.version = None
                self.skeleton = FakeSkeletonService()
                # fetch_root_skeletons stubs client.info.segmentation_cloudvolume
                # to skip cloudvolume root validation; the fake needs an assignable
                # `info` for that to work (a real CAVEclient has one).
                self.info = mock.Mock()

        with mock.patch.dict("sys.modules", {"caveclient": mock.Mock(CAVEclient=FakeClient)}):
            out = fetch_root_skeletons([101, 202], version=117)

        self.assertEqual(sorted(out.keys()), [101, 202])
        self.assertEqual(out[101].vertices.shape, (0, 3))
        self.assertEqual(out[202].edges.shape, (0, 2))


class BuildSkeletonConnectivityGraphTest(unittest.TestCase):
    def test_builds_candidate_graph_with_decoys(self):
        from neuronauts.fetch import RealBoxSpec, SkeletonData, SynapseTable
        from neuronauts.skeleton_graph import build_skeleton_connectivity_graph

        box = RealBoxSpec(center_nm=(10_000, 10_000, 10_000), side_um=6.0, mip=2)
        synapses = SynapseTable(
            pre_pt=np.array([[10, 10, 5], [11, 10, 5], [40, 40, 5]], dtype=np.float32),
            post_pt=np.array([[12, 12, 5], [13, 12, 5], [42, 42, 5]], dtype=np.float32),
            pre_root_id=np.array([1, 1, 2], dtype=np.int64),
            post_root_id=np.array([11, 11, 12], dtype=np.int64),
            synapse_id=np.array([0, 1, 2], dtype=np.int64),
        )

        fake_pre = {
            1: SkeletonData(1, 117, np.array([[0, 0, 0], [1000, 0, 0]], dtype=np.float32), np.array([[0, 1]], dtype=np.int64)),
            2: SkeletonData(2, 117, np.array([[2000, 2000, 0], [3000, 2000, 0]], dtype=np.float32), np.array([[0, 1]], dtype=np.int64)),
        }
        fake_post = {
            11: SkeletonData(11, 117, np.array([[0, 0, 0], [1200, 0, 0]], dtype=np.float32), np.array([[0, 1]], dtype=np.int64)),
            12: SkeletonData(12, 117, np.array([[2200, 2200, 0], [3200, 2200, 0]], dtype=np.float32), np.array([[0, 1]], dtype=np.int64)),
        }

        with mock.patch("neuronauts.skeleton_graph.fetch_root_skeletons", side_effect=[fake_pre, fake_post]):
            graph = build_skeleton_connectivity_graph(box, synapses, version=117)

        self.assertGreaterEqual(len(graph.neurons), 4)
        self.assertGreater(len(graph.edges), len(synapses.synapse_id))
        self.assertEqual(graph.unresolved_synapse_indices, [])
