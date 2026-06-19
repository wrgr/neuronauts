"""Tests for the L2-skeleton disk cache in neuronauts.data.lineage.

``l2_skeleton`` makes 2+ throttled network roundtrips per fragment, so building
a region with 1500+ fragments takes hours.  A skeleton is a pure function of the
(immutable) v117 root_id, so it caches cleanly to disk: pay the fetch once,
reuse across every train/eval run.  Only successful skeletons are cached so a
transient network failure does not poison the cache with a permanent None.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neuronauts.data import lineage as L


def _fake_skel(n: int) -> dict:
    rng = np.random.default_rng(0)
    return {
        "vertices_nm": rng.random((n, 3)).astype(np.float32),
        "edges": np.column_stack([np.arange(n - 1), np.arange(1, n)]).astype(np.int64),
        "radii_nm": np.full(n, 200.0, dtype=np.float32),
        "l2_ids": np.arange(n, dtype=np.uint64),
    }


class L2CacheKeyTest(unittest.TestCase):
    ROOT = 864691135000000001

    def test_key_is_deterministic(self):
        k1 = L._l2_cache_key(self.ROOT, max_l2_nodes=2000, seed=0)
        k2 = L._l2_cache_key(self.ROOT, max_l2_nodes=2000, seed=0)
        self.assertEqual(k1, k2)

    def test_key_distinguishes_root_nodes_seed(self):
        base = L._l2_cache_key(self.ROOT, max_l2_nodes=2000, seed=0)
        self.assertNotEqual(
            base, L._l2_cache_key(self.ROOT + 1, max_l2_nodes=2000, seed=0))
        self.assertNotEqual(
            base, L._l2_cache_key(self.ROOT, max_l2_nodes=500, seed=0))
        self.assertNotEqual(
            base, L._l2_cache_key(self.ROOT, max_l2_nodes=2000, seed=1))


class L2CacheRoundTripTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self._prev = os.environ.get("NEURONAUTS_L2_CACHE_DIR")
        os.environ["NEURONAUTS_L2_CACHE_DIR"] = self._dir

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("NEURONAUTS_L2_CACHE_DIR", None)
        else:
            os.environ["NEURONAUTS_L2_CACHE_DIR"] = self._prev

    def test_save_then_load_round_trips_all_keys(self):
        skel = _fake_skel(6)
        L._l2_cache_save("root_abc", skel)
        loaded = L._l2_cache_load("root_abc")
        self.assertIsNotNone(loaded)
        for k, v in skel.items():
            np.testing.assert_array_equal(loaded[k], v)

    def test_load_miss_returns_none(self):
        self.assertIsNone(L._l2_cache_load("does_not_exist"))

    def test_disabled_cache_returns_none(self):
        os.environ["NEURONAUTS_L2_CACHE_DIR"] = ""
        self.assertIsNone(L._l2_cache_dir())
        L._l2_cache_save("root_abc", _fake_skel(3))  # no-op, must not raise
        self.assertIsNone(L._l2_cache_load("root_abc"))


class L2SkeletonUsesCacheTest(unittest.TestCase):
    """l2_skeleton should serve from cache without any network call."""

    ROOT = 123456789

    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self._prev = os.environ.get("NEURONAUTS_L2_CACHE_DIR")
        os.environ["NEURONAUTS_L2_CACHE_DIR"] = self._dir

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("NEURONAUTS_L2_CACHE_DIR", None)
        else:
            os.environ["NEURONAUTS_L2_CACHE_DIR"] = self._prev

    def test_cache_hit_skips_network(self):
        key = L._l2_cache_key(self.ROOT, max_l2_nodes=2000, seed=0)
        L._l2_cache_save(key, _fake_skel(4))
        with patch.object(L, "root_leaves") as mrl, patch.object(L, "requests") as mreq:
            out = L.l2_skeleton(self.ROOT)
            mrl.assert_not_called()
            mreq.post.assert_not_called()
        self.assertIsNotNone(out)
        self.assertEqual(len(out["vertices_nm"]), 4)


if __name__ == "__main__":
    unittest.main()
