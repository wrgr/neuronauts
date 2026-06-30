"""Tests for the deterministic synapse-fetch cache in neuronauts.data.lineage.

The materialization query applies a server-side ``limit`` with no stable sort
order, so an over-limit bbox returns a different arbitrary subset each call.
The disk cache + canonical ordering make repeated fetches reproducible, which
is required for valid multi-run experiments (adding a training region, sweeping
a hyperparameter).
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


def _fake_result(n: int, *, with_ids: bool = True) -> dict:
    rng = np.random.default_rng(0)
    return {
        "positions_nm": rng.random((n, 3)).astype(np.float32),
        "supervoxel_ids": np.arange(n, dtype=np.uint64),
        "root_ids": np.arange(n, dtype=np.uint64),
        "other_root_ids": np.arange(n, dtype=np.uint64),
        "other_positions_nm": rng.random((n, 3)).astype(np.float32),
        "other_supervoxel_ids": np.arange(n, dtype=np.uint64),
        "synapse_ids": (np.arange(n, dtype=np.int64) if with_ids
                        else np.full(n, -1, dtype=np.int64)),
    }


class CacheKeyTest(unittest.TestCase):
    BBOX = ((750_000, 930_000, 780_000), (950_000, 1_000_000, 880_000))

    def test_key_is_deterministic(self):
        k1 = L._synapse_cache_key(self.BBOX, version=1718, side="pre", limit=10_000)
        k2 = L._synapse_cache_key(self.BBOX, version=1718, side="pre", limit=10_000)
        self.assertEqual(k1, k2)

    def test_key_distinguishes_side_limit_version(self):
        base = L._synapse_cache_key(self.BBOX, version=1718, side="pre", limit=10_000)
        self.assertNotEqual(
            base, L._synapse_cache_key(self.BBOX, version=1718, side="post", limit=10_000))
        self.assertNotEqual(
            base, L._synapse_cache_key(self.BBOX, version=1718, side="pre", limit=5_000))
        self.assertNotEqual(
            base, L._synapse_cache_key(self.BBOX, version=343, side="pre", limit=10_000))

    def test_key_distinguishes_bbox(self):
        base = L._synapse_cache_key(self.BBOX, version=1718, side="pre", limit=10_000)
        other_bbox = ((0, 0, 0), (1, 1, 1))
        self.assertNotEqual(
            base, L._synapse_cache_key(other_bbox, version=1718, side="pre", limit=10_000))


class CacheRoundTripTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self._prev = os.environ.get("NEURONAUTS_SYNAPSE_CACHE_DIR")
        os.environ["NEURONAUTS_SYNAPSE_CACHE_DIR"] = self._dir

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("NEURONAUTS_SYNAPSE_CACHE_DIR", None)
        else:
            os.environ["NEURONAUTS_SYNAPSE_CACHE_DIR"] = self._prev

    def test_save_then_load_round_trips_all_keys(self):
        result = _fake_result(7)
        L._synapse_cache_save("abc123", result)
        loaded = L._synapse_cache_load("abc123")
        self.assertIsNotNone(loaded)
        for k, v in result.items():
            np.testing.assert_array_equal(loaded[k], v)

    def test_load_miss_returns_none(self):
        self.assertIsNone(L._synapse_cache_load("does_not_exist"))

    def test_disabled_cache_returns_none(self):
        os.environ["NEURONAUTS_SYNAPSE_CACHE_DIR"] = ""
        self.assertIsNone(L._synapse_cache_dir())
        L._synapse_cache_save("abc123", _fake_result(3))  # no-op, must not raise
        self.assertIsNone(L._synapse_cache_load("abc123"))


class FetchUsesCacheTest(unittest.TestCase):
    """fetch_region_synapses should serve from cache on the second call."""

    BBOX = ((0, 0, 0), (1_000, 1_000, 1_000))

    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self._prev = os.environ.get("NEURONAUTS_SYNAPSE_CACHE_DIR")
        os.environ["NEURONAUTS_SYNAPSE_CACHE_DIR"] = self._dir

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("NEURONAUTS_SYNAPSE_CACHE_DIR", None)
        else:
            os.environ["NEURONAUTS_SYNAPSE_CACHE_DIR"] = self._prev

    def test_second_call_does_not_hit_network(self):
        key = L._synapse_cache_key(self.BBOX, version=1718, side="pre", limit=100)
        # Pre-populate the cache so the function must return it without any HTTP.
        L._synapse_cache_save(key, _fake_result(4))

        with patch.object(L, "requests") as mock_req:
            out = L.fetch_region_synapses(self.BBOX, version=1718, side="pre", limit=100)
            mock_req.post.assert_not_called()
        self.assertIsNotNone(out)
        self.assertEqual(len(out["positions_nm"]), 4)


if __name__ == "__main__":
    unittest.main()
