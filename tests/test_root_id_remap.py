from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest

import numpy as np

# Make scripts/ importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _import_train_module():
    """Import scripts/train.py as a module (not a package)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_script",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "train.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class MaybeMapSynapseRootsTest(unittest.TestCase):
    def setUp(self):
        self.mod = _import_train_module()

    def test_maybe_map_synapse_roots_no_mapping_returns_identity(self):
        # Ensure no mapping is loaded.
        if hasattr(self.mod._maybe_map_synapse_roots, "_root_mapping"):
            delattr(self.mod._maybe_map_synapse_roots, "_root_mapping")
        self.mod._maybe_map_synapse_roots._warned_no_mapping = False  # type: ignore[attr-defined]

        from neuronauts.fetch import SynapseTable

        syn = SynapseTable(
            pre_pt=np.zeros((3, 3), dtype=np.float32),
            post_pt=np.zeros((3, 3), dtype=np.float32),
            pre_root_id=np.array([1, 2, 3], dtype=np.int64),
            post_root_id=np.array([10, 20, 30], dtype=np.int64),
            synapse_id=np.arange(3, dtype=np.int64),
        )

        out = self.mod._maybe_map_synapse_roots(
            syn, base_version=117, target_version=1412
        )
        self.assertEqual(len(out.pre_root_id), 3)
        np.testing.assert_array_equal(out.pre_root_id, syn.pre_root_id)
        np.testing.assert_array_equal(out.post_root_id, syn.post_root_id)

    def test_maybe_map_synapse_roots_applies_mapping_and_drops_zero(self):
        # Provide a precomputed mapping dict.
        mapping = {
            1: 101,
            2: 102,
            3: 0,
            4: 104,
            10: 201,
            20: 0,
            30: 203,
            40: 204,
        }
        self.mod._maybe_map_synapse_roots._root_mapping = mapping  # type: ignore[attr-defined]

        from neuronauts.fetch import SynapseTable

        syn = SynapseTable(
            pre_pt=np.zeros((4, 3), dtype=np.float32),
            post_pt=np.zeros((4, 3), dtype=np.float32),
            pre_root_id=np.array([1, 2, 3, 4], dtype=np.int64),
            post_root_id=np.array([10, 20, 30, 40], dtype=np.int64),
            synapse_id=np.arange(4, dtype=np.int64),
        )

        out = self.mod._maybe_map_synapse_roots(
            syn, base_version=117, target_version=1412
        )

        # Keep synapses where both mapped pre and mapped post are != 0.
        # Index 0: pre 1->101, post 10->201 kept
        # Index 1: post 20->0 dropped
        # Index 2: pre 3->0 dropped
        # Index 3: pre 4->104, post 40->204 kept
        self.assertEqual(len(out.pre_root_id), 2)
        np.testing.assert_array_equal(out.pre_root_id, np.array([101, 104], dtype=np.int64))
        np.testing.assert_array_equal(out.post_root_id, np.array([201, 204], dtype=np.int64))
        np.testing.assert_array_equal(out.synapse_id, np.array([0, 3], dtype=np.int64))


class RemapRootArrayTest(unittest.TestCase):
    def setUp(self):
        self.mod = _import_train_module()

    def test_remap_root_array_fast(self):
        mapping = {1: 10, 2: 0, 3: 30}
        arr = np.array([1, 2, 2, 3], dtype=np.int64)
        out = self.mod._remap_root_array(arr, mapping)
        np.testing.assert_array_equal(out, np.array([10, 0, 0, 30], dtype=np.int64))


class RemapCacheRootsTest(unittest.TestCase):
    def setUp(self):
        self.mod = _import_train_module()

    def test_remap_cache_roots_recomputes_positive_pairs(self):
        from neuronauts.dataset_builder import BoxCache
        from neuronauts.fetch import RealBoxSpec, SynapseTable, VolumeChunk
        from neuronauts.dataset_builder import count_positive_pairs

        with tempfile.TemporaryDirectory() as d_in, tempfile.TemporaryDirectory() as d_out:
            cache_in = BoxCache(d_in)
            cache_out_dir = d_out

            spec = RealBoxSpec(center_nm=(1_000_000, 2_000_000, 100_000), side_um=6.0, mip=2)

            syn = SynapseTable(
                pre_pt=np.zeros((5, 3), dtype=np.float32),
                post_pt=np.zeros((5, 3), dtype=np.float32),
                pre_root_id=np.array([1, 1, 2, 3, 3], dtype=np.int64),
                post_root_id=np.array([10, 10, 10, 11, 12], dtype=np.int64),
                synapse_id=np.arange(5, dtype=np.int64),
            )

            # Save synapse-only box for speed/simplicity.
            cache_in.save_synapse_only(
                spec,
                syn,
                n_positive_pairs=count_positive_pairs(syn),
                root_id_version=117,
            )

            # Mapping: pre 1->101, 2->0, 3->103; post 10->201, 11->0, 12->202
            mapping_tsv = os.path.join(d_in, "root_map.tsv")
            with open(mapping_tsv, "w", encoding="utf-8") as fh:
                fh.write("root_base\troot_target\n")
                for b, t in [
                    (1, 101),
                    (2, 0),
                    (3, 103),
                    (10, 201),
                    (11, 0),
                    (12, 202),
                ]:
                    fh.write(f"{b}\t{t}\n")

            args = types.SimpleNamespace(
                cache_dir=d_in,
                out_cache_dir=cache_out_dir,
                root_remap_tsv=mapping_tsv,
                base_version=117,
                target_version=1412,
            )

            rc = self.mod.cmd_remap_cache_roots(args)
            self.assertEqual(rc, 0)

            cache_out = BoxCache(cache_out_dir)
            recs = cache_out.all_records()
            self.assertEqual(len(recs), 1)
            rec = recs[0]

            _, syn_out = cache_out.load(rec)

            # After remap:
            # keep synapse indices where both mapped pre and post != 0: indices 0,1,4
            np.testing.assert_array_equal(syn_out.synapse_id, np.array([0, 1, 4], dtype=np.int64))
            np.testing.assert_array_equal(syn_out.pre_root_id, np.array([101, 101, 103], dtype=np.int64))
            np.testing.assert_array_equal(syn_out.post_root_id, np.array([201, 201, 202], dtype=np.int64))

            # n_positive_pairs must be recomputed in target label space.
            # pre pairs: 101 count2 -> 1 pair; 103 count1 -> 0
            # post pairs: 201 count2 -> 1 pair; 202 count1 -> 0
            self.assertEqual(rec.n_positive_pairs, 2)
            self.assertEqual(rec.root_id_version, 1412)

