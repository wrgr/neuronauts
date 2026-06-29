"""Tests for neuronauts/dataset_builder.py.

Covers BoxRecord, BoxCache (save/load/contains/iter/duplicate-no-op),
select_random_boxes, select_boxes_from_nucleus_table, load_dataset, and
the synapse-count filtering in build_dataset (using a mock fetch).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neuronauts.dataset_builder import (
    MINNIE65_X_NM,
    MINNIE65_Y_NM,
    MINNIE65_Z_NM,
    BoxCache,
    BoxRecord,
    build_dataset,
    load_dataset,
    select_random_boxes,
)
from neuronauts.fetch import RealBoxSpec, SynapseTable, VolumeChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_volume(shape=(20, 20, 20)) -> VolumeChunk:
    return VolumeChunk(
        data=np.zeros(shape, dtype=np.uint8),
        voxel_size_nm=(32, 32, 40),
        bbox_voxels=((0, 0, 0), shape),
        mip=2,
    )


def _make_synapses(n: int = 8) -> SynapseTable:
    rng = np.random.default_rng(0)
    return SynapseTable(
        pre_pt=rng.random((n, 3), dtype=np.float32) * 10,
        post_pt=rng.random((n, 3), dtype=np.float32) * 10,
        pre_root_id=np.arange(1, n + 1, dtype=np.int64),
        post_root_id=np.arange(n + 1, 2 * n + 1, dtype=np.int64),
        synapse_id=np.arange(n, dtype=np.int64),
    )


def _make_synapses_with_seg_ids(n: int = 4) -> SynapseTable:
    syn = _make_synapses(n)
    return SynapseTable(
        pre_pt=syn.pre_pt,
        post_pt=syn.post_pt,
        pre_root_id=syn.pre_root_id,
        post_root_id=syn.post_root_id,
        synapse_id=syn.synapse_id,
        pre_seg_id=np.ones(n, dtype=np.int64),
        post_seg_id=np.ones(n, dtype=np.int64) * 2,
    )


def _make_spec(seed: int = 0) -> RealBoxSpec:
    rng = np.random.default_rng(seed)
    cx = int(rng.integers(500_000, 3_500_000))
    cy = int(rng.integers(500_000, 2_500_000))
    cz = int(rng.integers(50_000, 700_000))
    return RealBoxSpec(center_nm=(cx, cy, cz), side_um=6.0, mip=2)


# ---------------------------------------------------------------------------
# BoxRecord
# ---------------------------------------------------------------------------

class BoxRecordTest(unittest.TestCase):

    def test_to_spec_round_trip(self):
        spec = _make_spec(seed=1)
        rec = BoxRecord(
            box_hash=spec.cache_key,
            center_nm=spec.center_nm,
            side_um=spec.side_um,
            mip=spec.mip,
            n_synapses=5,
        )
        restored = rec.to_spec()
        self.assertEqual(restored.center_nm, spec.center_nm)
        self.assertAlmostEqual(restored.side_um, spec.side_um)
        self.assertEqual(restored.mip, spec.mip)

    def test_to_spec_cache_key_matches(self):
        spec = _make_spec(seed=2)
        rec = BoxRecord(
            box_hash=spec.cache_key,
            center_nm=spec.center_nm,
            side_um=spec.side_um,
            mip=spec.mip,
            n_synapses=3,
        )
        self.assertEqual(rec.to_spec().cache_key, spec.cache_key)


# ---------------------------------------------------------------------------
# BoxCache
# ---------------------------------------------------------------------------

class BoxCacheTest(unittest.TestCase):

    def _populated_cache(self, tmpdir, n_specs=3):
        cache = BoxCache(tmpdir)
        specs = select_random_boxes(n=n_specs, seed=7)
        for i, spec in enumerate(specs):
            vol = _make_volume()
            syn = _make_synapses(n=i + 4)
            cache.save(spec, vol, syn)
        return cache, specs

    def test_initially_empty(self):
        with tempfile.TemporaryDirectory() as d:
            cache = BoxCache(d)
            self.assertEqual(len(cache), 0)
            self.assertEqual(cache.all_records(), [])

    def test_save_increments_length(self):
        with tempfile.TemporaryDirectory() as d:
            cache = BoxCache(d)
            spec = _make_spec(seed=0)
            cache.save(spec, _make_volume(), _make_synapses(5))
            self.assertEqual(len(cache), 1)

    def test_save_and_load_volume_shape(self):
        with tempfile.TemporaryDirectory() as d:
            cache = BoxCache(d)
            spec = _make_spec(seed=0)
            vol = _make_volume((32, 32, 16))
            syn = _make_synapses(6)
            rec = cache.save(spec, vol, syn)
            vol2, syn2 = cache.load(rec)
            self.assertEqual(vol2.data.shape, (32, 32, 16))

    def test_save_and_load_synapses(self):
        with tempfile.TemporaryDirectory() as d:
            cache = BoxCache(d)
            spec = _make_spec(seed=0)
            syn = _make_synapses(7)
            rec = cache.save(spec, _make_volume(), syn)
            _, syn2 = cache.load(rec)
            self.assertEqual(len(syn2.pre_pt), 7)
            np.testing.assert_array_equal(syn2.pre_root_id, syn.pre_root_id)

    def test_save_and_load_seg_ids(self):
        with tempfile.TemporaryDirectory() as d:
            cache = BoxCache(d)
            spec = _make_spec(seed=0)
            syn = _make_synapses_with_seg_ids(n=5)
            rec = cache.save(spec, _make_volume(), syn)
            _, syn2 = cache.load(rec)
            self.assertIsNotNone(syn2.pre_seg_id)
            np.testing.assert_array_equal(syn2.pre_seg_id, syn.pre_seg_id)

    def test_save_without_seg_ids_loads_none(self):
        with tempfile.TemporaryDirectory() as d:
            cache = BoxCache(d)
            spec = _make_spec(seed=0)
            syn = _make_synapses(4)  # no seg_ids
            rec = cache.save(spec, _make_volume(), syn)
            _, syn2 = cache.load(rec)
            self.assertIsNone(syn2.pre_seg_id)
            self.assertIsNone(syn2.post_seg_id)

    def test_duplicate_save_is_noop(self):
        with tempfile.TemporaryDirectory() as d:
            cache = BoxCache(d)
            spec = _make_spec(seed=0)
            rec1 = cache.save(spec, _make_volume(), _make_synapses(5))
            rec2 = cache.save(spec, _make_volume(), _make_synapses(99))  # different data
            self.assertEqual(rec1.box_hash, rec2.box_hash)
            self.assertEqual(len(cache), 1)
            # n_synapses should be from the FIRST save
            self.assertEqual(cache.all_records()[0].n_synapses, 5)

    def test_contains_true_after_save(self):
        with tempfile.TemporaryDirectory() as d:
            cache = BoxCache(d)
            spec = _make_spec(seed=0)
            cache.save(spec, _make_volume(), _make_synapses(4))
            self.assertTrue(cache.contains(spec))

    def test_contains_false_for_unsaved(self):
        with tempfile.TemporaryDirectory() as d:
            cache = BoxCache(d)
            self.assertFalse(cache.contains(_make_spec(seed=99)))

    def test_all_records_returns_all(self):
        with tempfile.TemporaryDirectory() as d:
            cache, specs = self._populated_cache(d, n_specs=3)
            self.assertEqual(len(cache.all_records()), 3)

    def test_iter_records_yields_all(self):
        with tempfile.TemporaryDirectory() as d:
            cache, _ = self._populated_cache(d, n_specs=4)
            recs = list(cache.iter_records())
            self.assertEqual(len(recs), 4)

    def test_iter_records_shuffled(self):
        with tempfile.TemporaryDirectory() as d:
            cache, _ = self._populated_cache(d, n_specs=10)
            rng = np.random.default_rng(1)
            recs_default = list(cache.iter_records())
            recs_shuffled = list(cache.iter_records(shuffle=True, rng=rng))
            # Same elements, different order (with overwhelming probability for 10 items)
            self.assertEqual(
                sorted(r.box_hash for r in recs_default),
                sorted(r.box_hash for r in recs_shuffled),
            )

    def test_index_persists_across_instantiations(self):
        with tempfile.TemporaryDirectory() as d:
            cache = BoxCache(d)
            spec = _make_spec(seed=0)
            cache.save(spec, _make_volume(), _make_synapses(5))
            # Create a new BoxCache instance pointing to the same directory.
            cache2 = BoxCache(d)
            self.assertEqual(len(cache2), 1)
            self.assertTrue(cache2.contains(spec))

    def test_record_n_synapses_matches_actual(self):
        with tempfile.TemporaryDirectory() as d:
            cache = BoxCache(d)
            spec = _make_spec(seed=0)
            rec = cache.save(spec, _make_volume(), _make_synapses(11))
            self.assertEqual(rec.n_synapses, 11)

    def test_volume_dtype_stored_as_uint8(self):
        with tempfile.TemporaryDirectory() as d:
            cache = BoxCache(d)
            spec = _make_spec(seed=0)
            vol = VolumeChunk(
                data=np.ones((10, 10, 5), dtype=np.float32) * 200,
                voxel_size_nm=(32, 32, 40),
                bbox_voxels=((0, 0, 0), (10, 10, 5)),
                mip=2,
            )
            rec = cache.save(spec, vol, _make_synapses(3))
            vol2, _ = cache.load(rec)
            self.assertEqual(vol2.data.dtype, np.uint8)


# ---------------------------------------------------------------------------
# select_random_boxes
# ---------------------------------------------------------------------------

class SelectRandomBoxesTest(unittest.TestCase):

    def test_returns_correct_count(self):
        specs = select_random_boxes(n=10, seed=0)
        self.assertEqual(len(specs), 10)

    def test_returns_real_box_specs(self):
        specs = select_random_boxes(n=3, seed=0)
        for s in specs:
            self.assertIsInstance(s, RealBoxSpec)

    def test_centers_within_default_bounds(self):
        half_nm = int(6.0 * 1000 / 2)
        specs = select_random_boxes(n=50, seed=42)
        for s in specs:
            cx, cy, cz = s.center_nm
            self.assertGreaterEqual(cx, MINNIE65_X_NM[0] + half_nm)
            self.assertLessEqual(cx, MINNIE65_X_NM[1] - half_nm)
            self.assertGreaterEqual(cy, MINNIE65_Y_NM[0] + half_nm)
            self.assertLessEqual(cy, MINNIE65_Y_NM[1] - half_nm)
            self.assertGreaterEqual(cz, MINNIE65_Z_NM[0] + half_nm)
            self.assertLessEqual(cz, MINNIE65_Z_NM[1] - half_nm)

    def test_seed_reproducibility(self):
        s1 = select_random_boxes(n=5, seed=7)
        s2 = select_random_boxes(n=5, seed=7)
        for a, b in zip(s1, s2):
            self.assertEqual(a.center_nm, b.center_nm)

    def test_different_seeds_different_boxes(self):
        s1 = select_random_boxes(n=5, seed=1)
        s2 = select_random_boxes(n=5, seed=2)
        # Very unlikely to be identical
        self.assertNotEqual(
            [x.center_nm for x in s1], [x.center_nm for x in s2]
        )

    def test_side_um_propagated(self):
        specs = select_random_boxes(n=3, box_side_um=8.0, seed=0)
        for s in specs:
            self.assertAlmostEqual(s.side_um, 8.0)

    def test_mip_propagated(self):
        specs = select_random_boxes(n=3, mip=3, seed=0)
        for s in specs:
            self.assertEqual(s.mip, 3)

    def test_zero_boxes_returns_empty(self):
        self.assertEqual(select_random_boxes(n=0), [])

    def test_custom_bounds(self):
        specs = select_random_boxes(
            n=20, seed=99,
            x_range_nm=(1_000_000, 2_000_000),
            y_range_nm=(1_000_000, 2_000_000),
            z_range_nm=(100_000, 200_000),
        )
        half = int(3000)
        for s in specs:
            self.assertGreaterEqual(s.center_nm[0], 1_000_000 + half)
            self.assertLessEqual(s.center_nm[0], 2_000_000 - half)


# ---------------------------------------------------------------------------
# select_boxes_from_nucleus_table
# ---------------------------------------------------------------------------

class SelectBoxesFromNucleusTableTest(unittest.TestCase):

    def _make_csv_files(self, tmpdir, n_roots=20):
        """Create minimal mock counts TSV and nucleus CSV files."""
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        rng = np.random.default_rng(42)
        root_ids = np.arange(1, n_roots + 1)
        syn_counts = rng.integers(20, 200, size=n_roots)

        counts_path = os.path.join(tmpdir, "counts.tsv")
        counts_df = pd.DataFrame({
            "root_id": root_ids,
            "pre_synapse_count": syn_counts // 2,
            "post_synapse_count": syn_counts // 2,
            "total_synapse_count": syn_counts,
            "has_soma": [True] * n_roots,
        })
        counts_df.to_csv(counts_path, sep="\t", index=False)

        # Nucleus CSV with position columns in supervoxel voxels (8nm/vox).
        nucleus_path = os.path.join(tmpdir, "nucleus.csv")
        nuc_df = pd.DataFrame({
            "pt_root_id": root_ids,
            "pt_position_x": rng.integers(40_000, 450_000, size=n_roots),
            "pt_position_y": rng.integers(40_000, 340_000, size=n_roots),
            "pt_position_z": rng.integers(1_500, 20_000, size=n_roots),
        })
        nuc_df.to_csv(nucleus_path, index=False)
        return counts_path, nucleus_path

    def test_returns_expected_count(self):
        from neuronauts.dataset_builder import select_boxes_from_nucleus_table
        with tempfile.TemporaryDirectory() as d:
            counts_path, nuc_path = self._make_csv_files(d, n_roots=20)
            specs = select_boxes_from_nucleus_table(
                counts_tsv=counts_path, nucleus_csv=nuc_path, n=5,
                min_syn=20, max_syn=200,
            )
            self.assertLessEqual(len(specs), 5)
            self.assertGreater(len(specs), 0)

    def test_returns_real_box_specs(self):
        from neuronauts.dataset_builder import select_boxes_from_nucleus_table
        with tempfile.TemporaryDirectory() as d:
            counts_path, nuc_path = self._make_csv_files(d, n_roots=10)
            specs = select_boxes_from_nucleus_table(
                counts_tsv=counts_path, nucleus_csv=nuc_path, n=3,
                min_syn=20, max_syn=200,
            )
            for s in specs:
                self.assertIsInstance(s, RealBoxSpec)

    def test_raises_on_no_matching_roots(self):
        from neuronauts.dataset_builder import select_boxes_from_nucleus_table
        with tempfile.TemporaryDirectory() as d:
            counts_path, nuc_path = self._make_csv_files(d, n_roots=10)
            with self.assertRaises(ValueError):
                select_boxes_from_nucleus_table(
                    counts_tsv=counts_path, nucleus_csv=nuc_path, n=5,
                    min_syn=10_000, max_syn=20_000,  # impossible range
                )

    def test_raises_on_missing_position_columns(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")
        from neuronauts.dataset_builder import select_boxes_from_nucleus_table
        with tempfile.TemporaryDirectory() as d:
            counts_path, _ = self._make_csv_files(d, n_roots=5)
            # Write nucleus CSV without position columns.
            bad_nuc = os.path.join(d, "bad_nuc.csv")
            pd.DataFrame({"pt_root_id": [1, 2, 3]}).to_csv(bad_nuc, index=False)
            with self.assertRaises(ValueError):
                select_boxes_from_nucleus_table(
                    counts_tsv=counts_path, nucleus_csv=bad_nuc, n=3,
                    min_syn=0, max_syn=10_000,
                )

    def test_seed_reproducibility(self):
        from neuronauts.dataset_builder import select_boxes_from_nucleus_table
        with tempfile.TemporaryDirectory() as d:
            counts_path, nuc_path = self._make_csv_files(d, n_roots=20)
            s1 = select_boxes_from_nucleus_table(
                counts_tsv=counts_path, nucleus_csv=nuc_path, n=5,
                min_syn=20, max_syn=200, seed=1,
            )
            s2 = select_boxes_from_nucleus_table(
                counts_tsv=counts_path, nucleus_csv=nuc_path, n=5,
                min_syn=20, max_syn=200, seed=1,
            )
            self.assertEqual([s.center_nm for s in s1], [s.center_nm for s in s2])


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

class LoadDatasetTest(unittest.TestCase):

    def _make_cache(self, tmpdir, n_boxes=5):
        specs = select_random_boxes(n=n_boxes, seed=10)
        cache = BoxCache(tmpdir)
        for i, spec in enumerate(specs):
            cache.save(spec, _make_volume(), _make_synapses(n=i + 5))
        return cache

    def test_returns_cache_and_records(self):
        with tempfile.TemporaryDirectory() as d:
            self._make_cache(d, n_boxes=4)
            cache, records = load_dataset(d)
            self.assertIsInstance(cache, BoxCache)
            self.assertEqual(len(records), 4)

    def test_min_synapses_filter(self):
        with tempfile.TemporaryDirectory() as d:
            self._make_cache(d, n_boxes=5)
            # n_synapses are 5, 6, 7, 8, 9 from the loop above
            _, records = load_dataset(d, min_synapses=7)
            self.assertTrue(all(r.n_synapses >= 7 for r in records))

    def test_max_synapses_filter(self):
        with tempfile.TemporaryDirectory() as d:
            self._make_cache(d, n_boxes=5)
            _, records = load_dataset(d, max_synapses=7)
            self.assertTrue(all(r.n_synapses <= 7 for r in records))

    def test_empty_cache_returns_empty_records(self):
        with tempfile.TemporaryDirectory() as d:
            _, records = load_dataset(d)
            self.assertEqual(records, [])


# ---------------------------------------------------------------------------
# build_dataset (mock fetch)
# ---------------------------------------------------------------------------

class BuildDatasetTest(unittest.TestCase):
    """Tests build_dataset with monkeypatched fetch functions."""

    def _mock_fetch(self, n_synapses: int):
        """Return mock fetch_volume and fetch_synapses that return fixed data."""
        vol = _make_volume()
        syn = _make_synapses(n_synapses)

        def mock_fetch_volume(bbox_nm, mip=2):
            return vol

        def mock_fetch_synapses(bbox_nm, mip=2, token=None, **kwargs):
            return syn

        return mock_fetch_volume, mock_fetch_synapses

    def test_saves_boxes_within_synapse_range(self):
        import neuronauts.dataset_builder as db_module
        orig_fv = db_module.fetch_volume
        orig_fs = db_module.fetch_synapses
        try:
            mock_fv, mock_fs = self._mock_fetch(n_synapses=20)
            db_module.fetch_volume = mock_fv
            db_module.fetch_synapses = mock_fs

            with tempfile.TemporaryDirectory() as d:
                cache = BoxCache(d)
                specs = select_random_boxes(n=3, seed=0)
                records = build_dataset(
                    specs, cache, min_synapses=10, max_synapses=30,
                    min_root_synapses=0, verbose=False,
                )
                self.assertEqual(len(records), 3)
        finally:
            db_module.fetch_volume = orig_fv
            db_module.fetch_synapses = orig_fs

    def test_skips_boxes_below_min_synapses(self):
        import neuronauts.dataset_builder as db_module
        orig_fv = db_module.fetch_volume
        orig_fs = db_module.fetch_synapses
        try:
            mock_fv, mock_fs = self._mock_fetch(n_synapses=3)  # too few
            db_module.fetch_volume = mock_fv
            db_module.fetch_synapses = mock_fs

            with tempfile.TemporaryDirectory() as d:
                cache = BoxCache(d)
                specs = select_random_boxes(n=3, seed=0)
                records = build_dataset(
                    specs, cache, min_synapses=10, max_synapses=50,
                    min_root_synapses=0, verbose=False,
                )
                self.assertEqual(len(records), 0)
        finally:
            db_module.fetch_volume = orig_fv
            db_module.fetch_synapses = orig_fs

    def test_skips_boxes_above_max_synapses(self):
        import neuronauts.dataset_builder as db_module
        orig_fv = db_module.fetch_volume
        orig_fs = db_module.fetch_synapses
        try:
            mock_fv, mock_fs = self._mock_fetch(n_synapses=500)  # too many
            db_module.fetch_volume = mock_fv
            db_module.fetch_synapses = mock_fs

            with tempfile.TemporaryDirectory() as d:
                cache = BoxCache(d)
                specs = select_random_boxes(n=3, seed=0)
                records = build_dataset(
                    specs, cache, min_synapses=10, max_synapses=100,
                    min_root_synapses=0, verbose=False,
                )
                self.assertEqual(len(records), 0)
        finally:
            db_module.fetch_volume = orig_fv
            db_module.fetch_synapses = orig_fs

    def test_already_cached_boxes_are_skipped(self):
        """build_dataset should not re-fetch already-cached boxes."""
        import neuronauts.dataset_builder as db_module
        orig_fv = db_module.fetch_volume
        orig_fs = db_module.fetch_synapses
        call_count = [0]

        def counting_fetch_synapses(bbox_nm, mip=2, token=None, **kwargs):
            call_count[0] += 1
            return _make_synapses(20)

        def mock_fv(bbox_nm, mip=2):
            return _make_volume()

        try:
            db_module.fetch_volume = mock_fv
            db_module.fetch_synapses = counting_fetch_synapses

            with tempfile.TemporaryDirectory() as d:
                cache = BoxCache(d)
                specs = select_random_boxes(n=2, seed=0)

                # First call: fetches both.
                build_dataset(
                    specs, cache, min_synapses=10, max_synapses=30,
                    min_root_synapses=0, verbose=False,
                )
                count_after_first = call_count[0]

                # Second call with same specs: should not re-fetch.
                build_dataset(
                    specs, cache, min_synapses=10, max_synapses=30,
                    min_root_synapses=0, verbose=False,
                )
                self.assertEqual(call_count[0], count_after_first,
                                 "build_dataset re-fetched already-cached boxes")
        finally:
            db_module.fetch_volume = orig_fv
            db_module.fetch_synapses = orig_fs

    def test_fetch_errors_are_tolerated(self):
        """Fetch failures should be skipped, not raise."""
        import neuronauts.dataset_builder as db_module
        orig_fv = db_module.fetch_volume
        orig_fs = db_module.fetch_synapses
        try:
            def failing_fetch(*args, **kwargs):
                raise RuntimeError("network error")

            db_module.fetch_synapses = failing_fetch
            db_module.fetch_volume = lambda *a, **kw: _make_volume()

            with tempfile.TemporaryDirectory() as d:
                cache = BoxCache(d)
                specs = select_random_boxes(n=3, seed=0)
                # Should not raise.
                records = build_dataset(specs, cache, verbose=False)
                self.assertEqual(records, [])
        finally:
            db_module.fetch_volume = orig_fv
            db_module.fetch_synapses = orig_fs


# ---------------------------------------------------------------------------
# synapse_root_counts_static helpers
# ---------------------------------------------------------------------------

class SynapseRootCountsStaticTest(unittest.TestCase):
    """Lightweight tests for helper functions in synapse_root_counts_static.py."""

    def test_load_synapse_header_basic(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")
        from neuronauts.synapse_root_counts_static import load_synapse_header

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("pre_pt_root_id,int64\n")
            f.write("post_pt_root_id,int64\n")
            f.write("ctr_pt_position,object\n")
            fname = f.name
        try:
            mapping = load_synapse_header(fname)
            self.assertEqual(mapping[0], "pre_pt_root_id")
            self.assertEqual(mapping[1], "post_pt_root_id")
        finally:
            os.unlink(fname)

    def test_build_root_table_basic(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")
        from neuronauts.synapse_root_counts_static import build_root_table

        pre_counts = {1: 10, 2: 5, 3: 1}
        post_counts = {1: 3, 4: 7}
        soma_roots = pd.Series([1, 2])

        df = build_root_table(pre_counts, post_counts, soma_roots)
        self.assertIn("root_id", df.columns)
        self.assertIn("total_synapse_count", df.columns)
        self.assertIn("has_soma", df.columns)

        root1_row = df[df["root_id"] == 1].iloc[0]
        self.assertEqual(root1_row["pre_synapse_count"], 10)
        self.assertEqual(root1_row["post_synapse_count"], 3)
        self.assertEqual(root1_row["total_synapse_count"], 13)
        self.assertTrue(root1_row["has_soma"])

    def test_build_root_table_soma_flag(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")
        from neuronauts.synapse_root_counts_static import build_root_table

        df = build_root_table({1: 5}, {2: 3}, pd.Series([1]))
        self.assertTrue(df[df["root_id"] == 1]["has_soma"].values[0])
        self.assertFalse(df[df["root_id"] == 2]["has_soma"].values[0])

    def test_build_root_table_sorted_descending(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")
        from neuronauts.synapse_root_counts_static import build_root_table

        df = build_root_table({1: 1, 2: 100, 3: 50}, {}, pd.Series([]))
        counts = df["total_synapse_count"].tolist()
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_get_soma_roots_from_csv(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")
        from neuronauts.synapse_root_counts_static import get_soma_roots_from_csv

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("pt_root_id\n1001\n1002\n0\n1003\n")
            fname = f.name
        try:
            roots = get_soma_roots_from_csv(fname)
            self.assertNotIn(0, roots.tolist())
            self.assertIn(1001, roots.tolist())
            self.assertEqual(len(roots), 3)
        finally:
            os.unlink(fname)


if __name__ == "__main__":
    unittest.main()
