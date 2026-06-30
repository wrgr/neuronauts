"""Tests for neuronauts.data.loaders — all mocked, no network required."""

from __future__ import annotations

import gzip
import io
import struct
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: build fake HTTP responses
# ---------------------------------------------------------------------------

def _gz_csv_response(rows: list[dict]) -> MagicMock:
    """Return a mock requests.Response whose .content is a gzip'd CSV."""
    import pandas as pd

    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    # Write raw CSV bytes matching the format the real table uses:
    # col0,col1,col2,root_id (index 3)
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for i, row in enumerate(rows):
            line = f"a,b,c,{row['root_id']}\n"
            gz.write(line.encode())

    mock = MagicMock()
    mock.status_code = 200
    mock.content = buf.getvalue()
    mock.raise_for_status = MagicMock()
    return mock


def _skeleton_bytes(n_verts: int = 10, n_edges: int = 9) -> bytes:
    """Build a synthetic neuroglancer precomputed skeleton binary payload."""
    buf = bytearray()
    buf += struct.pack("<I", n_verts)
    buf += struct.pack("<I", n_edges)
    # vertices: n_verts × 3 float32
    verts = np.arange(n_verts * 3, dtype=np.float32).reshape(n_verts, 3)
    buf += verts.tobytes()
    # edges: n_edges × 2 uint32  (simple chain)
    edges = np.stack([np.arange(n_edges), np.arange(1, n_edges + 1)], axis=1).astype(np.uint32)
    buf += edges.tobytes()
    # radii: n_verts float32
    radii = np.ones(n_verts, dtype=np.float32) * 250.0
    buf += radii.tobytes()
    return bytes(buf)


def _skeleton_response(n_verts: int = 10, n_edges: int = 9) -> MagicMock:
    mock = MagicMock()
    mock.status_code = 200
    mock.content = _skeleton_bytes(n_verts, n_edges)
    return mock


# ---------------------------------------------------------------------------
# load_nucleus_table
# ---------------------------------------------------------------------------

class TestLoadNucleusTable:
    def test_load_nucleus_table_shape(self):
        """Mock HTTP → tiny CSV → DataFrame with root_id column and correct row count."""
        fake_rows = [{"root_id": 1000 + i} for i in range(5)]
        mock_resp = _gz_csv_response(fake_rows)

        with patch("requests.get", return_value=mock_resp) as mock_get:
            from neuronauts.data.loaders import load_nucleus_table

            df = load_nucleus_table()

        mock_get.assert_called_once()
        assert "root_id" in df.columns
        assert len(df) == 5
        assert df["root_id"].dtype == np.int64

    def test_load_nucleus_table_excludes_zero_root_ids(self):
        """Rows with root_id == 0 must be filtered out."""
        fake_rows = [{"root_id": 0}, {"root_id": 42}, {"root_id": 99}]
        mock_resp = _gz_csv_response(fake_rows)
        # Overwrite content to include the zero row
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            for row in fake_rows:
                gz.write(f"a,b,c,{row['root_id']}\n".encode())
        mock_resp.content = buf.getvalue()

        with patch("requests.get", return_value=mock_resp):
            from neuronauts.data.loaders import load_nucleus_table

            df = load_nucleus_table()

        assert 0 not in df["root_id"].values
        assert len(df) == 2

    def test_load_nucleus_table_cache_roundtrip(self, tmp_path):
        """Written cache file is read back without hitting the network again."""
        fake_rows = [{"root_id": 200 + i} for i in range(3)]
        mock_resp = _gz_csv_response(fake_rows)
        cache = str(tmp_path / "nucleus.csv.gz")

        with patch("requests.get", return_value=mock_resp) as mock_get:
            from neuronauts.data.loaders import load_nucleus_table

            df1 = load_nucleus_table(cache_path=cache)
            assert mock_get.call_count == 1

            df2 = load_nucleus_table(cache_path=cache)
            assert mock_get.call_count == 1  # no second network call

        assert len(df1) == len(df2) == 3


# ---------------------------------------------------------------------------
# load_skeleton / parse
# ---------------------------------------------------------------------------

class TestLoadSkeleton:
    def test_load_skeleton_parse(self):
        """Mock HTTP binary → correct vertex/edge/radii shapes."""
        n_v, n_e = 12, 11
        mock_resp = _skeleton_response(n_verts=n_v, n_edges=n_e)

        with patch("requests.get", return_value=mock_resp):
            from neuronauts.data.loaders import load_skeleton

            skel = load_skeleton(root_id=123456789)

        assert skel is not None
        assert skel["vertices_nm"].shape == (n_v, 3)
        assert skel["edges"].shape == (n_e, 2)
        assert skel["radii_nm"].shape == (n_v,)
        assert skel["vertices_nm"].dtype == np.float32
        assert skel["edges"].dtype == np.int64
        assert skel["radii_nm"].dtype == np.float32

    def test_load_skeleton_returns_none_on_404(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("requests.get", return_value=mock_resp):
            from neuronauts.data.loaders import load_skeleton

            assert load_skeleton(root_id=0) is None

    def test_load_skeleton_returns_none_on_too_few_vertices(self):
        """n_verts < 3 → placeholder skeleton → None."""
        buf = bytearray()
        buf += struct.pack("<I", 2)  # n_verts = 2
        buf += struct.pack("<I", 1)  # n_edges = 1
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = bytes(buf)

        with patch("requests.get", return_value=mock_resp):
            from neuronauts.data.loaders import load_skeleton

            assert load_skeleton(root_id=1) is None

    def test_load_skeleton_falls_back_radii_when_absent(self):
        """When radii bytes are missing, default 300 nm fallback is used."""
        n_v, n_e = 5, 4
        buf = bytearray()
        buf += struct.pack("<I", n_v)
        buf += struct.pack("<I", n_e)
        verts = np.zeros((n_v, 3), dtype=np.float32)
        buf += verts.tobytes()
        edges = np.zeros((n_e, 2), dtype=np.uint32)
        buf += edges.tobytes()
        # Deliberately omit radii bytes

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = bytes(buf)

        with patch("requests.get", return_value=mock_resp):
            from neuronauts.data.loaders import load_skeleton

            skel = load_skeleton(root_id=99)

        assert skel is not None
        assert np.all(skel["radii_nm"] == 300.0)


# ---------------------------------------------------------------------------
# load_skeletons (concurrent)
# ---------------------------------------------------------------------------

class TestLoadSkeletonsConcurrent:
    def test_load_skeletons_concurrent(self):
        """3 roots, all succeed → all 3 returned in output dict."""
        n_v, n_e = 8, 7
        mock_resp = _skeleton_response(n_verts=n_v, n_edges=n_e)

        with patch("requests.get", return_value=mock_resp):
            from neuronauts.data.loaders import load_skeletons

            result = load_skeletons([10, 20, 30], max_workers=2, progress=False)

        assert set(result.keys()) == {10, 20, 30}
        for rid, skel in result.items():
            assert skel["vertices_nm"].shape == (n_v, 3)

    def test_load_skeletons_omits_failures(self):
        """Failed fetches (404) are silently omitted."""
        ok_resp = _skeleton_response()
        fail_resp = MagicMock()
        fail_resp.status_code = 404

        call_count = {"n": 0}

        def _side_effect(url, **kwargs):
            call_count["n"] += 1
            # Fail the second call
            if call_count["n"] == 2:
                return fail_resp
            return ok_resp

        with patch("requests.get", side_effect=_side_effect):
            from neuronauts.data.loaders import load_skeletons

            result = load_skeletons([1, 2, 3], max_workers=1, progress=False)

        assert 2 not in result
        assert len(result) == 2


# ---------------------------------------------------------------------------
# sample_neurons
# ---------------------------------------------------------------------------

class TestSampleNeurons:
    def _mock_nucleus(self, n: int = 100) -> MagicMock:
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            for i in range(n):
                gz.write(f"a,b,c,{1_000_000 + i}\n".encode())
        mock = MagicMock()
        mock.status_code = 200
        mock.content = buf.getvalue()
        mock.raise_for_status = MagicMock()
        return mock

    def test_sample_neurons_count(self):
        """Returns exactly n root IDs."""
        with patch("requests.get", return_value=self._mock_nucleus(100)):
            from neuronauts.data.loaders import sample_neurons

            result = sample_neurons(10, seed=7)

        assert len(result) == 10

    def test_sample_neurons_reproducible(self):
        """Same seed → same result across two calls."""
        mock_resp = self._mock_nucleus(100)

        with patch("requests.get", return_value=mock_resp):
            from neuronauts.data.loaders import sample_neurons

            r1 = sample_neurons(5, seed=42)

        with patch("requests.get", return_value=mock_resp):
            r2 = sample_neurons(5, seed=42)

        assert r1 == r2

    def test_sample_neurons_different_seeds_differ(self):
        """Different seeds → different results (with high probability for n=10)."""
        mock_resp = self._mock_nucleus(200)

        with patch("requests.get", return_value=mock_resp):
            from neuronauts.data.loaders import sample_neurons

            r1 = sample_neurons(10, seed=0)

        with patch("requests.get", return_value=mock_resp):
            r2 = sample_neurons(10, seed=1)

        assert r1 != r2

    def test_sample_neurons_returns_ints(self):
        """All returned values are plain Python ints."""
        with patch("requests.get", return_value=self._mock_nucleus(50)):
            from neuronauts.data.loaders import sample_neurons

            result = sample_neurons(5, seed=0)

        assert all(isinstance(r, int) for r in result)

    def test_sample_neurons_cell_type_unavailable_raises(self):
        """cell_type filter with unavailable table → ValueError."""
        with patch("requests.get", return_value=self._mock_nucleus(50)), \
             patch("neuronauts.data.loaders.load_cell_types", return_value=None):
            from neuronauts.data.loaders import sample_neurons

            with pytest.raises(ValueError, match="cell_type filter"):
                sample_neurons(3, cell_type="E")

    def test_sample_neurons_cell_type_filter(self):
        """cell_type filter returns only matching root IDs."""
        import pandas as pd

        mock_resp = self._mock_nucleus(100)
        # Build a fake cell type table: only root IDs 1_000_000..1_000_009 are "E"
        ct_df = pd.DataFrame({
            "root_id": list(range(1_000_000, 1_000_010)),
            "cell_type": ["E"] * 10,
        })

        with patch("requests.get", return_value=mock_resp), \
             patch("neuronauts.data.loaders.load_cell_types", return_value=ct_df):
            from neuronauts.data.loaders import sample_neurons

            result = sample_neurons(5, cell_type="E", seed=0)

        assert len(result) == 5
        allowed = set(range(1_000_000, 1_000_010))
        assert all(r in allowed for r in result)
