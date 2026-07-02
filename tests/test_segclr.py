"""Offline tests for the SegCLR loader (no network)."""
import numpy as np
import pytest

from neuronauts import segclr


def test_md5_shard_deterministic_and_in_range():
    sid = 864691134133779548
    s1 = segclr.md5_shard(sid)
    s2 = segclr.md5_shard(sid)
    assert s1 == s2
    assert 0 <= s1 < segclr.NUM_SHARDS


def test_md5_shard_bytewidth_matters():
    # The released data uses bytewidth=64 bytes; the naive bytewidth=8 differs.
    sid = 864691134133779548
    assert segclr.md5_shard(sid, bytewidth=64) != segclr.md5_shard(sid, bytewidth=8)


def test_parse_csv_layout():
    # rows are: node_id, x, y, z, e0..e63  (68 columns)
    emb0 = list(range(segclr.EMB_DIM))
    emb1 = list(range(100, 100 + segclr.EMB_DIM))
    row0 = ",".join(str(v) for v in [7, 1000.0, 2000.0, 3000.0, *emb0])
    row1 = ",".join(str(v) for v in [8, 4000.0, 5000.0, 6000.0, *emb1])
    raw = (row0 + "\n" + row1 + "\n").encode("utf-8")
    pts, emb = segclr._parse_csv(raw)
    assert pts.shape == (2, 3)
    assert emb.shape == (2, segclr.EMB_DIM)
    np.testing.assert_allclose(pts[0], [1000.0, 2000.0, 3000.0])
    np.testing.assert_allclose(emb[0], emb0)
    np.testing.assert_allclose(emb[1][0], 100.0)


def test_parse_csv_skips_short_rows():
    raw = b"1,2,3\n7,1.0,2.0,3.0," + b",".join(b"0" for _ in range(segclr.EMB_DIM)) + b"\n"
    pts, emb = segclr._parse_csv(raw)
    assert pts.shape == (1, 3)  # the short first row is skipped


def test_assign_points_to_vertices_coverage():
    pts = np.array([[0.0, 0, 0], [1000, 0, 0], [2000, 0, 0]])
    emb = np.arange(3 * segclr.EMB_DIM, dtype=np.float32).reshape(3, segclr.EMB_DIM)
    verts = np.array([[0.0, 0, 0],          # exact hit
                      [1000, 10, 0],        # within cap
                      [50_000, 0, 0]])      # far -> uncovered
    asg = segclr.assign_points_to_vertices(verts, pts, emb, max_dist_nm=2000.0)
    assert asg.covered.tolist() == [True, True, False]
    assert asg.coverage_fraction == pytest.approx(2 / 3)
    np.testing.assert_allclose(asg.embedding[0], emb[0])
    assert np.isnan(asg.embedding[2]).all()
