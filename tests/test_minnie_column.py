"""Unit tests for experiments/minnie_column (no CAVE network)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.minnie_column.dedup import synapse_stable_key, tube_overlap_weights
from experiments.minnie_column.attach_assets import enrich_manifest
from experiments.minnie_column.asset_urls import mat_dbs_synapse_gz
from experiments.minnie_column.discover_column_bbox import compute_bbox_nm
from experiments.minnie_column.paradigm import difficulty_from_proofread_row, tube_radius_um_for_difficulty
from experiments.minnie_column.spatial import assign_bins_xy, parse_bbox_nm, train_test_split_by_bin


def test_parse_bbox_nm():
    bb = parse_bbox_nm("0,0,0,1000,2000,3000")
    assert bb == ((0, 0, 0), (1000, 2000, 3000))


def test_assign_bins_xy_two_bins():
    bbox = ((0, 0, 0), (200_000, 100_000, 100_000))
    # 50 um = 50_000 nm width, 100 um = 100_000 nm height
    x = np.array([25_000, 75_000], dtype=np.float64)
    y = np.array([50_000, 50_000], dtype=np.float64)
    bins = assign_bins_xy(x, y, bbox, bin_width_um=50.0, bin_height_um=100.0)
    assert bins[0] != bins[1]


def test_train_test_median():
    bid = np.array([0, 0, 1, 1], dtype=np.int64)
    s = train_test_split_by_bin(bid, auto_median_test=True)
    assert set(s.tolist()) <= {"train", "test"}


def test_synapse_stable_key_deterministic():
    ctr = np.array([[100.0, 200.0, 300.0]], dtype=np.float64)
    k1 = synapse_stable_key(ctr)
    k2 = synapse_stable_key(ctr)
    assert np.array_equal(k1, k2)


def test_tube_overlap_weights():
    w = tube_overlap_weights(np.array([1, 2]), tube_membership=[[1], [1, 2]])
    assert abs(w[0] - 1.0) < 1e-6
    assert abs(w[1] - 0.5) < 1e-6


def test_tube_radius_by_difficulty():
    assert tube_radius_um_for_difficulty("easy") < tube_radius_um_for_difficulty("hard")


def test_compute_bbox_nm_margin():
    bb = compute_bbox_nm(
        np.array([100.0, 200.0]),
        np.array([300.0, 400.0]),
        np.array([50.0, 60.0]),
        margin_nm=1000,
    )
    assert bb[0][0] == 100 - 1000
    assert bb[1][1] == 400 + 1000


def test_mat_dbs_url_contains_version():
    u = mat_dbs_synapse_gz(1718)
    assert "1718" in u and "synapses_pni_2" in u


def test_enrich_manifest_columns():
    import pandas as pd

    df = pd.DataFrame({"id": [1], "pt_root_id": [12345]})
    out = enrich_manifest(df, version=1718, static_synapse_version=None)
    assert "asset_em_cloudvolume" in out.columns
    assert "asset_skeleton_swc_proofread_url" in out.columns
    assert out["pt_root_id"].iloc[0] == 12345


def test_difficulty_from_proofread():
    d = difficulty_from_proofread_row(
        status_dendrite=True,
        status_axon=True,
        strategy_dendrite="dendrite_clean",
        strategy_axon="axon_partially_extended",
    )
    assert d == "easy"
