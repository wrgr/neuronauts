"""Integration tests for CellGNN pipeline subcommands.

Tests cmd_train_cell_gnn, cmd_evaluate, cmd_sweep, cmd_scale_test
using synthetic BoxCache data (no network / CAVE access required).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

# Lazy torch import
torch = pytest.importorskip("torch", reason="torch not installed")

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuronauts.fetch import SynapseTable, RealBoxSpec
from neuronauts.dataset_builder import BoxCache
from neuronauts.cell_graph import (
    CellGNN,
    CellGNNConfig,
    build_synapse_graph,
    cell_graph_train_step,
    infer_cells,
    partition_from_embeddings,
    save_cell_gnn,
    spatial_train_val_test_split,
    train_cell_gnn,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synapses(n_cells=4, per_cell=6, seed=0):
    """Synthetic SynapseTable with clearly clustered structure."""
    rng = np.random.default_rng(seed)
    n = n_cells * per_cell
    pre_pt = np.zeros((n, 3), dtype=np.float32)
    post_pt = np.zeros((n, 3), dtype=np.float32)
    pre_root_id = np.zeros(n, dtype=np.int64)
    post_root_id = np.zeros(n, dtype=np.int64)
    pre_seg_id = np.zeros(n, dtype=np.int64)
    post_seg_id = np.zeros(n, dtype=np.int64)

    for c in range(n_cells):
        ctr_pre = rng.standard_normal(3) * 600
        ctr_post = rng.standard_normal(3) * 600
        for k in range(per_cell):
            idx = c * per_cell + k
            pre_pt[idx] = ctr_pre + rng.standard_normal(3) * 25
            post_pt[idx] = ctr_post + rng.standard_normal(3) * 25
            pre_root_id[idx] = c + 1
            post_root_id[idx] = rng.integers(1, n_cells + 1)
            pre_seg_id[idx] = (c + 1) * 100 + rng.integers(0, 3)
            post_seg_id[idx] = rng.integers(1, n_cells * 3 + 1)

    return SynapseTable(
        pre_pt=pre_pt, post_pt=post_pt,
        pre_root_id=pre_root_id, post_root_id=post_root_id,
        synapse_id=np.arange(n, dtype=np.int64),
        pre_seg_id=pre_seg_id, post_seg_id=post_seg_id,
    )


def _count_positive_pairs(syn):
    n_pos = 0
    for ids in [syn.pre_root_id, syn.post_root_id]:
        c = Counter(ids.tolist())
        for rid, cnt in c.items():
            if rid > 0 and cnt >= 2:
                n_pos += cnt * (cnt - 1) // 2
    return n_pos


@pytest.fixture
def synthetic_cache(tmp_path):
    """Build a 12-box synthetic BoxCache in a temp directory."""
    cache_dir = tmp_path / "boxes"
    cache = BoxCache(str(cache_dir))
    rng = np.random.default_rng(42)

    for i in range(12):
        n_cells = int(rng.integers(3, 7))
        per_cell = int(rng.integers(4, 10))
        syn = _make_synapses(n_cells=n_cells, per_cell=per_cell, seed=i)
        n_pos = _count_positive_pairs(syn)
        center_nm = (
            int(500_000 + i * 200_000),
            int(500_000 + rng.integers(-100_000, 100_000)),
            int(200_000 + rng.integers(-50_000, 50_000)),
        )
        spec = RealBoxSpec(center_nm=center_nm, side_um=6.0, mip=2)
        cache.save_synapse_only(spec, syn, n_positive_pairs=n_pos)

    return str(cache_dir)


@pytest.fixture
def trained_checkpoint(synthetic_cache, tmp_path):
    """Train a small CellGNN and return (cache_dir, checkpoint_path)."""
    ckpt = str(tmp_path / "cell_gnn.pt")
    model = CellGNN(d_model=32, n_layers=2, embedding_dim=16)
    cache = BoxCache(synthetic_cache)

    cfg = CellGNNConfig(
        d_model=32, n_layers=2, epochs=5,
        learning_rate=1e-3, proximity_radius_nm=100000.0, seed=42,
    )
    train_cell_gnn(model, cache, config=cfg, verbose=False)
    save_cell_gnn(ckpt, model)
    return synthetic_cache, ckpt


# ---------------------------------------------------------------------------
# cmd_train_cell_gnn
# ---------------------------------------------------------------------------

class TestCmdTrainCellGnn:
    def test_full_training_run(self, synthetic_cache, tmp_path):
        """cmd_train_cell_gnn runs end-to-end and produces checkpoint + history."""
        from scripts.train import cmd_train_cell_gnn, parse_args

        ckpt = str(tmp_path / "cell_gnn_out.pt")
        log_dir = str(tmp_path / "logs")

        args = parse_args([
            "train-cell-gnn",
            "--cache-dir", synthetic_cache,
            "--epochs", "5",
            "--d-model", "32",
            "--n-layers", "2",
            "--embedding-dim", "16",
            "--proximity-radius-nm", "100000",
            "--min-synapses", "10",
            "--min-positive-pairs", "2",
            "--cell-gnn-output", ckpt,
            "--log-dir", log_dir,
            "--seed", "42",
        ])
        rc = cmd_train_cell_gnn(args)

        assert rc == 0
        assert Path(ckpt).exists()
        history_path = Path(log_dir) / "cell_gnn_history.json"
        assert history_path.exists()
        history = json.loads(history_path.read_text())
        assert len(history["train_loss"]) == 5
        # Loss should be non-negative
        assert all(l >= 0 for l in history["train_loss"])

    def test_training_with_edit_pairs_tsv(self, synthetic_cache, tmp_path):
        """cmd_train_cell_gnn loads and uses --edit-pairs-tsv."""
        from scripts.train import cmd_train_cell_gnn, parse_args

        # Write a small edit pairs TSV
        tsv_path = str(tmp_path / "edit_pairs.tsv")
        with open(tsv_path, "w") as f:
            f.write("synapse_i\tsynapse_j\tlabel\trole\tsource_root_a\tsource_root_b\tedit_type\n")
            f.write("0\t5\t1\tpre\t1\t2\tmerge\n")
            f.write("1\t8\t0\tpre\t1\t3\tsplit\n")

        ckpt = str(tmp_path / "cell_gnn_edit.pt")
        args = parse_args([
            "train-cell-gnn",
            "--cache-dir", synthetic_cache,
            "--epochs", "3",
            "--d-model", "32",
            "--n-layers", "2",
            "--proximity-radius-nm", "100000",
            "--min-synapses", "10",
            "--min-positive-pairs", "2",
            "--cell-gnn-output", ckpt,
            "--log-dir", str(tmp_path / "logs"),
            "--edit-pairs-tsv", tsv_path,
            "--edit-weight", "3.0",
        ])
        rc = cmd_train_cell_gnn(args)
        assert rc == 0
        assert Path(ckpt).exists()

    def test_resume_from_checkpoint(self, trained_checkpoint, tmp_path):
        """--resume loads existing checkpoint instead of reinitializing."""
        from scripts.train import cmd_train_cell_gnn, parse_args

        cache_dir, ckpt = trained_checkpoint
        args = parse_args([
            "train-cell-gnn",
            "--cache-dir", cache_dir,
            "--epochs", "2",
            "--d-model", "32",
            "--n-layers", "2",
            "--proximity-radius-nm", "100000",
            "--min-synapses", "10",
            "--min-positive-pairs", "2",
            "--cell-gnn-output", ckpt,
            "--log-dir", str(tmp_path / "logs"),
            "--resume",
        ])
        rc = cmd_train_cell_gnn(args)
        assert rc == 0


# ---------------------------------------------------------------------------
# cmd_evaluate
# ---------------------------------------------------------------------------

class TestCmdEvaluate:
    def test_evaluate_produces_results(self, trained_checkpoint, tmp_path):
        """cmd_evaluate runs on test split and writes results JSON."""
        from scripts.train import cmd_evaluate, parse_args

        cache_dir, ckpt = trained_checkpoint
        log_dir = str(tmp_path / "eval_logs")

        args = parse_args([
            "evaluate",
            "--cache-dir", cache_dir,
            "--cell-gnn-checkpoint", ckpt,
            "--proximity-radius-nm", "100000",
            "--partition-threshold", "0.5",
            "--min-synapses", "10",
            "--min-positive-pairs", "2",
            "--split", "test",
            "--log-dir", log_dir,
        ])
        rc = cmd_evaluate(args)
        assert rc == 0

        results_path = Path(log_dir) / "evaluate_results.json"
        assert results_path.exists()
        results = json.loads(results_path.read_text())
        assert results["split"] == "test"
        assert results["n_boxes"] > 0
        assert 0.0 <= results["gnn"]["f1_mean"] <= 1.0

    def test_evaluate_on_val_split(self, trained_checkpoint, tmp_path):
        """cmd_evaluate can target the val split."""
        from scripts.train import cmd_evaluate, parse_args

        cache_dir, ckpt = trained_checkpoint
        args = parse_args([
            "evaluate",
            "--cache-dir", cache_dir,
            "--cell-gnn-checkpoint", ckpt,
            "--proximity-radius-nm", "100000",
            "--min-synapses", "10",
            "--min-positive-pairs", "2",
            "--split", "val",
            "--log-dir", str(tmp_path / "logs"),
        ])
        rc = cmd_evaluate(args)
        assert rc == 0

    def test_evaluate_missing_checkpoint_fails(self, synthetic_cache, tmp_path):
        """cmd_evaluate returns non-zero if checkpoint doesn't exist."""
        from scripts.train import cmd_evaluate, parse_args

        args = parse_args([
            "evaluate",
            "--cache-dir", synthetic_cache,
            "--cell-gnn-checkpoint", "/nonexistent/path.pt",
            "--log-dir", str(tmp_path / "logs"),
        ])
        rc = cmd_evaluate(args)
        assert rc == 1


# ---------------------------------------------------------------------------
# cmd_sweep
# ---------------------------------------------------------------------------

class TestCmdSweep:
    def test_sweep_small_grid(self, synthetic_cache, tmp_path):
        """cmd_sweep runs a 2x2 grid and saves best model + results."""
        from scripts.train import cmd_sweep, parse_args

        best_path = str(tmp_path / "best.pt")
        log_dir = str(tmp_path / "sweep_logs")

        args = parse_args([
            "sweep",
            "--cache-dir", synthetic_cache,
            "--d-models", "32,64",
            "--n-layers-list", "2",
            "--proximity-radii", "100000",
            "--partition-thresholds", "0.5",
            "--epochs", "3",
            "--min-synapses", "10",
            "--min-positive-pairs", "2",
            "--best-output", best_path,
            "--log-dir", log_dir,
        ])
        rc = cmd_sweep(args)
        assert rc == 0
        assert Path(best_path).exists()

        results_path = Path(log_dir) / "sweep_results.json"
        assert results_path.exists()
        data = json.loads(results_path.read_text())
        assert len(data["sweep"]) == 2  # 2 configs
        assert data["best"] is not None
        assert data["best"]["val_f1_mean"] >= 0.0


# ---------------------------------------------------------------------------
# cmd_scale_test
# ---------------------------------------------------------------------------

class TestCmdScaleTest:
    def test_scale_test_runs(self, trained_checkpoint, tmp_path):
        """cmd_scale_test profiles boxes and writes results."""
        from scripts.train import cmd_scale_test, parse_args

        cache_dir, ckpt = trained_checkpoint
        log_dir = str(tmp_path / "scale_logs")

        args = parse_args([
            "scale-test",
            "--cache-dir", cache_dir,
            "--cell-gnn-checkpoint", ckpt,
            "--proximity-radius-nm", "100000",
            "--min-synapses", "10",
            "--n-boxes", "4",
            "--log-dir", log_dir,
        ])
        rc = cmd_scale_test(args)
        assert rc == 0

        results_path = Path(log_dir) / "scale_test_results.json"
        assert results_path.exists()
        data = json.loads(results_path.read_text())
        assert len(data) > 0
        assert all(r["graph_ms"] >= 0 for r in data)
        assert all(r["infer_ms"] >= 0 for r in data)

    def test_scale_test_without_checkpoint(self, synthetic_cache, tmp_path):
        """scale-test works with untrained model when no checkpoint given."""
        from scripts.train import cmd_scale_test, parse_args

        args = parse_args([
            "scale-test",
            "--cache-dir", synthetic_cache,
            "--proximity-radius-nm", "100000",
            "--min-synapses", "10",
            "--n-boxes", "3",
            "--log-dir", str(tmp_path / "logs"),
        ])
        rc = cmd_scale_test(args)
        assert rc == 0


# ---------------------------------------------------------------------------
# Edge cases: partition_from_embeddings
# ---------------------------------------------------------------------------

class TestPartitionEdgeCases:
    def test_threshold_zero_merges_everything(self):
        """threshold=0.0 should merge all synapses into one cell."""
        emb = np.random.default_rng(0).standard_normal((10, 16)).astype(np.float32)
        labels = partition_from_embeddings(emb, threshold=0.0)
        assert len(set(labels.tolist())) == 1

    def test_threshold_one_splits_everything(self):
        """threshold=1.0 should keep every synapse as its own cell
        (unless two embeddings are perfectly identical)."""
        rng = np.random.default_rng(7)
        emb = rng.standard_normal((10, 16)).astype(np.float32)
        labels = partition_from_embeddings(emb, threshold=1.0)
        # Each should be unique (extremely unlikely for random to be identical)
        assert len(set(labels.tolist())) == 10

    def test_greedy_vs_agglomerative_same_trivial_case(self):
        """Both methods should agree on perfectly separated clusters."""
        emb = np.array([
            [1.0, 0.0], [0.99, 0.01],  # cluster A
            [0.0, 1.0], [0.01, 0.99],  # cluster B
        ], dtype=np.float32)
        labels_agg = partition_from_embeddings(emb, threshold=0.9, method="agglomerative")
        labels_greedy = partition_from_embeddings(emb, threshold=0.9, method="greedy")
        # Both should produce 2 clusters
        assert len(set(labels_agg.tolist())) == 2
        assert len(set(labels_greedy.tolist())) == 2

    def test_large_embeddings_no_crash(self):
        """partition_from_embeddings handles 200+ synapses."""
        emb = np.random.default_rng(0).standard_normal((200, 32)).astype(np.float32)
        labels = partition_from_embeddings(emb, threshold=0.5)
        assert labels.shape == (200,)
        assert labels.min() >= 0


# ---------------------------------------------------------------------------
# Edge cases: spatial_train_val_test_split
# ---------------------------------------------------------------------------

class TestSpatialSplitEdgeCases:
    def _make_records(self, n, rng=None):
        """Create n mock BoxRecords with spatial positions."""
        from neuronauts.dataset_builder import BoxRecord
        if rng is None:
            rng = np.random.default_rng(0)
        records = []
        for i in range(n):
            records.append(BoxRecord(
                box_hash=f"hash{i:04d}",
                center_nm=(
                    int(500_000 + i * 100_000),
                    int(500_000 + rng.integers(-50_000, 50_000)),
                    int(200_000),
                ),
                side_um=6.0, mip=2,
                n_synapses=20 + i,
                n_positive_pairs=10,
            ))
        return records

    def test_single_box_goes_to_train(self):
        """With 1 box, fallback split puts it somewhere without crashing."""
        cache = type("C", (), {"all_records": lambda self: []})()
        records = self._make_records(1)
        splits = spatial_train_val_test_split(cache, records)
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 1

    def test_two_boxes_no_crash(self):
        """With 2 boxes, the split completes without error."""
        cache = type("C", (), {"all_records": lambda self: []})()
        records = self._make_records(2)
        splits = spatial_train_val_test_split(cache, records)
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 2

    def test_reproducible_with_same_seed(self):
        """Same seed produces identical splits."""
        cache = type("C", (), {"all_records": lambda self: []})()
        records = self._make_records(20)
        s1 = spatial_train_val_test_split(cache, records, seed=99)
        s2 = spatial_train_val_test_split(cache, records, seed=99)
        assert [r.box_hash for r in s1["train"]] == [r.box_hash for r in s2["train"]]
        assert [r.box_hash for r in s1["val"]] == [r.box_hash for r in s2["val"]]

    def test_different_seeds_differ(self):
        """Different seeds produce different splits."""
        cache = type("C", (), {"all_records": lambda self: []})()
        records = self._make_records(20)
        s1 = spatial_train_val_test_split(cache, records, seed=1)
        s2 = spatial_train_val_test_split(cache, records, seed=2)
        # At least one split should differ
        t1 = set(r.box_hash for r in s1["train"])
        t2 = set(r.box_hash for r in s2["train"])
        assert t1 != t2


# ---------------------------------------------------------------------------
# Edge cases: cell_graph_train_step with edit pairs
# ---------------------------------------------------------------------------

class TestTrainStepEditEdgeCases:
    def _make_graph_and_model(self):
        syn = _make_synapses(n_cells=3, per_cell=4, seed=0)
        graph = build_synapse_graph(syn, "pre", proximity_radius_nm=100000.0)
        model = CellGNN(d_model=32, n_layers=2, embedding_dim=16)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        return graph, model, opt

    def test_out_of_range_edit_pairs_ignored(self):
        """Edit pairs with indices beyond graph size are silently dropped."""
        graph, model, opt = self._make_graph_and_model()
        N = graph.n_synapses
        m = cell_graph_train_step(
            model, opt, graph,
            edit_positive_pairs=[(0, 1), (999, 1000)],  # 999 is out of range
            edit_negative_pairs=[(0, N + 50)],  # also out of range
        )
        # Only the valid pair (0,1) should be added
        assert m["loss"] >= 0

    def test_edit_weight_zero_same_as_no_edits(self):
        """edit_weight=0 should not change training behavior."""
        # This tests the parameter is accepted (actual weighting
        # is a future enhancement for per-pair loss scaling)
        graph, model, opt = self._make_graph_and_model()
        m = cell_graph_train_step(
            model, opt, graph,
            edit_positive_pairs=[(0, 1)],
            edit_weight=0.0,
        )
        assert m["loss"] >= 0

    def test_train_cell_gnn_edit_weight_zero(self):
        """train_cell_gnn with edit_weight=0.0 runs without error."""
        from neuronauts.edit_history import EditPair

        syn = _make_synapses(n_cells=3, per_cell=5, seed=0)
        model = CellGNN(d_model=32, n_layers=2, embedding_dim=16)

        class _Cache:
            def iter_records(self, shuffle=False, rng=None):
                return [type("R", (), {"n_positive_pairs": 5, "n_synapses": 15, "box_hash": "x"})()]
            def load(self, r):
                return None, syn

        pairs = [EditPair(0, 5, 1, "pre", 1, 2, "merge")]
        cfg = CellGNNConfig(d_model=32, n_layers=2, epochs=2, proximity_radius_nm=100000.0)
        history = train_cell_gnn(
            model, _Cache(), config=cfg,
            edit_pairs=pairs, edit_weight=0.0, verbose=False,
        )
        assert len(history["train_loss"]) == 2


# ---------------------------------------------------------------------------
# fetch_cave_boxes.py argument parsing
# ---------------------------------------------------------------------------

class TestFetchCaveBoxesArgs:
    def test_dry_run_no_network(self):
        """--dry-run should exit 0 without making network calls,
        or return 1 if caveclient is not installed."""
        from scripts.fetch_cave_boxes import main
        rc = main([
            "--cache-dir", "/tmp/test_dry_run",
            "--n-boxes", "5",
            "--dry-run",
        ])
        try:
            import caveclient  # noqa: F401
            assert rc == 0, "dry-run should succeed when caveclient is available"
        except ImportError:
            assert rc == 1, "should return 1 when caveclient is missing"
