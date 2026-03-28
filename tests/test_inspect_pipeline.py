"""Smoke tests for scripts/inspect_pipeline.py stage builders."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from neuronauts.fetch import SynapseTable

torch = pytest.importorskip("torch", reason="torch not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_box_record():
    return SimpleNamespace(
        center_nm=[100000.0, 100000.0, 100000.0],
        side_um=10.0,
        box_hash="abcdef1234567890",
        n_synapses=12,
        n_positive_pairs=6,
    )


def _make_synapses(n=12, seed=42):
    rng = np.random.default_rng(seed)
    n_cells = 3
    per_cell = n // n_cells
    pre_pt = np.zeros((n, 3), dtype=np.float32)
    pre_root_id = np.zeros(n, dtype=np.int64)
    post_root_id = np.zeros(n, dtype=np.int64)
    post_pt = rng.standard_normal((n, 3)).astype(np.float32) * 50

    for c in range(n_cells):
        centre = rng.standard_normal(3) * 500
        for k in range(per_cell):
            idx = c * per_cell + k
            pre_pt[idx] = centre + rng.standard_normal(3) * 20
            pre_root_id[idx] = c + 1
            post_root_id[idx] = rng.integers(1, n_cells + 1)

    # Add seg_ids so scaffold stage works
    pre_seg_id = pre_root_id * 100  # distinct from root_id but correlated
    post_seg_id = post_root_id * 100

    return SynapseTable(
        pre_pt=pre_pt,
        post_pt=post_pt,
        pre_root_id=pre_root_id,
        post_root_id=post_root_id,
        synapse_id=np.arange(n, dtype=np.int64),
        pre_seg_id=pre_seg_id,
        post_seg_id=post_seg_id,
    )


@pytest.fixture
def mock_neuroglancer():
    """Minimal mock of the neuroglancer module for stage builder tests."""
    ng = MagicMock()

    # Make PointAnnotation and LineAnnotation return SimpleNamespace-like objects
    ng.PointAnnotation = lambda **kw: SimpleNamespace(**kw)
    ng.LineAnnotation = lambda **kw: SimpleNamespace(**kw)
    ng.LocalAnnotationLayer = lambda **kw: SimpleNamespace(kind="annotation", **kw)

    return ng


# ---------------------------------------------------------------------------
# Stage 1: synapses
# ---------------------------------------------------------------------------

class TestStageSynapses:
    def test_returns_two_layers(self, mock_neuroglancer):
        from inspect_pipeline import stage_synapses

        synapses = _make_synapses()
        box_record = _make_box_record()
        layers = stage_synapses(synapses, box_record, mock_neuroglancer)
        assert len(layers) == 2
        names = [name for name, _ in layers]
        assert "syn_pre" in names
        assert "syn_post" in names

    def test_annotations_count(self, mock_neuroglancer):
        from inspect_pipeline import stage_synapses

        n = 12
        synapses = _make_synapses(n=n)
        box_record = _make_box_record()
        layers = stage_synapses(synapses, box_record, mock_neuroglancer)
        for _, layer in layers:
            assert len(layer.annotations) == n


# ---------------------------------------------------------------------------
# Stage 2: scaffold
# ---------------------------------------------------------------------------

class TestStageScaffold:
    def test_returns_layers_with_seg_ids(self, mock_neuroglancer):
        from inspect_pipeline import stage_scaffold

        synapses = _make_synapses()
        box_record = _make_box_record()
        layers = stage_scaffold(synapses, box_record, mock_neuroglancer)
        assert len(layers) > 0
        for name, _ in layers:
            assert "scaffold_" in name

    def test_no_seg_ids_returns_empty(self, mock_neuroglancer):
        from inspect_pipeline import stage_scaffold

        synapses = _make_synapses()
        # Remove seg_ids
        delattr(synapses, "pre_seg_id")
        delattr(synapses, "post_seg_id")
        box_record = _make_box_record()
        layers = stage_scaffold(synapses, box_record, mock_neuroglancer)
        assert layers == []


# ---------------------------------------------------------------------------
# Stage 3: evidence graph
# ---------------------------------------------------------------------------

class TestStageEvidenceGraph:
    def test_returns_layers(self, mock_neuroglancer):
        from inspect_pipeline import stage_evidence_graph

        synapses = _make_synapses()
        box_record = _make_box_record()
        layers = stage_evidence_graph(
            synapses, box_record, mock_neuroglancer,
            proximity_radius_nm=50000.0,  # wide radius to get edges
        )
        # Should have at least one layer per role (pre/post)
        assert len(layers) >= 1
        for name, layer in layers:
            assert "evidence_" in name


# ---------------------------------------------------------------------------
# Stage 4: grammar scores
# ---------------------------------------------------------------------------

class TestStageGrammarScores:
    def test_returns_layers_with_grammar_fn(self, mock_neuroglancer):
        from inspect_pipeline import stage_grammar_scores

        synapses = _make_synapses()
        box_record = _make_box_record()

        # Simple grammar function: cosine similarity of mean positions
        def dummy_grammar(feats_a, feats_b):
            return float(0.5)

        layers = stage_grammar_scores(
            synapses, box_record, mock_neuroglancer, dummy_grammar,
            proximity_radius_nm=50000.0,
        )
        # May return 0 or more layers depending on scaffold groups
        for name, _ in layers:
            assert "grammar_" in name


# ---------------------------------------------------------------------------
# Stage 5 & 6 require CellGNN model — test with a real tiny model
# ---------------------------------------------------------------------------

class TestStageCellLabels:
    def test_returns_layers(self, mock_neuroglancer):
        from inspect_pipeline import stage_cell_labels
        from neuronauts.cell_graph import CellGNN

        synapses = _make_synapses()
        box_record = _make_box_record()
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=2, n_heads=2, embedding_dim=8)
        model.eval()

        layers = stage_cell_labels(
            synapses, box_record, mock_neuroglancer, model,
            proximity_radius_nm=50000.0,
        )
        assert len(layers) >= 1
        for name, _ in layers:
            assert "cell_" in name


class TestStageAssembly:
    def test_returns_layers_and_metrics(self, mock_neuroglancer):
        from inspect_pipeline import stage_assembly
        from neuronauts.cell_graph import CellGNN

        synapses = _make_synapses()
        box_record = _make_box_record()
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=2, n_heads=2, embedding_dim=8)
        model.eval()

        result = stage_assembly(
            synapses, box_record, mock_neuroglancer, model,
            proximity_radius_nm=50000.0,
        )
        layers, metrics = result
        # Should return some layers (correct and/or wrong)
        assert isinstance(layers, list)
        assert hasattr(metrics, "f1")


# ---------------------------------------------------------------------------
# CLI parse_args
# ---------------------------------------------------------------------------

class TestCLI:
    def test_parse_args_defaults(self):
        from inspect_pipeline import parse_args

        args = parse_args(["--cache-dir", "/tmp/test"])
        assert args.cache_dir == "/tmp/test"
        assert args.box_idx == 0
        assert args.proximity_radius_nm == 5000.0
        assert args.partition_threshold == 0.5

    def test_parse_args_all_options(self):
        from inspect_pipeline import parse_args

        args = parse_args([
            "--cache-dir", "/tmp/test",
            "--grammar-path", "models/gram.pt",
            "--cell-gnn-path", "models/gnn.pt",
            "--box-idx", "3",
            "--proximity-radius-nm", "8000",
            "--partition-threshold", "0.7",
            "--list-boxes",
            "--min-synapses", "20",
        ])
        assert args.grammar_path == "models/gram.pt"
        assert args.cell_gnn_path == "models/gnn.pt"
        assert args.box_idx == 3
        assert args.proximity_radius_nm == 8000.0
        assert args.partition_threshold == 0.7
        assert args.list_boxes is True
        assert args.min_synapses == 20
