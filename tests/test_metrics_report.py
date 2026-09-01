"""Key registry, formatting, JSON round-trip: neuronauts.metrics.report.

The registry test is the important one: it fails the build if a metric
function starts returning a key that nobody documented, which is how this
suite stays self-describing as it grows.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from neuronauts.metrics import (
    KEY_DOCS,
    describe_key,
    evaluate_partition_suite,
    format_metrics,
    metrics_to_json,
    to_jsonable,
    undocumented_keys,
)


def test_describe_key_known_exact_key():
    assert describe_key("ari") is not None


def test_describe_key_unknown_key_is_none():
    assert describe_key("totally_made_up_key_xyz") is None


def test_describe_key_matches_line_graph_pattern():
    assert describe_key("lg_and_metric_precision") is not None
    assert describe_key("lg_bogus_variant_precision") is None


def test_every_key_docs_entry_is_a_nonempty_string():
    for k, v in KEY_DOCS.items():
        assert isinstance(v, str) and v, k


def test_full_suite_output_has_no_undocumented_keys():
    """Registry drift guard: every key evaluate_partition_suite can emit,
    across every optional block, must be in the registry."""
    rng = np.random.default_rng(0)
    n = 20
    pred = rng.integers(0, 5, size=n)
    true = rng.integers(1, 6, size=n)
    true_post = rng.integers(100, 103, size=n)
    pred_post = rng.integers(0, 4, size=n)
    fragment_id = rng.integers(0, 6, size=n)
    root_label_map = {i: {rng.integers(1, 5)} for i in range(6)}
    weights = rng.uniform(1, 10, size=n)
    src = rng.integers(0, n, size=15)
    dst = rng.integers(0, n, size=15)

    m = evaluate_partition_suite(
        pred, true, weights=weights, src=src, dst=dst,
        fragment_id=fragment_id, root_label_map=root_label_map,
        true_post=true_post, pred_post=pred_post,
    )
    missing = undocumented_keys(m.keys())
    assert missing == [], f"undocumented metric keys: {missing}"


def test_format_metrics_groups_into_sections_and_formats_nan():
    m = {"ari": float("nan"), "n_items": 5, "merge_precision": 0.5}
    text = format_metrics(m, title="demo")
    assert "demo" in text
    assert "n/a" in text
    assert "0.5000" in text
    assert "[PARTITION]" in text
    assert "[EDGES]" in text


def test_to_jsonable_converts_nan_and_numpy_types():
    payload = {"a": float("nan"), "b": np.int64(3), "c": np.array([1.0, 2.0]), "d": np.bool_(True)}
    out = to_jsonable(payload)
    assert out["a"] is None
    assert out["b"] == 3 and isinstance(out["b"], int)
    assert out["c"] == [1.0, 2.0]
    assert out["d"] is True
    json.dumps(out)  # must not raise


def test_metrics_to_json_round_trips_through_standard_json():
    m = {"ari": 0.5, "n_items": 3, "vi": float("nan")}
    s = metrics_to_json(m)
    back = json.loads(s)
    assert back["ari"] == 0.5
    assert back["vi"] is None
