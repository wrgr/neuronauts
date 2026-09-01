"""Tests for SynapseTable.filter_clutter and the line_graph.evaluate* clutter filter.

The filter drops synapses whose pre- or post-root has fewer than
``min_root_synapses`` total occurrences (counted across both columns).  This
removes 0-degree roots and small reconstruction fragments before connectome
F1 is computed.
"""

import unittest

import numpy as np
import pytest

# The implementation this file tests is not in main. `SynapseTable.filter_clutter`
# and `line_graph._clutter_keep_indices` live only on the unmerged branch
# `claude/remove-connectome-clutter-CkKag` (see experiments/README.md, "Needs a
# decision"), but the test was merged without them -- so collecting this module
# has been raising ImportError and aborting a bare `pytest` run. Skip honestly
# rather than delete: un-skip when that branch lands or the feature is dropped.
pytest.skip(
    "SynapseTable.filter_clutter / line_graph._clutter_keep_indices were never "
    "merged to main; test came from branch claude/remove-connectome-clutter-CkKag",
    allow_module_level=True,
)

from neuronauts.fetch import SynapseTable  # noqa: E402
from neuronauts.line_graph import (  # noqa: E402
    _clutter_keep_indices,
    evaluate,
    evaluate_from_root_ids,
)
from neuronauts.merge import ConnectivityGraph, MergedNeuron


def _make_table(pre, post, *, with_seg=False):
    n = len(pre)
    pre = np.asarray(pre, dtype=np.int64)
    post = np.asarray(post, dtype=np.int64)
    pre_pt = np.zeros((n, 3), dtype=np.float32)
    post_pt = np.zeros((n, 3), dtype=np.float32)
    syn_id = np.arange(n, dtype=np.int64)
    if with_seg:
        return SynapseTable(
            pre_pt=pre_pt,
            post_pt=post_pt,
            pre_root_id=pre,
            post_root_id=post,
            synapse_id=syn_id,
            pre_seg_id=pre.copy(),
            post_seg_id=post.copy(),
        )
    return SynapseTable(
        pre_pt=pre_pt,
        post_pt=post_pt,
        pre_root_id=pre,
        post_root_id=post,
        synapse_id=syn_id,
    )


class FilterClutterTest(unittest.TestCase):
    def test_disabled_below_threshold_two(self):
        # min_root_synapses <= 1 is a no-op.
        table = _make_table([1, 2, 3], [4, 5, 6])
        out = table.filter_clutter(min_root_synapses=1)
        self.assertEqual(out.n_synapses, 3)
        out = table.filter_clutter(min_root_synapses=0)
        self.assertEqual(out.n_synapses, 3)

    def test_drops_zero_degree_singletons(self):
        # Root 99 appears once on the pre side and once on the post side as
        # singletons; it has total occurrence 2 in the table, which is below
        # the threshold of 5.  Roots that pass form a single big cluster.
        # Synapses 0..4: pre=1 (5x), post=10 (5x) — 1 and 10 each have count 5.
        # Synapse 5: pre=99, post=10 — 99 has count 1, drop.
        pre = [1, 1, 1, 1, 1, 99]
        post = [10, 10, 10, 10, 10, 10]
        table = _make_table(pre, post)
        out = table.filter_clutter(min_root_synapses=5)
        self.assertEqual(out.n_synapses, 5)
        self.assertTrue(np.all(out.pre_root_id == 1))
        self.assertTrue(np.all(out.post_root_id == 10))

    def test_drops_when_either_endpoint_below_threshold(self):
        # Pre-root 1 appears 5 times → keep on pre side.
        # Post-root 99 appears once on synapse 0 → that synapse drops.
        pre = [1, 1, 1, 1, 1]
        post = [99, 10, 10, 10, 10]  # 99 has count 1; 10 has count 4
        table = _make_table(pre, post)
        out = table.filter_clutter(min_root_synapses=5)
        # Post-root 10 has count 4 (<5) so all surviving rows would also drop.
        self.assertEqual(out.n_synapses, 0)

    def test_threshold_counts_pre_and_post_combined(self):
        # Root 7 appears 3 times on pre and 2 times on post → total 5 → keep.
        pre = [7, 7, 7, 8, 8]
        post = [8, 8, 8, 7, 7]
        table = _make_table(pre, post)
        out = table.filter_clutter(min_root_synapses=5)
        self.assertEqual(out.n_synapses, 5)

    def test_preserves_seg_ids(self):
        pre = [1, 1, 1, 1, 1, 99]
        post = [10, 10, 10, 10, 10, 10]
        table = _make_table(pre, post, with_seg=True)
        out = table.filter_clutter(min_root_synapses=5)
        self.assertIsNotNone(out.pre_seg_id)
        self.assertIsNotNone(out.post_seg_id)
        self.assertEqual(len(out.pre_seg_id), 5)
        self.assertEqual(len(out.post_seg_id), 5)

    def test_empty_table_is_safe(self):
        table = _make_table([], [])
        out = table.filter_clutter(min_root_synapses=5)
        self.assertEqual(out.n_synapses, 0)


def _make_neuron(neuron_id, synapses):
    return MergedNeuron(
        neuron_id=neuron_id,
        agent_ids=[],
        path_points=np.zeros((1, 3), dtype=np.float32),
        synapse_indices=list(synapses),
        role="pre",
    )


def _make_graph(*neurons):
    return ConnectivityGraph(
        neurons={n.neuron_id: n for n in neurons},
        edges=[],
        unresolved_synapse_indices=[],
    )


class LineGraphFilterTest(unittest.TestCase):
    def test_evaluate_min_root_synapses_zero_is_unchanged(self):
        # Backwards-compatible default: min_root_synapses=0 reproduces baseline.
        pre = np.array([1, 1, 2, 2, 99], dtype=np.int64)
        post = np.array([10, 10, 11, 11, 12], dtype=np.int64)
        graph = _make_graph(
            _make_neuron(0, [0, 1]),
            _make_neuron(1, [2, 3]),
            _make_neuron(2, [4]),
        )
        m_default = evaluate(graph, pre, post)
        m_zero = evaluate(graph, pre, post, min_root_synapses=0)
        self.assertEqual(m_default.tp, m_zero.tp)
        self.assertEqual(m_default.fp, m_zero.fp)
        self.assertEqual(m_default.fn, m_zero.fn)

    def test_evaluate_drops_singleton_root_from_metric(self):
        # Synapses 0,1 belong to true neuron (root 1, post 10); synapses 2,3
        # belong to true neuron (root 2, post 11); synapse 4 has unique pre 99
        # and unique post 12 — total occurrences 1+1=2 across both columns.
        # The estimated graph mistakenly merges synapse 4 into cluster 1.
        pre = np.array([1, 1, 2, 2, 99], dtype=np.int64)
        post = np.array([10, 10, 11, 11, 12], dtype=np.int64)
        graph = _make_graph(
            _make_neuron(0, [0, 1]),
            _make_neuron(1, [2, 3, 4]),
        )
        # Without filter: synapse 4 is a false positive in cluster 1.
        m_raw = evaluate(graph, pre, post, min_root_synapses=0)
        # Threshold 3: roots 1 and 10 keep (count 2 each — wait recount).
        # pre col: 1,1,2,2,99 → counts 1:2, 2:2, 99:1
        # post col: 10,10,11,11,12 → counts 10:2, 11:2, 12:1
        # combined: 1:2, 2:2, 99:1, 10:2, 11:2, 12:1
        # threshold 2 keeps {1,2,10,11}; drops synapse 4 (pre=99, post=12).
        m_filt = evaluate(graph, pre, post, min_root_synapses=2)
        self.assertGreater(m_filt.f1, m_raw.f1)
        # Synapse 4 dropped → cluster 1 has only synapses 2,3 against true edge.
        self.assertEqual(m_filt.fp, 0)

    def test_keep_indices_helper(self):
        # pre col: 1,1,1,99 → counts 1:3, 99:1
        # post col: 10,10,10,10 → 10:4
        # threshold 3 keeps {1, 10}; drops synapse 3 (pre=99).
        pre = np.array([1, 1, 1, 99], dtype=np.int64)
        post = np.array([10, 10, 10, 10], dtype=np.int64)
        keep = _clutter_keep_indices(pre, post, min_root_synapses=3)
        self.assertEqual(sorted(keep.tolist()), [0, 1, 2])

    def test_evaluate_from_root_ids_filter(self):
        # 5 synapses: root 1 (4x pre) + root 99 (1x pre); root 10 (5x post).
        # combined counts: 1:4, 99:1, 10:5.
        # threshold 4 keeps {1, 10}; drops synapse 4 (pre=99).
        true_pre = np.array([1, 1, 1, 1, 99], dtype=np.int64)
        true_post = np.array([10, 10, 10, 10, 10], dtype=np.int64)
        est_pre = true_pre.copy()
        est_post = true_post.copy()
        m = evaluate_from_root_ids(
            est_pre, est_post, true_pre, true_post, min_root_synapses=4
        )
        # All filtered-in synapses agree perfectly → F1 == 1.0.
        self.assertAlmostEqual(m.f1, 1.0)
        self.assertEqual(m.n_synapses, 4)


if __name__ == "__main__":
    unittest.main()
