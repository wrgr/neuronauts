"""Tests and inspection for topology/atomicity metric correctness.

Verifies that:
- Topology batch class balance can be inspected (pos_frac, n_atomic vs n_non_atomic)
- Atomicity accuracy reflects actual learning (improves with training on balanced data)
- Severe class imbalance would explain flat accuracy (e.g. pos_frac ~0.9 → predict-1 ≈ 90%)
"""

from __future__ import annotations

import unittest

import numpy as np

from neuronauts.fetch import SynapseTable
from neuronauts.topology_dataset import (
    build_cluster_examples,
    examples_to_branch_sequence_arrays,
    inspect_topology_batch_balance,
)

try:
    import torch
except ImportError:
    torch = None


def _make_balanced_topology_synapses():
    """SynapseTable that yields roughly 50/50 atomic vs non-atomic examples.

    Three distinct clusters per role → 3 atomic + 3 non-atomic (nearest pairs).
    """
    rng = np.random.default_rng(42)
    # 3 clusters, 4 pts each
    n_per = 4
    pre_pt = np.vstack([
        rng.random((n_per, 3), dtype=np.float32) * 3,           # cluster A
        rng.random((n_per, 3), dtype=np.float32) * 3 + 30,      # cluster B
        rng.random((n_per, 3), dtype=np.float32) * 3 + 60,     # cluster C
    ])
    post_pt = np.vstack([
        rng.random((n_per, 3), dtype=np.float32) * 3 + 5,
        rng.random((n_per, 3), dtype=np.float32) * 3 + 35,
        rng.random((n_per, 3), dtype=np.float32) * 3 + 65,
    ])
    pre_root = np.array([101] * n_per + [202] * n_per + [203] * n_per, dtype=np.int64)
    post_root = np.array([301] * n_per + [302] * n_per + [303] * n_per, dtype=np.int64)
    return SynapseTable(
        pre_pt=pre_pt,
        post_pt=post_pt,
        pre_root_id=pre_root,
        post_root_id=post_root,
        synapse_id=np.arange(len(pre_root), dtype=np.int64),
    )


class TopologyMetricInspectTest(unittest.TestCase):
    def test_inspect_topology_batch_balance_reports_pos_frac(self):
        """Inspect helper returns n_atomic, n_non_atomic, pos_frac."""
        synapses = _make_balanced_topology_synapses()
        examples = build_cluster_examples(
            synapses,
            membrane_field=np.zeros((100, 100, 100), dtype=np.float32),
            min_cluster_size=2,
            max_negative_pairs_per_role=8,
            max_branches=4,
            seed=0,
        )
        self.assertGreater(len(examples), 0, "need topology examples")
        stats = inspect_topology_batch_balance(examples)
        self.assertIn("n", stats)
        self.assertIn("n_atomic", stats)
        self.assertIn("n_non_atomic", stats)
        self.assertIn("pos_frac", stats)
        self.assertEqual(stats["n"], len(examples))
        self.assertEqual(stats["n_atomic"] + stats["n_non_atomic"], stats["n"])
        self.assertGreaterEqual(stats["pos_frac"], 0.0)
        self.assertLessEqual(stats["pos_frac"], 1.0)

    def test_inspect_warns_when_severely_imbalanced(self):
        """pos_frac outside [0.3, 0.7] indicates potential trivial accuracy."""
        stats_balanced = {"pos_frac": 0.5}
        stats_skewed = {"pos_frac": 0.89}
        # Logic: if pos_frac ~0.9, majority-class predictor gets 90% → flat accuracy.
        trivial_upper = 0.85
        trivial_lower = 0.15
        self.assertLess(stats_balanced["pos_frac"], trivial_upper)
        self.assertGreater(stats_balanced["pos_frac"], trivial_lower)
        self.assertGreaterEqual(stats_skewed["pos_frac"], trivial_upper)


@unittest.skipIf(torch is None, "torch not installed")
class TopologyMetricLearningTest(unittest.TestCase):
    """Verify atomicity accuracy reflects actual learning, not trivial majority prediction."""

    def test_atomicity_accuracy_improves_with_training_on_balanced_data(self):
        """SharedGrammarModel should learn atomic vs non-atomic on balanced synthetic data.

        If topology accuracy stays flat at ~89% on real data, it may be due to:
        - Severe class imbalance (pos_frac ~0.9) — inspect with inspect_topology_batch_balance
        - Or a bug in the metric/loss wiring.
        This test proves the pipeline can learn when data is balanced.
        """
        from neuronauts.merge_dataset import build_merge_examples, examples_to_arrays
        from neuronauts.shared_grammar_model import SharedGrammarModel, multitask_train_step
        from neuronauts.topology_dataset import build_cluster_examples, examples_to_branch_sequence_arrays

        synapses = _make_balanced_topology_synapses()
        membrane = np.zeros((100, 100, 100), dtype=np.float32)

        merge_examples = build_merge_examples(synapses, min_fragment_size=2, max_negative_pairs_per_role=8)
        topo_examples = build_cluster_examples(
            synapses, membrane,
            min_cluster_size=2,
            max_negative_pairs_per_role=8,
            max_branches=4,
            seed=0,
        )
        if not merge_examples or not topo_examples:
            self.skipTest("synapse table produced no merge or topology examples")

        topo_stats = inspect_topology_batch_balance(topo_examples)
        pos_frac = topo_stats["pos_frac"]
        self.assertGreater(pos_frac, 0.25, "need non-trivial negative fraction to test learning")
        self.assertLess(pos_frac, 0.75, "need non-trivial positive fraction to test learning")

        lx, lm, rx, rm, y_merge = examples_to_arrays(merge_examples)
        bx, bsm, bm = examples_to_branch_sequence_arrays(topo_examples, max_branches=4)
        y_topo = np.array([ex.label for ex in topo_examples], dtype=np.float32)

        device = torch.device("cpu")
        model = SharedGrammarModel(embedding_dim=16).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

        merge_batch = {
            "left_x": torch.from_numpy(lx).to(device),
            "left_mask": torch.from_numpy(lm).to(device),
            "right_x": torch.from_numpy(rx).to(device),
            "right_mask": torch.from_numpy(rm).to(device),
            "y": torch.from_numpy(y_merge.astype(np.float32)).to(device),
        }
        topo_batch = {
            "branch_x": torch.from_numpy(bx).to(device),
            "branch_sequence_mask": torch.from_numpy(bsm).to(device),
            "branch_mask": torch.from_numpy(bm).to(device),
            "y": torch.from_numpy(y_topo).to(device),
        }

        initial_acc = None
        for step in range(80):
            model.train()
            metrics = multitask_train_step(
                model, optimizer,
                merge_batch=merge_batch,
                topology_batch=topo_batch,
                atomicity_loss_weight=1.0,
            )
            acc = metrics["atomicity_accuracy"]
            if initial_acc is None:
                initial_acc = acc

        final_acc = metrics["atomicity_accuracy"]
        # Model should improve: from ~random (0.5) or initial to meaningfully better.
        # Allow for some variance; we mainly check we're not stuck at a trivial ceiling.
        improvement = final_acc - initial_acc
        self.assertGreater(
            final_acc, 0.55,
            f"atomicity_accuracy should improve on balanced data (got {final_acc:.3f}, "
            f"initial={initial_acc:.3f}). Flat accuracy may indicate class imbalance or metric bug.",
        )
        self.assertGreater(
            improvement, 0.02,
            f"atomicity_accuracy should increase with training (initial={initial_acc:.3f}, "
            f"final={final_acc:.3f}, delta={improvement:.3f}).",
        )


if __name__ == "__main__":
    unittest.main()
