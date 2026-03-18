"""Additional tests for neuronauts/run.py covering gaps from the coverage audit.

Covers (HIGH / MEDIUM priority):
- R1: evaluate_synthetic_case — primary benchmark entry point
- R2: _build_bridge_graph with a bridge_score_fn (sign-convention check)
- R3: _load_shared_atomicity_score_fn — used in learned pipeline
"""

from __future__ import annotations

import tempfile
import unittest

import numpy as np


class EvaluateSyntheticCaseTest(unittest.TestCase):
    """R1 — HIGH: evaluate_synthetic_case is the outer benchmark entry point.

    A regression anywhere in run() or make_test_volume silently produces wrong
    metrics without this test.
    """

    def test_returns_line_graph_metrics_with_valid_f1(self):
        from neuronauts.fetch import SyntheticBenchmarkConfig
        from neuronauts.line_graph import LineGraphMetrics
        from neuronauts.run import evaluate_synthetic_case

        config = SyntheticBenchmarkConfig(
            shape=(32, 32, 32),
            n_synapses=8,
            anchor_margin=4,
            min_neuron_groups=2,
            max_neuron_groups=3,
        )
        result = evaluate_synthetic_case(config, volume_seed=0, run_seed=0, verbose=False)
        self.assertIsInstance(result, LineGraphMetrics)
        self.assertGreaterEqual(result.f1, 0.0)
        self.assertLessEqual(result.f1, 1.0)

    def test_precision_recall_non_negative(self):
        from neuronauts.fetch import SyntheticBenchmarkConfig
        from neuronauts.run import evaluate_synthetic_case

        config = SyntheticBenchmarkConfig(
            shape=(32, 32, 32),
            n_synapses=8,
            anchor_margin=4,
            min_neuron_groups=2,
            max_neuron_groups=3,
        )
        result = evaluate_synthetic_case(config, volume_seed=1, run_seed=1, verbose=False)
        self.assertGreaterEqual(result.precision, 0.0)
        self.assertGreaterEqual(result.recall, 0.0)
        self.assertLessEqual(result.precision, 1.0)
        self.assertLessEqual(result.recall, 1.0)

    def test_scaffold_vs_no_scaffold_both_return_metrics(self):
        from neuronauts.fetch import SyntheticBenchmarkConfig
        from neuronauts.line_graph import LineGraphMetrics
        from neuronauts.run import evaluate_synthetic_case

        config = SyntheticBenchmarkConfig(
            shape=(32, 32, 32),
            n_synapses=8,
            anchor_margin=4,
            min_neuron_groups=2,
            max_neuron_groups=3,
        )
        r_scaffold = evaluate_synthetic_case(
            config, volume_seed=2, run_seed=2, verbose=False, use_scaffold=True
        )
        r_no_scaffold = evaluate_synthetic_case(
            config, volume_seed=2, run_seed=2, verbose=False, use_scaffold=False
        )
        self.assertIsInstance(r_scaffold, LineGraphMetrics)
        self.assertIsInstance(r_no_scaffold, LineGraphMetrics)

    def test_different_seeds_produce_valid_results(self):
        from neuronauts.fetch import SyntheticBenchmarkConfig
        from neuronauts.run import evaluate_synthetic_case

        config = SyntheticBenchmarkConfig(
            shape=(32, 32, 32),
            n_synapses=8,
            anchor_margin=4,
            min_neuron_groups=2,
            max_neuron_groups=3,
        )
        for seed in (0, 5, 99):
            result = evaluate_synthetic_case(
                config, volume_seed=seed, run_seed=seed, verbose=False
            )
            self.assertGreaterEqual(result.f1, 0.0, f"seed={seed}")
            self.assertLessEqual(result.f1, 1.0, f"seed={seed}")

    def test_integer_counts_non_negative(self):
        from neuronauts.fetch import SyntheticBenchmarkConfig
        from neuronauts.run import evaluate_synthetic_case

        config = SyntheticBenchmarkConfig(
            shape=(32, 32, 32),
            n_synapses=10,
            anchor_margin=4,
            min_neuron_groups=2,
            max_neuron_groups=4,
        )
        result = evaluate_synthetic_case(config, verbose=False)
        self.assertGreaterEqual(result.tp, 0)
        self.assertGreaterEqual(result.fp, 0)
        self.assertGreaterEqual(result.fn, 0)


# ---------------------------------------------------------------------------
# R2 — _build_bridge_graph with bridge_score_fn
# ---------------------------------------------------------------------------

class BuildBridgeGraphWithScoreFnTest(unittest.TestCase):
    """R2 — MEDIUM: verify cost sign convention when bridge_score_fn is used.

    Positive logits → cost=0 (free edge).
    Negative logits → cost>0 (penalised edge).
    A sign-flip here silently routes bridges towards the wrong endpoints.
    """

    def _make_neurons(self, n_neurons: int = 3):
        """Return a minimal neurons dict as produced by _merge_role_groups."""
        from types import SimpleNamespace

        rng = np.random.default_rng(0)
        neurons = {}
        for i in range(n_neurons):
            pts = rng.random((8, 3), dtype=np.float32) * 100
            neurons[i] = SimpleNamespace(
                path_points=pts,
                role="pre",
            )
        return neurons

    def _iter_edges(self, bridge):
        """Yield unique (u, v, cost) from BridgeGraph._adj (deduplicated)."""
        seen = set()
        for u, neighbors in bridge._adj.items():
            for v, cost in neighbors:
                key = (min(u, v), max(u, v))
                if key not in seen:
                    seen.add(key)
                    yield u, v, cost

    def test_positive_logit_score_fn_gives_zero_cost_inter_edges(self):
        from neuronauts.run import _build_bridge_graph

        def always_positive(seq_a, seq_b):
            return 5.0  # large positive logit → cost should be 0

        neurons = self._make_neurons(3)
        bridge = _build_bridge_graph(neurons, bridge_score_fn=always_positive)
        for u, v, cost in self._iter_edges(bridge):
            if u // 2 != v // 2:  # inter-neuron edge
                self.assertAlmostEqual(
                    cost, 0.0, places=6,
                    msg=f"expected cost=0 for positive logit, got {cost} on edge ({u},{v})",
                )

    def test_negative_logit_score_fn_gives_positive_cost_inter_edges(self):
        from neuronauts.run import _build_bridge_graph

        def always_negative(seq_a, seq_b):
            return -3.0  # negative logit → cost = max(0, -(-3)) = 3

        neurons = self._make_neurons(3)
        bridge = _build_bridge_graph(neurons, bridge_score_fn=always_negative)
        inter_costs = [
            cost
            for u, v, cost in self._iter_edges(bridge)
            if u // 2 != v // 2
        ]
        self.assertTrue(len(inter_costs) > 0, "no inter-neuron edges produced")
        for cost in inter_costs:
            self.assertGreater(cost, 0.0, "expected positive cost for negative logit")

    def test_score_fn_cost_equals_max_zero_neg_logit(self):
        """cost = max(0, -logit) for each pair."""
        from neuronauts.run import _build_bridge_graph

        fixed_logit = -2.5
        expected_cost = 2.5

        def fixed(seq_a, seq_b):
            return fixed_logit

        neurons = self._make_neurons(2)
        bridge = _build_bridge_graph(neurons, bridge_score_fn=fixed)
        inter_costs = [
            cost
            for u, v, cost in self._iter_edges(bridge)
            if u // 2 != v // 2
        ]
        self.assertTrue(len(inter_costs) > 0)
        for cost in inter_costs:
            self.assertAlmostEqual(cost, expected_cost, places=5)

    def test_max_bridge_cost_prunes_high_cost_edges(self):
        from neuronauts.run import _build_bridge_graph

        def always_negative(seq_a, seq_b):
            return -100.0  # cost = 100

        neurons = self._make_neurons(3)
        bridge = _build_bridge_graph(
            neurons, bridge_score_fn=always_negative, max_bridge_cost=50.0
        )
        inter_costs = [
            cost
            for u, v, cost in self._iter_edges(bridge)
            if u // 2 != v // 2
        ]
        # All should be pruned since cost=100 > max=50.
        self.assertEqual(len(inter_costs), 0, "expected all inter edges pruned")

    def test_intra_neuron_edges_always_zero_cost(self):
        from neuronauts.run import _build_bridge_graph

        def always_negative(seq_a, seq_b):
            return -5.0

        neurons = self._make_neurons(2)
        bridge = _build_bridge_graph(neurons, bridge_score_fn=always_negative)
        for u, v, cost in self._iter_edges(bridge):
            if u // 2 == v // 2:  # intra-neuron
                self.assertAlmostEqual(cost, 0.0, places=6)


# ---------------------------------------------------------------------------
# R3 — _load_shared_atomicity_score_fn
# ---------------------------------------------------------------------------

class LoadSharedAtomicityScoreFnTest(unittest.TestCase):
    """R3 — MEDIUM: _load_shared_atomicity_score_fn is used in the learned pipeline.

    An error here falls through to atomicity_fn=None and silently disables the
    learned atomicity scorer.
    """

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

    def test_returns_callable(self):
        from neuronauts.run import _load_shared_atomicity_score_fn
        from neuronauts.shared_grammar_model import SharedGrammarModel, save_shared_grammar_model

        with tempfile.TemporaryDirectory() as d:
            ckpt = f"{d}/grammar.pt"
            model = SharedGrammarModel(embedding_dim=16)
            save_shared_grammar_model(ckpt, model)
            score_fn = _load_shared_atomicity_score_fn(ckpt)
            self.assertTrue(callable(score_fn))

    def test_returns_finite_float_on_valid_input(self):
        from neuronauts.run import _load_shared_atomicity_score_fn
        from neuronauts.shared_grammar_model import SharedGrammarModel, save_shared_grammar_model

        with tempfile.TemporaryDirectory() as d:
            ckpt = f"{d}/grammar.pt"
            model = SharedGrammarModel(embedding_dim=16)
            save_shared_grammar_model(ckpt, model)
            score_fn = _load_shared_atomicity_score_fn(ckpt)

            rng = np.random.default_rng(0)
            # branch_sequences: tuple of 2-D arrays (steps, 3)
            seqs = tuple(rng.random((5, 3)).astype(np.float32) for _ in range(3))
            result = score_fn(seqs)
            self.assertIsInstance(result, float)
            self.assertFalse(np.isnan(result), "score is NaN")
            self.assertFalse(np.isinf(result), "score is Inf")

    def test_score_fn_accepts_variable_branch_counts(self):
        from neuronauts.run import _load_shared_atomicity_score_fn
        from neuronauts.shared_grammar_model import SharedGrammarModel, save_shared_grammar_model

        with tempfile.TemporaryDirectory() as d:
            ckpt = f"{d}/grammar.pt"
            model = SharedGrammarModel(embedding_dim=16)
            save_shared_grammar_model(ckpt, model)
            score_fn = _load_shared_atomicity_score_fn(ckpt)
            rng = np.random.default_rng(1)
            for n_branches in (1, 2, 5):
                seqs = tuple(rng.random((4, 3)).astype(np.float32) for _ in range(n_branches))
                result = score_fn(seqs)
                self.assertIsInstance(result, float, f"failed for n_branches={n_branches}")

    def test_two_calls_with_same_input_are_deterministic(self):
        """eval() mode means same input → same output."""
        from neuronauts.run import _load_shared_atomicity_score_fn
        from neuronauts.shared_grammar_model import SharedGrammarModel, save_shared_grammar_model

        with tempfile.TemporaryDirectory() as d:
            ckpt = f"{d}/grammar.pt"
            model = SharedGrammarModel(embedding_dim=16)
            save_shared_grammar_model(ckpt, model)
            score_fn = _load_shared_atomicity_score_fn(ckpt)
            rng = np.random.default_rng(2)
            seqs = tuple(rng.random((6, 3)).astype(np.float32) for _ in range(2))
            r1 = score_fn(seqs)
            r2 = score_fn(seqs)
            self.assertAlmostEqual(r1, r2, places=6)


if __name__ == "__main__":
    unittest.main()
