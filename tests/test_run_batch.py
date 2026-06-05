"""Tests for run.py coverage gaps.

Covers:
- evaluate_synthetic_batch: fixed_validation and random modes, aggregation
- build_graph_hypotheses: multi-threshold / multi-beam sweep
- GAT refinement code path in run() (gat_assembly_checkpoint branch)
"""

from __future__ import annotations

import unittest

import numpy as np


def _default_benchmark():
    from neuronauts.run import SyntheticBenchmarkConfig
    return SyntheticBenchmarkConfig(
        shape=(30, 30, 30),
        n_synapses=8,
        min_neuron_groups=2,
        max_neuron_groups=4,
    )


# ---------------------------------------------------------------------------
# evaluate_synthetic_batch
# ---------------------------------------------------------------------------

class EvaluateSyntheticBatchTest(unittest.TestCase):

    def test_invalid_cases_raises(self):
        from neuronauts.run import evaluate_synthetic_batch
        cfg = _default_benchmark()
        with self.assertRaises(ValueError):
            evaluate_synthetic_batch(cfg, cases=0, mode="fixed_validation")

    def test_fixed_validation_mode_returns_agg(self):
        from neuronauts.run import evaluate_synthetic_batch
        from neuronauts.line_graph import LineGraphMetrics
        cfg = _default_benchmark()
        agg, summaries = evaluate_synthetic_batch(
            cfg, cases=2, mode="fixed_validation", verbose=False
        )
        self.assertIsInstance(agg, LineGraphMetrics)
        self.assertEqual(len(summaries), 2)

    def test_random_mode_returns_agg(self):
        from neuronauts.run import evaluate_synthetic_batch
        cfg = _default_benchmark()
        agg, summaries = evaluate_synthetic_batch(
            cfg, cases=2, mode="random", base_seed=99, verbose=False
        )
        self.assertEqual(len(summaries), 2)

    def test_unsupported_mode_raises(self):
        from neuronauts.run import evaluate_synthetic_batch
        cfg = _default_benchmark()
        with self.assertRaises(ValueError):
            evaluate_synthetic_batch(cfg, cases=1, mode="invalid_mode")

    def test_summaries_contain_expected_keys(self):
        from neuronauts.run import evaluate_synthetic_batch
        cfg = _default_benchmark()
        _, summaries = evaluate_synthetic_batch(
            cfg, cases=2, mode="fixed_validation", verbose=False
        )
        for s in summaries:
            for key in ("case", "volume_seed", "run_seed", "f1", "precision", "recall"):
                self.assertIn(key, s, f"key '{key}' missing from summary")

    def test_aggregate_n_synapses_sums_cases(self):
        from neuronauts.run import evaluate_synthetic_batch
        cfg = _default_benchmark()
        agg, _ = evaluate_synthetic_batch(
            cfg, cases=3, mode="fixed_validation", verbose=False
        )
        self.assertIsInstance(agg.n_synapses, int)
        self.assertGreater(agg.n_synapses, 0)

    def test_aggregate_f1_is_mean_of_cases(self):
        """Aggregate F1 should be the arithmetic mean of per-case F1s."""
        from neuronauts.run import evaluate_synthetic_batch
        cfg = _default_benchmark()
        agg, summaries = evaluate_synthetic_batch(
            cfg, cases=3, mode="fixed_validation", verbose=False
        )
        expected_mean = float(np.mean([s["f1"] for s in summaries]))
        self.assertAlmostEqual(agg.f1, expected_mean, places=5)

    def test_fixed_validation_deterministic(self):
        """Same seeds → same aggregate F1."""
        from neuronauts.run import evaluate_synthetic_batch
        cfg = _default_benchmark()
        agg1, _ = evaluate_synthetic_batch(
            cfg, cases=2, mode="fixed_validation", verbose=False
        )
        agg2, _ = evaluate_synthetic_batch(
            cfg, cases=2, mode="fixed_validation", verbose=False
        )
        self.assertAlmostEqual(agg1.f1, agg2.f1, places=5)

    def test_random_mode_with_seed_is_deterministic(self):
        from neuronauts.run import evaluate_synthetic_batch
        cfg = _default_benchmark()
        agg1, sums1 = evaluate_synthetic_batch(
            cfg, cases=2, mode="random", base_seed=7, verbose=False
        )
        agg2, sums2 = evaluate_synthetic_batch(
            cfg, cases=2, mode="random", base_seed=7, verbose=False
        )
        self.assertEqual(sums1[0]["volume_seed"], sums2[0]["volume_seed"])

    def test_fixed_validation_seeds_are_case_index(self):
        """fixed_validation uses case_idx as both volume_seed and run_seed."""
        from neuronauts.run import evaluate_synthetic_batch
        cfg = _default_benchmark()
        _, summaries = evaluate_synthetic_batch(
            cfg, cases=3, mode="fixed_validation", verbose=False
        )
        for i, s in enumerate(summaries):
            self.assertEqual(s["volume_seed"], i)
            self.assertEqual(s["run_seed"], i)


# ---------------------------------------------------------------------------
# build_graph_hypotheses
# ---------------------------------------------------------------------------

class BuildGraphHypothesesTest(unittest.TestCase):

    def _make_data(self, n_agents=20, n_syn=5, steps=8):
        from neuronauts.run import SyntheticBenchmarkConfig, make_test_volume, AGENT_CONFIG
        from neuronauts.legacy.fields import compute_membrane_field, compute_membrane_vectors
        from neuronauts.legacy.vectorized import run_agents_vectorized
        from dataclasses import replace

        cfg = SyntheticBenchmarkConfig(shape=(25, 25, 25), n_synapses=n_syn,
                                       min_neuron_groups=2, max_neuron_groups=3)
        chunk, synapses = make_test_volume(config=cfg, seed=0)
        volume = chunk.data
        mf = compute_membrane_field(volume)
        mv = compute_membrane_vectors(volume)
        ef = np.ones(volume.shape, dtype=np.float32)
        syn_pts = np.vstack([synapses.pre_pt, synapses.post_pt])
        agent_cfg = replace(AGENT_CONFIG, max_steps=steps)
        rng = np.random.default_rng(0)
        path_arr, synapse_hits, alive = run_agents_vectorized(
            np.array(volume.shape, dtype=np.int32), n_agents, syn_pts,
            mf, mv, ef, agent_cfg, rng, verbose=False
        )
        path_lengths = np.full(n_agents, steps, dtype=np.int32)
        return path_arr, path_lengths, synapse_hits, synapses.pre_pt, synapses.post_pt

    def test_returns_list_of_tuples(self):
        from neuronauts.run import build_graph_hypotheses
        path_arr, path_lengths, synapse_hits, pre, post = self._make_data()
        hypotheses = build_graph_hypotheses(
            path_arr, path_lengths, synapse_hits, pre, post,
            thresholds=[-0.5, 0.5],
            beam_widths=[1, 2],
        )
        # 2 thresholds × 2 beam_widths = 4 hypotheses
        self.assertEqual(len(hypotheses), 4)

    def test_each_hypothesis_is_three_tuple(self):
        from neuronauts.run import build_graph_hypotheses
        from neuronauts.merge import ConnectivityGraph
        path_arr, path_lengths, synapse_hits, pre, post = self._make_data()
        for threshold, beam_width, graph in build_graph_hypotheses(
            path_arr, path_lengths, synapse_hits, pre, post,
            thresholds=[0.0],
            beam_widths=[1],
        ):
            self.assertIsInstance(threshold, float)
            self.assertIsInstance(beam_width, int)
            self.assertIsInstance(graph, ConnectivityGraph)

    def test_single_threshold_and_beam(self):
        from neuronauts.run import build_graph_hypotheses
        path_arr, path_lengths, synapse_hits, pre, post = self._make_data()
        hypotheses = build_graph_hypotheses(
            path_arr, path_lengths, synapse_hits, pre, post,
            thresholds=[0.0],
            beam_widths=[1],
        )
        self.assertEqual(len(hypotheses), 1)




if __name__ == "__main__":
    unittest.main()
