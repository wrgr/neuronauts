import tempfile
import unittest
from pathlib import Path

import numpy as np

from neuronauts.assembly_dataset import (
    ASSEMBLY_FEATURE_NAMES,
    build_hypothesis_examples,
    examples_to_arrays,
    hypothesis_features,
    save_hypothesis_examples_npz,
)
from neuronauts.hypothesis_reranker import load_linear_reranker, save_linear_reranker, train_linear_reranker
from neuronauts.line_graph import LineGraphMetrics
from neuronauts.merge import ConnectivityGraph, MergedNeuron
from neuronauts.run import select_hypothesis_with_reranker


class AssemblyRankingTest(unittest.TestCase):
    def test_build_hypothesis_examples_marks_best_hypothesis(self):
        examples = build_hypothesis_examples(
            "box_0",
            [
                ("h0", np.array([0.0] * len(ASSEMBLY_FEATURE_NAMES), dtype=np.float32), LineGraphMetrics(0, 0, 0, 0.0, 0.0, 0.25, 0, 0, 0)),
                ("h1", np.array([1.0] * len(ASSEMBLY_FEATURE_NAMES), dtype=np.float32), LineGraphMetrics(0, 0, 0, 0.0, 0.0, 0.75, 0, 0, 0)),
            ],
        )
        self.assertEqual(len(examples), 2)
        self.assertEqual(sum(example.is_best for example in examples), 1)

    def test_hypothesis_dataset_and_reranker_round_trip(self):
        examples = build_hypothesis_examples(
            "box_0",
            [
                ("h0", np.array([0.0] * len(ASSEMBLY_FEATURE_NAMES), dtype=np.float32), LineGraphMetrics(0, 0, 0, 0.0, 0.0, 0.1, 0, 0, 0)),
                ("h1", np.array([1.0] * len(ASSEMBLY_FEATURE_NAMES), dtype=np.float32), LineGraphMetrics(0, 0, 0, 0.0, 0.0, 0.9, 0, 0, 0)),
            ],
        )
        x, y_f1, y_best = examples_to_arrays(examples)
        self.assertEqual(x.shape[1], len(ASSEMBLY_FEATURE_NAMES))
        self.assertEqual(y_best.tolist(), [0, 1])

        model, metrics = train_linear_reranker(x, y_f1, list(ASSEMBLY_FEATURE_NAMES))
        self.assertIn("mse", metrics)
        preds = model.predict(x)
        self.assertEqual(preds.shape, y_f1.shape)

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "assembly_dataset.npz"
            save_hypothesis_examples_npz(dataset_path, examples)
            loaded = np.load(dataset_path, allow_pickle=True)
            self.assertEqual(loaded["x"].shape[1], len(ASSEMBLY_FEATURE_NAMES))

            model_path = Path(tmpdir) / "reranker.npz"
            save_linear_reranker(model_path, model)
            loaded_model = load_linear_reranker(model_path)
            np.testing.assert_allclose(loaded_model.predict(x), preds, atol=1e-6)

    def test_select_hypothesis_with_reranker_prefers_higher_predicted_score(self):
        graph_a = ConnectivityGraph(
            neurons={0: MergedNeuron(0, [0], np.zeros((2, 3), dtype=np.float32), [0, 1])},
            edges=[],
            unresolved_synapse_indices=[0, 1, 2],
        )
        graph_b = ConnectivityGraph(
            neurons={
                0: MergedNeuron(0, [0], np.zeros((2, 3), dtype=np.float32), [0]),
                1: MergedNeuron(1, [1], np.zeros((2, 3), dtype=np.float32), [1]),
            },
            edges=[],
            unresolved_synapse_indices=[0],
        )
        x = np.stack(
            [
                hypothesis_features(graph_a, merge_threshold=-0.5, beam_width=1, n_synapses=3),
                hypothesis_features(graph_b, merge_threshold=0.5, beam_width=4, n_synapses=3),
            ],
            axis=0,
        )
        y = np.array([0.1, 0.9], dtype=np.float32)
        model, _ = train_linear_reranker(x, y, list(ASSEMBLY_FEATURE_NAMES))
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "reranker.npz"
            save_linear_reranker(model_path, model)
            threshold, beam_width, _ = select_hypothesis_with_reranker(
                [
                    (-0.5, 1, graph_a),
                    (0.5, 4, graph_b),
                ],
                reranker_checkpoint=str(model_path),
                n_synapses=3,
            )
            self.assertEqual((threshold, beam_width), (0.5, 4))


if __name__ == "__main__":
    unittest.main()
