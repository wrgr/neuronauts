import tempfile
import unittest
from pathlib import Path

import numpy as np

from neuronauts.fetch import SynapseTable
from neuronauts.topology_dataset import FEATURE_NAMES, build_cluster_examples, examples_to_arrays, save_examples_npz
from neuronauts.topology_model import load_logistic_model, save_logistic_model, train_logistic_model


class TopologyLearningTest(unittest.TestCase):
    def test_cluster_examples_include_atomic_and_non_atomic_labels(self):
        synapses = SynapseTable(
            pre_pt=np.array(
                [
                    [1, 1, 1],
                    [2, 1, 1],
                    [10, 10, 10],
                    [11, 10, 10],
                ],
                dtype=np.float32,
            ),
            post_pt=np.array(
                [
                    [1, 5, 1],
                    [2, 5, 1],
                    [10, 15, 10],
                    [11, 15, 10],
                ],
                dtype=np.float32,
            ),
            pre_root_id=np.array([101, 101, 202, 202], dtype=np.int64),
            post_root_id=np.array([301, 301, 402, 402], dtype=np.int64),
            synapse_id=np.arange(4, dtype=np.int64),
        )
        membrane = np.zeros((20, 20, 20), dtype=np.float32)

        examples = build_cluster_examples(
            synapses,
            membrane,
            min_cluster_size=2,
            max_negative_pairs_per_role=4,
            seed=7,
        )

        self.assertGreaterEqual(len(examples), 4)
        labels = {example.label for example in examples}
        self.assertEqual(labels, {0, 1})
        x, y = examples_to_arrays(examples)
        self.assertEqual(x.shape[1], len(FEATURE_NAMES))
        self.assertEqual(len(x), len(y))

    def test_logistic_model_round_trip(self):
        x = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.2],
                [2.0, 2.0],
                [2.2, 2.1],
            ],
            dtype=np.float32,
        )
        y = np.array([0, 0, 1, 1], dtype=np.int64)
        model, metrics = train_logistic_model(x, y, ["a", "b"])
        self.assertGreaterEqual(metrics["accuracy"], 0.5)
        probs = model.predict_proba(x)
        self.assertTrue(np.all(np.isfinite(probs)))

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.npz"
            save_logistic_model(model_path, model)
            loaded = load_logistic_model(model_path)
            np.testing.assert_allclose(model.predict_proba(x), loaded.predict_proba(x), atol=1e-6)

    def test_save_examples_npz(self):
        synapses = SynapseTable(
            pre_pt=np.array([[1, 1, 1], [2, 1, 1], [10, 10, 10], [11, 10, 10]], dtype=np.float32),
            post_pt=np.array([[1, 5, 1], [2, 5, 1], [10, 15, 10], [11, 15, 10]], dtype=np.float32),
            pre_root_id=np.array([101, 101, 202, 202], dtype=np.int64),
            post_root_id=np.array([301, 301, 402, 402], dtype=np.int64),
            synapse_id=np.arange(4, dtype=np.int64),
        )
        membrane = np.zeros((20, 20, 20), dtype=np.float32)
        examples = build_cluster_examples(synapses, membrane, min_cluster_size=2, max_negative_pairs_per_role=4, seed=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset.npz"
            save_examples_npz(output, examples)
            loaded = np.load(output, allow_pickle=True)
            self.assertIn("x", loaded.files)
            self.assertIn("y", loaded.files)
            self.assertIn("feature_names", loaded.files)
            self.assertEqual(loaded["x"].shape[1], len(FEATURE_NAMES))


if __name__ == "__main__":
    unittest.main()
