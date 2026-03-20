import tempfile
import unittest
from pathlib import Path

import numpy as np

from neuronauts.fetch import SynapseTable
from neuronauts.merge_dataset import MERGE_FEATURE_NAMES, build_merge_examples, examples_to_arrays, save_merge_examples_npz


class MergeLearningDatasetTest(unittest.TestCase):
    def test_build_merge_examples_contains_positive_and_negative_pairs(self):
        synapses = SynapseTable(
            pre_pt=np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [2, 0, 0],
                    [3, 0, 0],
                    [10, 0, 0],
                    [11, 0, 0],
                    [12, 0, 0],
                    [13, 0, 0],
                ],
                dtype=np.float32,
            ),
            post_pt=np.array(
                [
                    [0, 5, 0],
                    [1, 5, 0],
                    [2, 5, 0],
                    [3, 5, 0],
                    [10, 5, 0],
                    [11, 5, 0],
                    [12, 5, 0],
                    [13, 5, 0],
                ],
                dtype=np.float32,
            ),
            pre_root_id=np.array([101, 101, 101, 101, 202, 202, 202, 202], dtype=np.int64),
            post_root_id=np.array([301, 301, 301, 301, 402, 402, 402, 402], dtype=np.int64),
            synapse_id=np.arange(8, dtype=np.int64),
        )

        examples = build_merge_examples(
            synapses,
            min_fragment_size=2,
            max_negative_pairs_per_role=4,
        )

        self.assertTrue(examples)
        self.assertEqual({example.label for example in examples}, {0, 1})
        for example in examples:
            self.assertGreaterEqual(example.left_sequence.shape[1], 3 if len(example.left_sequence) else 0)
            if example.label == 1:
                self.assertEqual(example.left_root_ids, example.right_root_ids)
            else:
                self.assertNotEqual(example.left_root_ids, example.right_root_ids)

    def test_examples_to_arrays_shapes(self):
        synapses = SynapseTable(
            pre_pt=np.array(
                [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [10, 0, 0], [11, 0, 0], [12, 0, 0], [13, 0, 0]],
                dtype=np.float32,
            ),
            post_pt=np.array(
                [[0, 5, 0], [1, 5, 0], [2, 5, 0], [3, 5, 0], [10, 5, 0], [11, 5, 0], [12, 5, 0], [13, 5, 0]],
                dtype=np.float32,
            ),
            pre_root_id=np.array([101, 101, 101, 101, 202, 202, 202, 202], dtype=np.int64),
            post_root_id=np.array([301, 301, 301, 301, 402, 402, 402, 402], dtype=np.int64),
            synapse_id=np.arange(8, dtype=np.int64),
        )
        examples = build_merge_examples(synapses, min_fragment_size=2, max_negative_pairs_per_role=4)
        left_x, left_mask, right_x, right_mask, y = examples_to_arrays(examples)
        self.assertEqual(left_x.ndim, 3)
        self.assertEqual(right_x.ndim, 3)
        self.assertEqual(left_mask.shape[:1], y.shape)
        self.assertEqual(right_mask.shape[:1], y.shape)
        self.assertEqual(left_x.shape[2], len(MERGE_FEATURE_NAMES))
        self.assertEqual(right_x.shape[2], len(MERGE_FEATURE_NAMES))

    def test_save_merge_examples_npz(self):
        synapses = SynapseTable(
            pre_pt=np.array(
                [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [10, 0, 0], [11, 0, 0], [12, 0, 0], [13, 0, 0]],
                dtype=np.float32,
            ),
            post_pt=np.array(
                [[0, 5, 0], [1, 5, 0], [2, 5, 0], [3, 5, 0], [10, 5, 0], [11, 5, 0], [12, 5, 0], [13, 5, 0]],
                dtype=np.float32,
            ),
            pre_root_id=np.array([101, 101, 101, 101, 202, 202, 202, 202], dtype=np.int64),
            post_root_id=np.array([301, 301, 301, 301, 402, 402, 402, 402], dtype=np.int64),
            synapse_id=np.arange(8, dtype=np.int64),
        )
        examples = build_merge_examples(synapses, min_fragment_size=2, max_negative_pairs_per_role=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "merge_dataset.npz"
            save_merge_examples_npz(output, examples)
            loaded = np.load(output, allow_pickle=True)
            self.assertIn("left_x", loaded.files)
            self.assertIn("left_mask", loaded.files)
            self.assertIn("right_x", loaded.files)
            self.assertIn("right_mask", loaded.files)
            self.assertIn("y", loaded.files)
            self.assertEqual(tuple(loaded["feature_names"].tolist()), MERGE_FEATURE_NAMES)


if __name__ == "__main__":
    unittest.main()
