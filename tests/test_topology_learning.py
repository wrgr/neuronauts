import tempfile
import unittest
from pathlib import Path

import numpy as np

from neuronauts.fetch import SynapseTable
from neuronauts.grammar import (
    ArborEncoder,
    PathEncoder,
    TorchArborEncoder,
    TorchMergeScorer,
    TorchPathEncoder,
    build_path_batch,
    load_torch_grammar_component,
    save_torch_grammar_component,
)
from neuronauts.topology_dataset import (
    BRANCH_FEATURE_NAME,
    build_cluster_examples,
    examples_to_branch_sequence_arrays,
    examples_to_multi_branch_arrays,
    save_multi_branch_npz,
)
from neuronauts.training_batches import pad_path_sequences

try:
    import torch
except ImportError:
    torch = None


class GrammarAndTopologyDatasetTest(unittest.TestCase):
    def test_path_encoder_captures_coarse_sequence_profile(self):
        encoder = PathEncoder(output_dim=32)
        early_heavy = build_path_batch(
            edge_len=[5.0, 5.0, 1.0, 1.0, 1.0, 1.0],
            radius=[1.0] * 6,
            curvature=[0.0] * 6,
        )
        late_heavy = build_path_batch(
            edge_len=[1.0, 1.0, 1.0, 1.0, 5.0, 5.0],
            radius=[1.0] * 6,
            curvature=[0.0] * 6,
        )

        emb_a = encoder.encode(early_heavy)
        emb_b = encoder.encode(late_heavy)

        self.assertEqual(emb_a.shape, (32,))
        self.assertEqual(emb_b.shape, (32,))
        self.assertFalse(np.allclose(emb_a, emb_b))

    def test_arbor_encoder_combines_mean_and_max_pooling(self):
        encoder = ArborEncoder(output_dim=64)
        embeddings = [
            np.ones(32, dtype=np.float32),
            np.full(32, 3.0, dtype=np.float32),
        ]

        encoded = encoder.encode(embeddings)

        self.assertEqual(encoded.shape, (64,))
        np.testing.assert_allclose(encoded[:32], 2.0, atol=1e-6)
        np.testing.assert_allclose(encoded[32:], 3.0, atol=1e-6)

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
            max_branches=4,
            seed=7,
        )

        self.assertGreaterEqual(len(examples), 4)
        self.assertEqual({example.label for example in examples}, {0, 1})
        self.assertTrue(all(example.branch_embeddings for example in examples))

        x, y, mask = examples_to_multi_branch_arrays(examples, max_branches=4)
        self.assertEqual(len(x), len(y))
        self.assertEqual(mask.shape, (len(examples), 4))
        self.assertEqual(x.ndim, 3)
        self.assertTrue(np.any(~mask))

        branch_x, branch_sequence_mask, branch_mask = examples_to_branch_sequence_arrays(examples, max_branches=4)
        self.assertEqual(branch_x.ndim, 4)
        self.assertEqual(branch_sequence_mask.shape[:2], branch_mask.shape)
        self.assertEqual(branch_x.shape[:2], branch_mask.shape)

    def test_save_multi_branch_npz(self):
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
            save_multi_branch_npz(output, examples, max_branches=5)
            loaded = np.load(output, allow_pickle=True)
            self.assertIn("x", loaded.files)
            self.assertIn("y", loaded.files)
            self.assertIn("mask", loaded.files)
            self.assertIn("branch_x", loaded.files)
            self.assertIn("branch_sequence_mask", loaded.files)
            self.assertIn("branch_mask", loaded.files)
            self.assertIn("feature_names", loaded.files)
            self.assertEqual(loaded["feature_names"].tolist(), [BRANCH_FEATURE_NAME])
            self.assertEqual(loaded["x"].shape[1], 5)


@unittest.skipIf(torch is None, "torch not installed")
class TorchGrammarModuleTest(unittest.TestCase):
    def test_pad_path_sequences_builds_dense_batch_and_mask(self):
        batch = pad_path_sequences(
            [
                np.array([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32),
                np.array([[3.0, 1.0, 0.0]], dtype=np.float32),
            ]
        )
        self.assertEqual(batch.x.shape, (2, 2, 3))
        self.assertEqual(batch.mask.shape, (2, 2))
        self.assertTrue(batch.mask[1, 1])
        self.assertFalse(batch.mask[0, 1])

    def test_torch_path_encoder_forward_shape(self):
        model = TorchPathEncoder(output_dim=16)
        x = torch.tensor(
            [
                [[1.0, 1.0, 0.0], [2.0, 1.0, 0.1], [0.0, 0.0, 0.0]],
                [[3.0, 1.0, 0.2], [4.0, 1.0, 0.3], [5.0, 1.0, 0.4]],
            ],
            dtype=torch.float32,
        )
        mask = torch.tensor([[False, False, True], [False, False, False]])
        out = model(x, mask=mask)
        self.assertEqual(tuple(out.shape), (2, 16))

    def test_torch_merge_scorer_forward_shape(self):
        model = TorchMergeScorer(embedding_dim=16)
        left = torch.randn(4, 16)
        right = torch.randn(4, 16)
        out = model(left, right)
        self.assertEqual(tuple(out.shape), (4,))

    def test_torch_arbor_encoder_forward_shape(self):
        model = TorchArborEncoder(embedding_dim=16, output_dim=20)
        x = torch.randn(3, 5, 16)
        mask = torch.tensor(
            [
                [False, False, True, True, True],
                [False, False, False, False, True],
                [False, True, True, True, True],
            ]
        )
        out = model(x, mask=mask)
        self.assertEqual(tuple(out.shape), (3, 20))

    def test_torch_grammar_component_round_trip(self):
        model = TorchPathEncoder(output_dim=12)
        model.eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "path_encoder.pt"
            save_torch_grammar_component(path, model)
            loaded = load_torch_grammar_component(path, TorchPathEncoder)
            x = torch.randn(2, 4, 3)
            mask = torch.tensor([[False, False, False, True], [False, False, True, True]])
            with torch.no_grad():
                expected = model(x, mask=mask)
                actual = loaded(x, mask=mask)
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
