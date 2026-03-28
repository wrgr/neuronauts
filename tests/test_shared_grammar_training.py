import tempfile
import unittest
from pathlib import Path

import numpy as np

from neuronauts.fetch import SynapseTable
from neuronauts.merge_dataset import examples_to_arrays as merge_examples_to_arrays
from neuronauts.merge_dataset import build_merge_examples
from neuronauts.shared_grammar_model import (
    SharedGrammarModel,
    load_shared_grammar_model,
    multitask_train_step,
    save_shared_grammar_model,
)
from neuronauts.topology_dataset import build_cluster_examples, examples_to_branch_sequence_arrays

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class SharedGrammarTrainingTest(unittest.TestCase):
    def _make_synapses(self):
        return SynapseTable(
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

    def test_multitask_train_step_updates_shared_encoder(self):
        synapses = self._make_synapses()
        merge_examples = build_merge_examples(synapses, min_fragment_size=2, max_negative_pairs_per_role=4)
        left_x, left_mask, right_x, right_mask, merge_y = merge_examples_to_arrays(merge_examples)

        topology_examples = build_cluster_examples(
            synapses,
            membrane_field=np.zeros((20, 20, 20), dtype=np.float32),
            min_cluster_size=2,
            max_negative_pairs_per_role=4,
            max_branches=4,
            seed=7,
        )
        branch_x, branch_sequence_mask, branch_mask = examples_to_branch_sequence_arrays(topology_examples, max_branches=4)
        topology_y = np.array([example.label for example in topology_examples], dtype=np.float32)

        model = SharedGrammarModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = model.path_encoder.input_proj.weight.detach().clone()

        metrics = multitask_train_step(
            model,
            optimizer,
            merge_batch={
                "left_x": torch.from_numpy(left_x),
                "left_mask": torch.from_numpy(left_mask),
                "right_x": torch.from_numpy(right_x),
                "right_mask": torch.from_numpy(right_mask),
                "y": torch.from_numpy(merge_y.astype(np.float32)),
            },
            topology_batch={
                "branch_x": torch.from_numpy(branch_x),
                "branch_sequence_mask": torch.from_numpy(branch_sequence_mask),
                "branch_mask": torch.from_numpy(branch_mask),
                "y": torch.from_numpy(topology_y),
            },
        )

        after = model.path_encoder.input_proj.weight.detach().clone()
        self.assertIn("loss", metrics)
        self.assertFalse(torch.equal(before, after))

    def test_shared_grammar_model_round_trip(self):
        model = SharedGrammarModel()
        model.eval()
        D = model._init_kwargs["input_dim"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "shared_grammar.pt"
            save_shared_grammar_model(path, model)
            loaded = load_shared_grammar_model(path)
            left_x = torch.randn(2, 4, D)
            left_mask = torch.tensor([[False, False, False, True], [False, False, True, True]])
            right_x = torch.randn(2, 4, D)
            right_mask = torch.tensor([[False, False, False, True], [False, False, True, True]])
            with torch.no_grad():
                expected = model.score_merge(left_x, left_mask, right_x, right_mask)
                actual = loaded.score_merge(left_x, left_mask, right_x, right_mask)
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
