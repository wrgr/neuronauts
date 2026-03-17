import unittest

from neuronauts.assembly import CandidateMerge, beam_search_merge_groups


class AssemblySearchTest(unittest.TestCase):
    def test_beam_search_can_reject_chain_merge_when_atomicity_penalizes_large_cluster(self):
        groups = beam_search_merge_groups(
            [0, 1, 2],
            [
                CandidateMerge(0, 1, 0.8),
                CandidateMerge(1, 2, 0.7),
            ],
            beam_width=2,
            atomicity_score_fn=lambda members: -10.0 if len(members) >= 3 else 0.0,
            atomicity_weight=1.0,
        )
        self.assertEqual(len(groups), 2)

    def test_beam_search_can_favor_full_merge_when_atomicity_rewards_large_cluster(self):
        groups = beam_search_merge_groups(
            [0, 1, 2],
            [
                CandidateMerge(0, 1, 0.8),
                CandidateMerge(1, 2, 0.7),
            ],
            beam_width=2,
            atomicity_score_fn=lambda members: 1.0 if len(members) >= 3 else 0.0,
            atomicity_weight=1.0,
        )
        self.assertEqual(len(groups), 1)


if __name__ == "__main__":
    unittest.main()
