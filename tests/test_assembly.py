import unittest

from neuronauts.assembly import CandidateMerge, beam_search_merge_groups, repartition_low_atomicity_group


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

    def test_repartition_low_atomicity_group_splits_by_pairwise_affinity(self):
        groups = repartition_low_atomicity_group(
            (0, 1, 2),
            pair_score_fn=lambda left, right: 2.0 if {left, right} == {0, 1} else -1.0,
            atomicity_score_fn=lambda members: -2.0 if len(members) >= 3 else 1.0,
            atomicity_threshold=0.0,
            min_group_size=3,
            max_rounds=2,
        )
        self.assertEqual(groups, ((0, 1), (2,)))

    def test_repartition_low_atomicity_group_keeps_high_atomicity_group_intact(self):
        groups = repartition_low_atomicity_group(
            (0, 1, 2),
            pair_score_fn=lambda left, right: 2.0,
            atomicity_score_fn=lambda members: 1.0,
            atomicity_threshold=0.0,
            min_group_size=3,
            max_rounds=2,
        )
        self.assertEqual(groups, ((0, 1, 2),))


if __name__ == "__main__":
    unittest.main()
