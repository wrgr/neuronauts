import math
import unittest

import numpy as np

from neuronauts.assembly import (
    CandidateMerge,
    beam_search_merge_groups,
    logit_to_probability,
    probability_to_log_odds,
    repartition_low_atomicity_group,
)


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


class ProbabilityConversionTest(unittest.TestCase):
    """Tests for logit_to_probability and probability_to_log_odds."""

    def test_logit_zero_gives_half(self):
        self.assertAlmostEqual(logit_to_probability(0.0), 0.5, places=6)

    def test_large_positive_logit_near_one(self):
        self.assertGreater(logit_to_probability(10.0), 0.999)

    def test_large_negative_logit_near_zero(self):
        self.assertLess(logit_to_probability(-10.0), 0.001)

    def test_temperature_flattens(self):
        """Higher temperature → closer to 0.5."""
        p_cold = logit_to_probability(3.0, temperature=0.5)
        p_warm = logit_to_probability(3.0, temperature=2.0)
        self.assertGreater(p_cold, p_warm)
        self.assertGreater(p_warm, 0.5)

    def test_temperature_sharpens(self):
        """Lower temperature → more extreme."""
        p_default = logit_to_probability(2.0, temperature=1.0)
        p_sharp = logit_to_probability(2.0, temperature=0.5)
        self.assertGreater(p_sharp, p_default)

    def test_roundtrip(self):
        """probability_to_log_odds inverts logit_to_probability at T=1."""
        for logit in [-3.0, -1.0, 0.0, 1.0, 5.0]:
            p = logit_to_probability(logit)
            recovered = probability_to_log_odds(p)
            self.assertAlmostEqual(recovered, logit, places=4)

    def test_probability_to_log_odds_clamps(self):
        """Edge cases near 0 and 1 don't raise."""
        lo = probability_to_log_odds(0.0)
        hi = probability_to_log_odds(1.0)
        self.assertTrue(math.isfinite(lo))
        self.assertTrue(math.isfinite(hi))
        self.assertLess(lo, -10.0)
        self.assertGreater(hi, 10.0)


class CandidateMergeProbabilityTest(unittest.TestCase):
    """Tests for the probability field on CandidateMerge."""

    def test_default_probability_is_half(self):
        c = CandidateMerge(left_agent=0, right_agent=1, score=0.0)
        self.assertEqual(c.probability, 0.5)

    def test_probability_stored(self):
        c = CandidateMerge(left_agent=0, right_agent=1, score=2.0, probability=0.88)
        self.assertAlmostEqual(c.probability, 0.88, places=5)

    def test_probability_from_logit(self):
        """Probability matches sigmoid of score when explicitly set."""
        score = 3.0
        p = logit_to_probability(score)
        c = CandidateMerge(left_agent=0, right_agent=1, score=score, probability=p)
        self.assertAlmostEqual(c.probability, 1.0 / (1.0 + np.exp(-score)), places=5)


class BeamSearchLogProbabilityTest(unittest.TestCase):
    """Tests for beam search with use_log_probability=True."""

    def test_log_probability_mode_runs(self):
        """Basic smoke test: log-probability beam search completes."""
        candidates = [
            CandidateMerge(0, 1, score=2.0, probability=0.88),
            CandidateMerge(1, 2, score=1.0, probability=0.73),
        ]
        groups = beam_search_merge_groups(
            [0, 1, 2],
            candidates,
            beam_width=2,
            use_log_probability=True,
        )
        self.assertTrue(len(groups) >= 1)

    def test_high_probability_favors_merge(self):
        """Candidates with high probability should be merged."""
        candidates = [
            CandidateMerge(0, 1, score=5.0, probability=0.99),
        ]
        groups = beam_search_merge_groups(
            [0, 1],
            candidates,
            beam_width=2,
            use_log_probability=True,
        )
        self.assertEqual(len(groups), 1)

    def test_low_probability_favors_reject(self):
        """Candidates with very low probability should be rejected."""
        candidates = [
            CandidateMerge(0, 1, score=-5.0, probability=0.01),
        ]
        groups = beam_search_merge_groups(
            [0, 1],
            candidates,
            beam_width=2,
            use_log_probability=True,
        )
        self.assertEqual(len(groups), 2)

    def test_log_prob_and_score_modes_agree_on_clear_cases(self):
        """Both modes should merge when probability is high."""
        candidates = [
            CandidateMerge(0, 1, score=4.0, probability=0.98),
            CandidateMerge(1, 2, score=3.5, probability=0.97),
        ]
        groups_score = beam_search_merge_groups(
            [0, 1, 2], candidates, beam_width=2, use_log_probability=False,
        )
        groups_prob = beam_search_merge_groups(
            [0, 1, 2], candidates, beam_width=2, use_log_probability=True,
        )
        self.assertEqual(len(groups_score), 1)
        self.assertEqual(len(groups_prob), 1)

    def test_atomicity_in_log_probability_mode(self):
        """Atomicity penalty prevents chain merge even in log-probability mode."""
        candidates = [
            CandidateMerge(0, 1, score=2.0, probability=0.88),
            CandidateMerge(1, 2, score=1.5, probability=0.82),
        ]
        groups = beam_search_merge_groups(
            [0, 1, 2],
            candidates,
            beam_width=2,
            use_log_probability=True,
            atomicity_score_fn=lambda members: -10.0 if len(members) >= 3 else 0.0,
            atomicity_weight=1.0,
        )
        self.assertEqual(len(groups), 2)


if __name__ == "__main__":
    unittest.main()
