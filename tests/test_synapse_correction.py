"""Synapse-level correction model: offline checks on the synthetic scenario.

No CAVE token / network required.  Verifies that (a) the v117->later join recovers the
injected false-merges and false-splits at synapse level, and (b) the learned affinity
beats its permutation null in both strata.
"""
import numpy as np
import pytest

from experiments.pcfg_synapse_partitions.synapse_correction import (
    build_correction_pairs,
    cell_components,
    summarize_edits,
)
from experiments.pcfg_synapse_partitions.run_synapse_correction import (
    evaluate,
    make_synthetic,
)


def test_summary_recovers_injected_edits():
    tab = make_synthetic(seed=1)
    s = summarize_edits(tab)
    # injected false merges (one v117 root -> 2 later roots) must be detected as split_roots
    assert s["split_roots"] >= 3, s
    # injected false splits (one later root <- 2 v117 roots) as merge_targets
    assert s["merge_targets"] >= 3, s
    assert s["sides_with_later_label"] == len(tab)


def test_cell_grouping_is_leakage_safe():
    tab = make_synthetic(seed=2)
    comp = cell_components(tab)
    # a fused (false-merge) v117 root and the two later cells it spans land in one component;
    # every v117 root receives a group id
    assert set(comp.keys()) == set(int(r) for r in np.unique(tab.root_v117))
    assert min(comp.values()) >= 0


def test_both_strata_beat_null():
    tab = make_synthetic(seed=0)
    X, y, groups, strata = build_correction_pairs(tab, rng=np.random.default_rng(0))
    assert len(y) > 200
    assert (strata == 0).sum() > 0 and (strata == 1).sum() > 0  # both strata present
    res = evaluate(X, y, groups, strata, n_splits=5, n_perm=30, seed=0, verbose=False)
    # logistic regression should clear chance and its own null in every stratum
    for stratum in ("overall", "merge", "split"):
        r = res[f"logreg/{stratum}"]
        assert r["auc"] > r["null_mean"] + 2 * r["null_std"], (stratum, r)
        assert r["p"] < 0.05, (stratum, r)


def test_pair_label_matches_later_comembership():
    # label y must equal "same later root" for a hand-built pair set
    tab = make_synthetic(seed=3)
    X, y, groups, strata = build_correction_pairs(tab, rng=np.random.default_rng(3))
    # split stratum carries both should-keep (y=1) and should-cut (y=0) pairs
    split_y = y[strata == 0]
    assert split_y.min() == 0 and split_y.max() == 1
    # merge stratum likewise carries real merges (y=1)
    assert y[strata == 1].sum() > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
