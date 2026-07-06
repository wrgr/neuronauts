"""Offline tests for merge-aware constrained joining (no network)."""
import numpy as np

from experiments.proofread.merge_aware_join import (
    fragment_types, constrained_union_find, apply_partition, AXON, DEND, UNKNOWN)


def test_fragment_typing_and_contamination():
    # frag 1: 5 pre, 0 post -> axon; frag 2: 0 pre, 5 post -> dend;
    # frag 3: 4 pre, 4 post -> mixed + contaminated; frag 4: 1 pre -> unknown (too few)
    pre = [1] * 5 + [3] * 4 + [4] * 1
    post = [2] * 5 + [3] * 4
    types, pc, qc, contam = fragment_types(pre, post, dom=0.6, min_syn=2, contam_min=2)
    assert types[1] == AXON and types[2] == DEND
    assert types[3] == UNKNOWN and 3 in contam
    assert types[4] == UNKNOWN and 4 not in contam


def test_ad_veto_blocks_axon_dendrite_join():
    # axon frag 1 <-> dendrite frag 2, high-weight edge, no soma -> A↔D veto must reject
    pre = {1: 6}; post = {2: 6}
    edges = [(0.9, 1, 2, 0)]
    dsu, rej = constrained_union_find(
        edges, pre_count=pre, post_count=post, soma_frags=set(), contaminated=set(),
        area_of={1: 1.0, 2: 1.0})
    assert rej["ad"] == 1 and rej["committed"] == 0
    assert dsu.find(1) != dsu.find(2)          # deferred, not merged
    # with the veto disabled it commits
    dsu2, rej2 = constrained_union_find(
        edges, pre_count=pre, post_count=post, soma_frags=set(), contaminated=set(),
        area_of={1: 1.0, 2: 1.0}, use_ad=False)
    assert dsu2.find(1) == dsu2.find(2)


def test_ad_join_allowed_through_soma():
    # same axon<->dendrite join but one side is a soma fragment -> legal, commits
    edges = [(0.9, 1, 2, 1)]
    dsu, rej = constrained_union_find(
        edges, pre_count={1: 6}, post_count={2: 6}, soma_frags={1}, contaminated=set(),
        area_of={1: 1.0, 2: 1.0})
    assert dsu.find(1) == dsu.find(2) and rej["ad"] == 0


def test_two_soma_veto():
    edges = [(0.9, 1, 2, 0)]
    dsu, rej = constrained_union_find(
        edges, pre_count={}, post_count={}, soma_frags={1, 2}, contaminated=set(),
        area_of={1: 1.0, 2: 1.0})
    assert rej["soma"] == 1 and dsu.find(1) != dsu.find(2)


def test_caliber_and_quarantine_vetoes():
    # caliber: area ratio 5 > 2.5 -> reject
    dsu, rej = constrained_union_find(
        [(0.9, 1, 2, 0)], pre_count={}, post_count={}, soma_frags=set(),
        contaminated=set(), area_of={1: 1.0, 2: 5.0})
    assert rej["caliber"] == 1
    # quarantine: frag 3 is contaminated -> any edge touching it rejected
    dsu, rej = constrained_union_find(
        [(0.9, 3, 4, 0)], pre_count={}, post_count={}, soma_frags=set(),
        contaminated={3}, area_of={3: 1.0, 4: 1.0})
    assert rej["quarantine"] == 1


def test_confident_first_and_partition_application():
    # two good axon joins commit; apply_partition groups them
    edges = [(0.9, 1, 2, 1), (0.8, 2, 5, 1)]
    dsu, rej = constrained_union_find(
        edges, pre_count={1: 4, 2: 4, 5: 4}, post_count={}, soma_frags=set(),
        contaminated=set(), area_of={1: 1., 2: 1., 5: 1.})
    lab = apply_partition(dsu, [1, 2, 5, 9])
    assert lab[0] == lab[1] == lab[2]          # 1,2,5 merged
    assert lab[3] == 9                          # untouched maps to self
