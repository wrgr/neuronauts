"""A box-local proposer must be scored on links reachable inside the box.

The motivating case is real: proofread cell 864691136011850926's full skeleton
is one connected component, but clipped to the 100 um harness cube it is eight,
because its axon leaves through the +y face, runs 90-220 um outside and comes
back. Six of the seven floating pieces are axon.
"""

import numpy as np
import pytest

from neuronauts.harness.box_truth import (
    box_components, restrict_links, spanning_target,
)

LO, HI = [-1, -1, -1], [50, 50, 50]


def _chain(positions, fragments):
    pos = np.array([[p, 0, 0] for p in positions], float)
    e = np.array([[i, i + 1] for i in range(len(positions) - 1)])
    return e, pos, np.asarray(fragments, np.int64)


def test_path_leaving_the_box_splits_two_fragments():
    """Both fragments sit inside; the only path between them detours outside."""
    e, pos, frag = _chain([0, 1, 100, 5, 6], [10, 10, 0, 20, 20])
    bt = box_components(e, pos, frag, LO, HI)
    assert bt.components == [[10], [20]]
    assert bt.n_fragments == 2
    assert spanning_target(bt) == []          # nothing joinable in this box


def test_same_pair_joins_when_the_box_holds_the_path():
    e, pos, frag = _chain([0, 1, 100, 5, 6], [10, 10, 0, 20, 20])
    bt = box_components(e, pos, frag, [-1, -1, -1], [200, 200, 200])
    assert bt.components == [[10, 20]]
    assert spanning_target(bt) == [[10, 20]]
    assert bt.frac_in_largest == 1.0


def test_a_fragment_straddling_the_edge_is_not_split_from_itself():
    """One fragment touching two in-box node components must stay one fragment."""
    e, pos, frag = _chain([0, 100, 1], [10, 10, 10])
    assert box_components(e, pos, frag, LO, HI).components == [[10]]


def test_largest_discards_a_smaller_joinable_component():
    """Two groups, each joinable inside the box, connected only outside it.

    {10,20} is a 2-fragment group and {30,40,50} a 3-fragment group. Both are
    real in-box targets; ``largest`` keeps only the bigger one, which is the
    43.7% of fragments that mode discards on the real 40-cell measurement.
    """
    pos = np.array([[0, 0, 0], [1, 0, 0], [100, 0, 0],
                    [5, 0, 0], [6, 0, 0], [7, 0, 0]], float)
    e = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    frag = np.array([10, 20, 0, 30, 40, 50], np.int64)
    bt = box_components(e, pos, frag, LO, HI)
    assert bt.components == [[30, 40, 50], [10, 20]]
    assert spanning_target(bt, mode="all_components") == [[30, 40, 50], [10, 20]]
    assert spanning_target(bt, mode="largest") == [[30, 40, 50]]
    assert bt.dropped == [10, 20]


def test_restrict_links_keeps_only_pairs_inside_the_target():
    links = {(1, 2), (2, 3), (3, 4)}
    assert restrict_links(links, [1, 2, 3]) == {(1, 2), (2, 3)}


def test_unknown_mode_is_rejected():
    e, pos, frag = _chain([0, 1], [10, 10])
    with pytest.raises(ValueError):
        spanning_target(box_components(e, pos, frag, LO, HI), mode="biggest")


def test_seeded_target_is_the_seeds_own_component():
    from neuronauts.harness.box_truth import seeded_target
    pos = np.array([[0, 0, 0], [1, 0, 0], [100, 0, 0],
                    [5, 0, 0], [6, 0, 0], [7, 0, 0]], float)
    e = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    frag = np.array([10, 20, 0, 30, 40, 50], np.int64)
    bt = box_components(e, pos, frag, LO, HI)
    assert seeded_target(bt, 30) == [30, 40, 50]      # the soma's own piece
    assert seeded_target(bt, 10) == [10, 20]          # a different root process
    assert seeded_target(bt, 999) == []               # no such fragment in the box


def test_seeded_target_singleton_has_nothing_to_grow_into():
    from neuronauts.harness.box_truth import seeded_target
    e, pos, frag = _chain([0, 1, 100, 5, 6], [10, 10, 0, 20, 20])
    bt = box_components(e, pos, frag, LO, HI)
    assert seeded_target(bt, 10) == []


def test_crosses_compartment_only_for_two_different_non_soma_compartments():
    from neuronauts.harness.box_truth import crosses_compartment
    assert crosses_compartment("axon", "dendrite")
    assert crosses_compartment("dendrite", "axon")
    assert not crosses_compartment("axon", "axon")
    assert not crosses_compartment("dendrite", "dendrite")
    assert not crosses_compartment("soma", "axon")      # everything attaches to the soma
    assert not crosses_compartment("axon", "soma")
    assert not crosses_compartment(None, "axon")        # unknown: keep, do not guess
    assert not crosses_compartment("?", "dendrite")


def test_drop_crossing_links_partitions_and_keeps_unknowns():
    from neuronauts.harness.box_truth import drop_crossing_links
    links = [{"compartment_a": "axon", "compartment_b": "axon", "gap_nm": 900},
             {"compartment_a": "axon", "compartment_b": "dendrite", "gap_nm": 7600},
             {"compartment_a": "soma", "compartment_b": "axon", "gap_nm": 400},
             {"compartment_a": None, "compartment_b": "axon", "gap_nm": 500}]
    kept, dropped = drop_crossing_links(links)
    assert [l["gap_nm"] for l in kept] == [900, 400, 500]
    assert [l["gap_nm"] for l in dropped] == [7600]
    assert len(kept) + len(dropped) == len(links)
