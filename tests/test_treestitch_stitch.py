"""Tests for treestitch.stitch — level-1 seam stitching."""

from __future__ import annotations

import numpy as np
import pytest

from neuronauts.schemas import Fragment
from treestitch.stitch import (
    StitchCandidate,
    build_super_fragments,
    candidate_stitch_edges,
    link_shared_observations,
    pairwise_merge_metrics,
    stitch_edge_precision,
    stitch_super_fragments,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_line_fragment(fid: int, start, direction, n: int = 5,
                       step: float = 1_000.0, dna=None) -> Fragment:
    """A straight-line skeleton fragment of n vertices."""
    start = np.asarray(start, dtype=np.float32)
    d = np.asarray(direction, dtype=np.float32)
    d = d / (np.linalg.norm(d) + 1e-9)
    verts = np.stack([start + d * step * i for i in range(n)]).astype(np.float32)
    edges = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int64)
    return Fragment(
        fragment_id=fid,
        region_id="test",
        base_root_id=fid,
        vertices_nm=verts,
        edges=edges,
        endpoints_nm=verts[[0, -1]],
        radius_nm=np.full(n, 100.0, dtype=np.float32),
        synapse_indices=np.zeros(0, dtype=np.int64),
        dna=None if dna is None else np.asarray(dna, dtype=np.float32),
    ).validate()


def make_super(tile_id, cluster_id, frag, *, obs_keys=(), n_obs=None,
               dna=None, has_soma=False, majority_label=0):
    from treestitch.stitch import SuperFragment
    keys = np.asarray(list(obs_keys), dtype=np.int64)
    return SuperFragment(
        tile_id=tile_id,
        cluster_id=cluster_id,
        atom_ids=frozenset([frag.base_root_id]),
        skeleton=frag,
        dna=None if dna is None else np.asarray(dna, dtype=np.float32),
        n_obs=n_obs if n_obs is not None else max(len(keys), 1),
        obs_keys=keys,
        has_soma=has_soma,
        majority_label=majority_label,
    )


# ---------------------------------------------------------------------------
# build_super_fragments
# ---------------------------------------------------------------------------

def test_build_super_fragments_groups_and_skips_abstained():
    f1 = make_line_fragment(1, (0, 0, 0), (1, 0, 0))
    f2 = make_line_fragment(2, (10_000, 0, 0), (1, 0, 0))
    pred = np.array([0, 0, 1, -1])
    frag_per_obs = np.array([1, 1, 2, 2])
    keys = np.array([10, 11, 12, 13])
    labels = np.array([7, 7, 8, 8])
    supers = build_super_fragments("A", [f1, f2], pred, frag_per_obs, keys,
                                   labels=labels)
    assert len(supers) == 2
    s0 = next(s for s in supers if s.cluster_id == 0)
    s1 = next(s for s in supers if s.cluster_id == 1)
    assert s0.atom_ids == frozenset([1])
    assert s0.n_obs == 2 and set(s0.obs_keys.tolist()) == {10, 11}
    assert s0.majority_label == 7
    assert s1.n_obs == 1  # abstained obs 13 excluded
    assert s1.majority_label == 8


def test_build_super_fragments_pools_dna_weighted():
    f1 = make_line_fragment(1, (0, 0, 0), (1, 0, 0), dna=[1.0, 0.0])
    f2 = make_line_fragment(2, (5_000, 0, 0), (1, 0, 0), dna=[0.0, 1.0])
    # 3 obs on frag 1, 1 obs on frag 2 → pooled direction closer to frag 1
    pred = np.array([0, 0, 0, 0])
    frag_per_obs = np.array([1, 1, 1, 2])
    keys = np.arange(4)
    supers = build_super_fragments("A", [f1, f2], pred, frag_per_obs, keys)
    assert len(supers) == 1
    dna = supers[0].dna
    assert dna is not None
    assert abs(np.linalg.norm(dna) - 1.0) < 1e-5
    assert dna[0] > dna[1] > 0


def test_build_super_fragments_soma_flag():
    f1 = make_line_fragment(1, (0, 0, 0), (1, 0, 0))
    supers = build_super_fragments(
        "A", [f1], np.array([0]), np.array([1]), np.array([0]),
        soma_atoms={1})
    assert supers[0].has_soma


# ---------------------------------------------------------------------------
# link_shared_observations
# ---------------------------------------------------------------------------

def test_link_shared_observations_mutual_best():
    f = make_line_fragment(1, (0, 0, 0), (1, 0, 0))
    a = make_super("A", 0, f, obs_keys=[1, 2, 3, 4])
    b = make_super("B", 0, f, obs_keys=[1, 2, 3, 9])   # 3 shared with a
    c = make_super("B", 1, f, obs_keys=[4, 20, 21])    # 1 shared with a
    pairs = link_shared_observations([a, b, c], min_shared=3)
    assert pairs == [(0, 1)]


def test_link_shared_observations_ignores_same_tile():
    f = make_line_fragment(1, (0, 0, 0), (1, 0, 0))
    a = make_super("A", 0, f, obs_keys=[1, 2, 3])
    b = make_super("A", 1, f, obs_keys=[1, 2, 3])
    assert link_shared_observations([a, b], min_shared=1) == []


def test_link_shared_observations_min_shared():
    f = make_line_fragment(1, (0, 0, 0), (1, 0, 0))
    a = make_super("A", 0, f, obs_keys=[1, 2])
    b = make_super("B", 0, f, obs_keys=[2, 3])
    assert link_shared_observations([a, b], min_shared=3) == []
    assert link_shared_observations([a, b], min_shared=1) == [(0, 1)]


def test_link_shared_atoms_cross_tile():
    from treestitch.stitch import SuperFragment, link_shared_atoms
    f1 = make_line_fragment(1, (0, 0, 0), (1, 0, 0))
    f2 = make_line_fragment(2, (50_000, 0, 0), (1, 0, 0))

    def sup(tile, cid, atoms):
        return SuperFragment(tile_id=tile, cluster_id=cid,
                             atom_ids=frozenset(atoms), skeleton=f1,
                             dna=None, n_obs=1,
                             obs_keys=np.zeros(0, dtype=np.int64))

    a = sup("A", 0, [1, 5])       # atom 5 straddles the seam
    b = sup("B", 0, [5, 9])
    c = sup("B", 1, [2])          # no overlap
    assert link_shared_atoms([a, b, c]) == [(0, 1)]
    # same tile never links
    assert link_shared_atoms([a, sup("A", 1, [5])]) == []


# ---------------------------------------------------------------------------
# candidate_stitch_edges
# ---------------------------------------------------------------------------

def test_candidates_cross_tile_only():
    fa = make_line_fragment(1, (0, 0, 0), (1, 0, 0))          # ends at x=4000
    fb = make_line_fragment(2, (6_000, 0, 0), (1, 0, 0))      # starts 2 µm away
    a = make_super("A", 0, fa)
    b_same = make_super("A", 1, fb)
    b_other = make_super("B", 1, fb)
    assert candidate_stitch_edges([a, b_same], endpoint_radius_nm=10_000) == []
    cands = candidate_stitch_edges([a, b_other], endpoint_radius_nm=10_000)
    assert len(cands) == 1
    c = cands[0]
    assert {c.i, c.j} == {0, 1}
    assert c.gap_nm == pytest.approx(2_000.0)
    assert c.score == pytest.approx(1.0 - 2_000.0 / 10_000.0)


def test_candidates_score_uses_dna():
    fa = make_line_fragment(1, (0, 0, 0), (1, 0, 0))
    fb = make_line_fragment(2, (6_000, 0, 0), (1, 0, 0))
    a = make_super("A", 0, fa, dna=[1.0, 0.0])
    b = make_super("B", 0, fb, dna=[-1.0, 0.0])   # opposite DNA → compat 0
    cands = candidate_stitch_edges([a, b], endpoint_radius_nm=10_000)
    assert len(cands) == 1
    assert cands[0].dna_cos == pytest.approx(-1.0)
    assert cands[0].score == pytest.approx(0.0)


def test_candidates_beyond_radius_absent():
    fa = make_line_fragment(1, (0, 0, 0), (1, 0, 0))
    fb = make_line_fragment(2, (100_000, 0, 0), (1, 0, 0))
    a = make_super("A", 0, fa)
    b = make_super("B", 0, fb)
    assert candidate_stitch_edges([a, b], endpoint_radius_nm=10_000) == []


def test_candidates_degree_cap():
    center = make_super("A", 0, make_line_fragment(1, (0, 0, 0), (1, 0, 0)))
    others = [
        make_super("B", k, make_line_fragment(
            10 + k, (5_000 + 100 * k, 200 * k, 0), (1, 0, 0)))
        for k in range(5)
    ]
    cands = candidate_stitch_edges([center] + others,
                                   endpoint_radius_nm=20_000,
                                   max_edges_per_super=2)
    n_center = sum(1 for c in cands if 0 in (c.i, c.j))
    assert n_center == 2


# ---------------------------------------------------------------------------
# stitch_super_fragments — constraints
# ---------------------------------------------------------------------------

def _manual_candidate(i, j, score, ep_i=0, ep_j=0):
    return StitchCandidate(score=score, i=i, j=j, ep_i=ep_i, ep_j=ep_j,
                           gap_nm=0.0, dna_cos=0.0)


def test_kruskal_cycle_rejection():
    frags = [make_line_fragment(k, (k * 6_000, 0, 0), (1, 0, 0)) for k in range(3)]
    supers = [make_super(t, 0, f) for t, f in zip("ABC", frags)]
    cands = [_manual_candidate(0, 1, 0.9, ep_i=1, ep_j=0),
             _manual_candidate(1, 2, 0.8, ep_i=1, ep_j=0),
             _manual_candidate(0, 2, 0.7, ep_i=0, ep_j=1)]
    res = stitch_super_fragments(supers, forced_pairs=[], candidates=cands)
    assert len(res.accepted) == 2
    assert res.rejected.get("cycle") == 1
    assert len(set(res.super_cluster.tolist())) == 1


def test_kruskal_endpoint_used_once():
    frags = [make_line_fragment(k, (k * 6_000, 0, 0), (1, 0, 0)) for k in range(3)]
    supers = [make_super(t, 0, f) for t, f in zip("ABC", frags)]
    # both candidates want endpoint 0 of super 1
    cands = [_manual_candidate(0, 1, 0.9, ep_i=1, ep_j=0),
             _manual_candidate(1, 2, 0.8, ep_i=0, ep_j=0)]
    res = stitch_super_fragments(supers, forced_pairs=[], candidates=cands)
    assert len(res.accepted) == 1
    assert res.rejected.get("endpoint_used") == 1


def test_kruskal_soma_cannot_link():
    fa = make_line_fragment(1, (0, 0, 0), (1, 0, 0))
    fb = make_line_fragment(2, (6_000, 0, 0), (1, 0, 0))
    a = make_super("A", 0, fa, has_soma=True)
    b = make_super("B", 0, fb, has_soma=True)
    cands = [_manual_candidate(0, 1, 0.9)]
    res = stitch_super_fragments([a, b], forced_pairs=[], candidates=cands)
    assert len(res.accepted) == 0
    assert res.rejected.get("soma_cannot_link") == 1
    assert len(set(res.super_cluster.tolist())) == 2

    res2 = stitch_super_fragments([a, b], forced_pairs=[], candidates=cands,
                                  enforce_single_soma=False)
    assert len(res2.accepted) == 1


def test_kruskal_obs_cap():
    fa = make_line_fragment(1, (0, 0, 0), (1, 0, 0))
    fb = make_line_fragment(2, (6_000, 0, 0), (1, 0, 0))
    a = make_super("A", 0, fa, n_obs=60)
    b = make_super("B", 0, fb, n_obs=50)
    cands = [_manual_candidate(0, 1, 0.9)]
    res = stitch_super_fragments([a, b], forced_pairs=[], candidates=cands,
                                 max_obs_per_cluster=100)
    assert len(res.accepted) == 0
    assert res.rejected.get("obs_cap") == 1


def test_kruskal_min_score():
    fa = make_line_fragment(1, (0, 0, 0), (1, 0, 0))
    fb = make_line_fragment(2, (6_000, 0, 0), (1, 0, 0))
    supers = [make_super("A", 0, fa), make_super("B", 0, fb)]
    cands = [_manual_candidate(0, 1, 0.01)]
    res = stitch_super_fragments(supers, forced_pairs=[], candidates=cands,
                                 min_score=0.05)
    assert len(res.accepted) == 0
    assert res.rejected.get("below_min_score") == 1


def test_forced_pairs_applied_and_soma_conflict_recorded():
    fa = make_line_fragment(1, (0, 0, 0), (1, 0, 0))
    fb = make_line_fragment(2, (6_000, 0, 0), (1, 0, 0))
    a = make_super("A", 0, fa, has_soma=True, obs_keys=[1, 2, 3])
    b = make_super("B", 0, fb, has_soma=True, obs_keys=[1, 2, 3])
    res = stitch_super_fragments([a, b], candidates=[])
    # forced merge (shared observations) is applied despite the two somata …
    assert len(set(res.super_cluster.tolist())) == 1
    # … but the conflict is surfaced for review
    assert res.soma_conflicts == [(0, 1)]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def test_pairwise_merge_metrics_exact():
    # pred: {0,1}, {2,3}   true: {0,1,2}, {3}
    pred = np.array([0, 0, 1, 1])
    true = np.array([5, 5, 5, 6])
    m = pairwise_merge_metrics(pred, true)
    # predicted merges: (0,1), (2,3). true merges: (0,1),(0,2),(1,2)
    # TP = (0,1) only → precision 1/2, recall 1/3
    assert m["merge_precision"] == pytest.approx(0.5)
    assert m["merge_recall"] == pytest.approx(1 / 3)


def test_pairwise_merge_metrics_huge_root_ids_no_overflow():
    pred = np.array([0, 0, 1, 1])
    true = np.array([864691135000000001, 864691135000000001,
                     864691135000000002, 864691135000000002])
    m = pairwise_merge_metrics(pred, true)
    assert m["merge_precision"] == pytest.approx(1.0)
    assert m["merge_recall"] == pytest.approx(1.0)


def test_pairwise_merge_metrics_ignores_abstained():
    pred = np.array([0, 0, -1])
    true = np.array([5, 5, 5])
    m = pairwise_merge_metrics(pred, true)
    assert m["n_obs"] == 2
    assert m["merge_precision"] == pytest.approx(1.0)


def test_stitch_edge_precision():
    f = make_line_fragment(1, (0, 0, 0), (1, 0, 0))
    supers = [make_super("A", 0, f, majority_label=7),
              make_super("B", 0, f, majority_label=7),
              make_super("B", 1, f, majority_label=8)]
    acc = [_manual_candidate(0, 1, 0.9), _manual_candidate(0, 2, 0.8)]
    m = stitch_edge_precision(supers, acc)
    assert m["n_correct"] == 1 and m["n_wrong"] == 1
    assert m["stitch_precision"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# End-to-end: tiled synthetic world, ground-truth per-tile partitions
# ---------------------------------------------------------------------------

def test_two_tile_stitch_recovers_cross_tile_objects():
    """Split a synthetic world into two x-tiles with a halo; per-tile
    partitions are perfect but tile-fragmented.  Stitching must reunify every
    cross-tile object (via shared halo observations and endpoint edges)."""
    from treestitch.synthetic import make_synthetic_world

    fragments, region, _ = make_synthetic_world(
        n_objects=6, n_pieces=3, observations_per_piece=10,
        object_spacing_nm=60_000.0, seed=7)

    pos = region.pre_pt_nm
    true = region.pre_root_id
    frag_ids = region.pre_seg_id
    keys = np.arange(len(pos), dtype=np.int64)
    root_to_frag = {int(f.base_root_id): f for f in fragments}

    x_mid = float(np.median(pos[:, 0]))
    halo = 40_000.0
    supers = []
    tile_masks = {}
    for tile_id, lo, hi in (("A", -np.inf, x_mid + halo),
                            ("B", x_mid - halo, np.inf)):
        mask = (pos[:, 0] >= lo) & (pos[:, 0] < hi)
        tile_masks[tile_id] = mask
        # perfect per-tile partition: cluster = true object id
        t_true = true[mask]
        _, t_pred = np.unique(t_true, return_inverse=True)
        t_frag_ids = frag_ids[mask]
        t_frags = [root_to_frag[int(r)] for r in np.unique(t_frag_ids)]
        supers.extend(build_super_fragments(
            tile_id, t_frags, t_pred, t_frag_ids, keys[mask], labels=t_true))

    res = stitch_super_fragments(supers, endpoint_radius_nm=15_000)

    # every accepted stitch is correct
    ep = stitch_edge_precision(supers, res.accepted)
    assert ep["n_wrong"] == 0

    # assemble per-observation global labels (owner tile = core side)
    global_label = np.full(len(pos), -1, dtype=np.int64)
    for tile_id, core_mask in (("A", pos[:, 0] < x_mid),
                               ("B", pos[:, 0] >= x_mid)):
        for si, s in enumerate(supers):
            if s.tile_id != tile_id:
                continue
            cluster = int(res.super_cluster[si])
            sel = np.isin(keys, s.obs_keys) & core_mask
            global_label[sel] = cluster

    assert (global_label >= 0).all()
    m = pairwise_merge_metrics(global_label, true)
    assert m["merge_precision"] == pytest.approx(1.0)
    assert m["merge_recall"] == pytest.approx(1.0)
