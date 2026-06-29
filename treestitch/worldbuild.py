"""Shared world-assembly helpers for synthetic and real data.

A *piece record* is a dict describing one connected sub-fragment:
    {"obj_id": int, "verts": [V,3] f32, "edges": [E,2] i64,
     "radii": [V] f32, "obs_pts": [M,3] f32}

``build_world_from_pieces`` turns a list of piece records (plus an assignment of
pieces to segment groups) into the (fragments, region, root_label_map) triple
the pipeline consumes.  ``frankenmerge_random`` and ``frankenmerge_adjacent``
produce the piece→segment assignment, optionally fusing pieces from *different*
objects into a shared v117 segment to simulate merge errors:

  - random   — fuse arbitrary cross-object pieces (often spatially distant).
  - adjacent — fuse cross-object pieces whose skeletons come within a radius
               (the realistic touch-based v117 merge between adjacent neurons).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np


def frankenmerge_random(
    pieces_rec: list[dict],
    frac: float,
    rng: np.random.Generator,
) -> tuple[list[int], int]:
    """Assign pieces to segment groups, fusing random cross-object pairs.

    Returns ``(seg_of_piece, n_franken)`` where ``seg_of_piece[i]`` is the
    segment-group id of piece ``i`` (pieces sharing an id form one v117
    segment / fragment).
    """
    seg_of_piece = list(range(len(pieces_rec)))
    if frac <= 0:
        return seg_of_piece, 0

    order = rng.permutation(len(pieces_rec))
    n_target = int(round(frac * len(pieces_rec) / 2))
    used: set[int] = set()
    made = 0
    for a in order:
        if made >= n_target:
            break
        if int(a) in used:
            continue
        partner = -1
        for b in order:
            if int(b) == int(a) or int(b) in used:
                continue
            if pieces_rec[int(b)]["obj_id"] != pieces_rec[int(a)]["obj_id"]:
                partner = int(b)
                break
        if partner < 0:
            continue
        seg_of_piece[partner] = seg_of_piece[int(a)]
        used.add(int(a))
        used.add(partner)
        made += 1
    return seg_of_piece, made


def frankenmerge_adjacent(
    pieces_rec: list[dict],
    frac: float,
    rng: np.random.Generator,
    *,
    radius_nm: float = 5_000.0,
) -> tuple[list[int], int]:
    """Assign pieces to segment groups, fusing *spatially adjacent* cross-object
    pieces.

    Builds candidate cross-object piece pairs whose skeleton vertices come
    within ``radius_nm`` of each other, then greedily fuses the closest pairs
    (each piece used at most once) until ``frac`` of pieces are merged.  This is
    the realistic v117 merge mode: two different neurons that physically touch
    get joined into one segment.

    Returns ``(seg_of_piece, n_franken)``.
    """
    from neuronauts._scipy_compat import cKDTree

    seg_of_piece = list(range(len(pieces_rec)))
    if frac <= 0 or len(pieces_rec) < 2:
        return seg_of_piece, 0

    # Gather skeleton vertices tagged by piece (subsample large pieces for speed).
    pts_list: list[np.ndarray] = []
    tag_list: list[np.ndarray] = []
    for pi, rec in enumerate(pieces_rec):
        v = np.asarray(rec["verts"], dtype=np.float32)
        if len(v) > 200:
            sel = rng.choice(len(v), 200, replace=False)
            v = v[sel]
        pts_list.append(v)
        tag_list.append(np.full(len(v), pi, dtype=np.int64))
    pts = np.concatenate(pts_list, axis=0)
    tags = np.concatenate(tag_list, axis=0)

    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=radius_nm, output_type="ndarray")

    # Reduce to cross-object piece pairs with their minimum observed distance.
    best: dict[tuple[int, int], float] = {}
    for pi_idx, pj_idx in pairs:
        a = int(tags[pi_idx])
        b = int(tags[pj_idx])
        if a == b:
            continue
        if pieces_rec[a]["obj_id"] == pieces_rec[b]["obj_id"]:
            continue
        d = float(np.linalg.norm(pts[pi_idx] - pts[pj_idx]))
        key = (min(a, b), max(a, b))
        if key not in best or d < best[key]:
            best[key] = d

    if not best:
        return seg_of_piece, 0

    ranked = sorted(best.items(), key=lambda kv: kv[1])
    n_target = int(round(frac * len(pieces_rec) / 2))
    used: set[int] = set()
    made = 0
    for (a, b), _d in ranked:
        if made >= n_target:
            break
        if a in used or b in used:
            continue
        # Fuse b's group into a's group.
        seg_of_piece[b] = seg_of_piece[a]
        used.add(a)
        used.add(b)
        made += 1
    return seg_of_piece, made


def build_world_from_pieces(
    pieces_rec: list[dict],
    seg_of_piece: Optional[list[int]] = None,
    *,
    region_id: str = "world",
    seg_version: int = 117,
    label_version: int = 1412,
    post_noise_nm: float = 2_000.0,
    seed: int = 0,
) -> tuple:
    """Assemble (fragments, region, root_label_map) from piece records.

    Pieces sharing a ``seg_of_piece`` id are merged into one Fragment (their
    skeletons concatenated, observations unioned) — this is how a franken
    segment ends up owning observations from two different objects.

    Returns
    -------
    (fragments, region, root_label_map)
    """
    from neuronauts.schemas import Fragment, Region

    if seg_of_piece is None:
        seg_of_piece = list(range(len(pieces_rec)))

    rng = np.random.default_rng(seed)

    group_pieces: dict[int, list[int]] = defaultdict(list)
    for pi, sid in enumerate(seg_of_piece):
        group_pieces[sid].append(pi)

    all_obs_pts: list[np.ndarray] = []
    all_frag_ids: list[int] = []
    all_obj_ids: list[int] = []
    fragments: list = []
    root_label_map: dict[int, set[int]] = {}
    obs_idx = 0

    for sid, members in group_pieces.items():
        frag_id = sid + 1
        vparts, eparts, rparts = [], [], []
        voff = 0
        syn_indices: list[int] = []
        for pi in members:
            rec = pieces_rec[pi]
            pv = np.asarray(rec["verts"], dtype=np.float32)
            pe = np.asarray(rec["edges"], dtype=np.int64)
            pr = np.asarray(rec["radii"], dtype=np.float32)
            obs_pts = np.asarray(rec["obs_pts"], dtype=np.float32)

            vparts.append(pv)
            rparts.append(pr)
            if len(pe):
                eparts.append(pe + voff)
            voff += len(pv)

            n_obs = len(obs_pts)
            obs_indices = np.arange(obs_idx, obs_idx + n_obs, dtype=np.int64)
            obs_idx += n_obs
            syn_indices.extend(obs_indices.tolist())
            all_obs_pts.append(obs_pts)
            all_frag_ids.extend([frag_id] * n_obs)
            all_obj_ids.extend([rec["obj_id"]] * n_obs)
            root_label_map.setdefault(frag_id, set()).add(rec["obj_id"])

        mv = np.concatenate(vparts, axis=0).astype(np.float32)
        mr = np.concatenate(rparts, axis=0).astype(np.float32)
        me = (np.concatenate(eparts, axis=0).astype(np.int64)
              if eparts else np.zeros((0, 2), dtype=np.int64))

        deg = np.zeros(len(mv), dtype=np.int64)
        if len(me):
            np.add.at(deg, me[:, 0], 1)
            np.add.at(deg, me[:, 1], 1)
        leaf_mask = deg <= 1
        endpoints = mv[leaf_mask] if leaf_mask.any() else mv[[0]]

        fragments.append(
            Fragment(
                fragment_id=frag_id,
                region_id=region_id,
                base_root_id=frag_id,
                vertices_nm=mv,
                edges=me,
                endpoints_nm=endpoints,
                radius_nm=mr,
                synapse_indices=np.asarray(sorted(syn_indices), dtype=np.int64),
                dna=None,
            ).validate()
        )

    if not fragments:
        raise RuntimeError("No usable pieces to assemble")

    all_pts = np.concatenate(all_obs_pts).astype(np.float32)
    post_pts = all_pts + rng.normal(0, post_noise_nm, all_pts.shape).astype(np.float32)
    mins, maxs = all_pts.min(0), all_pts.max(0)
    pad = 5000.0
    bbox = (tuple(float(v) for v in mins - pad), tuple(float(v) for v in maxs + pad))

    region = Region(
        region_id=region_id,
        bbox_nm=bbox,
        voxel_size_nm=(8.0, 8.0, 40.0),
        seg_version=seg_version,
        label_version=label_version,
        pre_pt_nm=all_pts,
        post_pt_nm=post_pts,
        pre_root_id=np.array(all_obj_ids, dtype=np.int64),
        post_root_id=np.zeros(obs_idx, dtype=np.int64),
        synapse_id=np.arange(obs_idx, dtype=np.int64),
        pre_seg_id=np.array(all_frag_ids, dtype=np.int64),
        post_seg_id=np.zeros(obs_idx, dtype=np.int64),
    ).validate()

    return fragments, region, root_label_map


__all__ = [
    "frankenmerge_random",
    "frankenmerge_adjacent",
    "build_world_from_pieces",
]
