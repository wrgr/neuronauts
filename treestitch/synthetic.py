"""Synthetic world generator for offline pipeline development.

``make_synthetic_world`` mirrors ``load_minnie65_world`` but generates random
branching tree skeletons instead of fetching real neurons from CAVE — so the
full pipeline (encode → graph → partition → evaluate) can be exercised and
benchmarked without network access.

Each parent object is a random branching tree.  The tree is split into
``n_pieces`` connected sub-fragments (simulating an over-segmentation), and
observations are placed near each piece's vertices.  The split keeps piece
endpoints adjacent across cuts, so endpoint-adjacency edges carry real signal —
matching the structure that makes the real-data partition tractable.

    from treestitch.synthetic import make_synthetic_world
    fragments, region, label_map = make_synthetic_world(n_objects=20, n_pieces=3)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from treestitch.data import _split_skeleton_n_pieces


def _random_tree(
    n_verts: int,
    rng: np.random.Generator,
    *,
    step_nm: float = 1500.0,
    branch_prob: float = 0.15,
    origin: Optional[np.ndarray] = None,
    drift: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grow a random branching tree skeleton.

    Returns (vertices_nm [V,3], edges [E,2], radii_nm [V]).  A persistent
    per-object ``drift`` direction gives each object a distinct global shape so
    that morphology (the fragment embedding) is discriminative.
    """
    if origin is None:
        origin = np.zeros(3, dtype=np.float32)
    if drift is None:
        drift = rng.normal(0, 1, 3).astype(np.float32)
        drift /= np.linalg.norm(drift) + 1e-8

    verts = [origin.astype(np.float32)]
    edges: list[tuple[int, int]] = []
    radii = [float(rng.uniform(200, 400))]
    frontier = [0]

    while len(verts) < n_verts:
        # Bias growth toward recent frontier vertices (tip-growth), occasionally
        # branch from an older vertex.
        if frontier and rng.random() > branch_prob:
            parent = frontier[-1]
        else:
            parent = int(rng.integers(len(verts)))

        direction = drift + rng.normal(0, 0.6, 3).astype(np.float32)
        direction /= np.linalg.norm(direction) + 1e-8
        new_v = verts[parent] + direction * step_nm
        new_idx = len(verts)
        verts.append(new_v.astype(np.float32))
        radii.append(max(50.0, radii[parent] * float(rng.uniform(0.85, 1.0))))
        edges.append((parent, new_idx))
        frontier.append(new_idx)
        if len(frontier) > 8:
            frontier.pop(0)

    return (
        np.asarray(verts, dtype=np.float32),
        np.asarray(edges, dtype=np.int64),
        np.asarray(radii, dtype=np.float32),
    )


def make_synthetic_world(
    n_objects: int = 20,
    n_pieces: int = 3,
    observations_per_piece: int = 12,
    *,
    verts_per_object: int = 120,
    object_spacing_nm: float = 60_000.0,
    synapse_noise_nm: float = 500.0,
    min_piece_verts: int = 8,
    frankenmerge_frac: float = 0.0,
    seed: int = 42,
    verbose: bool = False,
) -> tuple:
    """Build a synthetic (fragments, region, root_label_map) world.

    Parameters mirror ``load_minnie65_world``.  Objects are placed on a loose
    grid so different objects are spatially separated (the spatial k-NN channel
    is then mostly within-object near piece boundaries — realistic), while
    endpoint adjacency reliably connects adjacent pieces of the same object.

    ``frankenmerge_frac`` injects v117-style **merge errors**: that fraction of
    pieces is fused (in pairs, across *different* objects) under a single shared
    fragment / seg id.  The fused fragment owns observations from two distinct
    objects, so its same-fragment (type-0) edges are partly wrong and its DNA is
    morphologically ambiguous — exactly the corruption that learning
    f(v117 → v1412) must correct, and the case where threshold union-find
    over-merges irreversibly.

    Returns
    -------
    (fragments, region, root_label_map) — ready for the treestitch pipeline.
    """
    from neuronauts.schemas import Fragment, Region

    rng = np.random.default_rng(seed)

    grid = int(np.ceil(np.sqrt(n_objects)))

    # ---- Pass 1: generate pieces (skeleton + observations), one record each --
    pieces_rec: list[dict] = []  # {obj_id, verts, edges, radii, obs_pts}
    obj_counter = 0
    for obj_i in range(n_objects):
        gx, gy = obj_i % grid, obj_i // grid
        origin = np.array(
            [gx * object_spacing_nm, gy * object_spacing_nm, 0.0], dtype=np.float32
        )
        drift = rng.normal(0, 1, 3).astype(np.float32)
        drift /= np.linalg.norm(drift) + 1e-8

        verts, edges_raw, radii = _random_tree(
            verts_per_object, rng, origin=origin, drift=drift
        )
        pieces = _split_skeleton_n_pieces(
            verts, edges_raw, radii, n_pieces, min_verts=min_piece_verts
        )
        if len(pieces) < 2:
            continue
        obj_counter += 1
        obj_id = obj_counter
        for pv, pe, pr in pieces:
            anchor_idxs = rng.integers(0, len(pv), observations_per_piece)
            obs_pts = (
                pv[anchor_idxs]
                + rng.normal(0, synapse_noise_nm, (observations_per_piece, 3)).astype(
                    np.float32
                )
            )
            pieces_rec.append(
                {"obj_id": obj_id, "verts": pv,
                 "edges": pe if len(pe) else np.zeros((0, 2), dtype=np.int64),
                 "radii": pr, "obs_pts": obs_pts}
            )
        if verbose:
            print(f"  [{obj_counter:3d}] object {obj_id}  pieces={len(pieces)}")

    if not pieces_rec:
        raise RuntimeError("No usable synthetic objects generated")

    # ---- Pass 2: assign each piece a seg id; fuse some across objects --------
    seg_of_piece = list(range(len(pieces_rec)))  # piece index → seg group id
    n_franken = 0
    if frankenmerge_frac > 0:
        order = rng.permutation(len(pieces_rec))
        n_target = int(round(frankenmerge_frac * len(pieces_rec) / 2))
        used: set[int] = set()
        made = 0
        for a in order:
            if made >= n_target:
                break
            if a in used:
                continue
            # find a partner piece from a DIFFERENT object
            partner = -1
            for b in order:
                if b == a or b in used:
                    continue
                if pieces_rec[b]["obj_id"] != pieces_rec[a]["obj_id"]:
                    partner = int(b)
                    break
            if partner < 0:
                continue
            seg_of_piece[partner] = seg_of_piece[int(a)]  # share a's seg id
            used.add(int(a))
            used.add(partner)
            made += 1
        n_franken = made

    # ---- Pass 3: build fragments (one per seg group) + region arrays ---------
    from collections import defaultdict
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
        frag_id = sid + 1  # 1-based seg / fragment id

        # Merge member skeletons into one fragment (offset edge indices).
        vparts, eparts, rparts = [], [], []
        voff = 0
        syn_indices: list[int] = []
        for pi in members:
            rec = pieces_rec[pi]
            pv, pe, pr, obs_pts = rec["verts"], rec["edges"], rec["radii"], rec["obs_pts"]
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
                region_id="synthetic",
                base_root_id=frag_id,
                vertices_nm=mv,
                edges=me,
                endpoints_nm=endpoints,
                radius_nm=mr,
                synapse_indices=np.asarray(sorted(syn_indices), dtype=np.int64),
                dna=None,
            ).validate()
        )

    if verbose and frankenmerge_frac > 0:
        print(f"  franken-merges: {n_franken} cross-object seg fusions")

    all_pts = np.concatenate(all_obs_pts).astype(np.float32)
    post_pts = all_pts + rng.normal(0, 2000, all_pts.shape).astype(np.float32)
    mins, maxs = all_pts.min(0), all_pts.max(0)
    pad = 5000.0
    bbox = (
        tuple(float(v) for v in mins - pad),
        tuple(float(v) for v in maxs + pad),
    )

    region = Region(
        region_id="synthetic",
        bbox_nm=bbox,
        voxel_size_nm=(8.0, 8.0, 40.0),
        seg_version=117,
        label_version=1412,
        pre_pt_nm=all_pts,
        post_pt_nm=post_pts,
        pre_root_id=np.array(all_obj_ids, dtype=np.int64),
        post_root_id=np.zeros(obs_idx, dtype=np.int64),
        synapse_id=np.arange(obs_idx, dtype=np.int64),
        pre_seg_id=np.array(all_frag_ids, dtype=np.int64),
        post_seg_id=np.zeros(obs_idx, dtype=np.int64),
    ).validate()

    if verbose:
        print(
            f"  → {obj_counter} objects, {len(fragments)} fragments, {obs_idx} observations"
        )

    return fragments, region, root_label_map


__all__ = ["make_synthetic_world"]
