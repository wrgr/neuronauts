#!/usr/bin/env python3
"""Two-level stitch demo: per-tile partitions + level-1 seam stitching.

Experiment 1 of ``docs/tree_assembly_algorithm.md`` — the box-ceiling test.
The volume is split into a 2×2 grid of tiles with overlapping halos.  Each
tile runs the existing level-0 pipeline (FragmentEncoder → observation graph →
EdgePartitionGNN → GAEC).  Tile clusters become super-fragments; a level-1
pass reconciles them with (a) shared-halo-observation identity and (b)
endpoint-matching candidates accepted by a constrained maximum-weight forest
(cycle rejection, one use per endpoint, ≤1 soma per neuron).

The comparison is baseline (per-tile clusters, no stitching — the current
sparse-box regime) vs stitched, on the same per-tile predictions:

  - ARI and pairwise merge precision/recall over the union of tile cores
  - fraction of multi-tile objects fully assembled into one cluster
  - stitch-edge precision (accepted seam edges whose sides agree in truth)

Usage
-----
  # Offline synthetic world (no network needed)
  python scripts/two_level_stitch.py --synthetic

  # Real Minnie65 region (requires CAVE access): 2×2 tiles over the two
  # largest axes of the bbox
  python scripts/two_level_stitch.py \\
      --bbox 950000 930000 700000 1350000 1000000 800000 --halo-nm 20000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def slice_region(region, mask, region_id: str):
    """Tile-local Region from a boolean observation mask."""
    import numpy as np
    from neuronauts.schemas import Region

    pos = region.pre_pt_nm[mask]
    pad = 5000.0
    bbox = (tuple(float(v) for v in pos.min(0) - pad),
            tuple(float(v) for v in pos.max(0) + pad))
    return Region(
        region_id=region_id,
        bbox_nm=bbox,
        voxel_size_nm=region.voxel_size_nm,
        seg_version=region.seg_version,
        label_version=region.label_version,
        pre_pt_nm=pos,
        post_pt_nm=region.post_pt_nm[mask],
        pre_root_id=region.pre_root_id[mask],
        post_root_id=region.post_root_id[mask],
        synapse_id=region.synapse_id[mask],
        pre_seg_id=None if region.pre_seg_id is None else region.pre_seg_id[mask],
        post_seg_id=None if region.post_seg_id is None else region.post_seg_id[mask],
    ).validate()


def slice_fragments(fragments, frag_ids_tile):
    """Tile-local Fragment list: atoms present in the tile, with tile-local
    observation indices."""
    import numpy as np
    from neuronauts.schemas import Fragment

    out = []
    for f in fragments:
        idxs = np.where(frag_ids_tile == f.base_root_id)[0]
        if len(idxs) == 0:
            continue
        out.append(Fragment(
            fragment_id=f.fragment_id,
            region_id=f.region_id,
            base_root_id=f.base_root_id,
            vertices_nm=f.vertices_nm,
            edges=f.edges,
            endpoints_nm=f.endpoints_nm,
            radius_nm=f.radius_nm,
            synapse_indices=idxs.astype(np.int64),
            dna=None,
        ).validate())
    return out


def run_tile(tile_id, region_t, frags_t, label_map_t, args):
    """Level-0 pipeline on one tile: encoder → graph → edge classifier → GAEC."""
    from treestitch.embed import FragmentEncoder, encode_fragments, train_fragment_encoder
    from treestitch.graph import build_observation_graph
    from treestitch.partition import partition_observations_cc, train_edge_partition

    encoder = FragmentEncoder(node_input_dim=4, d_model=64, output_dim=32)
    if args.embed_epochs > 0:
        train_fragment_encoder(
            encoder, [frags_t], n_epochs=args.embed_epochs, lr=1e-3,
            device=args.device, root_label_map=label_map_t, log_every=0)
    frags_enc = encode_fragments(encoder, frags_t, device=args.device)

    graph = build_observation_graph(
        region_t, frags_enc, side="pre", k_spatial=args.k_spatial,
        endpoint_radius_nm=args.endpoint_radius_nm)

    model, _ = train_edge_partition(
        graph, n_epochs=args.partition_epochs, device=args.device,
        seed=args.seed, log_every=0)
    pred = partition_observations_cc(model, graph, bias=args.cc_bias,
                                     device=args.device)
    print(f"  [{tile_id}] {graph.n_nodes} obs, {len(frags_enc)} fragments, "
          f"{graph.n_edges} edges → {len(set(int(c) for c in pred if c >= 0))} clusters")
    return frags_enc, graph, pred


def multi_tile_assembly_fraction(pred_global, true, owner_tile, keep):
    """Fraction of objects spanning ≥2 tile cores whose core observations all
    land in one predicted cluster (abstained observations excluded)."""
    import numpy as np

    n_full = n_multi = 0
    for obj in np.unique(true[keep]):
        m = keep & (true == obj)
        if len(set(owner_tile[m].tolist())) < 2:
            continue
        n_multi += 1
        labs = set(pred_global[m].tolist())
        labs.discard(-1)
        if len(labs) == 1 and (pred_global[m] >= 0).all():
            n_full += 1
    return (n_full / n_multi if n_multi else float("nan")), n_multi


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", action="store_true",
                   help="offline synthetic world (no CAVE access)")
    # synthetic world
    p.add_argument("--n-objects", type=int, default=24)
    p.add_argument("--n-pieces", type=int, default=4)
    p.add_argument("--obs-per-piece", type=int, default=10)
    p.add_argument("--frankenmerge-frac", type=float, default=0.0)
    # real world
    p.add_argument("--bbox", type=float, nargs=6, metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
                   help="region bbox in nm (real mode)")
    p.add_argument("--version", type=int, default=1718)
    p.add_argument("--max-synapses", type=int, default=20_000,
                   help="per-tile synapse cap (real mode)")
    p.add_argument("--min-syn-per-fragment", type=int, default=5)
    # tiling
    p.add_argument("--halo-nm", type=float, default=40_000.0)
    # level-0 pipeline
    p.add_argument("--embed-epochs", type=int, default=40)
    p.add_argument("--partition-epochs", type=int, default=80)
    p.add_argument("--k-spatial", type=int, default=8)
    p.add_argument("--endpoint-radius-nm", type=float, default=10_000.0)
    p.add_argument("--cc-bias", type=float, default=-1.0)
    # level-1 stitch
    p.add_argument("--min-shared", type=int, default=3,
                   help="min shared halo observations for a forced merge")
    p.add_argument("--no-atom-links", action="store_true",
                   help="disable the shared-atom forced-merge channel")
    p.add_argument("--min-stitch-score", type=float, default=0.8,
                   help="min geometry score for endpoint candidates (the "
                        "geometry-only scorer is a weak placeholder — keep "
                        "this conservative until the learned scorer lands)")
    p.add_argument("--stitch-radius-nm", type=float, default=10_000.0,
                   help="endpoint radius for level-1 candidate edges")
    p.add_argument("--max-obs-per-cluster", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import numpy as np

    from treestitch.stitch import (
        build_super_fragments,
        link_shared_atoms,
        link_shared_observations,
        pairwise_merge_metrics,
        stitch_edge_precision,
        stitch_super_fragments,
    )

    # ------------------------------------------------------------------ world
    if args.synthetic:
        from treestitch.synthetic import make_synthetic_world
        fragments, region, root_label_map = make_synthetic_world(
            n_objects=args.n_objects, n_pieces=args.n_pieces,
            observations_per_piece=args.obs_per_piece,
            frankenmerge_frac=args.frankenmerge_frac,
            seed=args.seed, verbose=False)
        print(f"Synthetic world: {args.n_objects} objects, "
              f"{len(fragments)} fragments, {region.n_synapses} observations")
    else:
        if args.bbox is None:
            p.error("real mode requires --bbox (or pass --synthetic)")
        from treestitch.realworld import build_region_world
        lo = tuple(args.bbox[:3])
        hi = tuple(args.bbox[3:])
        fragments, region, root_label_map = build_region_world(
            (lo, hi), version=args.version, max_synapses=args.max_synapses,
            min_syn_per_fragment=args.min_syn_per_fragment, seed=args.seed)

    pos = region.pre_pt_nm
    true = region.pre_root_id
    frag_ids = region.pre_seg_id
    obs_keys = region.synapse_id.astype(np.int64)
    if len(np.unique(obs_keys)) != len(obs_keys):
        raise RuntimeError("observation keys are not unique — halo joins would be wrong")

    # ------------------------------------------------------ 2×2 tiling (core)
    extent = pos.max(0) - pos.min(0)
    ax0, ax1 = np.argsort(extent)[-2:]
    mid0 = float(np.median(pos[:, ax0]))
    mid1 = float(np.median(pos[:, ax1]))
    print(f"Tiling 2×2 on axes {int(ax0)}/{int(ax1)} at "
          f"({mid0:.0f}, {mid1:.0f}) nm, halo {args.halo_nm:.0f} nm")

    tiles = {}
    owner_tile = np.empty(len(pos), dtype=object)
    for tid, (s0, s1) in {"A": (-1, -1), "B": (+1, -1),
                          "C": (-1, +1), "D": (+1, +1)}.items():
        core = (((pos[:, ax0] >= mid0) if s0 > 0 else (pos[:, ax0] < mid0)) &
                ((pos[:, ax1] >= mid1) if s1 > 0 else (pos[:, ax1] < mid1)))
        halo = ((pos[:, ax0] >= (mid0 - args.halo_nm if s0 > 0 else -np.inf)) &
                (pos[:, ax0] <  (np.inf if s0 > 0 else mid0 + args.halo_nm)) &
                (pos[:, ax1] >= (mid1 - args.halo_nm if s1 > 0 else -np.inf)) &
                (pos[:, ax1] <  (np.inf if s1 > 0 else mid1 + args.halo_nm)))
        tiles[tid] = {"core": core, "mask": halo}
        owner_tile[core] = tid

    # -------------------------------------------------- level 0: tile pipeline
    print("\nLevel 0: per-tile partitions")
    supers = []
    tile_pred = {}
    for tid, t in tiles.items():
        mask = t["mask"]
        region_t = slice_region(region, mask, f"tile_{tid}")
        frags_t = slice_fragments(fragments, frag_ids[mask])
        atoms_t = {f.base_root_id for f in frags_t}
        label_map_t = {a: root_label_map[a] for a in atoms_t if a in root_label_map}
        frags_enc, graph, pred = run_tile(tid, region_t, frags_t, label_map_t, args)
        tile_pred[tid] = pred
        supers.extend(build_super_fragments(
            tid, frags_enc, pred, graph.fragment_id, obs_keys[mask],
            labels=graph.labels))

    print(f"\n{len(supers)} super-fragments across 4 tiles")

    # obs_key → super index per tile (for label assembly)
    key_to_super = {tid: {} for tid in tiles}
    for si, s in enumerate(supers):
        for k in s.obs_keys.tolist():
            key_to_super[s.tile_id][int(k)] = si

    def assemble_labels(super_to_cluster):
        out = np.full(len(pos), -1, dtype=np.int64)
        for oi in range(len(pos)):
            tid = owner_tile[oi]
            si = key_to_super[tid].get(int(obs_keys[oi]))
            if si is not None:
                out[oi] = super_to_cluster[si]
        return out

    # ------------------------------------------------ baseline: no stitching
    baseline_cluster = np.arange(len(supers), dtype=np.int64)  # every super its own neuron
    baseline = assemble_labels(baseline_cluster)

    # ---------------------------------------------------- level 1: stitching
    print("\nLevel 1: seam stitch")
    forced_obs = link_shared_observations(supers, min_shared=args.min_shared)
    forced_atom = ([] if args.no_atom_links else link_shared_atoms(supers))
    forced = sorted(set(forced_obs) | set(forced_atom))
    res = stitch_super_fragments(
        supers,
        endpoint_radius_nm=args.stitch_radius_nm,
        min_score=args.min_stitch_score,
        forced_pairs=forced,
        max_obs_per_cluster=args.max_obs_per_cluster,
    )
    stitched = assemble_labels(res.super_cluster)
    print(f"  forced merges: {len(forced)} "
          f"({len(forced_obs)} shared-obs, {len(forced_atom)} shared-atom)")
    print(f"  candidate edges accepted: {len(res.accepted)}  "
          f"rejected: {res.rejected or {}}")
    if res.soma_conflicts:
        print(f"  ⚠ soma conflicts in forced merges: {len(res.soma_conflicts)}")

    ep = stitch_edge_precision(supers, res.accepted)
    print(f"  stitch-edge precision: {ep['stitch_precision']:.3f} "
          f"({ep['n_correct']}/{ep['n_scored']} scored)")

    # ------------------------------------------------------------- evaluation
    keep = true != 0
    try:
        from sklearn.metrics import adjusted_rand_score
        def _ari(pred):
            m = keep & (pred >= 0)
            return adjusted_rand_score(true[m], pred[m])
    except ImportError:
        def _ari(pred):
            return float("nan")

    print("\n=== Baseline (per-tile clusters, no stitching) vs stitched ===")
    rows = []
    for name, lab in (("baseline", baseline), ("stitched", res and stitched)):
        m = pairwise_merge_metrics(lab, true)
        frac, n_multi = multi_tile_assembly_fraction(lab, true, owner_tile, keep)
        rows.append((name, _ari(lab), m, frac, n_multi))
    for name, ari, m, frac, n_multi in rows:
        print(f"  {name:9s} ARI={ari:.3f}  merge_P={m['merge_precision']:.3f} "
              f"merge_R={m['merge_recall']:.3f}  "
              f"multi-tile objects fully assembled: {frac:.1%} (of {n_multi})")

    b, s = rows[0], rows[1]
    print(f"\nΔARI = {s[1] - b[1]:+.3f}   Δmerge_R = "
          f"{s[2]['merge_recall'] - b[2]['merge_recall']:+.3f}   "
          f"Δmerge_P = {s[2]['merge_precision'] - b[2]['merge_precision']:+.3f}")
    print(f"Multi-tile assembly: {b[3]:.1%} → {s[3]:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
