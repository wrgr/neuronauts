#!/usr/bin/env python3
"""Multi-fragment ablation: split each skeleton into N≥3 parts.

Why this is harder than the 2-half split
-----------------------------------------
The 2-half split creates two fragments per neuron with a clear morphological
division at the balance edge.  A multi-fragment split creates N random subtrees
— shorter paths, more varied shapes, longer endpoint gaps between non-adjacent
fragments.  This is a more realistic simulation of the Phase 2 use case:

    Neuron → N unproofread seg-root fragments (arbitrary sub-arborisations)
    DNA encoder must assign similar embeddings to ALL N fragments of the same
    physical neuron.

An optional ``--volume-min`` / ``--volume-max`` filter restricts the evaluation
to neurons within a specific soma-volume range, biasing toward a more
morphologically homogeneous cohort (crude proxy for cell type without a CAVE
materialization token).

Usage
-----
  python scripts/multi_fragment_ablation.py \\
      --n-neurons 60 --n-splits 4 --epochs 80 \\
      --token $CAVE_TOKEN

  # Restrict to medium-volume neurons (crude pyramidal-cell proxy)
  python scripts/multi_fragment_ablation.py \\
      --n-neurons 60 --n-splits 4 \\
      --volume-min 300 --volume-max 3000 \\
      --epochs 80
"""
from __future__ import annotations

import argparse
import gzip
import io
import struct
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import NamedTuple

import numpy as np
import requests

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

NUCLEUS_URL = (
    "https://storage.googleapis.com/mat_dbs/public/minnie65_phase3_v1/"
    "v1412/nucleus_detection_v0_merged.csv.gz"
)
SKELETON_BASE = (
    "https://minnie.microns-daf.com/skeletoncache/api/v1/"
    "minnie65_public/precomputed/skeleton"
)


# ---------------------------------------------------------------------------
# Nucleus root fetch with optional volume filter
# ---------------------------------------------------------------------------

class NucleusRecord(NamedTuple):
    root_id: int
    volume_um3: float


def fetch_nucleus_records(
    *,
    min_volume: float | None = None,
    max_volume: float | None = None,
) -> list[NucleusRecord]:
    print("Fetching v1412 nucleus CSV …")
    resp = requests.get(NUCLEUS_URL, timeout=60)
    resp.raise_for_status()
    print(f"  Downloaded {len(resp.content) / 1e6:.1f} MB")

    records: list[NucleusRecord] = []
    with gzip.open(io.BytesIO(resp.content)) as f:
        for line in f:
            parts = line.decode().strip().split(',')
            if len(parts) < 4:
                continue
            try:
                vol = float(parts[1])
                root_id = int(parts[3])
            except ValueError:
                continue
            if root_id == 0:
                continue
            if min_volume is not None and vol < min_volume:
                continue
            if max_volume is not None and vol > max_volume:
                continue
            records.append(NucleusRecord(root_id=root_id, volume_um3=vol))

    print(f"  {len(records)} neurons pass volume filter")
    return records


# ---------------------------------------------------------------------------
# Skeleton fetch
# ---------------------------------------------------------------------------

def fetch_skeleton(root_id: int, token: str) -> tuple | None:
    url = f"{SKELETON_BASE}/{root_id}"
    try:
        resp = requests.get(url,
                            headers={"Authorization": f"Bearer {token}"},
                            timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.content
        off = 0
        nv = struct.unpack_from('<I', data, off)[0]; off += 4
        ne = struct.unpack_from('<I', data, off)[0]; off += 4
        if nv < 20:
            return None
        verts = np.frombuffer(data, dtype='<f4', count=nv * 3, offset=off).reshape(nv, 3).copy()
        off += nv * 3 * 4
        edges = np.frombuffer(data, dtype='<u4', count=ne * 2, offset=off).reshape(ne, 2).astype(np.int64).copy()
        off += ne * 2 * 4
        rem = len(data) - off
        radii = (np.frombuffer(data, dtype='<f4', count=nv, offset=off).copy()
                 if rem >= nv * 4 else np.ones(nv, np.float32) * 300.0)
        return verts, edges, radii
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Multi-fragment skeleton split
# ---------------------------------------------------------------------------

def _adj(n: int, edges: np.ndarray) -> list[list[int]]:
    a: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        a[int(u)].append(int(v))
        a[int(v)].append(int(u))
    return a


def _bfs_size(root: int, forbidden: int, adj: list[list[int]]) -> set[int]:
    """BFS from root excluding the edge to forbidden vertex."""
    visited: set[int] = {root}
    q = deque([root])
    while q:
        v = q.popleft()
        for w in adj[v]:
            if w != forbidden and w not in visited:
                visited.add(w)
                q.append(w)
    return visited


def split_skeleton_n_parts(
    verts: np.ndarray,
    edges: np.ndarray,
    radii: np.ndarray,
    n_splits: int,
    rng: np.random.Generator,
) -> list[tuple]:
    """Split a skeleton tree into up to n_splits sub-trees.

    Greedily removes edges (sampled with probability proportional to the
    min of the two sub-tree sizes) until n_splits components remain.

    Returns list of (sub_verts, sub_edges, sub_radii, endpoints_nm) tuples.
    Each sub-tree has at least min_part_verts vertices.
    """
    if len(verts) < n_splits * 3:
        return _extract_all([verts], [edges], [radii])

    adj = _adj(len(verts), edges)
    remaining_edges = set(map(tuple, edges.tolist()))  # (u, v) with u < v in original
    # Normalise edge orientation
    norm_edges = {(min(u, v), max(u, v)) for u, v in remaining_edges}

    components: list[set[int]] = []

    def _connected_components(n_v: int, edge_set: set[tuple]) -> list[set[int]]:
        visited = np.zeros(n_v, bool)
        comps: list[set[int]] = []
        adj_local: list[list[int]] = [[] for _ in range(n_v)]
        for u, v in edge_set:
            adj_local[u].append(v)
            adj_local[v].append(u)
        for start in range(n_v):
            if not visited[start]:
                comp: set[int] = set()
                q = deque([start])
                while q:
                    node = q.popleft()
                    if visited[node]:
                        continue
                    visited[node] = True
                    comp.add(node)
                    for nb in adj_local[node]:
                        if not visited[nb]:
                            q.append(nb)
                comps.append(comp)
        return comps

    # Iteratively remove one edge to create a new component.
    current_edges = set(norm_edges)
    n_v = len(verts)

    for _ in range(n_splits - 1):
        # Score edges by balance of the resulting split.
        edge_list = list(current_edges)
        if not edge_list:
            break
        # Build current adjacency.
        adj_cur: list[list[int]] = [[] for _ in range(n_v)]
        for u, v in current_edges:
            adj_cur[u].append(v)
            adj_cur[v].append(u)

        # Find the component containing each edge.
        visited = [False] * n_v
        comp_label = [-1] * n_v
        comp_id = 0
        comp_members: dict[int, list[int]] = {}
        for start in range(n_v):
            if not visited[start]:
                members: list[int] = []
                q = deque([start])
                while q:
                    node = q.popleft()
                    if visited[node]:
                        continue
                    visited[node] = True
                    comp_label[node] = comp_id
                    members.append(node)
                    for nb in adj_cur[node]:
                        if not visited[nb]:
                            q.append(nb)
                comp_members[comp_id] = members
                comp_id += 1

        # Only consider edges within the largest component.
        largest_comp_id = max(comp_members, key=lambda k: len(comp_members[k]))
        largest_nodes = set(comp_members[largest_comp_id])
        candidate_edges = [(u, v) for u, v in edge_list
                           if u in largest_nodes and v in largest_nodes]
        if not candidate_edges:
            break

        # Sample proportional to balance (min of sub-tree sizes).
        sub_sizes = []
        for u, v in candidate_edges:
            sv = len(_bfs_size(v, u, adj_cur))
            su = len(largest_nodes) - sv
            sub_sizes.append(min(sv, su))
        sub_sizes = np.array(sub_sizes, dtype=float)
        sub_sizes = np.clip(sub_sizes, 1, None)
        probs = sub_sizes / sub_sizes.sum()
        chosen_idx = int(rng.choice(len(candidate_edges), p=probs))
        cut_edge = candidate_edges[chosen_idx]
        current_edges.discard(cut_edge)

    # Extract sub-trees from remaining edge set.
    comps = _connected_components(n_v, current_edges)

    result = []
    for comp in comps:
        if len(comp) < 3:
            continue
        old_idx = sorted(comp)
        remap = np.full(n_v, -1, dtype=np.int64)
        for new_i, old_i in enumerate(old_idx):
            remap[old_i] = new_i
        sub_verts = verts[old_idx]
        sub_radii = radii[old_idx]
        keep = np.array([(u in comp and v in comp) for u, v in current_edges], dtype=bool)
        kept_edges = np.array([e for e in current_edges if e[0] in comp and e[1] in comp],
                               dtype=np.int64)
        if len(kept_edges) == 0:
            sub_edges = np.empty((0, 2), dtype=np.int64)
        else:
            sub_edges = remap[kept_edges]

        deg: Counter = Counter()
        for u, v in sub_edges:
            deg[int(u)] += 1
            deg[int(v)] += 1
        leaves = [i for i in range(len(sub_verts)) if deg.get(i, 0) <= 1]
        endpoints = sub_verts[leaves] if leaves else sub_verts[[0, -1]]
        result.append((sub_verts, sub_edges, sub_radii, endpoints))
    return result if result else [(verts, np.empty((0, 2), dtype=np.int64), radii,
                                   verts[[0, -1]])]


def _extract_all(verts_list, edges_list, radii_list):
    result = []
    for v, e, r in zip(verts_list, edges_list, radii_list):
        deg: Counter = Counter()
        for u, w in e:
            deg[int(u)] += 1
            deg[int(w)] += 1
        leaves = [i for i in range(len(v)) if deg.get(i, 0) <= 1]
        endpoints = v[leaves] if leaves else v[[0, -1]]
        result.append((v, e, r, endpoints))
    return result


# ---------------------------------------------------------------------------
# Build multi-fragment world
# ---------------------------------------------------------------------------

def build_multi_fragment_world(
    records: list[NucleusRecord],
    token: str,
    n_target: int,
    n_splits: int,
    synapses_per_fragment: int,
    rng: np.random.Generator,
    max_verts: int = 5_000,
) -> tuple:
    from neuronauts.schemas import Fragment, Region

    print(f"\nFetching skeletons (target {n_target} neurons, {n_splits} parts each) …")
    fragments: list[Fragment] = []
    root_label_map: dict[int, set[int]] = {}
    all_pre_roots: list[int] = []
    n_neurons = 0
    syn_idx = 0
    fid_counter = 20_000_000

    for rec in records:
        if n_neurons >= n_target:
            break
        result = fetch_skeleton(rec.root_id, token)
        if result is None:
            continue
        verts, edges, radii = result
        if len(verts) < max(30, n_splits * 10) or len(verts) > max_verts:
            continue

        parts = split_skeleton_n_parts(verts, edges, radii, n_splits, rng)
        if len(parts) < 2:
            continue

        label_root = n_neurons + 1
        n_neurons += 1

        for sv, se, sr, sep in parts:
            fid = fid_counter; fid_counter += 1
            syn_indices = list(range(syn_idx, syn_idx + synapses_per_fragment))
            syn_idx += synapses_per_fragment
            all_pre_roots.extend([label_root] * synapses_per_fragment)
            root_label_map[fid] = {label_root}
            frag = Fragment(
                fragment_id=fid,
                region_id="multi_split",
                base_root_id=fid,
                vertices_nm=sv,
                edges=se,
                endpoints_nm=sep,
                radius_nm=sr,
                synapse_indices=np.array(syn_indices, dtype=np.int64),
                dna=None,
            ).validate()
            fragments.append(frag)

        print(f"  [{n_neurons:3d}/{n_target}] root={rec.root_id}  V={len(verts)}"
              f"  vol={rec.volume_um3:.0f}µm³  parts={len(parts)}")
        time.sleep(0.05)

    if not fragments:
        raise RuntimeError("No usable skeletons fetched")

    n_syn = syn_idx
    pts = rng.uniform(0, 1_000_000, (n_syn, 3)).astype(np.float32)
    post = pts + rng.normal(0, 300, (n_syn, 3)).astype(np.float32)

    region = Region(
        region_id="multi_frag_ablation",
        bbox_nm=((0.0, 0.0, 0.0), (1_000_000.0, 1_000_000.0, 1_000_000.0)),
        voxel_size_nm=(8.0, 8.0, 40.0),
        seg_version=117,
        label_version=1412,
        pre_pt_nm=pts,
        post_pt_nm=post,
        pre_root_id=np.array(all_pre_roots, dtype=np.int64),
        post_root_id=np.zeros(n_syn, dtype=np.int64),
        synapse_id=np.arange(n_syn, dtype=np.int64),
    )
    print(f"\n→ {len(fragments)} fragments from {n_neurons} neurons"
          f" ({n_syn} synapses, spatial baseline ≈ chance)")
    return region, fragments, root_label_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-neurons", type=int, default=60)
    p.add_argument("--n-splits", type=int, default=4,
                   help="Number of skeleton parts per neuron (≥ 2)")
    p.add_argument("--synapses-per-fragment", type=int, default=10)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--output-dim", type=int, default=32)
    p.add_argument("--n-paths", type=int, default=8)
    p.add_argument("--max-pairs", type=int, default=2000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--token", default=None)
    p.add_argument("--volume-min", type=float, default=None,
                   help="Min nucleus volume µm³ (cell-type proxy)")
    p.add_argument("--volume-max", type=float, default=None,
                   help="Max nucleus volume µm³ (cell-type proxy)")
    p.add_argument("--encoder", choices=["path", "gnn"], default="path",
                   help="path=TreeDNAEncoder (hand-crafted features), "
                        "gnn=SkeletonGNN (data-driven, orientation-free)")
    args = p.parse_args()

    if args.n_splits < 2:
        p.error("--n-splits must be ≥ 2")

    rng = np.random.default_rng(args.seed)

    records = fetch_nucleus_records(
        min_volume=args.volume_min, max_volume=args.volume_max
    )
    if len(records) < args.n_neurons * 3:
        print(f"  Warning: only {len(records)} records pass volume filter "
              f"(need ~{args.n_neurons * 3}); reduce --n-neurons or widen volume range")

    rng.shuffle(records := list(records))
    region, fragments, root_label_map = build_multi_fragment_world(
        records,
        token=args.token,
        n_target=args.n_neurons,
        n_splits=args.n_splits,
        synapses_per_fragment=args.synapses_per_fragment,
        rng=rng,
    )

    vol_str = ""
    if args.volume_min or args.volume_max:
        lo = f"{args.volume_min:.0f}" if args.volume_min else "0"
        hi = f"{args.volume_max:.0f}" if args.volume_max else "∞"
        vol_str = f"  volume filter: {lo}–{hi} µm³ (same-type proxy)\n"

    print(f"\n{'='*60}")
    print(f"Multi-fragment ablation: {args.n_splits}-way split  (encoder={args.encoder})")
    print(f"  neurons: {args.n_neurons}  fragments: {len(fragments)}")
    print(vol_str, end="")
    print(f"{'='*60}")

    sys.path.insert(0, str(_ROOT / "scripts"))
    from ablate_dna import run_ablation
    run_ablation(
        region, fragments, root_label_map,
        n_epochs=args.epochs,
        lr=args.lr,
        d_model=args.d_model,
        output_dim=args.output_dim,
        n_paths=args.n_paths,
        max_pairs=args.max_pairs,
        device=args.device,
        seed=args.seed,
        encoder_type=args.encoder,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
