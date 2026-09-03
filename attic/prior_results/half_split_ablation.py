#!/usr/bin/env python3
"""Hard-split ablation: split each real skeleton in half → train DNA encoder.

Each proofread neuron skeleton is bisected at its balance edge (the edge whose
removal produces two sub-trees of most equal vertex count).  Both halves share
the same label_root so the encoder is trained to recognise them as the same
neuron despite seeing only partial skeletons.

This directly validates the Phase 2 use case: a neuron spans multiple seg
roots (unproofread); the DNA encoder must assign similar embeddings to all
fragments of the same physical neuron.

Usage
-----
  python scripts/half_split_ablation.py --n-neurons 40 --epochs 80
"""
from __future__ import annotations

import gzip
import io
import struct
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import requests

try:  # the token lives in the environment or ~/.cloudvolume, never here
    from neuronauts.auth import cave_token as _cave_token
except ImportError:  # keep standalone scripts runnable
    import json as _json, os as _os
    from pathlib import Path as _Path

    def _cave_token(required=False):
        t = _os.environ.get("CAVE_TOKEN")
        if t:
            return t.strip()
        f = _Path.home() / ".cloudvolume/secrets/cave-secret.json"
        return _json.loads(f.read_text())["token"].strip() if f.exists() else None


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

NUCLEUS_URL_V1412 = (
    "https://storage.googleapis.com/mat_dbs/public/minnie65_phase3_v1/"
    "v1412/nucleus_detection_v0_merged.csv.gz"
)
SKELETON_CACHE_BASE = (
    "https://minnie.microns-daf.com/skeletoncache/api/v1/"
    "minnie65_public/precomputed/skeleton"
)
TOKEN = _cave_token()
# ---------------------------------------------------------------------------
# Skeleton fetch (same as fetch_real_skeletons.py)
# ---------------------------------------------------------------------------

def fetch_skeleton(root_id: int) -> tuple | None:
    url = f"{SKELETON_CACHE_BASE}/{root_id}"
    try:
        resp = requests.get(
            url, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=30
        )
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
        remaining = len(data) - off
        radii = (np.frombuffer(data, dtype='<f4', count=nv, offset=off).copy()
                 if remaining >= nv * 4 else np.ones(nv, np.float32) * 300.0)
        return verts, edges, radii
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tree bisection
# ---------------------------------------------------------------------------

def _build_adj(n_verts: int, edges: np.ndarray) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n_verts)]
    for u, v in edges:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    return adj


def _subtree_sizes(root: int, adj: list[list[int]]) -> np.ndarray:
    """Compute subtree size at every node (treating the tree as rooted at root)."""
    n = len(adj)
    size = np.ones(n, dtype=np.int32)
    parent = np.full(n, -1, dtype=np.int32)
    order: list[int] = []
    visited = np.zeros(n, dtype=bool)
    q = deque([root])
    visited[root] = True
    while q:
        v = q.popleft()
        order.append(v)
        for w in adj[v]:
            if not visited[w]:
                visited[w] = True
                parent[w] = v
                q.append(w)
    for v in reversed(order):
        p = parent[v]
        if p >= 0:
            size[p] += size[v]
    return size, parent, order


def split_skeleton_balanced(
    verts: np.ndarray,
    edges: np.ndarray,
    radii: np.ndarray,
) -> tuple[tuple, tuple]:
    """Bisect tree at the edge whose removal yields most equal sub-tree sizes.

    Returns two (verts, edges, radii) tuples with local vertex numbering.
    """
    n = len(verts)
    if n < 4 or len(edges) == 0:
        # Degenerate: return both halves as the full skeleton
        return (verts, edges, radii), (verts, edges, radii)

    adj = _build_adj(n, edges)
    root = 0
    sizes, parent, order = _subtree_sizes(root, adj)

    # For each non-root vertex v, removing edge (parent[v], v) splits into
    # sub-tree of size sizes[v] and remainder of size n - sizes[v].
    best_v = -1
    best_diff = n + 1
    for v in order[1:]:  # skip root
        diff = abs(sizes[int(v)] - (n - sizes[int(v)]))
        if diff < best_diff:
            best_diff = diff
            best_v = int(v)

    if best_v < 0:
        return (verts, edges, radii), (verts, edges, radii)

    cut_parent = int(parent[best_v])

    # Side A: vertices in subtree rooted at best_v
    visited_a = np.zeros(n, dtype=bool)
    q = deque([best_v])
    visited_a[best_v] = True
    while q:
        v = q.popleft()
        for w in adj[v]:
            if not visited_a[w] and w != cut_parent:
                visited_a[w] = True
                q.append(w)

    # Side B: all remaining vertices
    visited_b = ~visited_a

    def _extract(mask: np.ndarray):
        old_idx = np.where(mask)[0]
        remap = np.full(n, -1, dtype=np.int64)
        remap[old_idx] = np.arange(len(old_idx), dtype=np.int64)
        sub_verts = verts[old_idx]
        sub_radii = radii[old_idx]
        # Keep edges entirely within this sub-tree
        keep = mask[edges[:, 0]] & mask[edges[:, 1]]
        sub_edges = remap[edges[keep]]
        # Leaf endpoints
        deg = Counter()
        for u, v in sub_edges:
            deg[int(u)] += 1; deg[int(v)] += 1
        leaves = [i for i in range(len(sub_verts)) if deg.get(i, 0) <= 1]
        endpoints = sub_verts[leaves] if leaves else sub_verts[[0, -1]]
        return sub_verts, sub_edges, sub_radii, endpoints

    vA, eA, rA, epA = _extract(visited_a)
    vB, eB, rB, epB = _extract(visited_b)
    return (vA, eA, rA, epA), (vB, eB, rB, epB)


def split_skeleton_n_ways(
    verts: np.ndarray,
    edges: np.ndarray,
    radii: np.ndarray,
    n_chunks: int,
    *,
    min_chunk_verts: int = 15,
) -> list[tuple]:
    """Split a skeleton into n_chunks pieces by recursive balanced bisection.

    At each step the largest remaining piece is bisected at its balance edge.
    Returns a list of (verts, edges, radii, endpoints) tuples, one per chunk.
    The list may be shorter than n_chunks if pieces become too small to split.
    """
    pieces = [split_skeleton_balanced(verts, edges, radii)]
    # pieces is a list of (vA,eA,rA,epA),(vB,eB,rB,epB) pairs initially;
    # flatten to list of individual pieces.
    pieces = list(pieces[0])  # → [(vA,eA,rA,epA), (vB,eB,rB,epB)]

    while len(pieces) < n_chunks:
        # Find the largest piece that can still be split.
        splittable = [
            (i, p) for i, p in enumerate(pieces) if len(p[0]) >= min_chunk_verts * 2
        ]
        if not splittable:
            break
        # Pick the largest.
        idx, piece = max(splittable, key=lambda x: len(x[1][0]))
        halves = split_skeleton_balanced(piece[0], piece[1], piece[2])
        # Reject if either half is below min size.
        if len(halves[0][0]) < min_chunk_verts or len(halves[1][0]) < min_chunk_verts:
            break
        pieces.pop(idx)
        pieces.extend(halves)

    return pieces


# ---------------------------------------------------------------------------
# Build world
# ---------------------------------------------------------------------------

def build_split_world(
    root_ids: list[int],
    n_target: int,
    synapses_per_half: int,
    rng: np.random.Generator,
    max_verts: int = 5000,
    min_half_verts: int = 15,
    n_chunks: int = 2,
) -> tuple:
    """Fetch skeletons, split each into n_chunks pieces, return Region + Fragments + root_label_map.

    n_chunks=2 (default) is the original bisection.  Higher values produce smaller
    fragments, simulating more aggressively fragmented unproofread data.
    All chunks from one neuron share the same label_root (positive training pairs).
    """
    from neuronauts.schemas import Fragment, Region

    chunk_word = "bisect" if n_chunks == 2 else f"{n_chunks}-way split"
    print(f"\nFetching and {chunk_word}ing skeletons (target {n_target} neurons) …")
    fragments: list[Fragment] = []
    root_label_map: dict[int, set[int]] = {}
    all_roots_list: list[int] = []
    syn_idx = 0
    label_root_counter = 0
    frag_id_counter = 10_000_000

    for root_id in root_ids:
        if label_root_counter >= n_target:
            break
        result = fetch_skeleton(root_id)
        if result is None:
            continue
        verts, edges, radii = result
        if len(verts) < 30 * n_chunks or len(verts) > max_verts:
            continue

        chunks = split_skeleton_n_ways(verts, edges, radii, n_chunks,
                                       min_chunk_verts=min_half_verts)
        if len(chunks) < 2:
            continue
        if any(len(c[0]) < min_half_verts for c in chunks):
            continue

        label_root_counter += 1
        label_root = label_root_counter

        chunk_sizes = "/".join(str(len(c[0])) for c in chunks)
        print(f"  [{label_root_counter:3d}] root={root_id}  V={len(verts)}"
              f"  chunks={chunk_sizes} vertices")

        for sv, se, sr, sep in chunks:
            fid = frag_id_counter; frag_id_counter += 1
            syn_indices = list(range(syn_idx, syn_idx + synapses_per_half))
            syn_idx += synapses_per_half
            all_roots_list.extend([label_root] * synapses_per_half)
            root_label_map[fid] = {label_root}
            frag = Fragment(
                fragment_id=fid,
                region_id="split",
                base_root_id=fid,
                vertices_nm=sv,
                edges=se,
                endpoints_nm=sep,
                radius_nm=sr,
                synapse_indices=np.array(syn_indices, dtype=np.int64),
                dna=None,
            ).validate()
            fragments.append(frag)

        time.sleep(0.05)

    if not fragments:
        raise RuntimeError("No usable skeletons fetched")

    n_syn = syn_idx
    pts = rng.uniform(0, 1_000_000, (n_syn, 3)).astype(np.float32)
    post_pts = pts + rng.normal(0, 300, (n_syn, 3)).astype(np.float32)

    region = Region(
        region_id="split_ablation",
        bbox_nm=((0.0, 0.0, 0.0), (1_000_000.0, 1_000_000.0, 1_000_000.0)),
        voxel_size_nm=(8.0, 8.0, 40.0),
        seg_version=117,
        label_version=1412,
        pre_pt_nm=pts,
        post_pt_nm=post_pts,
        pre_root_id=np.array(all_roots_list, dtype=np.int64),
        post_root_id=np.zeros(n_syn, dtype=np.int64),
        synapse_id=np.arange(n_syn, dtype=np.int64),
    )
    print(f"  → {len(fragments)} fragments ({n_syn} synapses, spatial baseline ≈ chance)")
    return region, fragments, root_label_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-neurons", type=int, default=40)
    p.add_argument("--n-chunks", type=int, default=2,
                   help="Number of pieces to cut each skeleton into (default 2 = bisect)")
    p.add_argument("--synapses-per-half", type=int, default=10)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--output-dim", type=int, default=32)
    p.add_argument("--n-paths", type=int, default=6)
    p.add_argument("--max-pairs", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--encoder", choices=["path", "gnn"], default="path",
                   help="path=TreeDNAEncoder (hand-crafted 6-D), gnn=SkeletonGNN (raw graph)")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # Nucleus roots
    print("Fetching v1412 nucleus CSV …")
    resp = requests.get(NUCLEUS_URL_V1412, timeout=60)
    resp.raise_for_status()
    root_ids: list[int] = []
    with gzip.open(io.BytesIO(resp.content)) as f:
        for line in f:
            parts = line.decode().strip().split(',')
            if len(parts) >= 4:
                try:
                    r = int(parts[3])
                    if r != 0:
                        root_ids.append(r)
                except ValueError:
                    pass
    print(f"  {len(root_ids)} proofread neurons at v1412")
    candidates = rng.choice(root_ids, size=min(args.n_neurons * 6, len(root_ids)), replace=False).tolist()

    region, fragments, root_label_map = build_split_world(
        candidates,
        n_target=args.n_neurons,
        synapses_per_half=args.synapses_per_half,
        rng=rng,
        n_chunks=args.n_chunks,
    )

    # Run ablation via shared helper
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
