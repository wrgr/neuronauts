#!/usr/bin/env python3
"""Fetch real CAVE skeletons for proofread neurons and run DNA ablation.

Workflow
--------
1. Download v1412 nucleus CSV from GCS → select N proofread root IDs.
2. Fetch each skeleton from the CAVE skeleton cache (precomputed neuroglancer
   format) → parse vertices, edges, radii.
3. Build Fragment objects (one per neuron root).
4. Sample synthetic synapses near each skeleton (since the synapse table is
   unavailable; this validates morphological discriminability with real shapes).
5. Run evaluate_dna_auc before and after training TreeDNAEncoder.

Usage
-----
  python attic/prior_results/fetch_real_skeletons.py \
      --n-neurons 40 --epochs 80 --token <CAVE_AUTH_TOKEN>
"""
from __future__ import annotations

import argparse
import gzip
import io
import struct
import sys
import time
from pathlib import Path

import numpy as np
import requests

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


# ---------------------------------------------------------------------------
# Data fetching helpers
# ---------------------------------------------------------------------------

def fetch_nucleus_roots(n: int, rng: np.random.Generator) -> list[int]:
    """Return n randomly sampled proofread v1412 root IDs."""
    print(f"Fetching v1412 nucleus CSV …")
    resp = requests.get(NUCLEUS_URL_V1412, timeout=60)
    resp.raise_for_status()
    print(f"  Downloaded {len(resp.content)/1e6:.1f} MB")

    root_ids: list[int] = []
    with gzip.open(io.BytesIO(resp.content)) as f:
        for line in f:
            parts = line.decode().strip().split(',')
            if len(parts) < 4:
                continue
            try:
                root_id = int(parts[3])
                if root_id != 0:
                    root_ids.append(root_id)
            except ValueError:
                pass

    print(f"  {len(root_ids)} proofread neurons at v1412")
    chosen = rng.choice(root_ids, size=min(n * 4, len(root_ids)), replace=False).tolist()
    return chosen  # caller will trim to n after filtering failures


def fetch_skeleton(root_id: int, token: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Fetch and parse a neuroglancer precomputed skeleton.

    Returns (vertices_nm [V,3], edges [E,2], radii_nm [V]) or None on failure.
    """
    url = f"{SKELETON_CACHE_BASE}/{root_id}"
    try:
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if resp.status_code != 200:
            return None
        data = resp.content
    except Exception:
        return None

    try:
        offset = 0
        n_verts = struct.unpack_from('<I', data, offset)[0]; offset += 4
        n_edges = struct.unpack_from('<I', data, offset)[0]; offset += 4
        if n_verts < 3:
            return None  # placeholder skeleton
        verts = np.frombuffer(data, dtype='<f4', count=n_verts * 3, offset=offset).reshape(n_verts, 3).copy()
        offset += n_verts * 3 * 4
        edges = np.frombuffer(data, dtype='<u4', count=n_edges * 2, offset=offset).reshape(n_edges, 2).astype(np.int64).copy()
        offset += n_edges * 2 * 4
        remaining = len(data) - offset
        if remaining >= n_verts * 4:
            radii = np.frombuffer(data, dtype='<f4', count=n_verts, offset=offset).copy()
        else:
            radii = np.ones(n_verts, dtype=np.float32) * 300.0
        return verts, edges, radii
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Build Region + Fragments from real skeletons
# ---------------------------------------------------------------------------

def build_real_world(
    root_ids: list[int],
    token: str,
    synapses_per_neuron: int,
    n_target: int,
    rng: np.random.Generator,
    max_verts: int = 6000,
    uniform_synapses: bool = True,
) -> tuple:
    """Fetch real skeletons, build Fragment + Region with skeleton-sampled synapses."""
    from neuronauts.schemas import Fragment, Region

    print(f"\nFetching skeletons for up to {len(root_ids)} candidates (target {n_target}) …")
    fragments: list[Fragment] = []
    root_label_map: dict[int, set[int]] = {}
    all_pre_pts: list[np.ndarray] = []
    all_post_pts: list[np.ndarray] = []
    all_pre_roots: list[int] = []
    all_syn_ids: list[int] = []
    syn_idx = 0

    for i, root_id in enumerate(root_ids):
        if len(fragments) >= n_target:
            break
        result = fetch_skeleton(root_id, token)
        if result is None:
            continue
        verts_nm, edges, radii_nm = result
        # Skip placeholder skeletons and very large neurons (memory)
        if len(verts_nm) < 10 or len(verts_nm) > max_verts:
            continue

        # Synapse positions: near skeleton vertices for now; if uniform_synapses
        # is True, these will be replaced with uniform-random positions below.
        syn_vert_idx = rng.integers(len(verts_nm), size=synapses_per_neuron)
        jitter = rng.normal(0, 500, (synapses_per_neuron, 3)).astype(np.float32)
        pre_pts = verts_nm[syn_vert_idx] + jitter
        post_pts = pre_pts + rng.normal(0, 300, (synapses_per_neuron, 3)).astype(np.float32)

        syn_indices = list(range(syn_idx, syn_idx + synapses_per_neuron))
        all_pre_pts.append(pre_pts)
        all_post_pts.append(post_pts)
        label_root = i + 1  # neuron identity (positional label)
        all_pre_roots.extend([label_root] * synapses_per_neuron)
        all_syn_ids.extend(syn_indices)
        syn_idx += synapses_per_neuron

        # Leaf vertices for endpoints_nm
        from collections import Counter
        deg: Counter = Counter()
        for u, v in edges:
            deg[int(u)] += 1
            deg[int(v)] += 1
        leaves = [j for j in range(len(verts_nm)) if deg.get(j, 0) <= 1]
        endpoints_nm = verts_nm[leaves] if leaves else verts_nm[[0, -1]]

        frag = Fragment(
            fragment_id=root_id,
            region_id="real",
            base_root_id=root_id,
            vertices_nm=verts_nm,
            edges=edges,
            endpoints_nm=endpoints_nm,
            radius_nm=radii_nm,
            synapse_indices=np.array(syn_indices, dtype=np.int64),
            dna=None,
        ).validate()
        fragments.append(frag)
        root_label_map[root_id] = {label_root}

        print(f"  [{len(fragments):3d}] root={root_id}  V={len(verts_nm)}  E={len(edges)}")
        time.sleep(0.05)  # gentle rate limiting

    if not fragments:
        raise RuntimeError("No skeletons fetched successfully")

    all_pts = np.concatenate(all_pre_pts).astype(np.float32)
    all_post = np.concatenate(all_post_pts).astype(np.float32)

    if uniform_synapses:
        # Replace all synapse positions with uniform-random positions in the
        # global bounding box of the fetched skeletons.  This decouples spatial
        # location from identity so the spatial-proximity baseline stays ≈ chance
        # and the DNA encoder must discriminate neurons by morphology alone.
        global_min = all_pts.min(axis=0)
        global_max = all_pts.max(axis=0)
        n_syn = len(all_pts)
        all_pts = (rng.uniform(0, 1, (n_syn, 3)) *
                   (global_max - global_min) + global_min).astype(np.float32)
        all_post = all_pts + rng.normal(0, 300, (n_syn, 3)).astype(np.float32)
        print("  → synapse positions randomised uniformly in global bbox "
              f"(spatial baseline will be ≈ chance)")

    region_min = all_pts.min(axis=0)
    region_max = all_pts.max(axis=0) + 1.0
    region = Region(
        region_id="real_minnie65",
        bbox_nm=((float(region_min[0]), float(region_min[1]), float(region_min[2])),
                 (float(region_max[0]), float(region_max[1]), float(region_max[2]))),
        voxel_size_nm=(8.0, 8.0, 40.0),
        seg_version=117,
        label_version=1412,
        pre_pt_nm=all_pts,
        post_pt_nm=all_post,
        pre_root_id=np.array(all_pre_roots, dtype=np.int64),
        post_root_id=np.zeros(syn_idx, dtype=np.int64),
        synapse_id=np.array(all_syn_ids, dtype=np.int64),
    )
    return region, fragments, root_label_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-neurons", type=int, default=40)
    p.add_argument("--synapses-per-neuron", type=int, default=12)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--output-dim", type=int, default=32)
    p.add_argument("--n-paths", type=int, default=12)
    p.add_argument("--max-pairs", type=int, default=2000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--token", default=None,
                   help="CAVE auth token")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # --- Fetch nucleus roots ---
    candidate_ids = fetch_nucleus_roots(args.n_neurons, rng)

    # --- Fetch skeletons and build world ---
    region, fragments, root_label_map = build_real_world(
        candidate_ids[: args.n_neurons * 5],  # extras to account for size filtering
        token=args.token,
        synapses_per_neuron=args.synapses_per_neuron,
        n_target=args.n_neurons,
        rng=rng,
        uniform_synapses=True,
    )

    print(f"\nBuilt Region: {region.n_synapses} synapses, {len(fragments)} neurons")

    # --- Run ablation ---
    # Import inline to avoid circular issues at top level
    sys.path.insert(0, str(_ROOT / "scripts"))
    from ablate_dna import run_ablation

    result = run_ablation(
        region, fragments, root_label_map,
        n_epochs=args.epochs,
        lr=args.lr,
        d_model=args.d_model,
        output_dim=args.output_dim,
        n_paths=args.n_paths,
        max_pairs=args.max_pairs,
        device=args.device,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
