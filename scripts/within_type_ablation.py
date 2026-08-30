#!/usr/bin/env python3
"""Within-type ablation: hard honest evaluation of the DNA encoder.

The hard-split ablation (half_split_ablation.py) samples neurons uniformly at
random across all cell types.  Negative pairs are almost certainly cross-type
(pyramidal vs interneuron), which are trivially distinguishable by morphology.
That test measures *type* discrimination, not *individual* discrimination.

This script fixes that:
  1. Load the MICrONS v1412 cell type annotation table (public GCS).
  2. Restrict the candidate pool to a single cell type (default: "23P",
     L2/3 pyramidal — the most abundant excitatory type, ~20K neurons).
  3. Run the standard hard-split bisection ablation on that cohort.
  4. All negative pairs are same-type neurons → the encoder must distinguish
     individuals within a morphologically similar population.

This is the honest test of whether DNA learns individual identity or just
cell-type identity.

Expected difficulty:
  23P neurons all have an apical dendrite, basal dendrites, similar calibre.
  AUC ≥ 0.75 after training → genuine individual-level discrimination.
  AUC ≈ 0.50 after training → DNA is type-discriminative only; need richer features.

Usage
-----
  # 40 L2/3 pyramidal cells, default settings
  python scripts/within_type_ablation.py --cell-type 23P --n-neurons 40

  # Try inhibitory basket cells (harder — axon dominates)
  python scripts/within_type_ablation.py --cell-type BC --n-neurons 30

  # List available cell types and counts
  python scripts/within_type_ablation.py --list-types
"""
from __future__ import annotations

import argparse
import gzip
import io
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CELL_TYPE_URL = (
    "https://storage.googleapis.com/mat_dbs/public/minnie65_phase3_v1/"
    "v1412/aibs_metamodel_celltypes_v661_merged.csv.gz"
)
SKELETON_CACHE_BASE = (
    "https://minnie.microns-daf.com/skeletoncache/api/v1/"
    "minnie65_public/precomputed/skeleton"
)
TOKEN = "a08cdcba8581846f48d5742a75c53311"


# ---------------------------------------------------------------------------
# Cell type table
# ---------------------------------------------------------------------------

def load_cell_type_roots(
    cell_type: str | None = None,
    *,
    cache_path: str | None = None,
) -> dict[str, list[int]]:
    """Download and parse the v1412 cell type annotation table.

    Returns
    -------
    dict mapping cell_type string → list of pt_root_id (int64).
    If cell_type is provided, only that key is returned.
    """
    if cache_path and Path(cache_path).exists():
        import json
        with open(cache_path) as f:
            data = json.load(f)
        if cell_type:
            return {cell_type: [int(r) for r in data.get(cell_type, [])]}
        return {k: [int(r) for r in v] for k, v in data.items()}

    print("Fetching cell type annotation table …")
    resp = requests.get(CELL_TYPE_URL, timeout=60)
    resp.raise_for_status()

    root_map: dict[str, list[int]] = {}
    with gzip.open(io.BytesIO(resp.content)) as f:
        for line in f:
            parts = line.decode().strip().split(",")
            if len(parts) < 7:
                continue
            ct = parts[3]
            try:
                rid = int(parts[6])
            except ValueError:
                continue
            root_map.setdefault(ct, []).append(rid)

    if cache_path:
        import json
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(root_map, f)

    counts = {ct: len(ids) for ct, ids in root_map.items()}
    print(f"  Loaded {sum(counts.values())} annotations across {len(counts)} types")

    if cell_type:
        return {cell_type: root_map.get(cell_type, [])}
    return root_map


def print_cell_type_summary(root_map: dict[str, list[int]]) -> None:
    counts = sorted(root_map.items(), key=lambda x: -len(x[1]))
    print(f"\n{'Cell type':<20}  {'Count':>8}  {'Notes'}")
    print("-" * 60)
    notes = {
        "23P": "L2/3 pyramidal (excitatory)",
        "4P":  "L4 pyramidal (excitatory)",
        "5P-IT": "L5 IT pyramidal (excitatory)",
        "5P-ET": "L5 ET pyramidal (excitatory)",
        "5P-NP": "L5 NP pyramidal (excitatory)",
        "6P-IT": "L6 IT pyramidal (excitatory)",
        "6P-CT": "L6 CT pyramidal (excitatory)",
        "BC":   "basket cell (inhibitory)",
        "MC":   "Martinotti cell (inhibitory)",
        "BPC":  "bipolar cell (inhibitory)",
        "NGC":  "neurogliaform cell (inhibitory)",
        "astrocyte": "non-neuron",
        "oligo":     "non-neuron",
        "OPC":       "non-neuron",
        "pericyte":  "non-neuron",
        "microglia": "non-neuron",
    }
    for ct, ids in counts:
        print(f"  {ct:<20}  {len(ids):>8}  {notes.get(ct, '')}")


# ---------------------------------------------------------------------------
# Skeleton fetch (identical to half_split_ablation.py)
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
        nv = struct.unpack_from("<I", data, off)[0]; off += 4
        ne = struct.unpack_from("<I", data, off)[0]; off += 4
        if nv < 20:
            return None
        verts = np.frombuffer(data, dtype="<f4", count=nv * 3, offset=off).reshape(nv, 3).copy()
        off += nv * 3 * 4
        edges = np.frombuffer(data, dtype="<u4", count=ne * 2, offset=off).reshape(ne, 2).astype(np.int64).copy()
        off += ne * 2 * 4
        remaining = len(data) - off
        radii = (
            np.frombuffer(data, dtype="<f4", count=nv, offset=off).copy()
            if remaining >= nv * 4
            else np.ones(nv, np.float32) * 300.0
        )
        return verts, edges, radii
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cell-type", default="23P",
                   help="Cell type to use for the cohort (default: 23P)")
    p.add_argument("--list-types", action="store_true",
                   help="Print available cell types and exit")
    p.add_argument("--n-neurons", type=int, default=40)
    p.add_argument("--n-chunks", type=int, default=2,
                   help="Pieces per neuron: 2=bisect (default), 4=quarters, etc. "
                        "More chunks = smaller fragments = harder task")
    p.add_argument("--synapses-per-half", type=int, default=10)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--output-dim", type=int, default=32)
    p.add_argument("--n-paths", type=int, default=6)
    p.add_argument("--max-pairs", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", default=None,
                   help="Directory to cache downloaded tables")
    p.add_argument("--encoder", choices=["path", "gnn", "vicreg"], default="path",
                   help="path=TreeDNAEncoder (hand-crafted 6-D), gnn=SkeletonGNN (raw graph)")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    cache_path = (
        str(Path(args.cache_dir) / "celltypes_v661.json")
        if args.cache_dir else None
    )

    if args.list_types:
        root_map = load_cell_type_roots(cache_path=cache_path)
        print_cell_type_summary(root_map)
        return 0

    # --- Load within-type candidate roots ---
    type_roots = load_cell_type_roots(args.cell_type, cache_path=cache_path)
    candidates = type_roots.get(args.cell_type, [])
    if not candidates:
        print(f"ERROR: cell type '{args.cell_type}' not found. Run --list-types.")
        return 1
    print(f"  {len(candidates)} neurons of type '{args.cell_type}' available")

    # Sample more than needed to account for skeleton failures / size filters
    n_sample = min(args.n_neurons * 8, len(candidates))
    sampled = rng.choice(candidates, size=n_sample, replace=False).tolist()

    # --- Bisect skeletons (reuse build_split_world from half_split_ablation) ---
    sys.path.insert(0, str(_ROOT / "scripts"))
    from half_split_ablation import build_split_world

    region, fragments, root_label_map = build_split_world(
        sampled,
        n_target=args.n_neurons,
        synapses_per_half=args.synapses_per_half,
        rng=rng,
        n_chunks=args.n_chunks,
    )

    n_neurons = len(set(v for vals in root_label_map.values() for v in vals))
    chunk_word = "bisect" if args.n_chunks == 2 else f"{args.n_chunks}-way split"

    print(f"\n{'='*60}")
    print(f"Within-type ablation: cell_type='{args.cell_type}', {chunk_word}")
    print(f"Region: {region.n_synapses} synapses, {len(fragments)} fragments")
    print(f"Neurons: {n_neurons}  (all same cell type — hard negatives)")
    print(f"{'='*60}")

    # --- Run ablation ---
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

    print(f"\nNote: negatives above are all within-type ('{args.cell_type}').")
    print("AUC >= 0.75 after training → individual-level discrimination.")
    print("AUC ≈ 0.50 after training → type-discriminative only; need richer features.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
