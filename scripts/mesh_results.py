#!/usr/bin/env python3
"""Mesh project results into a bundle Neuroglancer (or Blender/MeshLab/three.js)
can open — the single entry point for ``neuronauts.meshing``.

Subcommands
-----------
harness     v117 atoms from a fetched harness geometry dir
             (data/substrate/geom + optional data/substrate/topology/*.npz for
             the per-atom polarity/cable stats used as sortable columns).
kimimaro    every root skeleton in one kimimaro box archive
             (neuronauts.cell_graph.precompute_self_skeletons_for_cache output).
fragments   a neuronauts.schemas Fragment collection (schemas.save_fragments),
             optionally grouped by a fragment_to_neuron JSON (id -> group).
demo        a synthetic branching skeleton — smoke-test the pipeline with no
             data on disk.
serve       host a bundle (or a directory of bundles) over CORS HTTP so a
             browser-based Neuroglancer can load it.

Examples
--------
    python scripts/mesh_results.py demo --out viz/mesh/demo
    python scripts/mesh_results.py serve --dir viz/mesh
    # -> open the printed http://127.0.0.1:8000/demo/index.html

    python scripts/mesh_results.py harness \\
        --geom-dir data/substrate/geom \\
        --population data/substrate/c100um/population.npz \\
        --top-n 40 --min-synapses 10 \\
        --out viz/mesh/harness_top40

    python scripts/mesh_results.py kimimaro \\
        --archive data/boxes_30um/skeletons/<box_hash>.npz \\
        --out viz/mesh/box_<box_hash>

    python scripts/mesh_results.py fragments \\
        --fragments data/some_fragments.npz \\
        --fragment-to-neuron results/assembly_run/fragment_to_neuron.json \\
        --out viz/mesh/assembly_run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from neuronauts.meshing.bundle import MeshParams, export_bundle  # noqa: E402
from neuronauts.meshing.serve import DEFAULT_HOST, DEFAULT_PORT, serve_forever  # noqa: E402
from neuronauts.meshing.skeleton import SkeletonGeometry  # noqa: E402
from neuronauts.meshing.sources import (  # noqa: E402
    kimimaro_archive_skeletons, load_harness_atoms, top_atoms_by_synapse_count,
)


def _mesh_params_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--sides", type=int, default=6, help="ring vertex count (>=3)")
    ap.add_argument("--sphere-level", type=int, default=1, help="joint sphere subdivision (0-2)")
    ap.add_argument("--min-radius-nm", type=float, default=30.0)
    ap.add_argument("--max-radius-nm", type=float, default=None)
    ap.add_argument("--radius-scale", type=float, default=1.0)
    ap.add_argument("--caps", choices=["junctions", "all", "none"], default="junctions")
    ap.add_argument("--formats", nargs="+", default=["precomputed"],
                    choices=["precomputed", "obj", "ply"])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000",
                    help="URL prefix state.json will point at once served")
    ap.add_argument("--served-root", default=None,
                    help="directory that will be the HTTP server's document root "
                         "(default: --out's parent, i.e. `serve --dir <that parent>`)")


def _mesh_params(args) -> MeshParams:
    return MeshParams(sides=args.sides, sphere_level=args.sphere_level,
                      min_radius_nm=args.min_radius_nm, max_radius_nm=args.max_radius_nm,
                      radius_scale=args.radius_scale, caps=args.caps)


def _report(manifest: dict, out: Path) -> None:
    print(f"wrote {manifest['n_segments']} segments, {manifest['n_groups']} groups, "
          f"{manifest['n_faces_total']:,} triangles -> {out}", flush=True)
    url_file = out / "url.txt"
    if url_file.exists():
        print(f"  neuroglancer state: {url_file}")
        print(f"  (serve first: python scripts/mesh_results.py serve --dir {out.parent})")


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

def cmd_harness(args) -> None:
    if args.atom_ids:
        atom_ids = [int(a) for a in args.atom_ids]
    elif args.population:
        atom_ids = top_atoms_by_synapse_count(args.population, args.top_n,
                                              min_synapses=args.min_synapses)
    else:
        from neuronauts.harness.geometry import AtomGeometryStore
        all_ids = sorted(AtomGeometryStore(args.geom_dir).done_atoms())
        atom_ids = all_ids[:args.top_n]
    print(f"meshing {len(atom_ids)} atoms from {args.geom_dir} "
          f"(radius column: {args.radius_col}) ...", flush=True)

    geoms, coverage = load_harness_atoms(args.geom_dir, atom_ids, radius_col=args.radius_col,
                                        min_coverage=args.min_coverage)
    dropped = [a for a in atom_ids if a not in geoms]
    if dropped:
        print(f"  dropped {len(dropped)} atoms below --min-coverage {args.min_coverage} "
              f"(attribute cache may still be fetching): {dropped[:5]}"
              f"{'...' if len(dropped) > 5 else ''}", flush=True)

    numbers = None
    if args.topology:
        with np.load(args.topology, allow_pickle=False) as z:
            idx = {int(a): k for k, a in enumerate(z["atom_id"].tolist())}
            numbers = {}
            for col in ("n_synapses", "n_pre", "n_post", "cable_nm", "caliber_mean_nm"):
                if col in z.files:
                    arr = z[col]
                    numbers[col] = {a: float(arr[idx[a]]) for a in geoms if a in idx}
    elif args.population:
        from neuronauts.harness.population import load_population
        pop = load_population(args.population)
        idx = {int(a): k for k, a in enumerate(pop.atom_id.tolist())}
        numbers = {"n_synapses": {a: float(pop.n_synapses[idx[a]]) for a in geoms if a in idx}}

    manifest = export_bundle(
        args.out, geoms, labels={a: f"atom {a}" for a in geoms}, numbers=numbers,
        params=_mesh_params(args), formats=args.formats, obj_scale=args.obj_scale,
        title=args.title or Path(args.out).name, base_url=args.base_url,
        served_root=args.served_root, clean=args.clean, verbose=True,
        source={"kind": "harness_atom_l2", "geom_dir": str(args.geom_dir),
               "radius_col": args.radius_col, "coverage": coverage},
    )
    _report(manifest, Path(args.out))


# ---------------------------------------------------------------------------
# kimimaro
# ---------------------------------------------------------------------------

def cmd_kimimaro(args) -> None:
    geoms = kimimaro_archive_skeletons(args.archive)
    if args.min_vertices > 1:
        geoms = {r: g for r, g in geoms.items() if g.n_vertices >= args.min_vertices}
    print(f"meshing {len(geoms)} root skeletons from {args.archive} ...", flush=True)
    manifest = export_bundle(
        args.out, geoms, labels={r: f"root {r}" for r in geoms},
        params=_mesh_params(args), formats=args.formats, obj_scale=args.obj_scale,
        title=args.title or Path(args.archive).stem, base_url=args.base_url,
        served_root=args.served_root, clean=args.clean, verbose=True,
        source={"kind": "kimimaro_archive", "archive": str(args.archive)},
    )
    _report(manifest, Path(args.out))


# ---------------------------------------------------------------------------
# fragments (neuronauts.schemas Fragment collection)
# ---------------------------------------------------------------------------

def cmd_fragments(args) -> None:
    from neuronauts.schemas import load_fragments

    frags = load_fragments(args.fragments)
    geoms = {f.fragment_id: SkeletonGeometry.from_fragment(f) for f in frags}
    print(f"meshing {len(geoms)} fragments from {args.fragments} ...", flush=True)

    groups = None
    if args.fragment_to_neuron:
        raw = json.loads(Path(args.fragment_to_neuron).read_text())
        groups = {int(k): str(v) for k, v in raw.items()}

    manifest = export_bundle(
        args.out, geoms, groups=groups,
        labels={f.fragment_id: f"frag {f.fragment_id} (root {f.base_root_id})" for f in frags},
        params=_mesh_params(args), formats=args.formats, obj_scale=args.obj_scale,
        write_group_meshes=bool(groups), title=args.title or Path(args.fragments).stem,
        base_url=args.base_url, served_root=args.served_root, clean=args.clean, verbose=True,
        source={"kind": "fragment_collection", "fragments": str(args.fragments),
               "fragment_to_neuron": args.fragment_to_neuron},
    )
    _report(manifest, Path(args.out))


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------

def _synthetic_branch(seed: int, origin: np.ndarray) -> SkeletonGeometry:
    rng = np.random.default_rng(seed)
    verts = [origin.astype(np.float64)]
    edges = []
    radii = [400.0]

    def grow(tip_idx: int, direction: np.ndarray, depth: int, radius: float):
        if depth <= 0:
            return
        n_steps = rng.integers(4, 9)
        cur = tip_idx
        d = direction / np.linalg.norm(direction)
        for _ in range(n_steps):
            d = d + rng.normal(scale=0.25, size=3)
            d = d / np.linalg.norm(d)
            verts.append(verts[cur] + d * 900.0)
            radii.append(max(radius, 30.0))
            edges.append((cur, len(verts) - 1))
            cur = len(verts) - 1
        if depth > 1 and rng.random() < 0.9:
            for _ in range(2 if depth > 2 else rng.integers(1, 3)):
                branch_dir = d + rng.normal(scale=1.2, size=3)
                grow(cur, branch_dir, depth - 1, radius * 0.65)

    grow(0, rng.normal(size=3), depth=4, radius=280.0)
    return SkeletonGeometry(np.asarray(verts, np.float32),
                            np.asarray(edges, np.int64).reshape(-1, 2),
                            np.asarray(radii, np.float32))


def cmd_demo(args) -> None:
    n = args.n_neurons
    rng = np.random.default_rng(0)
    geoms: dict[int, SkeletonGeometry] = {}
    groups: dict[int, str] = {}
    for i in range(n):
        origin = rng.normal(scale=15_000.0, size=3) + np.array([1_250_000.0, 965_000.0, 830_000.0])
        n_pieces = rng.integers(2, 4)
        for p in range(n_pieces):
            sid = i * 10 + p
            geoms[sid] = _synthetic_branch(seed=i * 100 + p,
                                           origin=origin + rng.normal(scale=3000.0, size=3))
            groups[sid] = f"neuron_{i:02d}"
    print(f"generating a synthetic demo: {n} neurons, {len(geoms)} fragments ...", flush=True)
    manifest = export_bundle(
        args.out, geoms, groups=groups, params=_mesh_params(args), formats=args.formats,
        obj_scale=args.obj_scale, write_group_meshes=True,
        title=args.title or "meshing demo", base_url=args.base_url,
        served_root=args.served_root, clean=True, verbose=True,
        source={"kind": "synthetic_demo", "n_neurons": n},
    )
    _report(manifest, Path(args.out))


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

def cmd_serve(args) -> None:
    serve_forever(args.dir, host=args.host, port=args.port, quiet=args.quiet)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("harness", help="mesh v117 atoms from a harness geometry dir")
    p.add_argument("--geom-dir", required=True, help="data/substrate/geom")
    p.add_argument("--population", default=None,
                   help="population.npz, for --top-n selection and synapse-count column")
    p.add_argument("--topology", default=None,
                   help="topology/*.npz (build_atom_topology.py output) for extra columns; "
                        "overrides --population's numbers")
    p.add_argument("--atom-ids", nargs="+", default=None, help="explicit atom ids (skips selection)")
    p.add_argument("--top-n", type=int, default=25)
    p.add_argument("--min-synapses", type=int, default=10)
    p.add_argument("--min-coverage", type=float, default=0.99,
                   help="drop an atom if less than this fraction of its L2 nodes have a "
                        "coordinate yet (the attribute fetch can trail the topology fetch)")
    p.add_argument("--radius-col", default="mean_dt_nm", choices=["mean_dt_nm", "max_dt_nm"])
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--obj-scale", type=float, default=1e-3)
    p.add_argument("--clean", action="store_true")
    _mesh_params_args(p)
    p.set_defaults(func=cmd_harness)

    p = sub.add_parser("kimimaro", help="mesh a kimimaro box skeleton archive")
    p.add_argument("--archive", required=True)
    p.add_argument("--min-vertices", type=int, default=5)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--obj-scale", type=float, default=1e-3)
    p.add_argument("--clean", action="store_true")
    _mesh_params_args(p)
    p.set_defaults(func=cmd_kimimaro)

    p = sub.add_parser("fragments", help="mesh a neuronauts.schemas Fragment collection")
    p.add_argument("--fragments", required=True, help="npz written by schemas.save_fragments")
    p.add_argument("--fragment-to-neuron", default=None,
                   help="JSON {fragment_id: group_label} for assembly grouping")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--obj-scale", type=float, default=1e-3)
    p.add_argument("--clean", action="store_true")
    _mesh_params_args(p)
    p.set_defaults(func=cmd_fragments)

    p = sub.add_parser("demo", help="synthetic bundle; no data on disk required")
    p.add_argument("--n-neurons", type=int, default=8)
    p.add_argument("--out", default="viz/mesh/demo")
    p.add_argument("--title", default=None)
    p.add_argument("--obj-scale", type=float, default=1e-3)
    _mesh_params_args(p)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("serve", help="serve a bundle directory with CORS")
    p.add_argument("--dir", required=True)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_serve)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
