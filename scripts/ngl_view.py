#!/usr/bin/env python3
"""Build (and optionally serve) a Neuroglancer view of project data.

    ngl_view.py atom 864691135361314119            # one v117 atom from the harness cache
    ngl_view.py experiment EXP-056                 # the box a benchmark recorded
    ngl_view.py region --centre-um 663 591 860 --side-um 100
    ngl_view.py state results/reports/ngl/EXP-056_bbox.json --serve

Each subcommand writes a state JSON (``--out``) and prints a viewer URL when
the state is small enough to carry in one. ``--serve`` opens a local viewer
through the ``neuroglancer`` package instead, which is the route for atoms
with tens of thousands of L2 edges.

Atom views read the harness caches: ``data/substrate/geom`` (L2 ids and real
adjacency per atom), ``l2_attributes.npz`` (coordinates), and optionally the
topology table (endpoints) and population (synapses). Loading the attribute
cache costs a few seconds; pass several atom ids to amortise it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuronauts.report import ngl  # noqa: E402


def _emit(state: ngl.NglState | dict, out: Path | None, viewer: str, serve: bool,
          label: str) -> None:
    state_dict = state.to_dict() if isinstance(state, ngl.NglState) else state
    if out is not None:
        ngl.save_state(state_dict, out)
        print(f"{label}: state -> {out}")
    url = ngl.state_to_url(state_dict, viewer)
    if url:
        print(f"{label}: {url}")
    else:
        print(f"{label}: state exceeds {ngl.MAX_URL_BYTES:,} bytes as a URL; "
              f"use --serve or paste the JSON into the viewer")
    if serve:
        local = ngl.serve(state_dict)
        print(f"{label}: serving at {local}  (Ctrl-C to stop)")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


def cmd_atom(args) -> int:
    t0 = time.time()
    positions = ngl.L2Positions(Path(args.geom_dir) / "l2_attributes.npz")
    print(f"attribute cache: {len(positions.ids):,} L2 nodes in {time.time() - t0:.1f}s")
    for atom in args.atom_id:
        st, summary = ngl.atom_view(
            atom, geom_dir=args.geom_dir, positions=positions,
            topology_npz=args.topology, population_npz=args.population,
            seg_timestamp=None if args.current_segmentation else ngl.V117_TIMESTAMP,
            max_annotations=args.max_annotations)
        print(json.dumps(summary))
        out = Path(args.out) if args.out and len(args.atom_id) == 1 else \
            (Path(args.out_dir) / f"atom_{atom}.json")
        _emit(st, out, args.viewer, args.serve, f"atom {atom}")
    return 0


def cmd_experiment(args) -> int:
    from neuronauts.report.registry import discover
    recs = [r for r in discover("results", ROOT) if r.id.upper() == args.id.upper()]
    if not recs:
        print(f"no result named {args.id}", file=sys.stderr)
        return 2
    st = ngl.experiment_view(recs[0])
    if st is None:
        print(f"{args.id} recorded no bounding box or anchor", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else Path(args.out_dir) / f"{recs[0].id}_bbox.json"
    _emit(st, out, args.viewer, args.serve, recs[0].id)
    return 0


def cmd_region(args) -> int:
    st = ngl.region_view(args.centre_um, args.side_um)
    out = Path(args.out) if args.out else \
        Path(args.out_dir) / f"region_{'_'.join(f'{c:g}' for c in args.centre_um)}_{args.side_um:g}um.json"
    _emit(st, out, args.viewer, args.serve, "region")
    return 0


def cmd_state(args) -> int:
    state = json.loads(Path(args.path).read_text())
    _emit(state, None, args.viewer, args.serve, Path(args.path).stem)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--viewer", default=ngl.DEFAULT_VIEWER, choices=sorted(ngl.VIEWERS))
    ap.add_argument("--serve", action="store_true",
                    help="open a local viewer via the neuroglancer package")
    ap.add_argument("--out", default=None, help="state JSON path (single view)")
    ap.add_argument("--out-dir", default="results/reports/ngl")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("atom", help="one or more v117 atoms from the harness cache")
    a.add_argument("atom_id", type=int, nargs="+")
    a.add_argument("--geom-dir", default="data/substrate/geom")
    a.add_argument("--topology", default="data/substrate/topology/k10.npz",
                   help="endpoint table; pass '' to skip")
    a.add_argument("--population", default="data/substrate/c100um/population.npz",
                   help="synapse table; pass '' to skip")
    a.add_argument("--current-segmentation", action="store_true",
                   help="show the live segmentation instead of v117")
    a.add_argument("--max-annotations", type=int, default=20_000)
    a.set_defaults(fn=cmd_atom)

    e = sub.add_parser("experiment", help="the box a benchmark recorded")
    e.add_argument("id")
    e.set_defaults(fn=cmd_experiment)

    r = sub.add_parser("region", help="a cube of interest")
    r.add_argument("--centre-um", type=float, nargs=3, required=True)
    r.add_argument("--side-um", type=float, required=True)
    r.set_defaults(fn=cmd_region)

    s = sub.add_parser("state", help="print a URL for / serve an existing state JSON")
    s.add_argument("path")
    s.set_defaults(fn=cmd_state)

    args = ap.parse_args()
    if getattr(args, "topology", None) == "":
        args.topology = None
    if getattr(args, "population", None) == "":
        args.population = None
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
