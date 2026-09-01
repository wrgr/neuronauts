#!/usr/bin/env python3
"""Build a Markdown report, figures, and Neuroglancer views for every result.

Reads every JSON directly under ``results/`` (benchmarks ``expNNN_*.json``,
harness summaries, probes), normalises each into an ``ExperimentRecord``, and
writes under ``results/reports/``:

  README.md              index: status, commit, provenance grade, dependency graph
  <ID>.md                one report per result
  figures/<ID>_*.png     figures derived from the result's tables
  ngl/<ID>_*.json        Neuroglancer states for the spatial context recorded

Nothing here touches the network. Rerun after any experiment writes a result.

    uv run python scripts/build_reports.py
    uv run python scripts/build_reports.py --only EXP-056 --no-figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuronauts.report import ngl  # noqa: E402
from neuronauts.report.figures import figures_for_record  # noqa: E402
from neuronauts.report.registry import discover  # noqa: E402
from neuronauts.report.render import render_experiment, render_index  # noqa: E402


def ngl_views_for(rec, out_dir: Path, viewer: str) -> dict[str, dict]:
    """Region/experiment views from spatial provenance; atoms are per-atom (ngl_view.py)."""
    views: dict[str, dict] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    st = ngl.experiment_view(rec)
    if st is not None:
        path = st.save(out_dir / f"{rec.id}_bbox.json")
        url = st.to_url(viewer)
        views["experiment box"] = {
            "json": path, "url": url,
            "note": "bounding box, anchor soma and anchor root from `provenance`"}

    centre = rec.raw.get("config", {}).get("centre_um") if isinstance(rec.raw.get("config"), dict) else None
    side = rec.raw.get("config", {}).get("side_um") if isinstance(rec.raw.get("config"), dict) else None
    if centre is not None and side is not None:
        st = ngl.region_view(centre, side)
        path = st.save(out_dir / f"{rec.id}_region.json")
        views["region"] = {"json": path, "url": st.to_url(viewer),
                           "note": f"{side} um cube at {centre} um from `config`"}
    return views


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/reports")
    ap.add_argument("--only", action="append", default=[],
                    help="record id to build (repeatable); default all")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--viewer", default=ngl.DEFAULT_VIEWER,
                    choices=sorted(ngl.VIEWERS))
    args = ap.parse_args()

    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    records = discover(args.results, ROOT)
    if args.only:
        wanted = {w.upper() for w in args.only}
        records = [r for r in records if r.id.upper() in wanted]
        if not records:
            print(f"no record matches {sorted(wanted)}", file=sys.stderr)
            return 2

    report_paths: dict[str, Path] = {}
    for rec in records:
        figs = [] if args.no_figures else figures_for_record(rec, out_dir / "figures")
        views = ngl_views_for(rec, out_dir / "ngl", args.viewer)
        path = render_experiment(rec, figs, views, out_dir, ROOT)
        report_paths[rec.id] = path
        grade = rec.provenance_grade.get("score", 0)
        print(f"{rec.id:<24} {rec.status:<20} tables={len(rec.tables):<2} "
              f"figures={len(figs):<2} views={len(views)} provenance={grade:.0%}  "
              f"-> {path.relative_to(ROOT)}")

    if not args.only:
        index = render_index(records, report_paths, out_dir, ROOT)
        print(f"index -> {index.relative_to(ROOT)}")
    else:
        print("index not rebuilt (partial build); run without --only to refresh it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
