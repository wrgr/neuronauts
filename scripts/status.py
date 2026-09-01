#!/usr/bin/env python3
"""Where the consolidation and the experiment program stand, derived from disk.

    uv run python scripts/status.py              # terminal view
    uv run python scripts/status.py --markdown   # for a doc or an issue
    uv run python scripts/status.py --json       # for a machine
    uv run python scripts/status.py --next       # what could run right now
    uv run python scripts/status.py --fast       # skip the pytest collection

Nothing is hand-maintained: a phase is done when the thing it promised is on
disk, and an experiment has a state because its result file says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuronauts.report import tracker  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--html", action="store_true",
                    help="a shareable status board")
    ap.add_argument("--next", action="store_true",
                    help="list experiments whose prerequisites are met")
    ap.add_argument("--no-experiments", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="skip the pytest collection check")
    ap.add_argument("--out", default="", help="also write to this path")
    args = ap.parse_args()

    if args.fast:
        tracker._cache["collect"] = (-1, 0)   # reported as unknown, not asserted

    if args.next:
        from neuronauts.experiments.registry import blocked, next_runnable
        ready = next_runnable()
        if ready:
            print("ready to run:")
            for e in ready:
                mins = f" (~{e.est_minutes} min)" if e.est_minutes else ""
                print(f"  {e.id}  {e.spec.title}{mins}")
                print(f"      bar: {e.spec.criterion}")
        else:
            print("nothing is ready. blocked on:")
            for e, why in blocked()[:6]:
                print(f"  {e.id}  {why[0]}")
        return 0

    if args.html:
        text = tracker.render_html()
    elif args.json:
        text = json.dumps(tracker.as_dict(), indent=2)
    elif args.markdown:
        text = tracker.render_markdown()
    else:
        text = tracker.render_text(show_experiments=not args.no_experiments)

    if not (args.html and args.out):
        print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text if text.endswith("\n") else text + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
