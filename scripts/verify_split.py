#!/usr/bin/env python3
"""Independent leakage verification for a built dataset.

Deliberately a *separate code path* from `build_bench_v1.py`. A builder that
checks its own work with its own logic proves only that the logic is
self-consistent. This re-derives the checks from the written manifests, so a
bug in the builder's dedup shows up as a disagreement here.

Checks
------
1. **Root disjointness** — no v117 or label-version root may appear in two
   splits. A shared root means the model saw an evaluation neuron in training.
2. **Spatial separation** — every cross-split region pair must be separated by
   at least the declared seam buffer. Phase 2.11 measured what skipping this
   buys you: out-of-sample ARI 0.901 → 0.752 once the leak was removed, i.e.
   the un-buffered number was inflated by 0.149.
3. **Manifest integrity** — recorded stats match the recorded root lists, and
   the declared versions are the pinned ones.
4. **Gate status** — a dataset built with --relax-gates is reported as
   unusable for reported numbers.

Exits non-zero on any violation.

    python scripts/verify_split.py                     # data/bench_v1
    python scripts/verify_split.py --dataset-dir path
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from neuronauts.data.versions import BASE_VERSION, LABEL_VERSION  # noqa: E402

SPLITS = ("train", "val", "test")


def bbox_gap_nm(a, b) -> float:
    """Minimum axis-aligned separation between two boxes.

    Zero when they touch or overlap; otherwise the largest per-axis gap, since
    boxes separated on any single axis are disjoint in space.
    """
    a_lo, a_hi = np.asarray(a[0], float), np.asarray(a[1], float)
    b_lo, b_hi = np.asarray(b[0], float), np.asarray(b[1], float)
    per_axis = np.maximum(b_lo - a_hi, a_lo - b_hi)  # >0 means a gap on that axis
    return float(per_axis.max())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default="data/bench_v1")
    args = ap.parse_args()

    root = REPO / args.dataset_dir
    mdir = root / "manifests"
    if not mdir.exists():
        raise SystemExit(
            f"no manifests at {mdir}. Build the dataset first:\n"
            "    python scripts/build_bench_v1.py"
        )

    mans = {}
    for s in SPLITS:
        p = mdir / f"{s}.json"
        if not p.exists():
            raise SystemExit(f"missing manifest: {p}")
        mans[s] = json.loads(p.read_text())

    problems: list[str] = []
    notes: list[str] = []

    # -- 1. root disjointness ------------------------------------------------
    roots = {}
    for s in SPLITS:
        roots[s] = (set(mans[s]["base_root_ids"]) |
                    set(mans[s]["label_root_ids"]))
    for a, b in combinations(SPLITS, 2):
        shared = roots[a] & roots[b]
        if shared:
            sample = sorted(shared)[:5]
            problems.append(
                f"root leakage: {len(shared)} root(s) in both {a} and {b} "
                f"(e.g. {sample})"
            )
        else:
            notes.append(f"roots disjoint: {a} vs {b} "
                         f"({len(roots[a]):,} vs {len(roots[b]):,} roots)")

    # -- 2. spatial separation ----------------------------------------------
    buffers = {mans[s].get("seam_buffer_nm", 0.0) for s in SPLITS}
    if len(buffers) != 1:
        problems.append(f"splits declare different seam buffers: {buffers}")
    declared = max(buffers) if buffers else 0.0

    boxes: list[tuple[str, str, tuple]] = []
    for s in SPLITS:
        for name, info in mans[s]["regions"].items():
            bb = info["bbox_nm_buffered"]
            boxes.append((s, name, (tuple(bb[0]), tuple(bb[1]))))

    for (sa, na, ba), (sb, nb, bb_) in combinations(boxes, 2):
        if sa == sb:
            continue
        gap = bbox_gap_nm(ba, bb_)
        if gap < declared:
            problems.append(
                f"spatial leakage: {sa}/{na} and {sb}/{nb} are {gap:,.0f} nm "
                f"apart, below the declared {declared:,.0f} nm seam buffer"
            )
        else:
            notes.append(f"gap {sa}/{na} ↔ {sb}/{nb}: {gap:,.0f} nm "
                         f"(≥ {declared:,.0f})")

    # -- 3. manifest integrity ----------------------------------------------
    for s in SPLITS:
        m = mans[s]
        if m["base_version"] != BASE_VERSION or m["label_version"] != LABEL_VERSION:
            problems.append(
                f"{s}: versions {m['base_version']}->{m['label_version']} "
                f"do not match the pinned {BASE_VERSION}->{LABEL_VERSION}"
            )
        if m.get("synthetic", True):
            problems.append(f"{s}: manifest is not marked synthetic=false")
        n_declared = m["stats"]["n_base_roots"]
        n_listed = len(set(m["base_root_ids"]))
        if n_declared != n_listed:
            problems.append(
                f"{s}: stats say {n_declared} v117 roots but the root list has "
                f"{n_listed}"
            )

    # -- 4. gate status ------------------------------------------------------
    for s in SPLITS:
        if not mans[s].get("gates_passed", False):
            problems.append(
                f"{s}: built with failing acceptance gates — not usable for a "
                "reported number"
            )

    print("=== split verification ===")
    for n in notes:
        print(f"  ok  {n}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  !!  {p}")
        print(f"\n{len(problems)} problem(s). This dataset is not safe to "
              "report from.")
        return 1
    print("\nAll checks passed. Splits are root-disjoint and spatially "
          f"separated by ≥ {declared:,.0f} nm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
