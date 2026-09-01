#!/usr/bin/env python3
"""Step 2 — edit-signal survey over candidate regions.

Measures, per region, how much *real proofreading signal* it contains, so the
train/val/test splits can be chosen from evidence instead of convention.

Why this exists
---------------
EXP-051 ran a methodologically correct benchmark on a box containing exactly
**one** true merge pair among 21,175 candidate joins, and EXP-052's
soma-anchored box had 14. Neither could distinguish a good model from a bad
one. `docs/dataset_seeding_for_edit_pairs.md` measured the underlying cause:
ordinary spatial boxes show 0% v117≠current lineage divergence, while
proofread-anchored neighbourhoods show 28%. Region choice, not model choice,
decided those outcomes.

What is measured
----------------
For every synapse in a region we resolve its supervoxel to a root at both
versions: v117 (the segmentation being corrected) and the label version (the
proofread state). That gives two quantities that matter:

- **true merge pairs** — two v117 roots that resolve to the same label root.
  The segmentation split one neuron; a proofreader merged it. These are the
  positives any merge model must recover.
- **mixed-lineage (frankenmerge) roots** — one v117 root spanning two label
  roots. The segmentation merged two neurons; a proofreader split them. These
  are the label-noise floor: EXP-056 showed geometry alone cannot cleave them.

Fail-closed: no synthetic fallback, no silent empty result. A region that
cannot be fetched is recorded as an error, never as a zero.

Usage
-----
    python scripts/survey_regions.py                      # all regions
    python scripts/survey_regions.py --regions P1 A B     # a subset
    python scripts/survey_regions.py --limit 20000        # per-region cap
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Serve fetches from the repo cache so repeated runs see identical data. The
# server-side `limit` has no stable order, so an uncached over-limit bbox
# returns a different subset every call (lineage.py:595-598).
os.environ.setdefault("NEURONAUTS_SYNAPSE_CACHE_DIR", str(REPO / "cache" / "synapse"))
os.environ.setdefault("NEURONAUTS_L2_CACHE_DIR", str(REPO / "cache" / "l2_skeleton"))

import numpy as np  # noqa: E402

from neuronauts.data import lineage as L  # noqa: E402
from neuronauts.data.versions import (  # noqa: E402
    BASE_VERSION,
    LABEL_VERSION,
    verify_version_contract,
)

# ---------------------------------------------------------------------------
# Candidate regions (nm, synapse-table 4x4x40 frame)
#
# Sources, all already in the repo:
#   A-E   scripts/train_l2_partition.py:52-58 (matches spatial_variance.py
#         _ALL_TRAIN with seam_buffer=50k)
#   P1    scripts/train_l2_partition.py:47 — found by scanning nucleus edit
#         rate; ~100% of somata here have v117 != v1718
#   T1-T4 scripts/spatial_variance.py:308-331 — evaluation locations
#   OOC   out-of-column probes; v117≈v1718 there, so they carry no real labels
# ---------------------------------------------------------------------------

REGIONS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    # Densely proofread column core
    "P1": ((818_500, 685_000, 794_000), (918_500, 785_000, 994_000)),
    # P1 split into z-thirds. The survey showed the column's edit signal is
    # concentrated here, so val and test are drawn from P1 rather than from a
    # signal-poor region — otherwise they cannot measure anything. Defined
    # abutting; build_bench_v1.apply_seam_buffers opens the gaps between them,
    # and exact root dedup is the actual disjointness guarantee.
    "P1a": ((818_500, 685_000, 794_000), (918_500, 785_000, 860_667)),
    "P1b": ((818_500, 685_000, 860_667), (918_500, 785_000, 927_333)),
    "P1c": ((818_500, 685_000, 927_333), (918_500, 785_000, 994_000)),
    # Training-region registry
    "A": ((750_000, 930_000, 780_000), (950_000, 1_000_000, 880_000)),
    "B": ((950_000, 930_000, 780_000), (1_100_000, 1_000_000, 880_000)),
    "C": ((1_400_000, 930_000, 780_000), (1_550_000, 1_000_000, 880_000)),
    "D": ((1_600_000, 930_000, 875_000), (1_750_000, 1_000_000, 1_100_000)),
    "E": ((750_000, 1_000_000, 780_000), (950_000, 1_070_000, 880_000)),
    # Evaluation locations
    "T1": ((1_150_000, 930_000, 780_000), (1_350_000, 1_000_000, 880_000)),
    "T2": ((550_000, 930_000, 780_000), (750_000, 1_000_000, 880_000)),
    "T3": ((1_150_000, 870_000, 780_000), (1_350_000, 940_000, 880_000)),
    "T4": ((1_150_000, 1_000_000, 780_000), (1_350_000, 1_070_000, 880_000)),
    # Out-of-column (expected: no edit signal; used as a transfer probe only)
    "OOC1": ((200_000, 500_000, 700_000), (400_000, 570_000, 800_000)),
    "OOC2": ((1_200_000, 400_000, 700_000), (1_400_000, 470_000, 800_000)),
    "OOC3": ((600_000, 600_000, 700_000), (800_000, 670_000, 800_000)),
    # Validated densely-proofread centre used by the pcfg + EXP-051/056 work
    "PCFG": ((718_592, 498_592, 580_640), (748_592, 528_592, 610_640)),
}


def survey_region(
    name: str,
    bbox: tuple,
    *,
    base_ts: int,
    label_version: int,
    limit: int,
    side: str,
    min_syn_per_fragment: int,
    token: Optional[str] = None,
) -> dict:
    """Measure the real proofreading signal in one region. Never fabricates."""
    rec: dict = {
        "region": name,
        "bbox_nm": [list(bbox[0]), list(bbox[1])],
        "side": side,
        "limit": limit,
        "min_syn_per_fragment": min_syn_per_fragment,
    }
    extent = np.asarray(bbox[1], float) - np.asarray(bbox[0], float)
    rec["volume_um3"] = float(np.prod(extent / 1000.0))

    syn = L.fetch_region_synapses(
        bbox, version=label_version, side=side, limit=limit,
        **({"token": token} if token else {}),
    )
    if syn is None or len(syn.get("positions_nm", [])) == 0:
        rec["status"] = "fetch_failed_or_empty"
        rec["n_synapses"] = 0
        return rec

    svids = np.asarray(syn["supervoxel_ids"], dtype=np.uint64)
    label_roots = np.asarray(syn["root_ids"], dtype=np.uint64)
    n = len(svids)
    rec["n_synapses"] = int(n)
    # A fetch that exactly hits the cap was truncated in server order, which is
    # not stable; record it so the reader knows the counts are a lower bound.
    rec["limit_reached"] = bool(n >= limit)

    base_roots = L.roots_at(svids, base_ts, **({"token": token} if token else {}))
    if base_roots is None:
        rec["status"] = "base_lineage_resolution_failed"
        return rec
    base_roots = np.asarray(base_roots, dtype=np.uint64)

    valid = (base_roots != 0) & (label_roots != 0)
    rec["n_unresolved"] = int((~valid).sum())
    base_roots, label_roots = base_roots[valid], label_roots[valid]
    if len(base_roots) == 0:
        rec["status"] = "no_resolvable_synapses"
        return rec

    # Drop sliver fragments: too few observations to support any decision.
    counts = defaultdict(int)
    for b in base_roots:
        counts[int(b)] += 1
    keep = np.array([counts[int(b)] >= min_syn_per_fragment for b in base_roots])
    base_roots, label_roots = base_roots[keep], label_roots[keep]
    rec["n_observations_kept"] = int(len(base_roots))
    if len(base_roots) == 0:
        rec["status"] = "all_fragments_below_min_syn"
        return rec

    # label root -> distinct v117 roots  (a split the proofreader repaired)
    label_to_base: dict[int, set[int]] = defaultdict(set)
    # v117 root -> distinct label roots  (a merge error = frankenmerge)
    base_to_label: dict[int, set[int]] = defaultdict(set)
    for b, l in zip(base_roots.tolist(), label_roots.tolist()):
        label_to_base[l].add(b)
        base_to_label[b].add(l)

    multi = {l: bs for l, bs in label_to_base.items() if len(bs) > 1}
    n_pairs = sum(len(bs) * (len(bs) - 1) // 2 for bs in multi.values())
    mixed = {b: ls for b, ls in base_to_label.items() if len(ls) > 1}

    rec.update({
        "status": "ok",
        "n_base_roots": len(base_to_label),
        "n_label_roots": len(label_to_base),
        "n_multi_fragment_label_roots": len(multi),
        "n_true_merge_pairs": int(n_pairs),
        "n_mixed_lineage_base_roots": len(mixed),
        "frankenmerge_rate": (
            round(len(mixed) / max(1, len(base_to_label)), 5)
        ),
        "merge_pairs_per_1k_synapses": (
            round(1000.0 * n_pairs / max(1, len(base_roots)), 3)
        ),
        "max_fragments_per_label_root": (
            max((len(bs) for bs in label_to_base.values()), default=0)
        ),
    })
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--regions", nargs="*", default=None,
                    help="subset of region names (default: all)")
    ap.add_argument("--limit", type=int, default=20_000,
                    help="max synapses fetched per region")
    ap.add_argument("--side", default="pre", choices=["pre", "post"])
    ap.add_argument("--min-syn-per-fragment", type=int, default=1,
                    help="Keep roots with at least this many synapses. "
                         "DEFAULT 1 = keep everything. Raising it hides "
                         "the sliver/singleton confuser population and "
                         "the true positives that involve it.")
    ap.add_argument("--label-version", type=int, default=LABEL_VERSION)
    ap.add_argument("--base-version", type=int, default=BASE_VERSION)
    ap.add_argument("--out-json", default="results/region_inventory.json")
    ap.add_argument("--out-md", default="docs/region_inventory.md")
    args = ap.parse_args()

    # Fail closed before any fetch: pinned versions must actually exist.
    prov = verify_version_contract(args.base_version, args.label_version)
    base_ts = prov["base_timestamp"]
    print(f"version contract OK: v{args.base_version} (ts {base_ts}) "
          f"-> v{args.label_version} (ts {prov['label_timestamp']})", flush=True)

    names = args.regions or list(REGIONS)
    unknown = [n for n in names if n not in REGIONS]
    if unknown:
        raise SystemExit(f"unknown region(s): {unknown}. Known: {list(REGIONS)}")

    records = []
    for name in names:
        print(f"[{name}] surveying …", flush=True)
        try:
            rec = survey_region(
                name, REGIONS[name],
                base_ts=base_ts,
                label_version=args.label_version,
                limit=args.limit,
                side=args.side,
                min_syn_per_fragment=args.min_syn_per_fragment,
            )
        except Exception as exc:  # record, never silently zero
            rec = {"region": name, "status": f"error: {type(exc).__name__}: {exc}"}
        records.append(rec)
        if rec.get("status") == "ok":
            print(f"  synapses={rec['n_synapses']:,} "
                  f"v117_roots={rec['n_base_roots']:,} "
                  f"merge_pairs={rec['n_true_merge_pairs']:,} "
                  f"frankenmerges={rec['n_mixed_lineage_base_roots']:,}", flush=True)
        else:
            print(f"  status={rec['status']}", flush=True)

    ok = [r for r in records if r.get("status") == "ok"]
    ok.sort(key=lambda r: r["n_true_merge_pairs"], reverse=True)

    out = {
        "provenance": prov,
        "survey_params": {
            "limit": args.limit, "side": args.side,
            "min_syn_per_fragment": args.min_syn_per_fragment,
        },
        "synthetic": False,
        "regions": records,
        "ranking_by_merge_pairs": [r["region"] for r in ok],
    }
    pj = REPO / args.out_json
    pj.parent.mkdir(parents=True, exist_ok=True)
    pj.write_text(json.dumps(out, indent=2) + "\n")

    lines = [
        "# Region inventory — where the real proofreading signal is",
        "",
        "Generated by `scripts/survey_regions.py`. Real CAVE data only; no",
        "synthetic fallback. Regenerate with:",
        "",
        "```bash",
        f"python scripts/survey_regions.py --limit {args.limit}",
        "```",
        "",
        f"Base v{args.base_version} → labels v{args.label_version}. "
        f"Per-region cap {args.limit:,} synapses ({args.side}-side), "
        f"fragments with <{args.min_syn_per_fragment} synapses dropped.",
        "",
        "**true merge pairs** = two v117 roots sharing one label root (the",
        "positives a merge model must recover). **frankenmerges** = one v117",
        "root spanning two label roots (the label-noise floor).",
        "",
        "| Region | Vol (µm³) | Synapses | v117 roots | Label roots | "
        "True merge pairs | Frankenmerges | FK rate | Pairs/1k syn |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ok:
        lines.append(
            f"| {r['region']} | {r['volume_um3']:,.0f} | {r['n_synapses']:,} | "
            f"{r['n_base_roots']:,} | {r['n_label_roots']:,} | "
            f"**{r['n_true_merge_pairs']:,}** | {r['n_mixed_lineage_base_roots']:,} | "
            f"{r['frankenmerge_rate']:.4f} | {r['merge_pairs_per_1k_synapses']:.2f} |"
        )
    bad = [r for r in records if r.get("status") != "ok"]
    if bad:
        lines += ["", "## Regions that did not survey", "",
                  "| Region | Status |", "|---|---|"]
        lines += [f"| {r['region']} | `{r.get('status')}` |" for r in bad]
    truncated = [r["region"] for r in ok if r.get("limit_reached")]
    if truncated:
        lines += ["", f"> **Note:** {', '.join(truncated)} hit the "
                  f"{args.limit:,}-synapse cap, so their counts are lower "
                  "bounds. The server applies no stable ordering to a "
                  "truncated fetch; results are served from `cache/synapse/` "
                  "so repeated runs stay identical."]
    pm = REPO / args.out_md
    pm.parent.mkdir(parents=True, exist_ok=True)
    pm.write_text("\n".join(lines) + "\n")

    print(f"\nwrote {pj.relative_to(REPO)} and {pm.relative_to(REPO)}")
    if ok:
        top = ok[0]
        print(f"richest region: {top['region']} with "
              f"{top['n_true_merge_pairs']:,} true merge pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
