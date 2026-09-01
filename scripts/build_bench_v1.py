#!/usr/bin/env python3
"""Step 3 — build the canonical `neuronauts-bench v1` train/val/test dataset.

Design principles, each earned by a specific documented failure:

1. **Split by region, never by box or at random.** A cortical arbor spans
   hundreds of microns and therefore many boxes, so box-level randomisation
   puts the same neuron in train and eval (`scripts/train.py` did exactly
   this, and had no test set at all).
2. **Labels come only from real proofreading lineage.** A v117 root's label is
   the set of label-version roots its supervoxels resolve to. Nothing is
   generated.
3. **Seam buffers + root dedup.** Phase 2.11 measured the cost of omitting
   them: out-of-sample ARI fell 0.901 → 0.752 and fk_split 0.350 → 0.000 once
   boundary leakage was removed. The leaked numbers were the optimistic ones.
4. **Fail closed.** Every acceptance gate below aborts the build rather than
   emitting a dataset that looks fine and means nothing.
5. **The test set is locked** by a manifest checksum; changing it means
   `bench_v2`, not an edit.

Outputs (under `data/bench_v1/`):
    manifests/<split>.json   committed — bboxes, root ids, counts, checksums
    regions/<region>.npz     local bulk data (gitignored; rebuildable)
    manifests/dataset.json   top-level provenance + per-split manifest hashes

Usage
-----
    python scripts/build_bench_v1.py --dry-run     # gates only, no writes
    python scripts/build_bench_v1.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ.setdefault("NEURONAUTS_SYNAPSE_CACHE_DIR", str(REPO / "cache" / "synapse"))
os.environ.setdefault("NEURONAUTS_L2_CACHE_DIR", str(REPO / "cache" / "l2_skeleton"))

import numpy as np  # noqa: E402

from neuronauts.data import lineage as L  # noqa: E402
from neuronauts.data.versions import (  # noqa: E402
    BASE_VERSION,
    LABEL_VERSION,
    verify_version_contract,
)
from scripts.survey_regions import REGIONS  # noqa: E402

DATASET_NAME = "bench_v1"
DEFAULT_SEAM_BUFFER_NM = 50_000.0

# Default split assignment. Regions are chosen for spatial disjointness; the
# survey (docs/region_inventory.md) reports which carry enough edit signal, and
# the gates below refuse the build if a chosen region does not.
DEFAULT_SPLITS: dict[str, list[str]] = {
    "train": ["A", "B", "C"],
    "val": ["E"],
    "test": ["P1"],
}

# Acceptance gates. A split that cannot clear these cannot support a claim.
MIN_MERGE_PAIRS = {"train": 20, "val": 5, "test": 10}
MIN_FRANKENMERGES = {"train": 5, "val": 1, "test": 1}


def shrink_bbox(bbox, buffer_nm: float):
    """Inset a bbox by *buffer_nm* on every face.

    The seam buffer is applied to each region independently, so any two
    regions end up separated by at least 2*buffer even if they originally
    abutted. Fragments straddling a boundary are excluded from both sides
    rather than assigned to one.
    """
    lo = np.asarray(bbox[0], float) + buffer_nm
    hi = np.asarray(bbox[1], float) - buffer_nm
    if np.any(hi <= lo):
        raise SystemExit(
            f"seam buffer {buffer_nm:.0f} nm collapses bbox {bbox}; "
            "use a smaller buffer or a larger region"
        )
    return (tuple(lo.tolist()), tuple(hi.tolist()))


def load_region(
    name: str,
    bbox,
    *,
    base_ts: int,
    label_version: int,
    limit: int,
    side: str,
    min_syn_per_fragment: int,
    tiled: bool = False,
    tile_x_nm: float = 40_000.0,
) -> dict:
    """Fetch one region and resolve real v117 -> label-version lineage."""
    if tiled:
        # Tiling avoids the per-request row cap, so the region is covered
        # rather than truncated in unstable server order.
        syn = L.fetch_region_synapses_tiled(
            bbox, version=label_version, side=side,
            tile_x_nm=tile_x_nm, per_tile_limit=limit)
    else:
        syn = L.fetch_region_synapses(
            bbox, version=label_version, side=side, limit=limit)
    if syn is None or len(syn.get("positions_nm", [])) == 0:
        raise SystemExit(
            f"[{name}] synapse fetch returned nothing. Refusing to continue — "
            "an empty fetch must never be treated as an empty region."
        )
    svids = np.asarray(syn["supervoxel_ids"], dtype=np.uint64)
    label_roots = np.asarray(syn["root_ids"], dtype=np.uint64)
    positions = np.asarray(syn["positions_nm"], dtype=np.float32)
    syn_ids = np.asarray(syn.get("synapse_ids", -np.ones(len(svids))), dtype=np.int64)

    base_roots = L.roots_at(svids, base_ts)
    if base_roots is None:
        raise SystemExit(f"[{name}] v{BASE_VERSION} lineage resolution failed.")
    base_roots = np.asarray(base_roots, dtype=np.uint64)

    valid = (base_roots != 0) & (label_roots != 0)
    base_roots, label_roots = base_roots[valid], label_roots[valid]
    positions, svids, syn_ids = positions[valid], svids[valid], syn_ids[valid]

    counts = defaultdict(int)
    for b in base_roots.tolist():
        counts[b] += 1
    keep = np.array([counts[b] >= min_syn_per_fragment for b in base_roots.tolist()],
                    dtype=bool)
    if not keep.any():
        raise SystemExit(f"[{name}] no fragment reached min_syn_per_fragment.")

    return {
        "region": name,
        "bbox_nm": bbox,
        "positions_nm": positions[keep],
        "supervoxel_ids": svids[keep],
        "synapse_ids": syn_ids[keep],
        "base_roots": base_roots[keep],
        "label_roots": label_roots[keep],
        "limit_reached": bool(len(svids) >= limit),
    }


def region_stats(base_roots: np.ndarray, label_roots: np.ndarray) -> dict:
    label_to_base: dict[int, set[int]] = defaultdict(set)
    base_to_label: dict[int, set[int]] = defaultdict(set)
    for b, l in zip(base_roots.tolist(), label_roots.tolist()):
        label_to_base[l].add(b)
        base_to_label[b].add(l)
    multi = {l: bs for l, bs in label_to_base.items() if len(bs) > 1}
    mixed = {b: ls for b, ls in base_to_label.items() if len(ls) > 1}
    return {
        "n_observations": int(len(base_roots)),
        "n_base_roots": len(base_to_label),
        "n_label_roots": len(label_to_base),
        "n_true_merge_pairs": int(
            sum(len(bs) * (len(bs) - 1) // 2 for bs in multi.values())
        ),
        "n_multi_fragment_label_roots": len(multi),
        "n_mixed_lineage_base_roots": len(mixed),
    }


def sha256_obj(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-regions", nargs="*", default=DEFAULT_SPLITS["train"])
    ap.add_argument("--val-regions", nargs="*", default=DEFAULT_SPLITS["val"])
    ap.add_argument("--test-regions", nargs="*", default=DEFAULT_SPLITS["test"])
    ap.add_argument("--seam-buffer-nm", type=float, default=DEFAULT_SEAM_BUFFER_NM)
    ap.add_argument("--limit", type=int, default=200_000,
                    help="per-tile row cap when --tiled (default), else the "
                         "total per-region cap")
    ap.add_argument("--side", default="pre", choices=["pre", "post"])
    ap.add_argument("--min-syn-per-fragment", type=int, default=3)
    ap.add_argument("--tiled", action="store_true", default=True,
                    help="fetch by x-tiles to cover the region instead of "
                         "truncating at --limit in unstable server order "
                         "(default: on)")
    ap.add_argument("--no-tiled", dest="tiled", action="store_false")
    ap.add_argument("--tile-x-nm", type=float, default=40_000.0)
    ap.add_argument("--out-dir", default=f"data/{DATASET_NAME}")
    ap.add_argument("--dry-run", action="store_true",
                    help="run every gate but write nothing")
    ap.add_argument("--relax-gates", action="store_true",
                    help="report gate failures without aborting (diagnostic "
                         "builds only; the result may not be reported)")
    args = ap.parse_args()

    prov = verify_version_contract(BASE_VERSION, LABEL_VERSION)
    base_ts = prov["base_timestamp"]
    print(f"version contract OK: v{BASE_VERSION} -> v{LABEL_VERSION}", flush=True)

    split_regions = {
        "train": args.train_regions,
        "val": args.val_regions,
        "test": args.test_regions,
    }
    all_named = [r for rs in split_regions.values() for r in rs]
    if len(set(all_named)) != len(all_named):
        raise SystemExit(f"a region appears in more than one split: {all_named}")
    unknown = [r for r in all_named if r not in REGIONS]
    if unknown:
        raise SystemExit(f"unknown region(s): {unknown}")

    # ---- fetch every region, seam-buffered ---------------------------------
    loaded: dict[str, dict] = {}
    for split, names in split_regions.items():
        for name in names:
            bbox = shrink_bbox(REGIONS[name], args.seam_buffer_nm)
            print(f"[{split}/{name}] fetching (seam-buffered) …", flush=True)
            rec = load_region(
                name, bbox, base_ts=base_ts, label_version=LABEL_VERSION,
                limit=args.limit, side=args.side,
                min_syn_per_fragment=args.min_syn_per_fragment,
                tiled=args.tiled, tile_x_nm=args.tile_x_nm,
            )
            rec["split"] = split
            st = region_stats(rec["base_roots"], rec["label_roots"])
            rec["stats"] = st
            loaded[name] = rec
            print(f"    obs={st['n_observations']:,} v117_roots={st['n_base_roots']:,} "
                  f"merge_pairs={st['n_true_merge_pairs']:,} "
                  f"frankenmerges={st['n_mixed_lineage_base_roots']:,}", flush=True)

    # ---- root dedup: test wins, then val; train yields ----------------------
    # A root appearing in two splits would let the model see an eval neuron
    # during training. Priority order keeps evaluation sets intact.
    roots_by_split: dict[str, set[int]] = {}
    for split, names in split_regions.items():
        s: set[int] = set()
        for n in names:
            s |= set(loaded[n]["base_roots"].tolist())
            s |= set(loaded[n]["label_roots"].tolist())
        roots_by_split[split] = s

    dropped = {"train": set(), "val": set()}
    dropped["val"] = roots_by_split["val"] & roots_by_split["test"]
    dropped["train"] = roots_by_split["train"] & (
        roots_by_split["test"] | roots_by_split["val"]
    )
    for split in ("val", "train"):
        if not dropped[split]:
            continue
        print(f"[dedup] removing {len(dropped[split]):,} root(s) from {split} "
              f"that also occur in a higher-priority split", flush=True)
        for name in split_regions[split]:
            rec = loaded[name]
            drop = dropped[split]
            keep = np.array(
                [(b not in drop) and (l not in drop)
                 for b, l in zip(rec["base_roots"].tolist(),
                                 rec["label_roots"].tolist())],
                dtype=bool,
            )
            for k in ("positions_nm", "supervoxel_ids", "synapse_ids",
                      "base_roots", "label_roots"):
                rec[k] = rec[k][keep]
            rec["stats"] = region_stats(rec["base_roots"], rec["label_roots"])
            rec["n_dropped_by_dedup"] = int((~keep).sum())

    # ---- acceptance gates --------------------------------------------------
    failures: list[str] = []
    split_stats: dict[str, dict] = {}
    for split, names in split_regions.items():
        b = np.concatenate([loaded[n]["base_roots"] for n in names])
        l = np.concatenate([loaded[n]["label_roots"] for n in names])
        st = region_stats(b, l)
        split_stats[split] = st
        if st["n_true_merge_pairs"] < MIN_MERGE_PAIRS[split]:
            failures.append(
                f"{split}: {st['n_true_merge_pairs']} true merge pairs "
                f"< required {MIN_MERGE_PAIRS[split]} — this split cannot "
                "distinguish a good model from a bad one (the EXP-051 failure)"
            )
        if st["n_mixed_lineage_base_roots"] < MIN_FRANKENMERGES[split]:
            failures.append(
                f"{split}: {st['n_mixed_lineage_base_roots']} frankenmerges "
                f"< required {MIN_FRANKENMERGES[split]}"
            )

    # leakage gate — must be exactly zero after dedup
    final_roots = {}
    for split, names in split_regions.items():
        s = set()
        for n in names:
            s |= set(loaded[n]["base_roots"].tolist())
            s |= set(loaded[n]["label_roots"].tolist())
        final_roots[split] = s
    for a, b_ in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = final_roots[a] & final_roots[b_]
        if shared:
            failures.append(
                f"LEAKAGE: {len(shared)} root(s) shared between {a} and {b_}"
            )

    print("\n=== acceptance gates ===")
    for split in ("train", "val", "test"):
        st = split_stats[split]
        print(f"  {split:5s} obs={st['n_observations']:>7,} "
              f"v117_roots={st['n_base_roots']:>6,} "
              f"merge_pairs={st['n_true_merge_pairs']:>5,} "
              f"frankenmerges={st['n_mixed_lineage_base_roots']:>4,}")
    if failures:
        print("\nGATE FAILURES:")
        for f in failures:
            print(f"  - {f}")
        if not args.relax_gates:
            raise SystemExit(
                "\nRefusing to write a dataset that fails its own acceptance "
                "gates. Re-run the survey (scripts/survey_regions.py) and pick "
                "regions with more edit signal, or pass --relax-gates for a "
                "diagnostic build whose numbers may not be reported."
            )
        print("\n[W] --relax-gates: this dataset FAILS its gates and may not "
              "be used for a reported number.")
    else:
        print("  all gates PASS")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    # ---- write ------------------------------------------------------------
    out = REPO / args.out_dir
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "regions").mkdir(parents=True, exist_ok=True)

    manifest_hashes = {}
    for split, names in split_regions.items():
        for n in names:
            rec = loaded[n]
            np.savez_compressed(
                out / "regions" / f"{n}.npz",
                positions_nm=rec["positions_nm"],
                supervoxel_ids=rec["supervoxel_ids"],
                synapse_ids=rec["synapse_ids"],
                base_roots=rec["base_roots"],
                label_roots=rec["label_roots"],
            )
        man = {
            "dataset": DATASET_NAME,
            "split": split,
            "base_version": BASE_VERSION,
            "label_version": LABEL_VERSION,
            "synthetic": False,
            "seam_buffer_nm": args.seam_buffer_nm,
            "side": args.side,
            "limit": args.limit,
            "min_syn_per_fragment": args.min_syn_per_fragment,
            "regions": {
                n: {
                    "bbox_nm_buffered": [list(loaded[n]["bbox_nm"][0]),
                                         list(loaded[n]["bbox_nm"][1])],
                    "bbox_nm_original": [list(REGIONS[n][0]), list(REGIONS[n][1])],
                    "stats": loaded[n]["stats"],
                    "limit_reached": loaded[n]["limit_reached"],
                    "n_dropped_by_dedup": loaded[n].get("n_dropped_by_dedup", 0),
                }
                for n in names
            },
            "stats": split_stats[split],
            "base_root_ids": sorted(
                {int(x) for n in names
                 for x in loaded[n]["base_roots"].tolist()}
            ),
            "label_root_ids": sorted(
                {int(x) for n in names
                 for x in loaded[n]["label_roots"].tolist()}
            ),
            "gates_passed": not failures,
        }
        p = out / "manifests" / f"{split}.json"
        p.write_text(json.dumps(man, indent=2) + "\n")
        manifest_hashes[split] = sha256_obj(man)
        print(f"wrote {p.relative_to(REPO)}  sha256={manifest_hashes[split][:16]}…")

    dataset = {
        "dataset": DATASET_NAME,
        "synthetic": False,
        "provenance": prov,
        "splits": split_regions,
        "seam_buffer_nm": args.seam_buffer_nm,
        "manifest_sha256": manifest_hashes,
        "gates_passed": not failures,
        "gate_failures": failures,
        "split_stats": split_stats,
    }
    dp = out / "manifests" / "dataset.json"
    dp.write_text(json.dumps(dataset, indent=2) + "\n")
    combined = sha256_obj(manifest_hashes)
    print(f"\nwrote {dp.relative_to(REPO)}")
    print(f"dataset manifest sha256 = {combined}")
    print("Stamp this hash into every ResultsRecord produced from this dataset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
