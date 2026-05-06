#!/usr/bin/env python3
"""Pre-proofread CAVE seg-ID baseline F1.

For each box in --cache-dir, group synapses on each side by their pre/post
CAVE seg ID and evaluate F1 against the ground-truth pre/post root IDs.
Synapses with seg_id == 0 (unset) are each given a unique singleton label so
they cannot accidentally merge with other unsegmented points.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuronauts.dataset_builder import BoxCache
from neuronauts.cell_graph import connectivity_graph_from_cell_labels
from neuronauts.line_graph import evaluate as lg_evaluate


def _seg_to_labels(seg_ids: np.ndarray) -> np.ndarray:
    """Map seg IDs → contiguous labels; treat seg_id==0 as singletons."""
    seg_ids = np.asarray(seg_ids, dtype=np.int64)
    labels = np.empty_like(seg_ids)
    next_label = 0
    seg_to_label: dict[int, int] = {}
    for i, s in enumerate(seg_ids):
        s_int = int(s)
        if s_int == 0:
            labels[i] = next_label
            next_label += 1
        else:
            if s_int not in seg_to_label:
                seg_to_label[s_int] = next_label
                next_label += 1
            labels[i] = seg_to_label[s_int]
    return labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--min-synapses", type=int, default=10)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    cache = BoxCache(args.cache_dir)
    records = [r for r in cache.all_records() if r.n_synapses >= args.min_synapses]
    if not records:
        print(f"No boxes with >= {args.min_synapses} synapses in {args.cache_dir}")
        return 1

    print(f"Evaluating seg-ID baseline on {len(records)} boxes from {args.cache_dir}")
    t0 = time.perf_counter()

    f1s, precs, recs = [], [], []
    skipped = 0
    per_box = []
    for rec in records:
        try:
            _, syn = cache.load(rec)
        except Exception as exc:
            print(f"  [skip] {rec.box_hash[:8]}: load failed ({exc})")
            skipped += 1
            continue
        pre_seg = getattr(syn, "pre_seg_id", None)
        post_seg = getattr(syn, "post_seg_id", None)
        if pre_seg is None or post_seg is None:
            print(f"  [skip] {rec.box_hash[:8]}: no seg ids")
            skipped += 1
            continue

        pre_labels = _seg_to_labels(pre_seg)
        post_labels = _seg_to_labels(post_seg)
        cg = connectivity_graph_from_cell_labels(pre_labels, post_labels, syn)
        m = lg_evaluate(cg, syn.pre_root_id, syn.post_root_id)

        f1s.append(m.f1)
        precs.append(m.precision)
        recs.append(m.recall)
        per_box.append({
            "box_hash": rec.box_hash,
            "n_synapses": int(rec.n_synapses),
            "f1": float(m.f1),
            "precision": float(m.precision),
            "recall": float(m.recall),
        })
        print(
            f"  {rec.box_hash[:8]} ({rec.n_synapses:>5} syn): "
            f"F1={m.f1:.3f}  P={m.precision:.3f}  R={m.recall:.3f}"
        )

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"Cache:        {args.cache_dir}")
    print(f"Boxes:        {len(f1s)} evaluated, {skipped} skipped")
    print(f"Wall time:    {elapsed:.1f}s")
    if f1s:
        print(
            f"F1:           mean={np.mean(f1s):.4f}  median={np.median(f1s):.4f}  "
            f"min={min(f1s):.4f}  max={max(f1s):.4f}"
        )
        print(f"Precision:    mean={np.mean(precs):.4f}")
        print(f"Recall:       mean={np.mean(recs):.4f}")

    if args.output_json:
        out = {
            "cache_dir": args.cache_dir,
            "min_synapses": args.min_synapses,
            "n_boxes": len(f1s),
            "n_skipped": skipped,
            "wall_seconds": elapsed,
            "f1_mean": float(np.mean(f1s)) if f1s else None,
            "f1_median": float(np.median(f1s)) if f1s else None,
            "precision_mean": float(np.mean(precs)) if precs else None,
            "recall_mean": float(np.mean(recs)) if recs else None,
            "per_box": per_box,
        }
        Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
