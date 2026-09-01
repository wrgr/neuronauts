#!/usr/bin/env python3
"""EXP-052: proofread-anchor-seeded dense v117 soma-seeded grammar assembly.

Fixes EXP-051 sampling failure:
  1. Box centered on proofread soma ranked by synapse count (static nucleus table).
  2. v117 roots not in the static synapse table are discarded from the candidate
     population (non-synapse-bearing structural fragments / glia have no identity
     signal and inflated EXP-051 confuser set).
  3. Pre-flight asserts >= N true v117->v1412 merge pairs before inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuronauts.data import lineage as L
from neuronauts.grammar import featurize_path_points
from neuronauts.real_dense_soma import (
    Fragment,
    assert_real_root_ids,
    build_candidate_edges_batched,
    partition_metrics,
    single_soma_compliance,
    skeleton_from_observed_points,
    soma_seeded_assemble,
)
from neuronauts.shared_grammar_model import load_shared_grammar_model
from treestitch.realworld import _load_nucleus_somas

# (4, 4, 40) nm per voxel -- matches _load_nucleus_somas convention
VOX_NM = np.asarray([4.0, 4.0, 40.0], dtype=np.float64)

SYNAPSE_COUNTS_TSV = ROOT / "run_logs/synapse_root_counts_static.tsv"
NUCLEUS_CSV = ROOT / "data/microns_static/v1078/nucleus_detection_v0.csv"


# ---------------------------------------------------------------------------
# Anchor selection
# ---------------------------------------------------------------------------

def load_synapse_table(path: Path) -> tuple:
    """Single-pass read of the static synapse counts TSV.

    Returns:
        synapse_root_set  -- frozenset of root_ids (for filter_to_synapse_table)
        synapse_counts    -- dict[int, int] root_id -> total_synapse_count

    One pass over the 130M-row file; avoids reading it twice.
    """
    import csv
    roots: set = set()
    counts: dict = {}
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rid = int(row["root_id"])
            if rid <= 0:
                continue
            roots.add(rid)
            has_soma = row.get("has_soma", "False").strip().lower() in ("true", "1", "yes")
            if has_soma:
                counts[rid] = int(row["total_synapse_count"])
    return frozenset(roots), counts


def select_anchor_soma(
    nucleus_csv: Path,
    synapse_counts: dict,
    *,
    anchor_rank: int = 0,
) -> dict:
    """Return Nth-ranked proofread soma by total synapse count.

    ``synapse_counts`` is the dict[root_id -> total_synapse_count] returned by
    ``load_synapse_table`` (has_soma=True rows only).  No TSV re-read.
    """
    import csv
    nuc_by_root: dict = {}
    with nucleus_csv.open("r", newline="") as fh:
        for row in csv.DictReader(fh):
            root = int(row["pt_root_id"]); sv = int(row["pt_supervoxel_id"])
            if root <= 0 or sv <= 0: continue
            nuc_by_root[root] = {
                "nucleus_id": int(row["id"]), "root_id": root,
                "x_nm": float(int(row["pt_position_x"]) * VOX_NM[0]),
                "y_nm": float(int(row["pt_position_y"]) * VOX_NM[1]),
                "z_nm": float(int(row["pt_position_z"]) * VOX_NM[2]),
            }
    ranked: list = []
    for root, total in synapse_counts.items():
        if root not in nuc_by_root:
            continue
        entry = dict(nuc_by_root[root])
        entry["total_synapse_count"] = total
        ranked.append(entry)
    if not ranked:
        raise RuntimeError(
            "No proofread soma found in synapse counts. "
            f"Check {nucleus_csv} and synapse_counts input."
        )
    ranked.sort(key=lambda e: -e["total_synapse_count"])
    if anchor_rank >= len(ranked):
        raise RuntimeError(
            f"anchor_rank={anchor_rank} >= {len(ranked)} available somas")
    anchor = dict(ranked[anchor_rank]); anchor["anchor_rank"] = anchor_rank
    return anchor


# ---------------------------------------------------------------------------
# Synapse-table filter
# ---------------------------------------------------------------------------

def filter_to_synapse_table(all_roots: list, synapse_root_set: frozenset) -> tuple:
    """Return (kept, n_discarded): keep only roots present in static synapse table."""
    kept = [r for r in all_roots if r in synapse_root_set]
    return kept, len(all_roots) - len(kept)


# ---------------------------------------------------------------------------
# CAVE helpers (identical to EXP-051)
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as h:
        for block in iter(lambda: h.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def box_around_soma(center_nm: np.ndarray, side_nm: float) -> tuple:
    half = side_nm / 2.0
    return tuple(float(v) for v in center_nm - half), tuple(float(v) for v in center_nm + half)


def fetch_candidates(bbox: tuple, *, target_timestamp: int, token: str) -> tuple:
    from neuronauts.bulk_synapses import fetch_synapses_bulk
    cache_key = hashlib.sha1(json.dumps(bbox).encode()).hexdigest()[:16]
    cache_path = Path("/tmp") / f"neuronauts_exp052_synapses_{cache_key}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as c: bulk = {k: c[k] for k in c.files}
    else:
        raw = fetch_synapses_bulk(bbox, token, version=117, use_version_roots=True)
        keys = ("pre_pt_nm","post_pt_nm","pre_root_id","post_root_id","pre_supervoxel_id","post_supervoxel_id","synapse_id")
        bulk = {k: np.asarray(raw[k]) for k in keys}
        np.savez_compressed(cache_path, **bulk)
    mapped = {}; points: dict = {}; votes: dict = {}; role_counts: dict = {}
    for side, other in (("pre","post"),("post","pre")):
        roots = np.asarray(bulk[f"{side}_root_id"], dtype=np.uint64)
        svs = np.asarray(bulk[f"{side}_supervoxel_id"], dtype=np.uint64)
        labels = L.roots_at(svs, target_timestamp, token=token)
        if labels is None: raise RuntimeError(f"{side}-side v1412 lineage mapping failed")
        mapped[side] = {"v117": roots, "target": labels, "other_supervoxels": bulk[f"{other}_supervoxel_id"]}
        for root, label, pos in zip(roots, labels, bulk[f"{side}_pt_nm"]):
            rid, tid = int(root), int(label)
            if rid <= 0: continue
            points.setdefault(rid, []).append(np.asarray(pos, dtype=np.float32))
            role_counts.setdefault(rid, Counter())[side] += 1
            if tid > 0: votes.setdefault(rid, []).append(tid)
    label_map = {}
    for root, lbls in votes.items():
        u, c = np.unique(lbls, return_counts=True); b = int(np.argmax(c))
        label_map[root] = (int(u[b]), float(c[b]/c.sum()), int(len(u)))
    point_map = {r: np.unique(np.stack(v), axis=0).astype(np.float32, copy=False) for r,v in points.items()}
    context = {
        "synapse_ids": bulk["synapse_id"],
        "pre_v117": mapped["pre"]["v117"], "post_v117": mapped["post"]["v117"],
        "pre_target": mapped["pre"]["target"], "post_target": mapped["post"]["target"],
        "role_counts": {r: dict(c) for r,c in role_counts.items()},
    }
    return point_map, label_map, context


def exact_soma_counts(bbox: tuple, *, token: str) -> dict:
    somas = _load_nucleus_somas()
    lower, upper = np.asarray(bbox[0]), np.asarray(bbox[1])
    pos = np.stack([somas["x_nm"], somas["y_nm"], somas["z_nm"]], axis=1)
    inside = np.all((pos >= lower) & (pos < upper), axis=1)
    roots = L.roots_at(somas["sv"][inside], L.V117_TIMESTAMP, token=token)
    if roots is None: raise RuntimeError("exact nucleus-supervoxel to v117 mapping failed")
    return dict(Counter(int(r) for r in roots if int(r) > 0))


def circuit_f1(context: dict, prediction: dict) -> dict:
    pre, post = context["pre_v117"], context["post_v117"]
    pre_t, post_t = context["pre_target"], context["post_target"]
    keep = np.asarray([int(l) in prediction and int(r) in prediction and int(lt)>0 and int(rt)>0
                       for l,r,lt,rt in zip(pre,post,pre_t,post_t)])
    if not np.any(keep): return {"circuit_f1": None, "n_circuit_synapses": 0}
    pj = np.column_stack([[prediction[int(r)] for r in pre[keep]], [prediction[int(r)] for r in post[keep]]])
    tj = np.column_stack([pre_t[keep], post_t[keep]])
    _, ti = np.unique(tj, axis=0, return_inverse=True); _, pi = np.unique(pj, axis=0, return_inverse=True)
    n_p = int(pi.max())+1; joint = np.bincount(ti.astype(np.int64)*n_p+pi)
    tc, pc = np.bincount(ti), np.bincount(pi)
    c2 = lambda v: float(np.sum(v*(v-1)//2))
    tp=c2(joint); t_p=c2(tc); p_p=c2(pc)
    prec = tp/p_p if p_p else 1.0; rec = tp/t_p if t_p else 1.0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    return {"circuit_f1": float(f1), "circuit_precision": float(prec), "circuit_recall": float(rec), "n_circuit_synapses": int(keep.sum())}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor-rank", type=int, default=0)
    ap.add_argument("--box-side-nm", type=float, default=30_000.0)
    ap.add_argument("--target-version", type=int, default=1412)
    ap.add_argument("--checkpoint", type=Path, default=ROOT/"models/shared_grammar_raw_skel_50e.pt")
    ap.add_argument("--min-root-observations", type=int, default=10)
    ap.add_argument("--max-path-points", type=int, default=96)
    ap.add_argument("--max-distance-nm", type=float, default=2500.0)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--score-sweep", default="0,1,2,3,4,5,6")
    ap.add_argument("--min-merge-pairs", type=int, default=5,
                    help="Pre-flight: require >= N true v117->v1412 merge pairs (default: 5)")
    ap.add_argument("--max-fragments", type=int, default=0)
    ap.add_argument("--token-stdin", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print anchor + box coords, exit before CAVE calls")
    ap.add_argument("--synapse-counts-tsv", type=Path, default=SYNAPSE_COUNTS_TSV)
    ap.add_argument("--nucleus-csv", type=Path, default=NUCLEUS_CSV)
    ap.add_argument("--output", type=Path, default=ROOT/"results/exp052_real_dense.json")
    args = ap.parse_args()

    # 0. Single-pass TSV load: get filter set AND soma counts simultaneously
    print("[0/7] loading static synapse table (one pass) ...", flush=True)
    synapse_root_set, synapse_counts = load_synapse_table(args.synapse_counts_tsv)
    print(f"      {len(synapse_root_set):,} roots in static synapse table; "
          f"{len(synapse_counts):,} with soma", flush=True)

    # 1. Anchor
    print(f"[1/7] selecting anchor soma (rank={args.anchor_rank}) ...", flush=True)
    anchor = select_anchor_soma(args.nucleus_csv, synapse_counts, anchor_rank=args.anchor_rank)
    center_nm = np.asarray([anchor["x_nm"], anchor["y_nm"], anchor["z_nm"]])
    bbox = box_around_soma(center_nm, args.box_side_nm)
    print(f"      root_id={anchor['root_id']}  total_syn={anchor['total_synapse_count']:,}", flush=True)
    print(f"      center_nm=({center_nm[0]:.0f}, {center_nm[1]:.0f}, {center_nm[2]:.0f})", flush=True)
    print(f"      bbox_nm={bbox}", flush=True)
    if args.dry_run:
        print("--dry-run: exiting before CAVE calls"); return 0

    token = sys.stdin.readline().strip() if args.token_stdin else L.DEFAULT_TOKEN
    if not token: raise SystemExit("CAVE token required")
    if not args.checkpoint.is_file(): raise SystemExit(f"checkpoint missing: {args.checkpoint}")

    started = time.time()
    target_timestamp = L.version_timestamp(args.target_version, token=token)
    if target_timestamp is None: raise SystemExit("target timestamp unavailable")

    # 2. Fetch
    print("[2/7] fetching real synapses and exact endpoint lineage ...", flush=True)
    point_map, label_map, context = fetch_candidates(bbox, target_timestamp=target_timestamp, token=token)
    all_roots_raw = sorted(point_map)
    assert_real_root_ids(all_roots_raw)
    print(f"      synapses={len(context['synapse_ids'])}; roots before filter={len(all_roots_raw)}", flush=True)

    # 3. Synapse-table filter
    print("[3/7] applying synapse-table filter ...", flush=True)
    all_roots, n_disc = filter_to_synapse_table(all_roots_raw, synapse_root_set)
    print(f"      discarded {n_disc} non-synapse-bearing roots; {len(all_roots)} remain", flush=True)
    if not all_roots: raise SystemExit("All v117 roots discarded -- no synapse-bearing fragments in box.")
    point_map = {r: point_map[r] for r in all_roots if r in point_map}
    label_map = {r: label_map[r] for r in all_roots if r in label_map}

    # 4. Pre-flight
    print("[4/7] pre-flight: counting true v117->v1412 merge pairs ...", flush=True)
    target_counts: Counter = Counter(label_map[r][0] for r in all_roots if r in label_map and label_map[r][0]>0)
    true_merge_pairs = int(sum(c*(c-1)//2 for c in target_counts.values()))
    print(f"      roots with v1412 label={len(target_counts)}; true merge pairs={true_merge_pairs}", flush=True)
    if true_merge_pairs < args.min_merge_pairs:
        raise SystemExit(
            f"Pre-flight FAILED: {true_merge_pairs} merge pairs < {args.min_merge_pairs}. "
            "Try higher --anchor-rank or lower --min-merge-pairs.")
    print(f"      pre-flight PASSED ({true_merge_pairs} >= {args.min_merge_pairs})", flush=True)

    # 5. Soma seeds
    print("[5/7] resolving exact soma seeds ...", flush=True)
    soma_counts = exact_soma_counts(bbox, token=token)
    seeds = sorted(set(all_roots) & set(soma_counts))
    if not seeds: raise SystemExit("no soma seed in filtered candidate population")
    if any(soma_counts[r]>1 for r in seeds): raise SystemExit("multi-soma root cannot be assembled atomically")
    selected = sorted({r for r in all_roots if len(point_map.get(r,[]))>=args.min_root_observations} | set(seeds))
    if args.max_fragments:
        nonseeds = sorted((r for r in selected if r not in seeds), key=lambda r: (-len(point_map.get(r,[])), r))
        selected = sorted(seeds + nonseeds[:max(0,args.max_fragments-len(seeds))])
    print(f"      soma seeds={len(seeds)}; path roots={len(selected)}", flush=True)

    # 6. Build graphs, score, assemble
    print("[6/7] building path graphs and running grammar ...", flush=True)
    fragments = []; contaminated = 0
    for root in selected:
        pts = point_map.get(root)
        if pts is None or len(pts)==0: continue
        verts, edges = skeleton_from_observed_points(pts, max_points=args.max_path_points)
        if len(edges)==0: continue
        label, purity, n_labels = label_map.get(root, (0, 0.0, 0))
        contaminated += int(n_labels>1)
        fragments.append(Fragment(root, verts, edges, soma_counts.get(root,0), label, purity))
    if not (set(seeds) & {f.root_id for f in fragments}):
        raise SystemExit("no soma seed has sufficient real observations")

    grammar = load_shared_grammar_model(args.checkpoint)
    mode = grammar.path_feature_mode
    mip2_nm = np.asarray([32.0, 32.0, 40.0], dtype=np.float32)
    featurize = lambda pts: featurize_path_points(pts/mip2_nm, mode=mode)
    edges_scored = build_candidate_edges_batched(fragments, grammar, featurize, max_distance_nm=args.max_distance_nm)
    print(f"      path graphs={len(fragments)}; candidate joins={len(edges_scored)}", flush=True)

    def with_confusers(active: dict) -> dict:
        pred = dict(active); nxt = max(pred.values(), default=-1)+1
        for r in all_roots:
            if r not in pred: pred[r]=nxt; nxt+=1
        return pred

    prediction = with_confusers(soma_seeded_assemble(fragments, edges_scored, min_score=args.min_score))
    metrics = partition_metrics(fragments, prediction)
    metrics.update(single_soma_compliance(fragments, prediction))
    metrics.update(circuit_f1(context, prediction))
    baseline = partition_metrics(fragments, {f.root_id: i for i,f in enumerate(fragments)})
    sweep = {}
    for thr in [float(v) for v in args.score_sweep.split(",") if v]:
        sp = with_confusers(soma_seeded_assemble(fragments, edges_scored, min_score=thr))
        sweep[str(thr)] = partition_metrics(fragments, sp)

    # 7. Write
    side_um = (np.asarray(bbox[1])-np.asarray(bbox[0]))/1000.0
    volume_um3 = float(np.prod(side_um))
    commit = subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip()
    result = {
        "experiment": "EXP-052 proofread-anchor-seeded v117 soma-seeded grammar",
        "provenance": {
            "git_commit": commit,
            "anchor_root_id": anchor["root_id"], "anchor_rank": anchor["anchor_rank"],
            "anchor_total_synapse_count": anchor["total_synapse_count"],
            "anchor_nucleus_id": anchor["nucleus_id"],
            "bbox_nm": [list(bbox[0]), list(bbox[1])], "box_side_nm": args.box_side_nm,
            "segmentation_version": 117, "v117_timestamp": L.V117_TIMESTAMP,
            "target_version": args.target_version, "target_timestamp": target_timestamp,
            "checkpoint": str(args.checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256(args.checkpoint), "path_feature_mode": mode,
            "candidate_policy": ("all v117 roots at synapse endpoints in bbox "
                                 "(public Delta export); filtered to roots in static synapse table"),
            "seed_policy": "exact nucleus supervoxel lineage containment",
            "synapse_table_filter": True, "synapse_table_path": str(args.synapse_counts_tsv),
            "min_root_observations": args.min_root_observations,
            "min_merge_pairs_gate": args.min_merge_pairs,
            "ground_truth_used_during_inference": False,
            "synthetic_fallback": False, "score_sweep_is_post_hoc": True,
        },
        "population": {
            "volume_um3": volume_um3, "synapses": len(context["synapse_ids"]),
            "v117_roots_before_synapse_filter": len(all_roots_raw),
            "v117_roots_discarded_by_synapse_filter": n_disc,
            "synapse_bearing_v117_roots": len(all_roots),
            "selected_roots": len(selected),
            "singleton_confuser_roots": len(all_roots)-len(fragments),
            "usable_path_graphs": len(fragments), "exact_soma_seeds": len(seeds),
            "contaminated_v117_roots": contaminated,
            "fragment_density_per_1000_um3": 1000.0*len(all_roots)/volume_um3,
            "v1412_label_coverage": len(label_map)/max(len(all_roots),1),
            "n_v1412_target_roots_active": len(target_counts),
            "true_fragment_merge_pairs_active": true_merge_pairs,
        },
        "assembly": {
            "candidate_edges": len(edges_scored),
            "accepted_non_singleton_fragments": int(sum(Counter(prediction.values())[c]>1 for c in prediction.values())),
            "min_score": args.min_score, "max_distance_nm": args.max_distance_nm,
            "score_quantiles": ({str(q): float(np.quantile([e.score for e in edges_scored], q))
                                 for q in (0.0,0.25,0.5,0.75,0.9,0.99,1.0)} if edges_scored else {}),
        },
        "metrics": metrics, "untouched_v117_baseline": baseline,
        "post_hoc_threshold_sweep": sweep,
        "predicted_cluster_by_v117_root": {str(r): int(c) for r,c in prediction.items()},
        "elapsed_seconds": time.time()-started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print("[7/7] complete", flush=True)
    print(json.dumps({"anchor": {k: anchor[k] for k in ("root_id","anchor_rank","total_synapse_count")},
                      "population": result["population"], "metrics": metrics}, indent=2))
    print(f"result: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
