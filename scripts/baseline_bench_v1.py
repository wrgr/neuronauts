#!/usr/bin/env python3
"""Step 4 — honest baselines on neuronauts-bench v1.

Establishes the floor that any learned model must beat. Every number here is
produced on real data with a locked split and is stamped with the dataset
manifest hash, so it can be compared against later runs.

The task
--------
Partition synapse observations into neurons. Ground truth is the v1718
(proofread) root of each observation. The v117 segmentation supplies the
starting fragments.

The metric that matters
-----------------------
Aggregate pairwise F1 is dominated by pairs that share a v117 root, which are
correct for free — EXP-056 flagged exactly this class-imbalance artifact
(an atomic baseline scored 0.914 pair-F1 while resolving nothing). So the
headline numbers here are **cross-fragment**: restricted to observation pairs
whose v117 roots differ. Those are the pairs a merge decision actually has to
get right.

  cross_merge_precision — of the pairs we joined across v117 roots, how many
                          truly share a v1718 neuron
  cross_merge_recall    — of the pairs that truly should be joined across
                          v117 roots, how many we recovered

Baselines
---------
1. untouched_v117   — predict no merges; each v117 root is its own neuron.
                      This is the "do nothing" floor. Its cross-merge recall is
                      0 by construction, and any model that cannot beat its
                      precision is not adding information.
2. proximity_unionfind — merge two v117 roots when any of their observations
                      lie within `d` nm. `d` is calibrated on VAL ONLY, then
                      applied once to TEST.
3. oracle_fragment  — cluster by v117 root, but relabel using the true v1718
                      root. Not a method: it reports the ceiling reachable
                      without splitting frankenmerged fragments, which EXP-056
                      showed geometry cannot do.

Usage
-----
    python scripts/baseline_bench_v1.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from neuronauts.data.versions import BASE_VERSION, LABEL_VERSION  # noqa: E402
from neuronauts.results_schema import ResultsRecord, write_results  # noqa: E402

DATASET_DIR = REPO / "data" / "bench_v1"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_split(split: str) -> dict:
    man = json.loads((DATASET_DIR / "manifests" / f"{split}.json").read_text())
    parts = []
    for region in man["regions"]:
        p = DATASET_DIR / "regions" / f"{region}.npz"
        if not p.exists():
            raise SystemExit(
                f"missing {p}. Rebuild with scripts/build_bench_v1.py "
                "(fetches are cached, so this is fast)."
            )
        d = np.load(p)
        parts.append({
            "region": region,
            "positions_nm": d["positions_nm"],
            "base_roots": d["base_roots"],
            "label_roots": d["label_roots"],
        })
    return {
        "split": split,
        "manifest": man,
        "positions_nm": np.concatenate([p["positions_nm"] for p in parts]),
        "base_roots": np.concatenate([p["base_roots"] for p in parts]),
        "label_roots": np.concatenate([p["label_roots"] for p in parts]),
        "region_of": np.concatenate([
            np.full(len(p["base_roots"]), i) for i, p in enumerate(parts)
        ]),
        "regions": [p["region"] for p in parts],
    }


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def _n_pairs(counts: np.ndarray) -> int:
    c = counts.astype(np.int64)
    return int((c * (c - 1) // 2).sum())


def pair_metrics(pred: np.ndarray, true: np.ndarray, base: np.ndarray) -> dict:
    """Exact pairwise metrics via contingency counting (no O(n^2) enumeration).

    Reports both the aggregate figures and the cross-fragment figures that
    isolate the actual merge decision.
    """
    df = pd.DataFrame({"pred": pred, "true": true, "base": base})

    pred_pairs = _n_pairs(df.groupby("pred").size().values)
    true_pairs = _n_pairs(df.groupby("true").size().values)
    tp = _n_pairs(df.groupby(["pred", "true"]).size().values)

    # Pairs that share a v117 root are correct for free; subtract them out.
    same_base_in_pred = _n_pairs(df.groupby(["pred", "base"]).size().values)
    same_base_in_true = _n_pairs(df.groupby(["true", "base"]).size().values)
    same_base_in_both = _n_pairs(df.groupby(["pred", "true", "base"]).size().values)

    x_pred = pred_pairs - same_base_in_pred
    x_true = true_pairs - same_base_in_true
    x_tp = tp - same_base_in_both

    def _f1(p, r):
        if p is None:
            return None
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def _sig(v, n=3):
        """Keep tiny precisions legible instead of rounding them to zero."""
        if v is None:
            return None
        if v == 0:
            return 0.0
        from math import floor, log10
        return round(v, max(6, -int(floor(log10(abs(v)))) + n))

    prec = tp / pred_pairs if pred_pairs else 0.0
    rec = tp / true_pairs if true_pairs else 0.0
    # Precision is undefined when nothing was predicted; reporting 0.0 there
    # would read as "wrong" rather than "declined to answer".
    xp = (x_tp / x_pred) if x_pred else None
    xr = x_tp / x_true if x_true else 0.0

    return {
        "n_observations": int(len(df)),
        "n_pred_clusters": int(df["pred"].nunique()),
        "n_true_neurons": int(df["true"].nunique()),
        "n_v117_fragments": int(df["base"].nunique()),
        "pair_precision": round(prec, 6),
        "pair_recall": round(rec, 6),
        "pair_f1": round(_f1(prec, rec), 6),
        "cross_merge_precision": _sig(xp),
        "cross_merge_recall": _sig(xr),
        "cross_merge_f1": _sig(_f1(xp, xr)),
        "n_cross_pairs_predicted": int(x_pred),
        "n_cross_pairs_true": int(x_true),
        "n_cross_pairs_correct": int(x_tp),
    }


def adjusted_rand(pred: np.ndarray, true: np.ndarray) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(true, pred))


def fmt(v) -> str:
    """Format a metric so a 1.7e-05 precision never prints as 0.0000."""
    if v is None:
        return "  n/a "
    if v != 0 and abs(v) < 1e-3:
        return f"{v:.1e}"
    return f"{v:.4f}"


def evaluate(pred: np.ndarray, data: dict) -> dict:
    m = pair_metrics(pred, data["label_roots"], data["base_roots"])
    m["ari"] = round(adjusted_rand(pred, data["label_roots"]), 6)
    return m


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

def baseline_untouched(data: dict) -> np.ndarray:
    """Predict no merges: every v117 root stands alone."""
    return data["base_roots"].copy()


def baseline_proximity(data: dict, radius_nm: float) -> np.ndarray:
    """Union two v117 roots when any of their observations lie within radius.

    Union-find over fragments, seeded by spatial adjacency of observations.
    Uses a KD-tree, so this is near-linear rather than all-pairs.
    """
    from scipy.spatial import cKDTree

    base = data["base_roots"]
    pos = data["positions_nm"].astype(np.float64)

    uniq, inv = np.unique(base, return_inverse=True)
    parent = np.arange(len(uniq))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    tree = cKDTree(pos)
    for i, j in tree.query_pairs(radius_nm, output_type="ndarray"):
        if inv[i] != inv[j]:
            union(inv[i], inv[j])

    roots = np.array([find(i) for i in range(len(uniq))])
    return roots[inv]


def oracle_fragment_ceiling(data: dict) -> np.ndarray:
    """Ceiling reachable without splitting frankenmerged v117 fragments.

    Each v117 fragment is assigned its majority v1718 label; observations
    inherit it. Perfect merging, zero splitting. Not a method — a bound.
    """
    df = pd.DataFrame({"base": data["base_roots"], "true": data["label_roots"]})
    majority = (
        df.groupby(["base", "true"]).size().reset_index(name="n")
        .sort_values("n", ascending=False)
        .drop_duplicates("base").set_index("base")["true"]
    )
    return df["base"].map(majority).values


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radii-nm", type=float, nargs="*",
                    default=[500, 1000, 2000, 3000, 5000, 8000, 12000],
                    help="proximity radii swept on VAL")
    ap.add_argument("--out-dir", default="results/bench_v1")
    args = ap.parse_args()

    dataset = json.loads((DATASET_DIR / "manifests" / "dataset.json").read_text())
    manifest_sha = __import__("hashlib").sha256(
        json.dumps(dataset["manifest_sha256"], sort_keys=True).encode()
    ).hexdigest()

    splits = {s: load_split(s) for s in ("train", "val", "test")}
    for s, d in splits.items():
        print(f"[{s}] {len(d['base_roots']):,} observations, "
              f"{len(np.unique(d['base_roots'])):,} v117 fragments, "
              f"{len(np.unique(d['label_roots'])):,} true neurons "
              f"({'+'.join(d['regions'])})", flush=True)

    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    records: list[tuple[str, str, dict]] = []

    # -- floor: predict nothing --------------------------------------------
    print("\n=== baseline 1: untouched v117 (predict no merges) ===")
    for s in ("val", "test"):
        m = evaluate(baseline_untouched(splits[s]), splits[s])
        print(f"  {s:5s} ARI={m['ari']:.4f}  pair_F1={m['pair_f1']:.4f}  "
              f"cross_merge P={fmt(m['cross_merge_precision'])} "
              f"R={fmt(m['cross_merge_recall'])}  "
              f"(predicted {m['n_cross_pairs_predicted']:,} cross joins; "
              f"{m['n_cross_pairs_true']:,} true ones exist)")
        records.append(("untouched_v117", s, m))

    # -- ceiling without splitting fragments -------------------------------
    print("\n=== bound: oracle fragment labelling (no fragment splitting) ===")
    for s in ("val", "test"):
        m = evaluate(oracle_fragment_ceiling(splits[s]), splits[s])
        print(f"  {s:5s} ARI={m['ari']:.4f}  pair_F1={m['pair_f1']:.4f}  "
              f"cross_merge P={fmt(m['cross_merge_precision'])} "
              f"R={fmt(m['cross_merge_recall'])} "
              f"({m['n_cross_pairs_correct']:,}/{m['n_cross_pairs_predicted']:,} "
              f"predicted joins correct)")
        records.append(("oracle_fragment_ceiling", s, m))

    # -- proximity, calibrated on VAL only ---------------------------------
    print("\n=== baseline 2: proximity union-find (calibrated on VAL) ===")
    sweep = []
    for r in args.radii_nm:
        m = evaluate(baseline_proximity(splits["val"], r), splits["val"])
        sweep.append((r, m))
        print(f"  val  d={r:>7,.0f}nm  ARI={m['ari']:>8.4f}  "
              f"cross_merge P={fmt(m['cross_merge_precision'])} "
              f"R={fmt(m['cross_merge_recall'])} "
              f"F1={fmt(m['cross_merge_f1'])}  "
              f"joins={m['n_cross_pairs_predicted']:>12,}  "
              f"clusters={m['n_pred_clusters']:,}")

    best_r, best_m = max(sweep, key=lambda t: t[1]["cross_merge_f1"] or 0.0)
    print(f"\n  -> selected d={best_r:,.0f} nm on val "
          f"(cross_merge_F1={fmt(best_m['cross_merge_f1'])}); applying once to test")
    records.append(("proximity_unionfind_val_sweep", "val", {
        "selected_radius_nm": best_r,
        "sweep": [{"radius_nm": r, **mm} for r, mm in sweep],
        **best_m,
    }))

    m_test = evaluate(baseline_proximity(splits["test"], best_r), splits["test"])
    print(f"  test d={best_r:>7,.0f}nm  ARI={m_test['ari']:>8.4f}  "
          f"cross_merge P={fmt(m_test['cross_merge_precision'])} "
          f"R={fmt(m_test['cross_merge_recall'])} "
          f"F1={fmt(m_test['cross_merge_f1'])}  "
          f"joins={m_test['n_cross_pairs_predicted']:,}  "
          f"clusters={m_test['n_pred_clusters']:,}")
    m_test["selected_radius_nm"] = best_r
    m_test["calibrated_on"] = "val"
    records.append(("proximity_unionfind", "test", m_test))

    # -- write stamped records ---------------------------------------------
    for name, split, metrics in records:
        rec = ResultsRecord(
            experiment=f"bench_v1_{name}",
            split=split,
            metrics=metrics,
            base_version=BASE_VERSION,
            label_version=LABEL_VERSION,
            synthetic=False,
            data_manifest_sha=manifest_sha,
            n_observations=metrics.get("n_observations"),
            notes=("Baseline on neuronauts-bench v1. Headline metrics are "
                   "cross-fragment (pairs whose v117 roots differ); aggregate "
                   "pair-F1 is class-imbalance dominated and is reported only "
                   "for completeness."),
        )
        write_results(rec, out / f"{name}__{split}.json")
    print(f"\nwrote {len(records)} stamped records to {out.relative_to(REPO)}")
    print(f"data_manifest_sha = {manifest_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
