"""Attach proofread ground truth to the label-blind atom population.

Two offline joins, both cached:

  1. every supervoxel the population touches -> its root at the target
     materialization (one batched ``roots_at`` call);
  2. the proofreading-status table at that version, for tiering.

Output is one NPZ (:class:`neuronauts.harness.labels.AtomLabels`) plus a JSON
summary of how much of the population carries usable ground truth.

    uv run python scripts/build_atom_labels.py --target-version 1822
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuronauts.data import lineage as L  # noqa: E402
from neuronauts.harness.labels import (  # noqa: E402
    proofread_tiers, summarize, tally_atom_targets,
)
from neuronauts.harness.population import load_population, map_supervoxels  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--population", default="data/substrate/c100um/population.npz")
    ap.add_argument("--target-version", type=int, default=1822)
    ap.add_argument("--proofread-table",
                    default="data/gt_manifest/proofreading_status_v1822.csv.gz")
    ap.add_argument("--min-side-count", type=int, default=2)
    ap.add_argument("--min-side-frac", type=float, default=0.05)
    ap.add_argument("--pure-min-owner-frac", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    pop = load_population(args.population)
    cache_dir = Path(args.population).parent
    out = Path(args.out or cache_dir / f"labels_v{args.target_version}.npz")
    report = Path(args.report
                 or f"results/atom_labels_v{args.target_version}.json")

    print(f"[1/3] target v{args.target_version} timestamp", flush=True)
    ts = L.version_timestamp(args.target_version)
    if ts is None:
        raise SystemExit(f"no timestamp for version {args.target_version}")

    with np.load(cache_dir / "sv_v117.npz", allow_pickle=False) as z:
        sv = z["sv"]
    print(f"[2/3] mapping {len(sv):,} supervoxels -> v{args.target_version} roots",
          flush=True)
    t0 = time.time()
    sv2target = map_supervoxels(
        sv, ts, workers=args.workers,
        cache_path=cache_dir / f"sv_v{args.target_version}.npz")
    print(f"      done in {time.time()-t0:.0f}s", flush=True)

    lut = np.zeros(len(sv), np.uint64)
    order = np.argsort(sv)
    sv_srt = sv[order]
    for i, s in enumerate(sv.tolist()):
        lut[i] = sv2target.get(s, 0)

    def target_of(side_sv: np.ndarray) -> np.ndarray:
        j = np.searchsorted(sv_srt, side_sv)
        jc = np.clip(j, 0, len(sv_srt) - 1)
        ok = sv_srt[jc] == side_sv
        return np.where(ok, lut[order][jc], 0).astype(np.uint64)

    # the population does not carry supervoxel ids directly; recompute the
    # pre/post side supervoxels from the underlying region+population join by
    # re-deriving them via the cached sv_v117 map is not possible without the
    # region file, so instead we tally directly from the atom-side arrays
    # already keyed by v117 root (syn_atom_pre/post) using the *same*
    # supervoxel roots that produced them. Since population.npz does not
    # store per-synapse supervoxels, target roots are looked up through the
    # region synapse table instead.
    print("[3/3] tallying per-atom target ownership", flush=True)
    region_npz = pop.meta.get("region_npz")
    if not region_npz:
        # default region used by build_population.py
        region_npz = "data/regions/dense_v1_synapses.npz"
    with np.load(region_npz, allow_pickle=False) as z:
        region_syn_id = z["synapse_id"]
        region_pre_sv = z["pre_sv"]
        region_post_sv = z["post_sv"]
    ord_r = np.argsort(region_syn_id)
    srt_r = region_syn_id[ord_r]
    j = np.searchsorted(srt_r, pop.syn_id)
    ok = srt_r[np.clip(j, 0, len(srt_r) - 1)] == pop.syn_id
    if not ok.all():
        raise SystemExit(f"{int((~ok).sum())} population synapses not found "
                         f"in {region_npz}")
    pre_sv = region_pre_sv[ord_r[j]]
    post_sv = region_post_sv[ord_r[j]]
    target_pre = target_of(pre_sv)
    target_post = target_of(post_sv)

    atom_of_side = np.concatenate([pop.syn_atom_pre, pop.syn_atom_post])
    target_of_side = np.concatenate([target_pre, target_post])

    pr = pd.read_csv(args.proofread_table)
    tiers = proofread_tiers(pr)

    labels = tally_atom_targets(
        atom_of_side, target_of_side, tiers=tiers,
        min_side_count=args.min_side_count, min_side_frac=args.min_side_frac,
        pure_min_owner_frac=args.pure_min_owner_frac,
        meta={"target_version": args.target_version,
              "population": str(args.population),
              "proofread_table": args.proofread_table})
    labels.save(out)

    summary = summarize(labels)
    summary["target_version"] = args.target_version
    summary["n_population_atoms"] = int(len(pop.atom_id))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"labels written to {out}")
    for k, v in summary.items():
        print(f"  {k:>26}: {v}")


if __name__ == "__main__":
    main()
