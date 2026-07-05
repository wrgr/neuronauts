"""Driver: run the two-cue complementarity eval on a cached column SideTable.

    PYTHONPATH=. python -m experiments.proofread.run_complementarity \
        --sidetable cache/sidetable/col_n1_v1718.npz \
        --encoder experiments/fingerprints/cutface_encoder.pt \
        --max-candidates 80 --max-pair-nm 6000 --out out/complementarity.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from experiments.pcfg.synapse_correction import SideTable, summarize_edits
from experiments.fingerprints.cutface.learned_cutface_encoder import (
    load_encoder, make_embed_fn)
from experiments.proofread.complementarity import run_complementarity


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sidetable", required=True)
    ap.add_argument("--encoder", default="experiments/fingerprints/cutface_encoder.pt")
    ap.add_argument("--max-candidates", type=int, default=80)
    ap.add_argument("--max-pair-nm", type=float, default=6000.0)
    ap.add_argument("--mip", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(args.sidetable)
    tab = SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])
    print("edit summary:", summarize_edits(tab))

    embed = make_embed_fn(load_encoder(args.encoder))
    res = run_complementarity(tab, embed, max_candidates=args.max_candidates,
                              max_pair_nm=args.max_pair_nm, mip=args.mip, seed=args.seed)

    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        dump = {k: v for k, v in res.items()
                if k not in ("cands", "local", "p_joint", "y", "strata")}
        # keep arrays in a JSON-friendly form
        for k in ("y", "strata"):
            if k in res:
                dump[k] = np.asarray(res[k]).tolist()
        if "local" in res:
            dump["local"] = np.asarray(res["local"]).tolist()
        if "p_joint" in res:
            dump["p_joint"] = [None if np.isnan(x) else float(x) for x in res["p_joint"]]
        with open(args.out, "w") as f:
            json.dump(dump, f, indent=2)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
