"""Driver for the seam-localized two-cue test (Task 26).

    PYTHONPATH=. token=$token python -m experiments.proofread.run_seam_test \
        --out out/seam_test.json
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import numpy as np

from experiments.fingerprints.cutface.learned_cutface_encoder import (
    load_encoder, make_embed_fn)
from experiments.proofread.seam_test import build_seam_rows, summarize

# real m343 false-merges (each -> two current roots) and clean proofread neurons
MERGES = [864691135233406809, 864691136010977187, 864691135589909259,
          864691136261867917, 864691135847908702, 864691137019954286,
          864691135467909772]
CLEAN = [864691135686494647, 864691136812081779, 864691136195284556,
         864691135975539779]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default="experiments/fingerprints/cutface_encoder.pt")
    ap.add_argument("--version", type=int, default=1822)
    ap.add_argument("--gap-nm", type=float, default=2000.0)
    ap.add_argument("--cont-per-neuron", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from caveclient import CAVEclient
    tok = os.environ["token"]
    client = CAVEclient("minnie65_public", auth_token=tok)
    client.version = args.version
    embed = make_embed_fn(load_encoder(args.encoder))

    rows = build_seam_rows(MERGES, CLEAN, embed, version=args.version, token=tok,
                           client=client, gap_nm=args.gap_nm,
                           cont_per_neuron=args.cont_per_neuron)
    summ = summarize(rows)
    print("\n=== SEAM-LOCALIZED TWO-CUE SEPARATION ===")
    for k, v in summ.items():
        print(f"  {k}: {v}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"summary": summ, "rows": [asdict(r) for r in rows]}, f, indent=2)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
