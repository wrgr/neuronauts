"""EXP-083 arm 2 -- the join sites and the pieces are real segmentation breaks.

Arm 1 cuts the arbor at an arbitrary vertex.  Here the cut is one of the 232
``split_challenges`` recorded in ``data/external/cell_cards``: a place where
the v117 segmentation actually broke the cell in two, with the 3D coordinates
of both sides of the gap.  The piece distal to that break is the fragment a
grower has to attach, and the wrong piece offered instead is another cell's
fragment taken at another real break.  Nothing about the wound or either
candidate is invented.
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exp083_shape_lib as L

R = Path(__file__).resolve().parents[1]
OUT = R / "results/EXP-083"
SEED = 1
MAX_SNAP_NM = 3000.0


def main():
    rng = np.random.default_rng(SEED)
    trees, ctype = {}, {}
    for f in sorted(glob.glob(str(L.SKEL / "*_skv4.npz"))):
        trees[int(Path(f).name.split("_")[0])] = L.load_tree(f)
    cards = {}
    for f in glob.glob(str(R / "data/external/cell_cards/*.json")):
        if "_aggregate" in f:
            continue
        j = json.load(open(f))
        cards[int(j["cell"])] = j
        ctype[int(j["cell"])] = j.get("cell_type", {}).get("final")

    # every real break -> the distal subtree it separates
    breaks = []
    snap, miss = [], 0
    for c, A in trees.items():
        j = cards.get(c)
        if not j:
            continue
        total = float(A["elen"].sum() / L.UM)
        for s in j.get("split_challenges", []):
            best = None
            for key in ("at_a_nm", "at_b_nm"):
                p = np.asarray(s[key], float)
                d = np.linalg.norm(A["V"] - p, axis=1)
                v = int(np.argmin(d))
                if best is None or A["depth"][v] > A["depth"][best[0]]:
                    best = (v, float(d[v]))
            v, dd = best
            snap.append(dd)
            cab = float(A["sub_cable"][v] / L.UM)
            if dd > MAX_SNAP_NM or v == 0 or A["par"][v] < 0 or cab < 1.0 \
                    or cab > 0.5 * total:
                miss += 1
                continue
            breaks.append(dict(cell=c, v=v, cable=cab, total=total,
                               comp=int(A["comp"][v]),
                               ca=s["compartment_a"], cb=s["compartment_b"]))
    snap = np.asarray(snap)
    print(f"{len(breaks)} usable breaks of {len(snap)} challenges "
          f"(snap distance median {np.median(snap):.0f} nm, "
          f"{miss} unusable)", flush=True)

    pool_cab = np.array([b["cable"] for b in breaks])
    X, meta, npair = [], [], 0
    t0 = time.time()
    for bi, br in enumerate(breaks):
        A = trees[br["cell"]]
        u = br["v"]
        total = br["total"]
        sc = A["sub_cable"] / L.UM
        zc = np.flatnonzero((sc > 0.03 * total) & (sc < 0.15 * total))
        if not len(zc):
            continue
        mu = L.subtree_mask(A, u)
        zc = zc[~mu[zc]]
        if not len(zc):
            continue
        z = int(rng.choice(zc))
        mz = L.subtree_mask(A, z)
        if mz[u] or mz[A["par"][u]]:
            continue
        # donor: another cell's real-break fragment, closest cable match
        other = np.array([i for i, b in enumerate(breaks)
                          if b["cell"] != br["cell"]])
        rel = np.abs(pool_cab[other] - br["cable"]) / br["cable"]
        cands = other[rel <= 0.15]
        if not len(cands):
            cands = other[np.argsort(rel)[:3]]
        di = int(rng.choice(cands))
        B = trees[breaks[di]["cell"]]
        w = breaks[di]["v"]
        good = L.assemble(A, [mz, mu], graft=(A, u, u))
        bad = L.assemble(A, [mz, mu], graft=(B, w, u))
        if good is None or bad is None:
            continue
        for lab, t in ((0, good), (1, bad)):
            f = L.descriptors(t)
            X.append([f[k] for k in L.ALL_COLS])
            meta.append(dict(pair=npair, y=lab, cell=br["cell"], bin=0,
                             added_um=br["cable"], donor=breaks[di]["cell"],
                             same_type=int(ctype.get(br["cell"])
                                           == ctype.get(breaks[di]["cell"])),
                             comp_u=br["comp"], comp_w=breaks[di]["comp"],
                             donor_um=breaks[di]["cable"], total_um=total))
        npair += 1
        if (bi + 1) % 40 == 0:
            print(f"  {bi+1}/{len(breaks)}  {npair} pairs  "
                  f"{time.time()-t0:.0f}s", flush=True)
    X = np.nan_to_num(np.asarray(X, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    np.savez_compressed(OUT / "features_arm2.npz", X=X,
                        cols=np.array(L.ALL_COLS), meta=np.array(json.dumps(meta)))
    print(f"arm2: {npair} pairs, X {X.shape}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
