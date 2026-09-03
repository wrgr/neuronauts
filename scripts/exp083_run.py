"""EXP-083 -- can a whole-cell shape score tell a correct assembly from one
corrupted by a single wrong join?

Design, per proofread cell A (103 of them, all one connected component):

  base        A's arbor with two subtrees removed: ``z`` (3-15% of the cable,
              so the correct assembly is itself partial, as a grower's is) and
              ``u`` (the branch under test).
  CORRECT     base + A's own subtree(u), put back exactly where it was.
  CORRUPTED   base + a subtree of a *different* proofread cell B, whose cable
              matches subtree(u) to within 15%, rigidly translated so its root
              lands on A's vertex u.

The replaced edge is identical in both: same length, same direction, same
parent.  The two assemblies have the same base, the same join point and the
same amount of cable.  The only difference is whose cable was added, so any
separation has to come from the shape of the whole tree, not from the join.

    python scripts/exp083_run.py
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
from neuronauts.harness.baselines import GradientBoostedStumps, LogisticRegression
from neuronauts.metrics.ranking import roc_auc

R = Path(__file__).resolve().parents[1]
OUT = R / "results/EXP-083"
SEED = 0
K_PER_BIN = 3
CABLE_TOL = 0.15                 # donor subtree cable, relative
BINS_UM = [1, 3, 10, 30, 100, 300, 1000, 1e9]
BIN_NAMES = ["1-3", "3-10", "10-30", "30-100", "100-300", "300-1000", ">1000"]


def bin_of(c):
    return int(np.searchsorted(BINS_UM, c, side="right") - 1)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    files = sorted(glob.glob(str(L.SKEL / "*_skv4.npz")))
    cells, trees = [], {}
    for f in files:
        c = int(Path(f).name.split("_")[0])
        cells.append(c)
        trees[c] = L.load_tree(f)
    print(f"loaded {len(cells)} arbors in {time.time()-t0:.1f}s", flush=True)

    ctype = {}
    for f in glob.glob(str(R / "data/external/cell_cards/*.json")):
        if "_aggregate" in f:
            continue
        j = json.load(open(f))
        ctype[int(j["cell"])] = j.get("cell_type", {}).get("final")

    # donor index: every subtree of every cell, by cable
    donor_cable = {c: trees[c]["sub_cable"] / L.UM for c in cells}
    for c in cells:                              # never donate the whole arbor
        donor_cable[c][0] = -1.0

    rows, X, meta = [], [], []
    t0 = time.time()
    for ci, A_id in enumerate(cells):
        A = trees[A_id]
        sc = A["sub_cable"] / L.UM
        total = float(A["elen"].sum() / L.UM)
        # the second removal: keeps the correct assembly partial too
        zc = np.flatnonzero((sc > 0.03 * total) & (sc < 0.15 * total))
        if not len(zc):
            continue
        z = int(rng.choice(zc))
        mz = L.subtree_mask(A, z)
        for b in range(len(BIN_NAMES)):
            lo, hi = BINS_UM[b], BINS_UM[b + 1]
            cand = np.flatnonzero((sc >= lo) & (sc < hi) & ~mz
                                  & (sc < 0.5 * total))
            cand = cand[cand != 0]
            if not len(cand):
                continue
            picks = rng.choice(cand, min(K_PER_BIN, len(cand)), replace=False)
            for u in picks.tolist():
                if mz[A["par"][u]] or A["par"][u] < 0:
                    continue
                mu = L.subtree_mask(A, u)
                if mu[z]:
                    continue
                target = float(sc[u])
                donor = None
                for _ in range(40):
                    B_id = int(rng.choice([c for c in cells if c != A_id]))
                    dc = donor_cable[B_id]
                    ok = np.flatnonzero(np.abs(dc - target) <= CABLE_TOL * target)
                    if len(ok):
                        donor = (B_id, int(rng.choice(ok)))
                        break
                if donor is None:
                    continue
                B_id, w = donor
                good = L.assemble(A, [mz, mu], graft=(A, u, u))
                bad = L.assemble(A, [mz, mu], graft=(trees[B_id], w, u))
                if good is None or bad is None:
                    continue
                fg, fb = L.descriptors(good), L.descriptors(bad)
                pid = len(rows)
                for lab, f in ((0, fg), (1, fb)):
                    X.append([f[k] for k in L.ALL_COLS])
                    meta.append(dict(pair=pid, y=lab, cell=A_id, bin=b,
                                     added_um=target, donor=B_id,
                                     same_type=int(ctype.get(A_id) == ctype.get(B_id)),
                                     comp_u=int(A["comp"][u]),
                                     comp_w=int(trees[B_id]["comp"][w]),
                                     donor_um=float(donor_cable[B_id][w]),
                                     total_um=total))
                rows.append(pid)
        if (ci + 1) % 20 == 0:
            print(f"  {ci+1}/{len(cells)} cells, {len(rows)} pairs, "
                  f"{time.time()-t0:.0f}s", flush=True)

    X = np.asarray(X, np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    np.savez_compressed(OUT / "features.npz", X=X, cols=np.array(L.ALL_COLS),
                        meta=np.array(json.dumps(meta)))
    print(f"{len(rows)} pairs, X {X.shape}, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
