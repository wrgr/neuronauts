"""EXP-083 null: shuffle which twin is called corrupted.  Everything else --
the same pairs, the same features, the same cell-disjoint folds, the same
conditional logistic -- is unchanged, so the pipeline must land on 0.5."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "scripts"))
from exp083_score2 import cond_logit, folds, load, boot, STRICT, N_FOLD

X, cols, meta = load("features.npz")
ci = {c: i for i, c in enumerate(cols)}
y = np.array([m["y"] for m in meta]); cell = np.array([m["cell"] for m in meta])
pair = np.array([m["pair"] for m in meta]); added = np.array([m["added_um"] for m in meta])
order = np.argsort(pair * 2 + y); i0, i1 = order[0::2], order[1::2]
fold = folds(cell); pfold = fold[i0]
Xs = X[:, [ci[c] for c in STRICT]]
sd = Xs[i0].std(0); sd[sd < 1e-12] = 1.0
Z = Xs / sd
E = [1, 3, 10, 30, 100, 300, 1000, 1e9]
BIN = ["1-3", "3-10", "10-30", "30-100", "100-300", "300-1000", ">1000"]
padd = added[i1]

for rep in range(3):
    rng = np.random.default_rng(100 + rep)
    flip = rng.random(len(i0)) < 0.5          # swap the two twins' roles
    a = np.where(flip, i1, i0); b = np.where(flip, i0, i1)
    D = Z[b] - Z[a]
    line = []
    for bi in range(7):
        m = np.flatnonzero((padd >= E[bi]) & (padd < E[bi + 1]))
        if len(m) < 60: line.append(np.nan); continue
        v = np.zeros(len(m))
        for k in range(N_FOLD):
            tr = m[pfold[m] != k]; te = m[pfold[m] == k]
            if len(tr) < 20 or not len(te): continue
            w = cond_logit(D[tr])
            a_, b_ = Z[a[te]] @ w, Z[b[te]] @ w
            v[np.isin(m, te)] = (b_ > a_).astype(float) + 0.5 * (b_ == a_)
        line.append(v.mean())
    print(f"  null seed {rep}: " + "  ".join(f"{BIN[i]}={line[i]:.3f}" for i in range(7)))
    # model-free single descriptor under the null
    if rep == 0:
        d = X[b, ci["frac_inward"]] - X[a, ci["frac_inward"]]
        print(f"    frac_inward alone, shuffled: {(d>0).mean()+0.5*(d==0).mean():.3f}")
