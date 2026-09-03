"""EXP-083 diagnostics: how far a wrong join moves the whole-cell descriptor,
and whether a size-matched model does better on the big grafts."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "scripts"))
from neuronauts.metrics.ranking import roc_auc
import exp083_shape_lib as L
from exp083_score2 import cond_logit, folds, load, boot, STRICT, N_FOLD

OUT = R / "results/EXP-083"
BIN = ["1-3", "3-10", "10-30", "30-100", "100-300", "300-1000", ">1000"]
E = [1, 3, 10, 30, 100, 300, 1000, 1e9]

X, cols, meta = load("features.npz")
ci = {c: i for i, c in enumerate(cols)}
y = np.array([m["y"] for m in meta]); cell = np.array([m["cell"] for m in meta])
pair = np.array([m["pair"] for m in meta]); added = np.array([m["added_um"] for m in meta])
total = np.array([m["total_um"] for m in meta])
order = np.argsort(pair * 2 + y); i0, i1 = order[0::2], order[1::2]
pcell, padd, pfrac = cell[i0], added[i1], added[i1] / total[i0]
fold = folds(cell); pfold = fold[i0]
idx = [ci[c] for c in STRICT]
Xs = X[:, idx]
# standardise by the spread ACROSS CELLS of correct assemblies, so a difference
# is read in "between-cell standard deviations of that descriptor"
sd = Xs[i0].std(0); sd[sd < 1e-12] = 1.0
Z = Xs / sd
D = Z[i1] - Z[i0]

rep = {}
print("how far one wrong join moves the whole-cell descriptor")
print(" bin um      pairs  median added  |shift| in between-cell sigma   abs AUC")
rep["shift"] = {}
for b in range(7):
    m = (padd >= E[b]) & (padd < E[b + 1])
    if m.sum() < 15: continue
    shift = np.linalg.norm(D[m], axis=1) / np.sqrt(D.shape[1])
    sel = np.isin(pair, np.flatnonzero(m))
    # absolute AUC of a model trained on all sizes, restricted to this bin
    print(f"  {BIN[b]:11s} {int(m.sum()):5d}  {np.median(padd[m]):9.1f}      "
          f"{np.median(shift):.3f}")
    rep["shift"][BIN[b]] = float(np.median(shift))

print("\nsize-stratified conditional logistic (fit inside the bin, cell-disjoint)")
print(" bin um      pairs   paired acc [95% CI]     absolute AUC in bin")
rep["stratified"] = {}
for b in range(7):
    m = np.flatnonzero((padd >= E[b]) & (padd < E[b + 1]))
    if len(m) < 60: continue
    v = np.zeros(len(m)); s_abs = np.zeros(2 * len(m))
    for k in range(N_FOLD):
        tr = m[pfold[m] != k]; te = m[pfold[m] == k]
        if len(tr) < 20 or not len(te): continue
        w = cond_logit(D[tr])
        st = Z[np.concatenate([i0[te], i1[te]])] @ w
        a_, b_ = Z[i0[te]] @ w, Z[i1[te]] @ w
        v[np.isin(m, te)] = (b_ > a_).astype(float) + 0.5 * (b_ == a_)
    lo, hi = boot(v, pcell[m])
    # absolute AUC needs one model; use fold 0's for a rough read
    print(f"  {BIN[b]:11s} {len(m):5d}   {v.mean():.3f} [{lo:.3f}, {hi:.3f}]")
    rep["stratified"][BIN[b]] = dict(n=int(len(m)), paired=float(v.mean()), ci=[lo, hi])

print("\nper-descriptor within-site accuracy by graft size (shape_geom only)")
hdr = "  %-22s" % "descriptor" + "".join(f"{BIN[b]:>10s}" for b in range(7))
print(hdr)
rep["desc_by_bin"] = {}
for c in STRICT:
    d = X[i1, ci[c]] - X[i0, ci[c]]
    row = []
    for b in range(7):
        m = (padd >= E[b]) & (padd < E[b + 1])
        row.append(float((d[m] > 0).mean() + 0.5 * (d[m] == 0).mean()) if m.sum() >= 15 else np.nan)
    if max(abs(np.nan_to_num(np.array(row), nan=0.5) - 0.5)) > 0.08:
        print("  %-22s" % c + "".join(f"{v:10.3f}" for v in row))
        rep["desc_by_bin"][c] = row
json.dump(rep, open(OUT / "diagnostics.json", "w"), indent=1)
print("\nwrote diagnostics.json")
