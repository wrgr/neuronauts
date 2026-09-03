"""EXP-083 controls: a same-cell displaced piece, and frankenmerge-scale grafts."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "scripts"))
from neuronauts.metrics.ranking import roc_auc
from exp083_score2 import cond_logit, folds, load, boot, STRICT, SETS, N_FOLD
import exp083_shape_lib as L
OUT = R / "results/EXP-083"


def evaluate(name, tag, frac_bins=None):
    X, cols, meta = load(name)
    ci = {c: i for i, c in enumerate(cols)}
    y = np.array([m["y"] for m in meta]); cell = np.array([m["cell"] for m in meta])
    pair = np.array([m["pair"] for m in meta])
    added = np.array([m["added_um"] for m in meta])
    total = np.array([m["total_um"] for m in meta])
    o = np.argsort(pair * 2 + y); i0, i1 = o[0::2], o[1::2]
    pcell, padd = cell[i0], added[i1]; pfrac = padd / total[i0]
    fold = folds(cell); pfold = fold[i0]
    out = dict(n_pairs=int(len(i0)),
               median_added_um=float(np.median(padd)),
               median_frac=float(np.median(pfrac)))
    print(f"\n===== {tag}: {len(i0)} pairs, wrong cable median "
          f"{np.median(padd):.0f} um = {np.median(pfrac)*100:.1f}% of the arbor")
    print("  feature set      absolute AUC   within-site paired [95% CI]")
    out["sets"] = {}
    keep = {}
    for sname, sc in SETS.items():
        Xs = X[:, [ci[c] for c in sc]]
        sd = Xs[i0].std(0); sd[sd < 1e-12] = 1.0
        Z = Xs / sd
        D = Z[i1] - Z[i0]
        s = np.zeros(len(y))
        for k in range(N_FOLD):
            tr = pfold != k; te = fold == k
            if tr.sum() < 20 or not te.any(): continue
            w = cond_logit(D[tr]); s[te] = Z[te] @ w
        auc = roc_auc(y == 1, s)
        v = (s[i1] > s[i0]).astype(float) + 0.5 * (s[i1] == s[i0])
        lo, hi = boot(v, pcell)
        print(f"  {sname:15s} {auc:.3f}         {v.mean():.3f} [{lo:.3f}, {hi:.3f}]")
        out["sets"][sname] = dict(abs_auc=float(auc), paired=float(v.mean()), ci=[lo, hi])
        keep[sname] = (s, v)
    if frac_bins:
        print("  shape_geom by graft size (share of arbor)")
        s, v = keep["shape_geom"]
        out["by_frac"] = {}
        for a, b in zip(frac_bins[:-1], frac_bins[1:]):
            m = (pfrac >= a) & (pfrac < b)
            if m.sum() < 15: continue
            sel = np.isin(pair, np.flatnonzero(m))
            lo, hi = boot(v[m], pcell[m])
            lab = f"{a*100:g}-{b*100:g}%"
            print(f"   {lab:9s} {int(m.sum()):4d} pairs  absolute AUC "
                  f"{roc_auc(y[sel]==1, s[sel]):.3f}   paired {v[m].mean():.3f} "
                  f"[{lo:.3f}, {hi:.3f}]")
            out["by_frac"][lab] = dict(n=int(m.sum()),
                                       abs_auc=float(roc_auc(y[sel] == 1, s[sel])),
                                       paired=float(v[m].mean()), ci=[lo, hi])
    # descriptor shift in between-cell sigma
    Xs = X[:, [ci[c] for c in STRICT]]
    sd = Xs[i0].std(0); sd[sd < 1e-12] = 1.0
    D = (Xs[i1] - Xs[i0]) / sd
    out["median_shift_sigma"] = float(np.median(np.linalg.norm(D, axis=1)
                                                / np.sqrt(D.shape[1])))
    print(f"  median descriptor shift: {out['median_shift_sigma']:.3f} "
          f"between-cell sigma")
    d = X[i1, ci["frac_inward"]] - X[i0, ci["frac_inward"]]
    fi = float((d > 0).mean() + 0.5 * (d == 0).mean())
    print(f"  frac_inward alone, model-free: {fi:.3f}")
    out["frac_inward_alone"] = fi
    return out


rep = {}
rep["same_cell_displaced"] = evaluate("features_samecell.npz",
                                      "CONTROL: the wrong piece is cell A's OWN "
                                      "cable moved to the site")
rep["frankenmerge_scale"] = evaluate("features_big.npz",
                                     "frankenmerge scale: 5-50% of the arbor is "
                                     "another cell's",
                                     [0.05, 0.10, 0.20, 0.35, 0.50])
json.dump(rep, open(OUT / "controls.json", "w"), indent=1)
print("\nwrote controls.json")
