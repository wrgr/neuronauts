"""Audit: what do the panel-era scoring rules do at a frontier base rate?

Every ranking number in this repository is measured on cut-centred panels,
which hand the grower the answer to "where should I look" and then ask only
"which candidate is right". EXP-081 measured the question actually asked: a
soma-seeded grower faces a median of 46 cut ends per cell, of which a median
of 1 is a real extension site (34 live sites in 2,137 tips, 1.6%).

This scores the same panels as a JOINT decision -- extend or decline, and if
extend, to which object -- and converts the per-decision rates into the
precision a grower would see at that base rate.

Scorers:
  treestitch   max(0, 1 - gap/10000) * (dna_cos + 1)/2, the rule in
               treestitch/stitch.py::candidate_stitch_edges. No pooled DNA
               exists for these objects, so the DNA factor is 1 and the rule
               reduces to "nearest object".
  geometry     along * collin * exp(-gap/500) * caliber-agreement, the current
               best stack (scripts/exp079_evaluate.py::base).

Negatives are decision sites where the honest answer is "nothing continues":
33 already-whole panels (data/external/panels/) plus the 25 EXP-076-corrected
terminal panels (data/external/panels_tip/), pooled to 58.

Thresholds are chosen on a held-out half and applied to the other half, so no
threshold is scored on the panels that selected it.

    python scripts/audit_frontier_precision.py
"""
import glob
import numpy as np

R = "/Users/wgray13/projects/neuronauts/"
EPS = 1e-9
TIPS_PER_CELL, LIVE_PER_CELL = 46.0, 1.0     # EXP-081, 40 cells


def load(f):
    z = np.load(f, allow_pickle=True)
    return {k: z[k] for k in z.files}


def scores(p):
    gap = p["gap_nm"].astype(np.float64)
    al, cl = p["along"].astype(np.float64), p["collin"].astype(np.float64)
    cc, cs = p["cal_cand"].astype(np.float64), float(p["cal_seed"])
    cal = np.minimum(cc, cs) / np.maximum(np.maximum(cc, cs), EPS)
    return {"treestitch": np.maximum(0.0, 1.0 - gap / 10_000.0),
            "geometry": al * cl * np.exp(-gap / 500.0) * cal}


def clopper_upper(k, n, alpha=0.05):
    """Upper 95% bound on a binomial rate; 3/n when k == 0."""
    from scipy.stats import beta
    return 1.0 if k == n else float(beta.ppf(1 - alpha, k + 1, n - k))


allp = list(map(load, sorted(glob.glob(R + "data/external/panels/*.npz"))))
cut = [p for p in allp if not bool(p["already_whole"]) and p["in_target"].sum() == 1]
neg = [p for p in allp if bool(p["already_whole"])]
neg += list(map(load, sorted(glob.glob(R + "data/external/panels_tip/*.npz"))))
print(f"cut panels {len(cut)}   negative (nothing-continues) panels {len(neg)}")
print(f"candidates per cut panel: median {np.median([len(p['obj']) for p in cut]):.0f}")

NAMES = ["treestitch", "geometry"]
top_c = {n: np.array([scores(p)[n].max() for p in cut]) for n in NAMES}
hit_c = {n: np.array([bool(p["in_target"][int(np.argmax(scores(p)[n]))]) for p in cut])
         for n in NAMES}
top_n = {n: np.array([scores(p)[n].max() for p in neg]) for n in NAMES}

print("\n== 1. RANKING on the 66 cut panels (the published protocol) ==")
print(f"{'scorer':12s} {'median rank':>11s} {'top-1':>8s} {'top-5':>8s} {'top-20':>8s}")
for n in NAMES:
    r = []
    for p in cut:
        s = scores(p)[n]
        i = int(np.flatnonzero(p["in_target"])[0])
        r.append((s > s[i]).sum() + ((s == s[i]).sum() + 1) / 2.0)
    r = np.array(r)
    print(f"{n:12s} {np.median(r):11.1f} {int((r <= 1.5).sum()):6d}/66 "
          f"{int((r <= 5).sum()):6d}/66 {int((r <= 20).sum()):6d}/66")
print(f"{'random':12s} {np.median([len(p['obj']) for p in cut])/2:11.0f} "
      f"{66/np.median([len(p['obj']) for p in cut]):8.3f}/66")


def frontier(p_tp, k_neg, n_neg):
    """Precision at the EXP-081 frontier composition, with a 95% upper bound
    on the negative fire rate carried through."""
    out = []
    for rate in (k_neg / n_neg, clopper_upper(k_neg, n_neg)):
        false = rate * (TIPS_PER_CELL - LIVE_PER_CELL)
        tp = p_tp * LIVE_PER_CELL
        out.append((rate, false, tp / (tp + false) if (tp + false) > 0 else float("nan")))
    return out


print("\n== 2. THE JOINT DECISION at the frontier composition "
      f"({TIPS_PER_CELL:.0f} tips/cell, {LIVE_PER_CELL:.0f} live) ==")
for n in NAMES:
    print(f"\n-- {n} --")
    print(f"{'thresh':>10s} {'joins':>6s} {'right':>6s} {'panel prec':>11s} "
          f"{'neg fire':>9s} {'false/cell':>11s} {'frontier prec':>14s} {'(95% ub)':>10s}")
    cand = np.unique(np.concatenate([top_c[n], top_n[n]]))
    grid = [-np.inf] + list(np.percentile(cand, [50, 75, 90, 95, 99, 99.9])) + [cand.max()]
    for t in grid:
        ext = top_c[n] >= t
        right = int((ext & hit_c[n]).sum())
        k = int((top_n[n] >= t).sum())
        p_tp = right / len(cut)
        (r0, f0, fp0), (r1, f1, fp1) = frontier(p_tp, k, len(neg))
        pp = right / int(ext.sum()) if ext.sum() else float("nan")
        print(f"{t:10.4g} {int(ext.sum()):6d} {right:6d} {pp:11.3f} "
              f"{r0:9.3f} {f0:11.2f} {fp0:14.3f} {fp1:10.3f}")

print("\n== 3. HELD-OUT THRESHOLD, criterion pre-registered by EXP-081 ==")
print("   EXP-081: 'the per-tip false-positive rate has to sit below roughly 2%'.")
print("   Threshold = lowest one meeting that on the selection half; scored on the other half.")
TARGET = 0.02
for n in NAMES:
    ic, inn = np.arange(len(cut)), np.arange(len(neg))
    selc, scoc = ic[::2], ic[1::2]
    seln, scon = inn[::2], inn[1::2]
    cand = np.sort(np.unique(np.concatenate([top_c[n][selc], top_n[n][seln]])))
    ok = [t for t in cand if (top_n[n][seln] >= t).mean() <= TARGET]
    if not ok:
        print(f"{n:12s} no threshold on the selection half reaches a 2% fire rate")
        continue
    t = ok[0]
    right = int(((top_c[n][scoc] >= t) & hit_c[n][scoc]).sum())
    k = int((top_n[n][scon] >= t).sum())
    p_tp = right / len(scoc)
    (r0, f0, fp0), (r1, f1, fp1) = frontier(p_tp, k, len(scon))
    print(f"{n:12s} t={t:.4g}  held out: {right}/{len(scoc)} cut panels joined correctly; "
          f"negatives fire {k}/{len(scon)} ({r0:.3f})")
    print(f"{'':12s}   {p_tp:.3f} correct and {f0:.2f} false joins per cell -> "
          f"frontier precision {fp0:.3f}")
    print(f"{'':12s}   carrying the 95% upper bound on the fire rate ({r1:.3f}): "
          f"{f1:.2f} false joins per cell, precision {fp1:.3f}")

print("\n== 4. IS THE CORPUS BIG ENOUGH TO CERTIFY 2%? ==")
need = int(np.ceil(3.0 / TARGET))
print(f"   zero false fires in n negatives gives a 95% upper bound of about 3/n.")
print(f"   certifying {TARGET:.0%} needs n >= {need} negative decision sites; we have {len(neg)}.")
print(f"   best upper bound available from {len(neg)} clean negatives: "
      f"{clopper_upper(0, len(neg)):.3f} -> "
      f"{clopper_upper(0, len(neg)) * (TIPS_PER_CELL - LIVE_PER_CELL):.1f} false joins per cell.")
print(f"   EXP-081 already located 2,103 dead tips; that is the corpus this needs.")
