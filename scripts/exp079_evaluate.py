"""EXP-079 -- score the corrected contact panels with a morphology grammar
whose productions are estimated from real proofread arbors, and compare it with
the pairwise geometric baseline of EXP-076.

Every production density is fitted with the scored panel's own cell held out.
The background density is the panel candidate field itself, which carries 66
positives among ~160,000 candidates and is therefore effectively unsupervised.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, wilcoxon

R = Path(__file__).resolve().parents[1]
EPS = 1e-9


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    x = np.concatenate([pos, neg])
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    r = rankdata(x)
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class LR:
    """log p(x | CONTINUE) - log p(x | background), as a binned density ratio.

    Counts are kept per cell so a cell can be dropped from both sides when its
    own panel is scored.
    """

    def __init__(self, num_x, num_cell, den_x, den_cell, n_bins=12, smooth=1.0):
        num_x, den_x = np.asarray(num_x, float), np.asarray(den_x, float)
        q = np.linspace(0, 100, n_bins + 1)[1:-1]
        self.edges = np.unique(np.percentile(np.concatenate([num_x, den_x]), q))
        self.nb = len(self.edges) + 1
        self.smooth = smooth
        self.num = self._tab(num_x, num_cell)
        self.den = self._tab(den_x, den_cell)

    def _tab(self, x, cell):
        b = np.digitize(x, self.edges)
        out = {}
        for c in np.unique(cell):
            out[int(c)] = np.bincount(b[cell == c], minlength=self.nb).astype(float)
        out["_all"] = np.bincount(b, minlength=self.nb).astype(float)
        return out

    def score(self, x, hold_out):
        n = self.num["_all"] - self.num.get(hold_out, 0.0)
        d = self.den["_all"] - self.den.get(hold_out, 0.0)
        pn = (n + self.smooth) / (n.sum() + self.smooth * self.nb)
        pd = (d + self.smooth) / (d.sum() + self.smooth * self.nb)
        return np.log(pn / pd)[np.digitize(np.asarray(x, float), self.edges)]


def main():
    panels = json.load(open(R / "data/external/exp079_panel_tree.json"))
    skel = json.load(open(R / "data/external/exp079_skel_tree.json"))
    cont = json.load(open(R / "data/external/exp079_continuations.json"))
    print(f"{len(panels)} panels, {len(skel)} skeleton tree rows, {len(cont)} continuation rows")

    # ---- assemble candidate tables -------------------------------------
    for p in panels:
        for c in p["cands"]:
            ext = c.get("extent")
            c["dsoma_frac"] = (c["dsoma"] / ext) if (c.get("covered") and ext and ext > 500) else np.nan
    cellof = {p["key"]: p["cell"] for p in panels}

    def cat(field, covered_only=False):
        xs, cs = [], []
        for p in panels:
            for c in p["cands"]:
                if covered_only and not c["covered"]:
                    continue
                v = c.get(field)
                if v is None or not np.isfinite(v):
                    continue
                xs.append(float(v)); cs.append(p["cell"])
        return np.array(xs), np.array(cs)

    # skeleton (CONTINUE) side
    sk_cell = np.array([r["cell"] for r in skel])
    sk = {f: np.array([r[f] if r[f] is not None else np.nan for r in skel], float)
          for f in ("oproj", "occupy", "dsoma", "extent")}
    sk["dsoma_frac"] = sk["dsoma"] / np.where(sk["extent"] > 500, sk["extent"], np.nan)
    co_cell = np.array([r["cell"] for r in cont])
    co = {f: np.array([r[f] for r in cont], float) for f in ("along", "collin")}

    lrs = {}
    for f in ("oproj", "occupy", "dsoma_frac"):
        m = np.isfinite(sk[f])
        bx, bc = cat(f, covered_only=True)
        lrs[f] = LR(sk[f][m], sk_cell[m], bx, bc)
    for f in ("along", "collin"):
        bx, bc = cat(f)
        lrs[f] = LR(co[f], co_cell, bx, bc)

    # ---- per-panel scores ----------------------------------------------
    TREE = ("oproj", "occupy", "dsoma_frac")
    GEOM = ("along", "collin")
    for p in panels:
        cs = p["cands"]
        ho = p["cell"]
        cov = np.array([c["covered"] and np.isfinite(c.get("dsoma_frac", np.nan)) for c in cs])
        n = len(cs)
        tree = np.zeros(n)
        for f in TREE:
            v = np.array([c.get(f, np.nan) if cov[i] else np.nan for i, c in enumerate(cs)], float)
            s = np.zeros(n)
            if cov.any():
                s[cov] = lrs[f].score(np.nan_to_num(v[cov]), ho)
            tree += s
        # objects with no cloud get the covered median, so having a cloud is
        # not itself evidence -- the mip-5 clouds hold every one of the 66
        # targets but only 54% of candidates
        if cov.any():
            tree[cov] -= np.median(tree[cov])
        tree[~cov] = 0.0
        geom = np.zeros(n)
        for f in GEOM:
            geom += lrs[f].score(np.array([c[f] for c in cs], float), ho)
        gap = np.array([c["gap"] for c in cs], float)
        al = np.array([c["along"] for c in cs], float)
        cl = np.array([c["collin"] for c in cs], float)
        cc = np.array([c["cal_cand"] for c in cs], float)
        cal = np.minimum(cc, p["cal_seed"]) / np.maximum(np.maximum(cc, p["cal_seed"]), EPS)
        base = al * cl * np.exp(-gap / 500.0) * cal
        p["scores"] = dict(
            baseline=np.log(np.maximum(base, 1e-300)),
            grammar_tree=tree,
            grammar_geom=geom,
            grammar=geom + tree,
            grammar_full=geom + tree + np.log(np.maximum(np.exp(-gap / 500.0) * cal, 1e-300)),
            baseline_plus_tree=np.log(np.maximum(base, 1e-300)) + tree,
            covered=cov.astype(float),
        )
        p["y"] = np.array([c["in_target"] for c in cs], bool)

    names = ["baseline", "grammar_geom", "grammar_tree", "grammar", "grammar_full",
             "baseline_plus_tree", "covered"]

    # ---- ranking ---------------------------------------------------------
    cut = [p for p in panels if not p["already_whole"] and p["y"].sum() == 1]
    print(f"\nRANKING  n={len(cut)} cut panels, median "
          f"{np.median([len(p['cands']) for p in cut]):.0f} candidates")
    print(f"{'score':22s} {'median rank':>11s} {'top-1':>6s} {'top-5':>6s} {'top-20':>7s}")
    ranks = {}
    for nm in names:
        rs = []
        for p in cut:
            s = p["scores"][nm]
            i = int(np.flatnonzero(p["y"])[0])
            rs.append((s > s[i]).sum() + ((s == s[i]).sum() + 1) / 2.0)
        ranks[nm] = np.array(rs, float)
        r = ranks[nm]
        print(f"{nm:22s} {np.median(r):11.1f} {int((r<=1.5).sum()):6d} "
              f"{int((r<=5).sum()):6d} {int((r<=20).sum()):7d}")
    for nm in names:
        if nm == "baseline":
            continue
        d = ranks["baseline"] - ranks[nm]
        if np.any(d != 0):
            st = wilcoxon(ranks["baseline"], ranks[nm])
            print(f"  {nm:22s} vs baseline: better on {int((d>0).sum())}/{len(d)} panels, "
                  f"worse on {int((d<0).sum())}, Wilcoxon p={st.pvalue:.3g}")

    # ---- stopping --------------------------------------------------------
    whole = [p for p in panels if p["already_whole"]]

    def stop_scores(nm):
        """One number per panel: how strongly it says a continuation exists."""
        if nm == "end_ratio":     # seed-side only: a cut face keeps its caliber
            return (np.array([-p["end_ratio"] for p in cut]),
                    np.array([-p["end_ratio"] for p in whole]))
        c = np.array([p["scores"][nm].max() for p in cut])
        w = np.array([p["scores"][nm].max() for p in whole])
        return c, w

    def zsum(parts):
        cs, ws = [], []
        for nm in parts:
            c, w = stop_scores(nm)
            m = np.nanmean(np.r_[c, w]); sd = np.nanstd(np.r_[c, w]) or 1.0
            cs.append((c - m) / sd); ws.append((w - m) / sd)
        return np.nansum(cs, 0), np.nansum(ws, 0)
    print(f"\nSTOPPING  {len(cut)} panels with a continuation vs {len(whole)} genuine terminals")
    d_cut = np.array([p["soma_nm"] for p in cut]) / 1000.0
    d_wh = np.array([p["soma_nm"] for p in whole]) / 1000.0
    print(f"  distance from soma: cuts {np.percentile(d_cut,[10,50,90]).round(0)} um, "
          f"terminals {np.percentile(d_wh,[10,50,90]).round(0)} um, "
          f"AUC on distality alone {auc(d_wh, d_cut):.3f}")
    band = (max(d_cut.min(), d_wh.min()), min(d_cut.max(), d_wh.max()))
    pairs = [(i, j) for i in range(len(whole)) for j in range(len(cut))
             if abs(d_wh[i] - d_cut[j]) <= 5.0 and band[0] <= d_wh[i] <= band[1]
             and band[0] <= d_cut[j] <= band[1]]
    print(f"  matched at 5 um: {len(pairs)} pairs from "
          f"{len(set(i for i,_ in pairs))} terminals and {len(set(j for _,j in pairs))} cuts")
    print(f"{'score':26s} {'AUC all':>8s} {'95% CI':>16s} {'matched':>8s}")
    rng = np.random.default_rng(0)
    stop_names = names + ["end_ratio", "end_ratio+grammar", "end_ratio+baseline"]
    for nm in stop_names:
        if nm == "end_ratio+grammar":
            sc, sw = zsum(["end_ratio", "grammar"])
        elif nm == "end_ratio+baseline":
            sc, sw = zsum(["end_ratio", "baseline"])
        else:
            sc, sw = stop_scores(nm)
        a = auc(sc, sw)
        boot = []
        for _ in range(2000):                    # cluster bootstrap over panels
            i = rng.integers(0, len(sc), len(sc)); j = rng.integers(0, len(sw), len(sw))
            boot.append(auc(sc[i], sw[j]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        m = np.mean([sc[j] > sw[i] for i, j in pairs]) + \
            0.5 * np.mean([sc[j] == sw[i] for i, j in pairs]) if pairs else np.nan
        print(f"{nm:26s} {a:8.3f} {f'[{lo:.2f}, {hi:.2f}]':>16s} {m:8.3f}")
    rd = np.mean([d_cut[j] > d_wh[i] for i, j in pairs]) if pairs else np.nan
    print(f"  residual distality over the matched pairs: {rd:.3f} "
          f"(0.5 = matched)")

    out = dict(
        ranking={nm: dict(median=float(np.median(ranks[nm])),
                          top1=int((ranks[nm] <= 1.5).sum()),
                          top5=int((ranks[nm] <= 5).sum()),
                          top20=int((ranks[nm] <= 20).sum())) for nm in names},
        n_cut=len(cut), n_whole=len(whole), n_pairs=len(pairs))
    json.dump(out, open(R / "results/EXP-079/result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
