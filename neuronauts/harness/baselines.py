"""Baselines for the atom-merge task.

The task: given a label-blind population of v117 atoms and a label-blind
candidate panel of atom pairs, decide which pairs belong to the same neuron,
then assemble. Every baseline here is a function from the panel (plus atom
context) to one score per pair, so they are all scored by the same ranking and
assembly code and differ only in the evidence they use.

The ladder, weakest to strongest, so a reported gain can be attributed:

1. ``do_nothing`` -- every atom stays alone. Not a scorer; the reference
   partition. On this substrate it is a strong baseline for *precision* by
   construction and the thing any method must beat on recall without losing
   precision. EXP-053A showed several learned scorers failing exactly here.
2. ``random`` -- a seeded shuffle of the panel. Fixes the base rate: with ~1%
   positives an AUC of 0.5 and an AP of 0.01 are what chance looks like, and
   any assembly metric at chance says the panel, not the scorer, did the work.
3. ``gap`` -- rank by endpoint gap alone (negated). The cheapest geometric
   signal and the one the tree-assembly work found carries no identity signal
   on its own; it belongs here as the null that the field's proximity
   heuristics reduce to.
4. ``facing`` / ``directed`` -- tangent agreement, alone and combined with gap
   and caliber continuity. This is the classical skeleton-stitch heuristic
   (RoboEM-style continuation, NEURD-style limb arbitration reduced to the
   evidence we actually have) and the honest non-learned competitor.
5. ``logistic`` -- multinomial-free logistic regression on the pair features,
   fit on the training half of the spatial split. The simplest learned model;
   it says how much of the task is a linear function of hand-made geometry.
6. ``gbdt`` -- gradient-boosted stumps on the same features, same split. The
   strongest non-neural learned baseline, and the bar a graph model has to
   clear to justify itself.

Implementation notes. The learned models are small and are implemented here in
numpy rather than pulled from scikit-learn (absent from this environment) so a
baseline run has no new dependency and is reproducible from a seed. Both are
plain, well-understood estimators; neither is a contribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from neuronauts.harness.candidates import PAIR_COLS, CandidatePanel

# ---------------------------------------------------------------------------
# feature construction
# ---------------------------------------------------------------------------

#: Pair features every scorer may use. Geometry from the panel, plus atom
#: context joined by atom id. All label-blind.
FEATURE_COLS = [
    "gap_um", "log_gap", "facing", "align_a", "align_b", "align_min",
    "caliber_min_nm", "caliber_ratio", "leaf_min_um",
    "n_syn_min", "n_syn_max", "cable_min_um", "cable_max_um",
    "polarity_agree", "same_polarity_pre", "same_polarity_post",
    "endpoints_min", "components_max",
]


@dataclass
class AtomContext:
    """Per-atom, label-blind context joined onto pairs by atom id."""

    atom_id: np.ndarray
    n_syn: np.ndarray
    cable_nm: np.ndarray
    n_end: np.ndarray
    n_comp: np.ndarray
    n_pre: np.ndarray
    n_post: np.ndarray
    _index: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._order = np.argsort(self.atom_id, kind="stable")
        self._sorted = self.atom_id[self._order]

    def rows(self, atoms: np.ndarray) -> np.ndarray:
        """Row index per atom id, -1 when absent."""
        a = np.asarray(atoms, self.atom_id.dtype)
        j = np.searchsorted(self._sorted, a)
        jc = np.clip(j, 0, max(len(self._sorted) - 1, 0))
        ok = (len(self._sorted) > 0) & (self._sorted[jc] == a)
        return np.where(ok, self._order[jc], -1).astype(np.int64)

    @classmethod
    def from_topology(cls, npz_path, population=None) -> "AtomContext":
        with np.load(npz_path, allow_pickle=False) as z:
            return cls(atom_id=z["atom_id"],
                       n_syn=(z["n_pre"] + z["n_post"]).astype(np.float32),
                       cable_nm=z["cable_nm"].astype(np.float32),
                       n_end=z["n_end"].astype(np.float32),
                       n_comp=z["n_comp"].astype(np.float32),
                       n_pre=z["n_pre"].astype(np.float32),
                       n_post=z["n_post"].astype(np.float32))


def pair_features(panel: CandidatePanel, ctx: AtomContext) -> np.ndarray:
    """[P, len(FEATURE_COLS)] label-blind features for every candidate pair."""
    gap = panel.col("gap_nm").astype(np.float64)
    facing = panel.col("facing").astype(np.float64)
    al_a = panel.col("align_a").astype(np.float64)
    al_b = panel.col("align_b").astype(np.float64)
    cal_a = panel.col("caliber_a").astype(np.float64)
    cal_b = panel.col("caliber_b").astype(np.float64)
    leaf_a = panel.col("leaf_len_a").astype(np.float64)
    leaf_b = panel.col("leaf_len_b").astype(np.float64)

    ra, rb = ctx.rows(panel.atom_a), ctx.rows(panel.atom_b)
    take = lambda arr, r: np.where(r >= 0, arr[np.clip(r, 0, len(arr) - 1)], np.nan)
    syn_a, syn_b = take(ctx.n_syn, ra), take(ctx.n_syn, rb)
    cab_a, cab_b = take(ctx.cable_nm, ra), take(ctx.cable_nm, rb)
    end_a, end_b = take(ctx.n_end, ra), take(ctx.n_end, rb)
    cmp_a, cmp_b = take(ctx.n_comp, ra), take(ctx.n_comp, rb)
    pre_a, pre_b = take(ctx.n_pre, ra), take(ctx.n_pre, rb)
    post_a, post_b = take(ctx.n_post, ra), take(ctx.n_post, rb)

    # polarity: an atom whose synapses are mostly presynaptic is axonal. Two
    # atoms of one neurite normally agree; an axon meeting a dendrite is a
    # merge across compartments and is usually wrong.
    frac = lambda pre, post: pre / np.maximum(pre + post, 1.0)
    fa, fb = frac(pre_a, post_a), frac(pre_b, post_b)
    polarity_agree = 1.0 - np.abs(fa - fb)

    cal_min = np.minimum(cal_a, cal_b)
    cal_max = np.maximum(cal_a, cal_b)
    feats = np.stack([
        gap / 1000.0,
        np.log1p(gap),
        facing,
        al_a,
        al_b,
        np.minimum(al_a, al_b),
        cal_min,
        cal_min / np.maximum(cal_max, 1e-6),
        np.minimum(leaf_a, leaf_b) / 1000.0,
        np.minimum(syn_a, syn_b),
        np.maximum(syn_a, syn_b),
        np.minimum(cab_a, cab_b) / 1000.0,
        np.maximum(cab_a, cab_b) / 1000.0,
        polarity_agree,
        ((fa > 0.5) & (fb > 0.5)).astype(np.float64),
        ((fa <= 0.5) & (fb <= 0.5)).astype(np.float64),
        np.minimum(end_a, end_b),
        np.maximum(cmp_a, cmp_b),
    ], axis=1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# unlearned scorers
# ---------------------------------------------------------------------------

def score_random(x: np.ndarray, *, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(len(x))


def score_gap(x: np.ndarray) -> np.ndarray:
    """Closer is better. The proximity null."""
    return -x[:, FEATURE_COLS.index("gap_um")]


def score_facing(x: np.ndarray) -> np.ndarray:
    """Tangent agreement alone: do the two tips point at each other."""
    return x[:, FEATURE_COLS.index("facing")]


def score_directed(x: np.ndarray, *, gap_scale_um: float = 2.0,
                   w_facing: float = 1.0, w_align: float = 1.0,
                   w_caliber: float = 0.5) -> np.ndarray:
    """Classical directed-continuation heuristic, in one score.

    Continuation plausibility falls off with the gap, rises when the tips
    face each other and each points at the other, and rises with caliber
    continuity. The weights are the conventional "all evidence counts about
    the same" choice, not fitted; ``logistic`` is the fitted version of this
    same feature set and the comparison between them is informative.
    """
    gap = x[:, FEATURE_COLS.index("gap_um")]
    facing = x[:, FEATURE_COLS.index("facing")]
    al_min = x[:, FEATURE_COLS.index("align_min")]
    cal_ratio = x[:, FEATURE_COLS.index("caliber_ratio")]
    return (-gap / gap_scale_um + w_facing * facing + w_align * al_min
            + w_caliber * cal_ratio)


# ---------------------------------------------------------------------------
# learned scorers (numpy; no new dependency)
# ---------------------------------------------------------------------------

@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        m = x.mean(axis=0)
        s = x.std(axis=0)
        s[s < 1e-9] = 1.0
        return cls(m.astype(np.float64), s.astype(np.float64))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, np.float64) - self.mean) / self.scale


@dataclass
class LogisticRegression:
    """L2-regularised logistic regression by full-batch gradient descent.

    Class weighting balances the positives, which are a small share of the
    panel; without it the fit collapses to predicting "never merge".
    """

    weight: np.ndarray
    bias: float
    std: Standardizer
    n_iter: int
    loss: float

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, *, l2: float = 1e-3,
            lr: float = 0.5, n_iter: int = 600, balance: bool = True,
            tol: float = 1e-7) -> "LogisticRegression":
        std = Standardizer.fit(x)
        z = std(x)
        y = np.asarray(y, np.float64)
        w = np.ones(len(y))
        if balance and 0 < y.sum() < len(y):
            w = np.where(y > 0, len(y) / (2.0 * y.sum()),
                         len(y) / (2.0 * (len(y) - y.sum())))
        w = w / w.mean()
        beta = np.zeros(z.shape[1])
        b = 0.0
        prev = np.inf
        loss = np.inf
        for it in range(n_iter):
            p = _sigmoid(z @ beta + b)
            g = (w * (p - y))
            grad_beta = z.T @ g / len(y) + l2 * beta
            grad_b = g.mean()
            beta -= lr * grad_beta
            b -= lr * grad_b
            loss = float(np.mean(w * _bce(y, p)) + 0.5 * l2 * float(beta @ beta))
            if abs(prev - loss) < tol:
                break
            prev = loss
        return cls(beta, float(b), std, it + 1, loss)

    def decision(self, x: np.ndarray) -> np.ndarray:
        return self.std(x) @ self.weight + self.bias

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(self.decision(x))


def _sigmoid(v: np.ndarray) -> np.ndarray:
    out = np.empty_like(v, dtype=np.float64)
    pos = v >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-v[pos]))
    e = np.exp(v[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _bce(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


@dataclass
class Stump:
    feature: int
    threshold: float
    left: float
    right: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.where(x[:, self.feature] <= self.threshold, self.left, self.right)


@dataclass
class GradientBoostedStumps:
    """Depth-1 gradient boosting on the logistic loss.

    Stumps are fit on quantile bins of each feature, so the cost is linear in
    the number of pairs per round and the model is exactly reproducible.
    """

    stumps: list
    init: float
    lr: float
    n_bins: int

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, *, n_rounds: int = 120,
            lr: float = 0.15, n_bins: int = 24, min_leaf: int = 40,
            balance: bool = True, seed: int = 0) -> "GradientBoostedStumps":
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        w = np.ones(len(y))
        if balance and 0 < y.sum() < len(y):
            w = np.where(y > 0, len(y) / (2.0 * y.sum()),
                         len(y) / (2.0 * (len(y) - y.sum())))
            w = w / w.mean()
        prior = float(np.clip((w * y).sum() / max(w.sum(), 1e-9), 1e-6, 1 - 1e-6))
        init = float(np.log(prior / (1 - prior)))
        f = np.full(len(y), init)

        edges = []
        codes = np.empty(x.shape, np.int16)
        for j in range(x.shape[1]):
            qs = np.unique(np.quantile(x[:, j], np.linspace(0, 1, n_bins + 1)[1:-1]))
            edges.append(qs)
            codes[:, j] = np.searchsorted(qs, x[:, j], side="right").astype(np.int16)

        stumps: list[Stump] = []
        for _ in range(n_rounds):
            p = _sigmoid(f)
            grad = w * (p - y)
            hess = w * np.maximum(p * (1 - p), 1e-6)
            best = None
            for j in range(x.shape[1]):
                nb = len(edges[j]) + 1
                if nb < 2:
                    continue
                g = np.bincount(codes[:, j], weights=grad, minlength=nb)
                h = np.bincount(codes[:, j], weights=hess, minlength=nb)
                c = np.bincount(codes[:, j], minlength=nb)
                gl, hl, cl = np.cumsum(g)[:-1], np.cumsum(h)[:-1], np.cumsum(c)[:-1]
                gr, hr = g.sum() - gl, h.sum() - hl
                cr = c.sum() - cl
                ok = (cl >= min_leaf) & (cr >= min_leaf)
                if not ok.any():
                    continue
                gain = np.where(ok, gl ** 2 / np.maximum(hl, 1e-9)
                                + gr ** 2 / np.maximum(hr, 1e-9), -np.inf)
                b = int(np.argmax(gain))
                if best is None or gain[b] > best[0]:
                    best = (float(gain[b]), j, b, float(-gl[b] / max(hl[b], 1e-9)),
                            float(-gr[b] / max(hr[b], 1e-9)))
            if best is None:
                break
            _, j, b, left, right = best
            thr = float(edges[j][b])
            s = Stump(j, thr, lr * left, lr * right)
            stumps.append(s)
            f = f + s.predict(x)
        return cls(stumps, init, lr, n_bins)

    def decision(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float64)
        f = np.full(len(x), self.init)
        for s in self.stumps:
            f = f + s.predict(x)
        return f

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(self.decision(x))


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

@dataclass
class Baseline:
    name: str
    kind: str                      # "reference" | "unlearned" | "learned"
    describe: str
    fit: Optional[Callable] = None
    score: Optional[Callable] = None


def unlearned_baselines(seed: int = 0) -> list[Baseline]:
    return [
        Baseline("random", "unlearned",
                 "seeded shuffle -- fixes the panel base rate",
                 score=lambda x, seed=seed: score_random(x, seed=seed)),
        Baseline("gap", "unlearned",
                 "rank by endpoint gap alone (proximity null)",
                 score=score_gap),
        Baseline("facing", "unlearned",
                 "tangent agreement between the two tips",
                 score=score_facing),
        Baseline("directed", "unlearned",
                 "gap + facing + alignment + caliber continuity, unfitted weights",
                 score=score_directed),
    ]


def learned_baselines(seed: int = 0) -> list[Baseline]:
    return [
        Baseline("logistic", "learned",
                 "balanced L2 logistic regression on the pair features",
                 fit=lambda x, y: LogisticRegression.fit(x, y),
                 score=None),
        Baseline("gbdt", "learned",
                 "gradient-boosted depth-1 stumps on the pair features",
                 fit=lambda x, y, seed=seed: GradientBoostedStumps.fit(x, y, seed=seed),
                 score=None),
    ]


def all_baselines(seed: int = 0) -> list[Baseline]:
    return unlearned_baselines(seed) + learned_baselines(seed)
