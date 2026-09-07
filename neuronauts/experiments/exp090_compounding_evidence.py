"""EXP-090 -- does tree-level evidence actually compound?

EXP-084 measured Cajal's material-conservation law (Murray's law) at real
bifurcations and found it holds -- exponent median 3.18 against an ideal 3.0
across 3,781 branch points -- and that a caliber-mismatched branch point
separates from a real one at area under the receiver operating characteristic
curve (AUC) 0.675, with no parameters and no training. Its "Honest reading"
then made a claim it did not test:

    0.675 is a real signal and a weak one, from a single branch point. Its
    value is that it COMPOUNDS: an assembled cell has many branch points, and a
    wrong join creates one bad branch among many good ones, so the evidence
    accumulates over a tree.

Nothing has tested that. This experiment does, and it is built so the answer is
not smuggled in, because the same-sounding claim already died once. EXP-083
scored whole-cell SHAPE the same way and got 0.505 held out by cell, 0.56 with
a third of the neuron foreign, for a reason that is a number rather than an
opinion: real neurons differ from each other far more than a chimera differs
from its host. Aggregating a weak per-branch signal over a tree aggregates its
noise too, and between-cell variation may swamp it exactly as it did there.

Question
--------
Does an aggregate of per-bifurcation conservation residuals separate a
correctly assembled arbor from one carrying k wrong joins, and does the
separation grow with k?

Construction (EXP-083's, reused deliberately)
---------------------------------------------
``scripts/exp083_shape_lib.py`` is imported, not reimplemented: the same
breadth-first rooted tree, the same subtree masks, and the same graft geometry.
Rebuilding it differently would make this result incomparable with EXP-083's,
which is most of its value. Per proofread cell A:

  base        A's arbor with subtree ``z`` removed (3-15% of the cable, so the
              correct assembly is itself partial, as a grower's is) and the k
              subtrees under test removed.
  CORRECT     base + A's own k subtrees, put back exactly where they were.
  FOREIGN     base + k subtrees of OTHER proofread cells, each cable-matched to
              within 15% and rigidly translated so its root lands on A's vertex
              u_i -- so each join edge has A's own length, direction and parent.
  SAME-CELL   base + k subtrees of A ITSELF taken from elsewhere in the arbor,
              same matching, same sites. This is EXP-083's control, and it is
              required: there the displaced own-cable arm was detected BETTER
              (0.710) than the foreign arm (0.642), which showed the shape score
              was reading local placement, not foreign identity. If the
              conservation aggregate reproduces that pattern it is measuring the
              same thing and the compounding story is wrong.

One deviation from EXP-083, and the reason for it: sites are restricted to
vertices ``u`` whose parent is a genuine bifurcation with exactly two children,
and whose sibling survives every removal. EXP-083 drew ``u`` from anywhere in
the arbor, but a graft replacing the only child of a path vertex creates no
branch point at all, so a conservation law has nothing to read there. The graft
mechanics are unchanged; the site set is a strict subset of EXP-083's, and that
is stated rather than hidden.

Scoring
-------
Every candidate bifurcation of an assembly yields two parameter-free residuals,
both oriented so larger is worse:

  murray   |p - 3| where p solves ``r0^p = r1^p + r2^p`` -- EXP-084's statistic.
  angle    ``1 - SantiagoCajalPriors.compute_bifurcation_angle_prior(...)`` --
           the Hess-Murray optimal branching angle, from the same module.

Two role assignments, because EXP-084 worked on an unrooted skeleton and this
works on a rooted one:

  unrooted  EXP-084's own convention: of a degree-3 vertex's three neighbours,
            the thickest is the mother and the other two are daughters. PRIMARY,
            so the comparison against 0.675 is like for like.
  rooted    mother is the parent, daughters are the two children. Structurally
            correct, but it discards every bifurcation whose measured parent
            radius is not the largest -- and a foreign daughter that is fatter
            than the mother is exactly the case it discards, which would hide
            the corruption. Reported alongside, never as the headline, and the
            discard rate is reported per arm as ``frac_invalid`` so that
            selection is visible rather than silent.

Aggregates over an assembly's residuals, all reported, none chosen after the
fact: mean, median, the 90th/95th/99th percentiles, the maximum, the count and
the fraction above a tail threshold (2.0 for murray, from EXP-084's published
medians; for the angle residual, which is bounded below 1, the residual of an
angle 45 degrees off the optimum), and the fraction of candidate bifurcations
with no admissible exponent. The primary is declared in
``SPEC.params`` before the run: the FRACTION above threshold, on the murray
residual, unrooted. The reasoning, written down first: the k wrong joins add k
bad branch points to a tree of several hundred, so the expected shift is
proportional to k over the bifurcation count; a mean is diluted by the
magnitude of the real spread (EXP-084's interquartile range is 2.17-4.65, wide),
and a maximum or a very high quantile is set by the real tail, which both
assemblies share. A tail COUNT is the aggregate whose expected shift tracks k
most directly. That is a prediction, not a finding.

Three further measurements, because "AUC 0.85 at k=5" alone would not settle
the question:

  single-join reference   the residual at ONE join bifurcation, correct against
                          corrupted, inside THIS construction. EXP-084's 0.675
                          came from a stronger corruption (both daughters
                          foreign); here only one daughter is foreign, so the
                          within-experiment single-site number is the reference
                          a whole-tree aggregate must beat for "compounding" to
                          mean anything. The declared bar still names 0.675.
  known-join aggregate    mean and max of the residual over the k join sites,
                          which is compounding WITH the join locations known.
                          An assembler knows where it joined; an auditor of
                          someone else's assembly does not. Separating this
                          from the blind whole-tree aggregate is the point.
  size-only control       AUC from the bifurcation count alone. EXP-083's
                          "size alone is 0.50, so the matching worked" check.

Held out by cell, five cell-disjoint folds. Nothing is fitted -- these are
parameter-free priors and a threshold taken from EXP-084's published medians
(real |p-3| 1.10, mismatched 1.94) before this data existed -- so the folds
report between-cell variability, they do not guard against overfitting, and
that is said plainly rather than dressed up. A label-shuffle null over the same
pipeline says what a nonzero separation is worth.

    python -m neuronauts.experiments.exp090_compounding_evidence
"""

from __future__ import annotations

import glob
import importlib.util
import sys
from pathlib import Path

import numpy as np

from neuronauts.experiments._runner import Context, Outcome, Spec, main
from neuronauts.metrics.ranking import roc_auc

SKEL_DIR = "data/external/cell_skeletons"
SHAPE_LIB = "scripts/exp083_shape_lib.py"

SEED = 0
#: Wrong joins per assembly. The k=1 sites are a prefix of the k=3 sites, which
#: are a prefix of the k=5 sites, and only cells that can supply all five are
#: used -- so the curve over k is not confounded by a changing cell set or a
#: changing choice of site.
K_LIST = (1, 3, 5)
N_SITES = max(K_LIST)
REPS_PER_CELL = 6

#: EXP-083's second removal, unchanged: the correct assembly is partial too.
Z_FRAC = (0.03, 0.15)
#: Site subtree cable. EXP-083's within-site signal peaked at 10-300 um and
#: collapsed above 300 um, so a band inside the range where a whole-tree view
#: had something to say does not handicap the comparison.
SITE_CABLE_UM = (30.0, 300.0)
#: EXP-083's donor tolerance, unchanged.
CABLE_TOL = 0.15
#: No single graft, and no set of grafts, may be half the arbor.
MAX_GRAFT_FRAC = 0.5
DONOR_TRIES = 60

#: EXP-084's bracket and radius floor, unchanged, so residuals are comparable.
P_BRACKET = (0.5, 8.0)
MIN_RADIUS_NM = 20.0
#: Tail thresholds, one per residual family, both fixed before this data
#: existed and both declared as parameters rather than buried.
#:
#: murray  from EXP-084's published medians -- real |p-3| 1.10, mismatched 1.94.
#: angle   the residual is ``1 - exp(-2|dtheta|)`` and so is bounded below 1; a
#:         shared threshold of 2.0 would never fire and the aggregate would be a
#:         constant zero. This is the residual of an observed angle 45 degrees
#:         away from the Hess-Murray optimum, which is a stated geometric
#:         choice rather than a number read off this data.
RESID_THRESHOLD = {"murray": 2.0,
                   "angle": float(1.0 - np.exp(-2.0 * np.pi / 4.0))}

N_FOLD = 5
N_NULL = 500

PRIMARY_VARIANT = "unrooted"
PRIMARY_RESIDUAL = "murray"
PRIMARY_AGG = "frac_over_thresh"

BAR_AUC_K3 = 0.80
#: EXP-084's single-branch-point number, named in the bar as the thing a
#: whole-tree aggregate has to beat.
SINGLE_POINT_REFERENCE = 0.675
#: The same-cell control must not score ABOVE the foreign arm. A strict
#: inequality on two noisy estimates would fail on nothing; the gap this clause
#: exists to catch is EXP-083's 0.710 against 0.642, which is 0.068, so a
#: tolerance well below that keeps the clause meaningful.
CONTROL_TOL = 0.02

AGG_NAMES = ("mean", "median", "p90", "p95", "p99", "max",
             "n_over_thresh", "frac_over_thresh", "frac_invalid", "n_bif",
             "n_candidate")
#: Aggregates that are not a residual statistic and are reported on their own
#: rather than swept into the grid: ``n_bif`` and ``n_candidate`` are counts,
#: and ``frac_invalid`` is identically zero for the angle residual.
NOT_A_RESIDUAL_STAT = ("n_bif", "n_candidate")
VARIANTS = ("unrooted", "rooted")
RESIDUALS = ("murray", "angle")
ARMS = ("foreign", "samecell")

SPEC = Spec(
    id="EXP-090",
    title="Does tree-level evidence compound?",
    question="Does an aggregate of per-bifurcation conservation residuals "
             "separate a correctly assembled arbor from one carrying k wrong "
             "joins, and does the separation grow with k?",
    criterion=(
        f"held out by cell, on the primary aggregate declared before the run "
        f"({PRIMARY_AGG} of the {PRIMARY_RESIDUAL} residual, {PRIMARY_VARIANT} "
        f"role assignment), ALL THREE of: (1) pooled AUC at least "
        f"{BAR_AUC_K3:.2f} at k=3 wrong joins, against "
        f"{SINGLE_POINT_REFERENCE:.3f} for EXP-084's single branch point; "
        f"(2) the AUC is monotone in k, k=1 < k=3 < k=5, since monotonicity is "
        f"the actual evidence for compounding and a high number at k=5 alone is "
        f"not; (3) the same-cell displacement control scores no higher than the "
        f"foreign-cable arm at k=3 (within {CONTROL_TOL:.2f}) -- if it scores "
        f"higher the aggregate is reading placement rather than conservation, "
        f"as EXP-083's shape score turned out to be, and the experiment fails "
        f"whatever the headline number. All aggregates, both role assignments, "
        f"both residual families, both controls and a label-shuffle null are "
        f"reported, so the primary was not chosen after the fact"),
    requires_ran=["EXP-083"],
    inputs=[SKEL_DIR, SHAPE_LIB],
    params={"k_list": list(K_LIST), "reps_per_cell": REPS_PER_CELL,
            "site_cable_um": list(SITE_CABLE_UM), "cable_tol": CABLE_TOL,
            "z_frac": list(Z_FRAC), "max_graft_frac": MAX_GRAFT_FRAC,
            "p_bracket": list(P_BRACKET), "min_radius_nm": MIN_RADIUS_NM,
            "resid_threshold": dict(RESID_THRESHOLD), "n_fold": N_FOLD,
            "n_null": N_NULL, "seed": SEED,
            "primary": {"variant": PRIMARY_VARIANT, "residual": PRIMARY_RESIDUAL,
                        "aggregate": PRIMARY_AGG},
            "bar_auc_k3": BAR_AUC_K3, "control_tol": CONTROL_TOL,
            "single_point_reference": SINGLE_POINT_REFERENCE},
    flags={"synthetic_fallback": False,
           "labels_used_only_for_evaluation": True},
)


# ---------------------------------------------------------------------------
# EXP-083's construction, loaded from where it already lives
# ---------------------------------------------------------------------------
def load_shape_lib(root: Path):
    """Import ``scripts/exp083_shape_lib.py`` by path.

    That module resolves its own repository root from ``__file__``, so it
    carries no hardcoded absolute path and works loaded this way. (Its sibling
    ``scripts/test_cajal_conservation.py`` does hardcode one; nothing here
    imports it.) Loading by path rather than by ``sys.path`` insertion keeps
    the import working from any working directory and leaves the path alone.
    """
    path = Path(root) / SHAPE_LIB
    if not path.exists():
        raise FileNotFoundError(f"EXP-083's construction is missing: {path}")
    name = "exp083_shape_lib"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def assemble_multi(L, A, drop_masks, grafts):
    """``L.assemble`` generalised to k grafts, with the geometry unchanged.

    Each graft is ``(B, w, at)``: the subtree of ``B`` rooted at ``w``, rigidly
    translated so ``w`` lands where A's vertex ``at`` sat, hung from A's parent
    of ``at``. Every line that decides geometry is the line ``L.assemble`` uses;
    the only change is that the appended blocks accumulate. Returns the tree
    with an extra ``join_nodes`` array giving each graft's anchor in the new
    indexing. ``_verify_against_exp083`` checks that with one graft this
    reproduces ``L.assemble`` exactly.
    """
    keep = np.ones(A["n"], bool)
    for m in drop_masks:
        keep &= ~m
    keep[0] = True
    idx = np.flatnonzero(keep)
    newid = np.full(A["n"], -1, np.int64)
    newid[idx] = np.arange(len(idx))
    par = np.where(A["par"][idx] >= 0, newid[A["par"][idx]], -1)
    par[0] = -1
    if (par[1:] < 0).any():                       # a kept vertex lost its parent
        return None

    Vs = [A["V"][idx]]
    rads = [A["rad"][idx]]
    comps = [A["comp"][idx]]
    pars = [par]
    offset = len(idx)
    joins = []
    for B, w, at in grafts:
        gm = L.subtree_mask(B, w)
        gidx = np.flatnonzero(gm)
        gnew = np.full(B["n"], -1, np.int64)
        gnew[gidx] = np.arange(len(gidx)) + offset
        gpar = gnew[B["par"][gidx]]
        anchor = newid[A["par"][at]]
        if anchor < 0:
            return None
        gpar[0] = anchor                          # gidx[0] == w (breadth-first)
        shift = A["V"][at] - B["V"][w]
        Vs.append(B["V"][gidx] + shift)
        rads.append(B["rad"][gidx])
        comps.append(B["comp"][gidx])
        pars.append(gpar)
        joins.append(int(anchor))
        offset += len(gidx)

    t = dict(V=np.concatenate(Vs), rad=np.concatenate(rads),
             comp=np.concatenate(comps), par=np.concatenate(pars),
             n=offset, n_graft=offset - len(idx))
    L._annotate(t)
    t["join_nodes"] = np.asarray(joins, np.int64)
    return t


# ---------------------------------------------------------------------------
# the two parameter-free residuals
# ---------------------------------------------------------------------------
def murray_exponent(r0, r1, r2, bracket=P_BRACKET, n_iter=64):
    """Vectorised solve of ``r0^p = r1^p + r2^p`` on EXP-084's bracket.

    Normalised to ``(r1/r0)^p + (r2/r0)^p = 1`` for numerical stability; the
    root is the same. Admission is ``g(lo) > 0 and g(hi) < 0``, which is
    equivalent to EXP-084's ``f(lo) * f(hi) > 0 -> skip``: if either ratio is at
    least 1 then ``g`` is bounded below by 0 at both ends, and if both are below
    1 then ``g`` is strictly decreasing, so a sign change can only run
    positive-to-negative. Verified against ``scipy.optimize.brentq`` on the raw
    form by :func:`_verify_solvers`.

    Returns ``(p, admitted)``; ``p`` is NaN where not admitted.
    """
    lo, hi = bracket
    r0 = np.asarray(r0, np.float64)
    a = np.asarray(r1, np.float64) / r0
    b = np.asarray(r2, np.float64) / r0
    with np.errstate(over="ignore", invalid="ignore"):
        glo = a ** lo + b ** lo - 1.0
        ghi = a ** hi + b ** hi - 1.0
        ok = np.isfinite(glo) & np.isfinite(ghi) & (glo > 0) & (ghi < 0)
        plo = np.full(a.shape, float(lo))
        phi = np.full(a.shape, float(hi))
        for _ in range(n_iter):
            mid = 0.5 * (plo + phi)
            lower = (a ** mid + b ** mid - 1.0) > 0
            plo = np.where(lower, mid, plo)
            phi = np.where(lower, phi, mid)
    p = 0.5 * (plo + phi)
    return np.where(ok, p, np.nan), ok


def angle_prior(r0, r1, r2, angle_rad):
    """Vectorised ``SantiagoCajalPriors.compute_bifurcation_angle_prior``.

    Same clamp, same formula, same epsilon. Checked element-for-element against
    the reference implementation by :func:`_verify_solvers`, which imports it
    from ``neuronauts.morpho_grammar.cajal_conservation_priors``.
    """
    r0 = np.maximum(10.0, np.asarray(r0, np.float64))
    r1 = np.maximum(10.0, np.asarray(r1, np.float64))
    r2 = np.maximum(10.0, np.asarray(r2, np.float64))
    cos_val = (r0 ** 2 + r1 ** 2 - r2 ** 2) / (2.0 * r0 * r1 + 1e-7)
    theta_opt = np.arccos(np.clip(cos_val, -1.0, 1.0))
    return np.exp(-2.0 * np.abs(np.asarray(angle_rad, np.float64) - theta_opt))


def bifurcations(t, variant):
    """Candidate branch points of an assembly and their three radii and angle.

    ``rooted``   mother is the parent vertex, daughters are the two children.
    ``unrooted`` EXP-084's convention: the thickest of the three neighbours is
                 the mother, the other two are daughters, and the angle is
                 between those two.

    In both, radii are read at the NEIGHBOURING vertices, which is what
    ``scripts/test_cajal_conservation.py`` did. Vertices whose smallest daughter
    or mother falls under ``MIN_RADIUS_NM`` are dropped, as there.
    """
    par, V, rad, n = t["par"], t["V"], t["rad"], t["n"]
    nkid = np.bincount(par[1:], minlength=n)
    node = np.flatnonzero((nkid == 2) & (np.arange(n) > 0))
    if not len(node):
        empty = np.zeros(0, np.float64)
        return dict(node=np.zeros(0, np.int64), r0=empty, r1=empty, r2=empty,
                    angle=empty)

    # children of each 2-child vertex, in index order
    child = np.flatnonzero(par >= 0)
    child = child[child > 0]
    cpar = par[child]
    sel = np.isin(cpar, node)
    child, cpar = child[sel], cpar[sel]
    order = np.argsort(cpar, kind="stable")
    child, cpar = child[order], cpar[order]
    c1, c2 = child[0::2], child[1::2]
    assert (cpar[0::2] == cpar[1::2]).all() and (cpar[0::2] == node).all()

    rp, r1, r2 = rad[par[node]], rad[c1], rad[c2]
    d1 = V[c1] - V[node]
    d2 = V[c2] - V[node]
    dp = V[par[node]] - V[node]

    if variant == "rooted":
        R0, R1, R2 = rp, r1, r2
        A1, A2 = d1, d2
    elif variant == "unrooted":
        trip_r = np.stack([rp, r1, r2], axis=1)
        trip_d = np.stack([dp, d1, d2], axis=1)
        top = np.argmax(trip_r, axis=1)
        rows = np.arange(len(node))
        R0 = trip_r[rows, top]
        keep = np.stack([np.where(top == 0, 1, 0), np.where(top == 2, 1, 2)], axis=1)
        R1 = trip_r[rows, keep[:, 0]]
        R2 = trip_r[rows, keep[:, 1]]
        A1 = trip_d[rows, keep[:, 0]]
        A2 = trip_d[rows, keep[:, 1]]
    else:                                          # pragma: no cover - guard
        raise ValueError(f"unknown variant {variant!r}")

    good = (np.minimum(R1, R2) >= MIN_RADIUS_NM) & (R0 >= MIN_RADIUS_NM)
    n1 = np.linalg.norm(A1, axis=1)
    n2 = np.linalg.norm(A2, axis=1)
    cos = np.sum(A1 * A2, axis=1) / np.maximum(n1 * n2, 1e-9)
    ang = np.arccos(np.clip(cos, -1.0, 1.0))
    good &= (n1 > 0) & (n2 > 0)
    return dict(node=node[good], r0=R0[good], r1=R1[good], r2=R2[good],
                angle=ang[good])


def residual_arrays(bif):
    """``{'murray': (values, admitted), 'angle': (values, admitted)}``.

    Both are oriented so that larger is worse. The murray residual is
    ``|p - 3|`` and exists only where the exponent is admissible; the angle
    residual is ``1 - prior`` and exists at every candidate bifurcation.
    """
    p, ok = murray_exponent(bif["r0"], bif["r1"], bif["r2"])
    murray = np.abs(p - 3.0)
    ang = 1.0 - angle_prior(bif["r0"], bif["r1"], bif["r2"], bif["angle"])
    return {"murray": (murray, ok),
            "angle": (ang, np.ones(len(ang), bool))}


def aggregate(values, admitted, n_candidate, threshold):
    """Every aggregate, reported together. ``values`` may hold NaN where not admitted.

    ``n_candidate`` counts the bifurcations offered to this residual family,
    ``n_bif`` the ones it could score, and ``frac_invalid`` the gap -- which is
    only meaningful for the murray residual, since the angle prior scores every
    candidate. ``n_candidate`` is carried through unchanged so it can serve as
    the size-only control: it is the assembly's bifurcation count, before any
    admissibility filter that the corruption itself could move.
    """
    v = np.asarray(values, np.float64)[admitted]
    n_bif = int(len(v))
    frac_invalid = float(1.0 - n_bif / n_candidate) if n_candidate else np.nan
    if n_bif == 0:
        out = {k: np.nan for k in AGG_NAMES}
        out["n_bif"] = 0.0
        out["n_candidate"] = float(n_candidate)
        out["frac_invalid"] = frac_invalid
        return out
    over = v > threshold
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)), "max": float(v.max()),
            "n_over_thresh": float(over.sum()),
            "frac_over_thresh": float(over.mean()),
            "frac_invalid": frac_invalid, "n_bif": float(n_bif),
            "n_candidate": float(n_candidate)}


def score_assembly(t):
    """Whole-tree aggregates, plus the residual at each known join site.

    Returns ``(whole, at_join)``. ``whole[variant][residual][agg]`` is a float;
    ``at_join[variant][residual]`` is the residual at each of the k join
    bifurcations, NaN where that bifurcation is not admissible -- and the NaN
    rate is itself reported, because "the join point has no admissible Murray
    exponent" is evidence, not a gap to drop quietly.
    """
    whole, at_join = {}, {}
    joins = t.get("join_nodes", np.zeros(0, np.int64))
    for variant in VARIANTS:
        bif = bifurcations(t, variant)
        res = residual_arrays(bif)
        whole[variant] = {r: aggregate(v, ok, len(bif["node"]),
                                       RESID_THRESHOLD[r])
                          for r, (v, ok) in res.items()}
        where = np.full(t["n"], -1, np.int64)
        where[bif["node"]] = np.arange(len(bif["node"]))
        rows = where[joins]
        at_join[variant] = {}
        for r, (v, ok) in res.items():
            got = np.full(len(joins), np.nan)
            hit = rows >= 0
            if hit.any():
                idx = rows[hit]
                got[hit] = np.where(ok[idx], v[idx], np.nan)
            at_join[variant][r] = got
    return whole, at_join


# ---------------------------------------------------------------------------
# evaluation helpers -- same definitions as scripts/exp083_score2.py
# ---------------------------------------------------------------------------
def cell_folds(cells, n_fold=N_FOLD, seed=SEED):
    uc = np.unique(cells)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uc))
    a = {int(uc[perm[i]]): i % n_fold for i in range(len(uc))}
    return np.array([a[int(c)] for c in cells])


def boot_ci(vals, cells, n=4000, seed=SEED):
    """Cell-level bootstrap of a mean, as EXP-083 did."""
    vals = np.asarray(vals, np.float64)
    cells = np.asarray(cells)
    rng = np.random.default_rng(seed)
    uc = np.unique(cells)
    by = {int(c): vals[cells == c] for c in uc}
    draws = [np.concatenate([by[int(c)] for c in rng.choice(uc, len(uc), True)]).mean()
             for _ in range(n)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _auc(y, s):
    m = np.isfinite(s)
    if m.sum() < 4 or len(np.unique(y[m])) < 2:
        return float("nan")
    return float(roc_auc(y[m] == 1, s[m]))


def evaluate(good, bad, cells, seed=SEED):
    """Absolute (pooled) AUC, paired win rate, folds, bootstrap and null.

    ``good`` and ``bad`` are one score per pair. The pooled AUC mixes assemblies
    from different cells, so between-cell variation is NOT removed -- that is
    the hard test EXP-083 failed at 0.505, and the one the bar is set on. The
    paired win rate holds the base fixed and asks only which of the two
    assemblies at one site scores worse; it is reported beside the pooled number,
    never instead of it.
    """
    good = np.asarray(good, np.float64)
    bad = np.asarray(bad, np.float64)
    cells = np.asarray(cells)
    both = np.isfinite(good) & np.isfinite(bad)
    s = np.concatenate([good, bad])
    y = np.concatenate([np.zeros(len(good)), np.ones(len(bad))])
    c = np.concatenate([cells, cells])
    out = {"n_pairs": int(len(good)), "n_cells": int(len(np.unique(cells))),
           "n_pairs_scored": int(both.sum()),
           "abs_auc": _auc(y, s)}

    win = np.where(both, (bad > good).astype(float) + 0.5 * (bad == good), np.nan)
    if both.any():
        out["paired"] = float(np.nanmean(win))
        lo, hi = boot_ci(win[both], cells[both], seed=seed)
        out["paired_ci"] = [lo, hi]
    else:
        out["paired"] = float("nan")
        out["paired_ci"] = [float("nan"), float("nan")]

    fold = cell_folds(cells, seed=seed)
    per_fold = []
    for k in range(N_FOLD):
        m = fold == k
        if m.sum() < 4:
            per_fold.append(float("nan"))
            continue
        per_fold.append(_auc(np.concatenate([np.zeros(m.sum()), np.ones(m.sum())]),
                             np.concatenate([good[m], bad[m]])))
    out["fold_auc"] = per_fold
    ok = np.isfinite(per_fold)
    out["fold_auc_mean"] = float(np.mean(np.asarray(per_fold)[ok])) if ok.any() else float("nan")
    out["fold_auc_spread"] = ([float(np.min(np.asarray(per_fold)[ok])),
                               float(np.max(np.asarray(per_fold)[ok]))]
                              if ok.any() else [float("nan")] * 2)

    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(N_NULL):
        nulls.append(_auc(rng.permutation(y), s))
    nulls = np.asarray(nulls, np.float64)
    nulls = nulls[np.isfinite(nulls)]
    out["null_abs_auc"] = {
        "mean": float(nulls.mean()) if len(nulls) else float("nan"),
        "p2.5": float(np.percentile(nulls, 2.5)) if len(nulls) else float("nan"),
        "p97.5": float(np.percentile(nulls, 97.5)) if len(nulls) else float("nan"),
        "n": int(len(nulls))}
    if both.any():
        flips = rng.random((N_NULL, int(both.sum()))) < 0.5
        w = win[both][None, :]
        pn = np.where(flips, 1.0 - w, w).mean(axis=1)
        out["null_paired"] = {"mean": float(pn.mean()),
                              "p2.5": float(np.percentile(pn, 2.5)),
                              "p97.5": float(np.percentile(pn, 97.5))}
    else:
        out["null_paired"] = {"mean": float("nan"), "p2.5": float("nan"),
                              "p97.5": float("nan")}
    return out


# ---------------------------------------------------------------------------
# self-checks: run once, before any science, and raise rather than warn
# ---------------------------------------------------------------------------
def _verify_solvers(seed=0, n=4000):
    """The vectorised solvers must equal the references they claim to reproduce."""
    from scipy.optimize import brentq

    from neuronauts.morpho_grammar.cajal_conservation_priors import (
        SantiagoCajalPriors as P)

    rng = np.random.default_rng(seed)
    r0 = rng.uniform(30.0, 1500.0, n)
    r1 = r0 * rng.uniform(0.15, 1.35, n)
    r2 = r0 * rng.uniform(0.15, 1.35, n)
    ang = rng.uniform(0.0, np.pi, n)

    p, ok = murray_exponent(r0, r1, r2)
    lo, hi = P_BRACKET
    ref = np.full(n, np.nan)
    refok = np.zeros(n, bool)
    for i in range(n):
        f = lambda q, i=i: r1[i] ** q + r2[i] ** q - r0[i] ** q   # noqa: E731
        if f(lo) * f(hi) > 0:
            continue
        ref[i] = brentq(f, lo, hi)
        refok[i] = True
    if not (ok == refok).all():
        raise AssertionError("murray_exponent admits a different set of "
                             "bifurcations than EXP-084's brentq bracket test")
    err = float(np.max(np.abs(p[ok] - ref[ok]))) if ok.any() else 0.0
    if err > 1e-8:
        raise AssertionError(f"murray_exponent disagrees with brentq by {err:.3g}")

    mine = angle_prior(r0, r1, r2, ang)
    theirs = np.array([P.compute_bifurcation_angle_prior(r0[i], r1[i], r2[i], ang[i])
                       for i in range(n)])
    aerr = float(np.max(np.abs(mine - theirs)))
    if aerr > 1e-12:
        raise AssertionError(f"angle_prior disagrees with the Cajal prior by {aerr:.3g}")
    return {"murray_vs_brentq_max_abs_err": err,
            "murray_admission_agrees": True,
            "angle_vs_cajal_prior_max_abs_err": aerr,
            "n_checked": int(n)}


def _verify_against_exp083(L, A, rng):
    """One graft through :func:`assemble_multi` must equal ``L.assemble`` exactly."""
    sc = A["sub_cable"] / L.UM
    total = float(A["elen"].sum() / L.UM)
    cand = np.flatnonzero((sc > 0.01 * total) & (sc < 0.2 * total))
    cand = cand[cand != 0]
    cand = cand[A["par"][cand] > 0]
    if len(cand) < 2:
        return None
    z, u = (int(x) for x in rng.choice(cand, 2, replace=False))
    mz, mu = L.subtree_mask(A, z), L.subtree_mask(A, u)
    if mu[z] or mz[u] or mz[A["par"][u]]:
        return None
    ref = L.assemble(A, [mz, mu], graft=(A, u, u))
    got = assemble_multi(L, A, [mz, mu], [(A, u, u)])
    if ref is None or got is None:
        return None
    for key in ("V", "rad", "comp", "par"):
        if not np.array_equal(np.asarray(ref[key]), np.asarray(got[key])):
            raise AssertionError(
                f"assemble_multi diverges from exp083_shape_lib.assemble on {key!r}; "
                "the construction would no longer be comparable with EXP-083")
    if int(ref["n"]) != int(got["n"]) or int(ref["n_graft"]) != int(got["n_graft"]):
        raise AssertionError("assemble_multi diverges from L.assemble on vertex counts")
    return {"n": int(got["n"]), "n_graft": int(got["n_graft"])}


# ---------------------------------------------------------------------------
# site and donor selection
# ---------------------------------------------------------------------------
def _ancestors(A, u):
    m = np.zeros(A["n"], bool)
    cur = int(u)
    while cur >= 0:
        m[cur] = True
        cur = int(A["par"][cur])
    return m


def _sibling(A, u, nkid_children):
    kids = nkid_children[int(A["par"][u])]
    other = [int(c) for c in kids if int(c) != int(u)]
    return other[0] if len(other) == 1 else -1


def choose_sites(L, A, mz, sc, total, nkid, children, rng):
    """Up to ``N_SITES`` graft sites, pairwise disjoint, each at a real bifurcation.

    A site ``u`` qualifies when: its parent exists, is not the soma's own
    removal, has EXACTLY two children (so replacing one leaves a bifurcation to
    read), the sibling survives every removal, ``u`` is outside ``z``'s subtree,
    and its cable is in the declared band and under the size cap. Sites are
    accepted greedily and are pairwise non-nested; the cumulative grafted cable
    is capped so the corrupted assembly is never mostly foreign.
    """
    lo, hi = SITE_CABLE_UM
    cand = np.flatnonzero((sc >= lo) & (sc < hi) & ~mz
                          & (sc < MAX_GRAFT_FRAC * total))
    cand = cand[cand != 0]
    cand = cand[A["par"][cand] >= 0]
    cand = cand[nkid[A["par"][cand]] == 2]
    cand = cand[~mz[A["par"][cand]]]
    if not len(cand):
        return []
    rng.shuffle(cand)

    taken, masks, sibs, cable = [], [], [], 0.0
    blocked = mz.copy()
    for u in cand.tolist():
        if len(taken) >= N_SITES:
            break
        sib = _sibling(A, u, children)
        if sib < 0 or blocked[sib] or blocked[u] or blocked[A["par"][u]]:
            continue
        mu = L.subtree_mask(A, u)
        if (mu & blocked).any():
            continue
        if any(mu[s] for s in sibs) or any(mu[A["par"][t]] for t in taken):
            continue
        if cable + float(sc[u]) > MAX_GRAFT_FRAC * total:
            continue
        taken.append(int(u))
        masks.append(mu)
        sibs.append(int(sib))
        blocked = blocked | mu
        cable += float(sc[u])
    if len(taken) < N_SITES:
        return []
    return list(zip(taken, masks))


def pick_foreign(trees, donor_cable, cells, a_id, target, rng):
    for _ in range(DONOR_TRIES):
        b_id = int(rng.choice([c for c in cells if c != a_id]))
        dc = donor_cable[b_id]
        ok = np.flatnonzero(np.abs(dc - target) <= CABLE_TOL * target)
        if len(ok):
            return b_id, int(rng.choice(ok))
    return None


def pick_samecell(A, donor_cable_a, forbidden, target, rng):
    """EXP-083's control donor: A's own cable, never the true piece, a piece of
    it, a removed piece, or an ancestor whose subtree contains the base."""
    ok = np.flatnonzero(np.abs(donor_cable_a - target) <= CABLE_TOL * target)
    ok = ok[~forbidden[ok]]
    if not len(ok):
        return None
    return int(rng.choice(ok))


# ---------------------------------------------------------------------------
def run(ctx: Context) -> Outcome:
    root = Path(ctx.root)
    L = load_shape_lib(root)

    checks = {"solvers": _verify_solvers()}
    print(f"  solvers verified against brentq and the Cajal prior: "
          f"{checks['solvers']}", flush=True)

    files = sorted(glob.glob(str(root / SKEL_DIR / "*_skv4.npz")))
    if not files:
        raise FileNotFoundError(f"no skeletons under {root / SKEL_DIR}")

    cells, trees, skipped_nonfinite, dropped = [], {}, 0, 0
    for f in files:
        d = np.load(f)
        if not np.isfinite(d["vertices"]).all():
            skipped_nonfinite += 1
            continue
        c = int(Path(f).name.split("_")[0])
        t = L.load_tree(f)
        dropped += int(t.get("n_dropped", 0) > 0)
        cells.append(c)
        trees[c] = t
    print(f"  loaded {len(cells)} arbors ({skipped_nonfinite} skipped for "
          f"non-finite vertices, {dropped} with vertices off the soma's "
          f"component)", flush=True)

    rng = np.random.default_rng(SEED)
    checks["assemble_multi_vs_exp083"] = None
    for c in cells[: min(8, len(cells))]:
        got = _verify_against_exp083(L, trees[c], rng)
        if got is not None:
            checks["assemble_multi_vs_exp083"] = got
            break
    if checks["assemble_multi_vs_exp083"] is None:
        raise AssertionError("could not verify assemble_multi against "
                             "exp083_shape_lib.assemble on any cell")
    print(f"  assemble_multi reproduces L.assemble exactly on one graft: "
          f"{checks['assemble_multi_vs_exp083']}", flush=True)

    donor_cable = {c: trees[c]["sub_cable"] / L.UM for c in cells}
    for c in cells:                                # never donate the whole arbor
        donor_cable[c][0] = -1.0

    # scores[(k, arm, variant, residual, agg)] -> list per pair
    rows = []
    n_rep_attempt = n_rep_used = 0
    for ci, a_id in enumerate(cells):
        A = trees[a_id]
        sc = A["sub_cable"] / L.UM
        total = float(A["elen"].sum() / L.UM)
        nkid = np.bincount(A["par"][1:], minlength=A["n"])
        children = {}
        for v in range(1, A["n"]):
            children.setdefault(int(A["par"][v]), []).append(v)

        for rep in range(REPS_PER_CELL):
            n_rep_attempt += 1
            zc = np.flatnonzero((sc > Z_FRAC[0] * total) & (sc < Z_FRAC[1] * total))
            if not len(zc):
                break
            z = int(rng.choice(zc))
            mz = L.subtree_mask(A, z)
            sites = choose_sites(L, A, mz, sc, total, nkid, children, rng)
            if not sites:
                continue

            forbidden = mz.copy()
            for u, mu in sites:
                forbidden = forbidden | mu | _ancestors(A, u)

            donors_f, donors_s, ok = [], [], True
            for u, _mu in sites:
                target = float(sc[u])
                f = pick_foreign(trees, donor_cable, cells, a_id, target, rng)
                s = pick_samecell(A, donor_cable[a_id], forbidden, target, rng)
                if f is None or s is None:
                    ok = False
                    break
                donors_f.append(f)
                donors_s.append(s)
            if not ok:
                continue
            n_rep_used += 1

            for k in K_LIST:
                sub = sites[:k]
                drops = [mz] + [m for _u, m in sub]
                us = [u for u, _m in sub]
                arms = {
                    "correct": [(A, u, u) for u in us],
                    "foreign": [(trees[donors_f[i][0]], donors_f[i][1], us[i])
                                for i in range(k)],
                    "samecell": [(A, donors_s[i], us[i]) for i in range(k)],
                }
                built = {}
                for name, grafts in arms.items():
                    t = assemble_multi(L, A, drops, grafts)
                    if t is None:
                        built = {}
                        break
                    built[name] = t
                if not built:
                    continue
                scored = {n: score_assembly(t) for n, t in built.items()}
                rows.append({"cell": a_id, "rep": rep, "k": k,
                             "graft_cable_um": float(sum(sc[u] for u in us)),
                             "arbor_cable_um": total,
                             "scored": scored})
        if (ci + 1) % 20 == 0:
            print(f"  {ci+1}/{len(cells)} cells, {len(rows)} assemblies scored",
                  flush=True)

    if not rows:
        return Outcome(
            passed=False,
            observed={"n_assemblies": 0},
            note="no assembly could be built: no cell supplied "
                 f"{N_SITES} disjoint graft sites at a two-child bifurcation "
                 f"with cable in {SITE_CABLE_UM} um and a size-matched donor. "
                 "That is a construction failure, not a result about "
                 "compounding.")

    # ---------------- assemble the score tables --------------------------
    def collect(k, arm, variant, resid, agg):
        g, b, c = [], [], []
        for r in rows:
            if r["k"] != k:
                continue
            g.append(r["scored"]["correct"][0][variant][resid][agg])
            b.append(r["scored"][arm][0][variant][resid][agg])
            c.append(r["cell"])
        return np.asarray(g), np.asarray(b), np.asarray(c)

    def collect_join(k, arm, variant, resid, how):
        g, b, c, miss = [], [], [], []
        for r in rows:
            if r["k"] != k:
                continue
            gv = r["scored"]["correct"][1][variant][resid]
            bv = r["scored"][arm][1][variant][resid]
            miss.append(float(np.isnan(bv).mean()))
            f = np.nanmean if how == "mean" else np.nanmax
            with np.errstate(invalid="ignore"):
                g.append(f(gv) if np.isfinite(gv).any() else np.nan)
                b.append(f(bv) if np.isfinite(bv).any() else np.nan)
            c.append(r["cell"])
        return (np.asarray(g), np.asarray(b), np.asarray(c),
                float(np.mean(miss)) if miss else float("nan"))

    whole_tree, known_join, size_only = {}, {}, {}
    for k in K_LIST:
        for arm in ARMS:
            for variant in VARIANTS:
                for resid in RESIDUALS:
                    for agg in AGG_NAMES:
                        if agg in NOT_A_RESIDUAL_STAT:
                            continue
                        if agg == "frac_invalid" and resid == "angle":
                            continue        # identically zero, not a statistic
                        g, b, c = collect(k, arm, variant, resid, agg)
                        if not len(g):
                            continue
                        key = f"k{k}/{arm}/{variant}/{resid}/{agg}"
                        whole_tree[key] = evaluate(g, b, c)
                    for how in ("mean", "max"):
                        g, b, c, miss = collect_join(k, arm, variant, resid, how)
                        if not len(g):
                            continue
                        e = evaluate(g, b, c)
                        e["join_residual_missing_rate"] = miss
                        known_join[f"k{k}/{arm}/{variant}/{resid}/{how}"] = e
            # size-only control: the raw bifurcation COUNT, taken before the
            # murray admissibility filter, which the corruption itself can move
            g, b, c = collect(k, arm, PRIMARY_VARIANT, PRIMARY_RESIDUAL,
                              "n_candidate")
            if len(g):
                size_only[f"k{k}/{arm}"] = evaluate(g, b, c)
                g, b, c = collect(k, arm, PRIMARY_VARIANT, PRIMARY_RESIDUAL,
                                  "n_bif")
                size_only[f"k{k}/{arm}/admitted_count"] = evaluate(g, b, c)

    prim = f"/{PRIMARY_VARIANT}/{PRIMARY_RESIDUAL}/{PRIMARY_AGG}"
    auc_by_k = {f"k{k}": whole_tree.get(f"k{k}/foreign{prim}", {}).get("abs_auc", float("nan"))
                for k in K_LIST}
    ctl_by_k = {f"k{k}": whole_tree.get(f"k{k}/samecell{prim}", {}).get("abs_auc", float("nan"))
                for k in K_LIST}
    single = known_join.get(f"k1/foreign/{PRIMARY_VARIANT}/{PRIMARY_RESIDUAL}/mean", {})
    single_auc = single.get("abs_auc", float("nan"))

    auc_k3 = auc_by_k.get("k3", float("nan"))
    ctl_k3 = ctl_by_k.get("k3", float("nan"))
    seq = [auc_by_k[f"k{k}"] for k in K_LIST]
    monotone = bool(np.all(np.isfinite(seq)) and all(seq[i] < seq[i + 1]
                                                     for i in range(len(seq) - 1)))
    control_ok = bool(np.isfinite(ctl_k3) and np.isfinite(auc_k3)
                      and ctl_k3 <= auc_k3 + CONTROL_TOL)
    bar_ok = bool(np.isfinite(auc_k3) and auc_k3 >= BAR_AUC_K3)
    passed = bar_ok and monotone and control_ok
    fails = [n for n, v in (("auc_k3", bar_ok), ("monotone_in_k", monotone),
                            ("same_cell_control", control_ok)) if not v]

    per_k_pairs = {f"k{k}": int(sum(1 for r in rows if r["k"] == k)) for k in K_LIST}
    fold = cell_folds(np.asarray([r["cell"] for r in rows]))
    split = {"n_folds": N_FOLD, "seed": SEED, "by_cell": True,
             "cells_per_fold": [int((fold == i).sum()) for i in range(N_FOLD)],
             "nothing_is_fitted": True,
             "note": "no parameters are estimated -- the priors are "
                     "parameter-free and the residual threshold was taken from "
                     "EXP-084's published medians before this data existed -- so "
                     "the folds report between-cell variability, they do not "
                     "guard against overfitting"}

    note_bits = [
        f"primary aggregate ({PRIMARY_AGG} of the {PRIMARY_RESIDUAL} residual, "
        f"{PRIMARY_VARIANT}): pooled held-out AUC "
        + ", ".join(f"k={k} {auc_by_k[f'k{k}']:.3f}" for k in K_LIST),
        f"single known join site inside this construction {single_auc:.3f} "
        f"(EXP-084's single branch point, under a stronger corruption, was "
        f"{SINGLE_POINT_REFERENCE:.3f})",
        f"same-cell displacement control at k=3 {ctl_k3:.3f} against the "
        f"foreign arm's {auc_k3:.3f}",
    ]
    if passed:
        note_bits.append("clears every clause: the aggregate rises with k and "
                         "the displacement control does not outscore foreign "
                         "cable, so the compounding claim survives its first test")
    else:
        note_bits.append("fails on " + ", ".join(fails) + ". EXP-084's "
                         "compounding claim is not supported by this test as "
                         "stated; what the numbers below license is written in "
                         "the evaluation, not inferred from the headline")

    return Outcome(
        passed=passed,
        observed={
            "auc_k3_primary": float(auc_k3),
            "monotone_in_k": monotone,
            "same_cell_control_auc_k3": float(ctl_k3),
            "auc_by_k_primary": auc_by_k,
            "same_cell_control_by_k": ctl_by_k,
            "single_join_site_auc_k1": float(single_auc),
            "single_point_reference_exp084": SINGLE_POINT_REFERENCE,
            "paired_by_k_primary": {
                f"k{k}": whole_tree.get(f"k{k}/foreign{prim}", {}).get("paired", float("nan"))
                for k in K_LIST},
            "null_abs_auc_k3_primary": whole_tree.get(
                f"k3/foreign{prim}", {}).get("null_abs_auc", {}),
            "null_paired_k3_primary": whole_tree.get(
                f"k3/foreign{prim}", {}).get("null_paired", {}),
            "size_only_control_auc": {k: v["abs_auc"] for k, v in size_only.items()},
            "failed_clauses": fails,
            "held_out_split": split,
            "graft_params": {
                "k_list": list(K_LIST), "sites_drawn_per_rep": N_SITES,
                "reps_per_cell": REPS_PER_CELL,
                "site_cable_um": list(SITE_CABLE_UM),
                "donor_cable_tolerance": CABLE_TOL,
                "second_removal_frac": list(Z_FRAC),
                "max_grafted_fraction_of_arbor": MAX_GRAFT_FRAC,
                "sites_are_two_child_bifurcation_parents": True,
                "k_sites_are_nested_prefixes": True,
                "murray_bracket": list(P_BRACKET),
                "min_radius_nm": MIN_RADIUS_NM,
                "residual_threshold": dict(RESID_THRESHOLD),
                "median_grafted_cable_um_by_k": {
                    f"k{k}": float(np.median([r["graft_cable_um"] for r in rows
                                              if r["k"] == k]) if per_k_pairs[f"k{k}"] else np.nan)
                    for k in K_LIST},
                "median_arbor_cable_um": float(np.median(
                    [r["arbor_cable_um"] for r in rows])),
            },
            "self_checks": checks,
        },
        population={
            "skeleton_files": len(files),
            "cells_loaded": len(cells),
            "cells_skipped_nonfinite": skipped_nonfinite,
            "cells_with_off_component_vertices": dropped,
            "cells_contributing": int(len(set(r["cell"] for r in rows))),
            "replicates_attempted": n_rep_attempt,
            "replicates_used": n_rep_used,
            "pairs_by_k": per_k_pairs,
            "assemblies_scored": 3 * len(rows),
        },
        tables={"whole_tree": whole_tree, "known_join_sites": known_join,
                "size_only_control": size_only},
        note="; ".join(note_bits),
    )


if __name__ == "__main__":
    raise SystemExit(main(SPEC, run))
