"""EXP-088 -- does the conservation prior separate a real join from a real wrong one?

EXP-084 found the first tree-level signal in this program: real bifurcations in
this tissue obey Murray's law (exponent median 3.18 against an ideal 3.0 over
3,781 branch points) and a caliber mismatch breaks it, separating at area under
the curve 0.675 with no parameters and no training. Its own closing line says
what it did not do: the mismatch was a *proxy* for a wrong join -- one branch
point's mother radius taken with another branch point's daughters -- and the
radii came from **proofread** skeletons, which a grower does not have.

Two things are therefore unestablished, and the prior is not usable until both
hold. This experiment asks them together, on one site set, so that a difference
between the answers is attributable:

1. **Real joins, not a proxy.** The 28,012 located post-v117 human merges of
   the EXP-082 corpus are real join sites. Does the prior prefer the tissue the
   proofreader actually attached over a different, nearby v117 object offered
   at the same cut end?
2. **Deployable caliber.** Does the signal survive when every radius is measured
   on v117 fragment geometry instead of read out of the finished
   reconstruction? EXP-082 names the same recomputation as the one verification
   its where-to-edit prior (area under the curve 0.779, radius alone 0.750)
   still needs, so the measurement lives in
   ``neuronauts.harness.v117_caliber`` and both experiments call it.

## One site set, three scorings

Every site is a post-v117 merge whose snapped skeleton vertex has degree three
and whose three arms carry, at v117, exactly the pattern ``{host, host,
added}`` -- so the bifurcation the law is applied to is one this merge actually
created, checked against the substrate rather than assumed. The three arms are
sampled at the same skeleton vertices in every arm of the experiment; only the
source of the radii and the identity of the wrong piece change.

  A. **proofread radii, permuted distractor** -- the EXP-084 construction moved
     onto join sites. This is the control that makes the rest readable. If it
     does not reproduce 0.675 within +-0.05, then the construction changed the
     number and nothing below may be attributed to resolution.
  B. **v117 radii, permuted distractor** -- same wrong piece, different ruler.
     A -> B isolates the radius source.
  C. **v117 radii, real distractor** -- the added arm replaced by a different
     v117 object that reaches the same cut end, excluding this cell's own
     tissue, nearest first. B -> C isolates how much easier a permuted caliber
     is than one a real decoy actually has. **The bar is on C.**

The distractor in C is chosen nearest-first and is sampled at the same stand-off
distance from the site as the true arm it replaces, so the comparison is not
secretly about distance -- the lesson of EXP-083, which had to equalise gap,
direction, parent and added cable inside each pair before its whole-cell score
meant anything.

## What this experiment does not claim

It does not measure whether the prior finds join sites (EXP-082 owns that), nor
whether whole-cell shape gates a join (EXP-083 measured that and the answer was
no: 0.505 absolute, and 0.64 within-site). It scores one branch point. EXP-084's
own reading applies unchanged -- a single branch point is weak evidence whose
value is that it compounds over a tree -- and this experiment does not test the
compounding.

    python -m neuronauts.experiments.exp088_conservation_joins
"""

from __future__ import annotations

import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import brentq
from scipy.spatial import cKDTree

from neuronauts.experiments._runner import Context, Outcome, Spec, main
from neuronauts.harness.baselines import LogisticRegression
from neuronauts.harness.v117_caliber import (
    DEFAULT_HALO_NM, DEFAULT_LOCAL_NM, VoxelCaliber, load_l2_caliber,
    read_v117_box, vertex_radii_from_l2,
)
from neuronauts.metrics.ranking import roc_auc

# ``neuronauts.morpho_grammar`` is an import shim onto ``attic/morpho_grammar``
# and warns on import. The warning is about the attic's *untrained engines*;
# these two priors are closed-form biophysics with no checkpoint and no random
# state, which is why EXP-084 used them and why the warning is suppressed here
# rather than worked around.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from neuronauts.morpho_grammar.cajal_conservation_priors import (
        SantiagoCajalPriors as CajalPriors,
    )

EDIT_JOIN = "data/external/edit_join_v082.npz"
SKEL_DIR = "data/external/cell_skeletons"
#: Optional cross-check only; never declared as an input, so its absence cannot
#: block the run. Where it covers a site, the level-2 cache's own distance
#: transform is compared against the voxel measurement.
L2_ATTRS = "data/substrate/geom/l2_attributes.npz"

#: v117 is the substrate every downstream method starts from; merges logged
#: after it are the ones a grower would still have to make.
V117_TIMESTAMP = 1623399000
V117_MS = V117_TIMESTAMP * 1000
MIP = 2

# -- site construction -------------------------------------------------------
#: An edit point further than this from the final skeleton is not located well
#: enough to name a branch point. Matches the filter EXP-082's own model used.
SNAP_MAX_NM = 2000.0
#: Arc length out from the join site at which each arm's caliber is read. Long
#: enough to leave the branch point's own swelling, short enough that a read box
#: covering all three arms stays small.
ARM_NM = 1500.0
MIN_ARM_NM = 700.0
#: Padding around every point the site could be measured at. Must exceed the
#: caliber window (``local + halo``), or a radius near the box face would be
#: measured against unread tissue and come back quietly too small -- which
#: would look exactly like a real caliber mismatch. The first version of this
#: padded only the three true arms and a distractor sampled off to the side
#: came back ``truncated`` every time.
BOX_PAD_NM = DEFAULT_LOCAL_NM + DEFAULT_HALO_NM + 200.0
#: The box is also forced to cover a sphere of this radius about the join site,
#: because a distractor may lie in a direction no true arm points in.
SITE_WINDOW_NM = 2000.0
#: A read this size is already ~7 um on a side, which is ~8.4M voxels at the
#: 32/32/40 nm the graphene segmentation serves at mip 2. Sites needing more are
#: skipped and counted; if too many are -- reading at a finer mip would do it --
#: the run says so instead of reporting an area under the curve computed on
#: whatever happened to fit.
MAX_BOX_VOXELS = 30_000_000
MAX_SKIP_FRACTION = 0.20
#: How far from a skeleton point an object may sit and still be read as this
#: cell's own tissue rather than a distractor.
SELF_TISSUE_NM = 400.0
#: Spacing at which the cell's skeleton is densified for that test; vertices are
#: ~2 um apart, so testing against vertices alone would call mid-edge cable
#: foreign.
SKELETON_SAMPLE_NM = 200.0
#: How far a skeleton vertex may sit from v117 tissue and still be read as
#: lying on it. The skeleton is a v1822 object; at v117 the same cable is there
#: but its centreline need not fall on the same voxel.
ARM_OBJECT_SNAP_NM = 200.0

# -- distractor construction -------------------------------------------------
DISTRACTOR_MAX_GAP_NM = 2000.0
N_DISTRACTORS = 3
#: Half-angle acceptance for sampling a distractor at range: cos 60 degrees.
DISTRACTOR_CONE_COS = 0.5
DISTRACTOR_RANGE_TOL_NM = 500.0

# -- scoring -----------------------------------------------------------------
#: The bracket EXP-084 solved the Murray exponent in, kept identical so the
#: control number is comparable rather than merely similar.
P_LO, P_HI = 0.5, 8.0
MIN_RADIUS_NM = 20.0
N_FOLDS = 5
SEED = 0

N_SITES = 1500
MIN_SITES = 300

BAR_AUC_V117 = 0.65
EXP084_AUC = 0.675
CONTROL_TOLERANCE = 0.05

SPEC = Spec(
    id="EXP-088",
    title="Conservation priors on real joins, v117 radii",
    question="Does the Murray/Cajal conservation prior separate a real human "
             "join from a plausible wrong one at the same site, and does it "
             "survive caliber measured on v117 fragments instead of a "
             "proofread skeleton?",
    criterion=f"one site set, three scorings, held out by cell in "
              f"{N_FOLDS} folds. Sites are post-v117 human merges from the "
              f"EXP-082 corpus whose snapped skeleton vertex has degree 3 and "
              f"whose three arms carry the v117 pattern host/host/added. PASS "
              f"requires all of: (1) at least {MIN_SITES} such sites survive "
              f"the funnel -- below that the run reports the funnel and no "
              f"area under the curve; (2) arm C, v117-measured radii against "
              f"real nearby wrong objects offered at the same cut end, reaches "
              f"held-out area under the curve at least {BAR_AUC_V117:.2f} by "
              f"the parameter-free Murray score |p-3|, pooled over up to "
              f"{N_DISTRACTORS} distractors per site; (3) arm A, the same "
              f"construction scored on PROOFREAD radii with EXP-084's permuted "
              f"distractor, reproduces EXP-084's {EXP084_AUC} within "
              f"+-{CONTROL_TOLERANCE}. Clause 3 is not decoration: if it "
              f"fails, the construction moved the number and no drop may be "
              f"attributed to the v117 radius. Also gated: at most "
              f"{MAX_SKIP_FRACTION:.0%} of otherwise-valid sites lost to the "
              f"read-box size guard. Proofread identity is used only to build "
              f"the site set, to say which offered object is correct, and to "
              f"keep the cell's own tissue out of the distractor pool; no "
              f"score reads a label",
    requires_ran=[],
    inputs=[EDIT_JOIN, SKEL_DIR],
    params={"v117_timestamp": V117_TIMESTAMP, "mip": MIP,
            "snap_max_nm": SNAP_MAX_NM, "arm_nm": ARM_NM,
            "min_arm_nm": MIN_ARM_NM, "box_pad_nm": BOX_PAD_NM,
            "site_window_nm": SITE_WINDOW_NM,
            "max_box_voxels": MAX_BOX_VOXELS,
            "self_tissue_nm": SELF_TISSUE_NM,
            "skeleton_sample_nm": SKELETON_SAMPLE_NM,
            "distractor_max_gap_nm": DISTRACTOR_MAX_GAP_NM,
            "n_distractors": N_DISTRACTORS,
            "distractor_cone_cos": DISTRACTOR_CONE_COS,
            "distractor_range_tol_nm": DISTRACTOR_RANGE_TOL_NM,
            "caliber_local_nm": DEFAULT_LOCAL_NM,
            "caliber_halo_nm": DEFAULT_HALO_NM,
            "murray_bracket": [P_LO, P_HI], "min_radius_nm": MIN_RADIUS_NM,
            "n_folds": N_FOLDS, "seed": SEED, "n_sites_target": N_SITES,
            "min_sites": MIN_SITES, "bar_auc_v117": BAR_AUC_V117,
            "exp084_auc": EXP084_AUC, "control_tolerance": CONTROL_TOLERANCE},
    flags={"network": True, "synthetic_fallback": False,
           "labels_used_only_for_evaluation": True},
)


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------

#: Fields EXP-082's ``build_join.py`` writes per operation endpoint. Checked
#: rather than assumed: the builder in the tree writes a structured ``.npy`` to
#: a scratchpad, and the file this experiment declares is the archived copy, so
#: a mismatch is a real possibility and must fail loudly at load time.
EDIT_JOIN_FIELDS = ("root", "op", "t_ms", "is_merge", "pt_idx", "n_pts",
                    "x", "y", "z", "d_skel", "radius", "comp", "deg", "vert")


def load_edit_join(path: Path) -> np.ndarray:
    """The EXP-082 operation-endpoint table, whatever container it arrived in."""
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as z:
            structured = [k for k in z.files if z[k].dtype.names]
            if len(structured) != 1:
                raise ValueError(
                    f"{path}: expected exactly one structured array, found "
                    f"{structured or 'none'} among {list(z.files)}")
            arr = z[structured[0]]
    names = arr.dtype.names or ()
    missing = [f for f in EDIT_JOIN_FIELDS if f not in names]
    if missing:
        raise ValueError(f"{path}: missing field(s) {missing}; have {list(names)}. "
                         "This is not the EXP-082 endpoint table.")
    return arr


def merge_sites(arr: np.ndarray) -> list[dict]:
    """Post-v117 merges with two well-located endpoints, grouped per operation."""
    keep = ((arr["is_merge"] == 1) & (arr["t_ms"] > V117_MS)
            & (arr["n_pts"] == 2) & (arr["d_skel"] <= SNAP_MAX_NM))
    rows = arr[keep]
    by_op: dict[tuple[int, int], list] = defaultdict(list)
    for r in rows:
        by_op[(int(r["root"]), int(r["op"]))].append(r)
    out = []
    for (root, op), rr in by_op.items():
        if len(rr) != 2:
            continue                      # one endpoint failed the snap gate
        a, b = sorted(rr, key=lambda r: int(r["pt_idx"]))
        out.append({"root": root, "op": op,
                    "pts": [np.array([float(x["x"]), float(x["y"]), float(x["z"])])
                            for x in (a, b)],
                    "vert": [int(a["vert"]), int(b["vert"])],
                    "deg_stored": [int(a["deg"]), int(b["deg"])],
                    "rad_stored": [float(a["radius"]), float(b["radius"])]})
    return out


# ---------------------------------------------------------------------------
# skeletons
# ---------------------------------------------------------------------------

class Skeleton:
    """One proofread arbor, with the adjacency and arm walk the sites need."""

    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as z:
            self.V = z["vertices"].astype(np.float64)
            self.E = z["edges"].astype(np.int64)
            self.rad = z["radius"].astype(np.float64)
            self.comp = (z["compartment"].astype(np.int64)
                         if "compartment" in z.files else None)
            self.lvl2 = (z["lvl2_ids"].astype(np.uint64)
                         if "lvl2_ids" in z.files else None)
        self.adj: list[list[int]] = [[] for _ in range(len(self.V))]
        for a, b in self.E:
            self.adj[int(a)].append(int(b))
            self.adj[int(b)].append(int(a))
        self.deg = np.array([len(n) for n in self.adj], np.int64)
        self._dense: Optional[np.ndarray] = None

    def arm(self, v: int, first: int) -> Optional[tuple[int, float, list[int]]]:
        """Walk away from ``v`` via ``first`` to ~``ARM_NM`` of arc.

        Returns ``(end_vertex, arc_nm, path)``. At a branch it takes the
        straightest continuation, so the arm stays one process rather than
        wandering into a sibling. None when the branch dies before
        ``MIN_ARM_NM``.
        """
        prev, cur = int(v), int(first)
        arc = float(np.linalg.norm(self.V[cur] - self.V[prev]))
        path = [prev, cur]
        while arc < ARM_NM:
            nbrs = [n for n in self.adj[cur] if n != prev]
            if not nbrs:
                break
            if len(nbrs) == 1:
                nxt = nbrs[0]
            else:
                d0 = self.V[cur] - self.V[prev]
                n0 = np.linalg.norm(d0)
                if n0 <= 0:
                    break
                d0 = d0 / n0
                cos = []
                for n in nbrs:
                    d1 = self.V[n] - self.V[cur]
                    n1 = np.linalg.norm(d1)
                    cos.append(float(d0 @ (d1 / n1)) if n1 > 0 else -np.inf)
                nxt = nbrs[int(np.argmax(cos))]
            arc += float(np.linalg.norm(self.V[nxt] - self.V[cur]))
            prev, cur = cur, nxt
            path.append(cur)
        if arc < MIN_ARM_NM:
            return None
        return cur, arc, path

    def dense_points(self) -> np.ndarray:
        """Skeleton resampled every ``SKELETON_SAMPLE_NM`` along its edges."""
        if self._dense is None:
            chunks = [self.V]
            for a, b in self.E:
                pa, pb = self.V[int(a)], self.V[int(b)]
                L = float(np.linalg.norm(pb - pa))
                k = int(L // SKELETON_SAMPLE_NM)
                if k > 1:
                    t = np.linspace(0.0, 1.0, k + 1)[1:-1][:, None]
                    chunks.append(pa + t * (pb - pa))
            self._dense = np.vstack(chunks)
        return self._dense


# ---------------------------------------------------------------------------
# the priors
# ---------------------------------------------------------------------------

def murray_exponent(r0: float, r1: float, r2: float) -> float:
    """``p`` solving ``r0^p = r1^p + r2^p``; NaN when unbracketed in [0.5, 8]."""
    if not all(np.isfinite([r0, r1, r2])):
        return float("nan")
    if min(r1, r2) < MIN_RADIUS_NM or r0 < MIN_RADIUS_NM:
        return float("nan")

    def f(p: float) -> float:
        return r1 ** p + r2 ** p - r0 ** p

    try:
        if f(P_LO) * f(P_HI) > 0:
            return float("nan")
        return float(brentq(f, P_LO, P_HI))
    except (ValueError, OverflowError):
        return float("nan")


def score_triple(radii: list[float], dirs: list[np.ndarray]) -> dict:
    """EXP-084's two scores for one branch point, ordered by caliber.

    Mother is the widest of the three arms and the angle is between the two
    others -- the same ordering ``scripts/test_cajal_conservation.py`` used, so
    the control arm is comparable to its 0.675 rather than merely alike.
    """
    order = np.argsort(-np.asarray(radii, float))
    r0 = float(radii[order[0]])
    r1, r2 = float(radii[order[1]]), float(radii[order[2]])
    d1, d2 = dirs[order[1]], dirs[order[2]]
    n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
    if n1 <= 0 or n2 <= 0:
        angle = float("nan")
    else:
        angle = float(np.arccos(np.clip(float(d1 @ d2) / (n1 * n2), -1.0, 1.0)))
    p = murray_exponent(r0, r1, r2)
    prior = (float(CajalPriors.compute_bifurcation_angle_prior(r0, r1, r2, angle))
             if np.isfinite(angle) else float("nan"))
    return {"p": p, "murray": -abs(p - 3.0) if np.isfinite(p) else float("nan"),
            "angle_rad": angle, "angle_prior": prior,
            "r_mother": r0, "r_d1": r1, "r_d2": r2}


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def fold_of(cells: np.ndarray, n_folds: int = N_FOLDS) -> np.ndarray:
    """Deterministic cell-disjoint fold assignment: a cell is never split."""
    uniq = np.unique(cells)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(uniq))
    f = {int(c): int(perm[i] % n_folds) for i, c in enumerate(uniq.tolist())}
    return np.array([f[int(c)] for c in cells], np.int64)


def heldout_auc(score: np.ndarray, y: np.ndarray, fold: np.ndarray) -> dict:
    """Pooled and per-fold area under the curve for a score that fits nothing.

    A parameter-free score has no training, so its pooled number is already
    held out; the per-fold spread is what says whether it is one or two cells
    carrying it. Reported together, never one without the other.
    """
    ok = np.isfinite(score)
    per = []
    for k in range(N_FOLDS):
        m = ok & (fold == k)
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            continue
        per.append(round(float(roc_auc(y[m], score[m])), 6))
    pooled = (float(roc_auc(y[ok], score[ok]))
              if ok.sum() and len(np.unique(y[ok])) == 2 else float("nan"))
    return {"auc": round(pooled, 6) if np.isfinite(pooled) else None,
            "auc_by_fold": per,
            "auc_fold_median": (round(float(np.median(per)), 6) if per else None),
            "auc_fold_min": (round(float(np.min(per)), 6) if per else None),
            "auc_fold_max": (round(float(np.max(per)), 6) if per else None),
            "n_scored": int(ok.sum()), "n_positive": int(y[ok].sum())}


def combined_auc(feats: np.ndarray, y: np.ndarray, fold: np.ndarray) -> dict:
    """Out-of-fold area under the curve for the two priors fitted together.

    This is the only thing here that fits anything, so it is the only thing for
    which "held out by cell" is load-bearing rather than descriptive.
    """
    ok = np.isfinite(feats).all(axis=1)
    oof = np.full(len(y), np.nan)
    for k in range(N_FOLDS):
        te = ok & (fold == k)
        tr = ok & (fold != k)
        if tr.sum() < 50 or te.sum() < 20 or len(np.unique(y[tr])) < 2:
            continue
        model = LogisticRegression.fit(feats[tr], y[tr])
        oof[te] = model.decision(feats[te])
    m = np.isfinite(oof)
    if m.sum() < 20 or len(np.unique(y[m])) < 2:
        return {"auc": None, "n_scored": int(m.sum())}
    return {"auc": round(float(roc_auc(y[m], oof[m])), 6),
            "n_scored": int(m.sum())}


def paired_rate(true_s: np.ndarray, wrong_s: np.ndarray) -> Optional[float]:
    """Share of sites where the true piece outscores its own distractor.

    The within-site statistic EXP-083 argued for: it holds the base fixed, so a
    per-site offset cannot flatter it.
    """
    m = np.isfinite(true_s) & np.isfinite(wrong_s)
    if not m.any():
        return None
    return round(float(np.mean((true_s[m] > wrong_s[m]).astype(float)
                               + 0.5 * (true_s[m] == wrong_s[m]))), 6)


def dist_summary(v: np.ndarray) -> dict:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)),
            "median": round(float(np.median(v)), 4),
            "q25": round(float(np.percentile(v, 25)), 4),
            "q75": round(float(np.percentile(v, 75)), 4)}


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def _majority_object(box: VoxelCaliber, pts: np.ndarray) -> int:
    """Object id along an arm: nearest non-background id at each sample, voted."""
    votes: dict[int, int] = defaultdict(int)
    for p in pts:
        o = box.object_at(p)
        if o == 0:
            near = box.objects_within(p, ARM_OBJECT_SNAP_NM)
            o = near[0][0] if near else 0
        if o:
            votes[int(o)] += 1
    if not votes:
        return 0
    return max(votes.items(), key=lambda kv: kv[1])[0]


def run(ctx: Context) -> Outcome:
    root = ctx.root
    t0 = time.time()
    funnel: dict[str, int] = defaultdict(int)

    arr = load_edit_join(root / EDIT_JOIN)
    ops = merge_sites(arr)
    funnel["post_v117_merges_two_located_endpoints"] = len(ops)
    print(f"  corpus: {len(arr):,} endpoint rows -> {len(ops):,} post-v117 "
          f"merges with two endpoints within {SNAP_MAX_NM:.0f} nm of the "
          f"skeleton", flush=True)

    skel_dir = root / SKEL_DIR
    rng = np.random.default_rng(SEED)
    rng.shuffle(ops)

    # CloudVolume is imported here, not at module scope: this is the only part
    # of the experiment that needs the network, and the module must stay
    # importable (and testable) without it.
    from caveclient import CAVEclient
    from cloudvolume import CloudVolume
    from datetime import datetime, timezone

    client = CAVEclient("minnie65_public")
    cv = CloudVolume(client.chunkedgraph.cloudvolume_path, mip=MIP,
                     use_https=True, progress=False, fill_missing=True,
                     agglomerate=True,
                     timestamp=datetime.fromtimestamp(V117_TIMESTAMP,
                                                      tz=timezone.utc))
    resolution_nm = np.asarray(cv.resolution, float).tolist()
    print(f"  v117 read: mip {MIP}, resolution {resolution_nm} nm", flush=True)

    l2cal = None
    if (root / L2_ATTRS).exists():
        l2cal = load_l2_caliber(root / L2_ATTRS)
        print(f"  level-2 caliber cross-check available: "
              f"{l2cal.coverage:,} nodes", flush=True)

    skels: dict[int, Skeleton] = {}
    sites: list[dict] = []

    for rec in ops:
        if len(sites) >= N_SITES:
            break
        cell = rec["root"]
        if cell not in skels:
            f = skel_dir / f"{cell}_skv4.npz"
            if not f.exists():
                funnel["no_skeleton"] += 1
                continue
            skels[cell] = Skeleton(f)
        sk = skels[cell]

        for side in (0, 1):
            # Funnel counts are per ENDPOINT attempt, not per operation: a merge
            # whose first endpoint fails a gate is retried on its second.
            funnel["endpoint_attempts"] += 1
            v = rec["vert"][side]
            p_site, p_other = rec["pts"][side], rec["pts"][1 - side]
            if v < 0 or v >= len(sk.V):
                funnel["vertex_out_of_range"] += 1
                continue
            if int(sk.deg[v]) != 3:
                funnel["site_not_degree_3"] += 1
                continue
            if int(sk.deg[v]) != rec["deg_stored"][side]:
                funnel["degree_disagrees_with_corpus"] += 1

            arms = [sk.arm(v, n) for n in sk.adj[v]]
            if any(a is None for a in arms):
                funnel["arm_too_short"] += 1
                continue

            anchor = sk.V[v]
            pts_needed = [p_site, p_other, anchor] + [sk.V[a[0]] for a in arms]
            lo = np.minimum(np.min(np.vstack(pts_needed), axis=0),
                            anchor - SITE_WINDOW_NM)
            hi = np.maximum(np.max(np.vstack(pts_needed), axis=0),
                            anchor + SITE_WINDOW_NM)
            box = read_v117_box(cv, lo, hi, pad_nm=BOX_PAD_NM,
                                max_voxels=MAX_BOX_VOXELS)
            if box is None:
                funnel["box_skipped_size_or_bounds"] += 1
                continue

            obj_a = box.object_at(p_site)
            obj_b = box.object_at(p_other)
            if obj_a == 0 or obj_b == 0:
                funnel["click_on_background"] += 1
                continue
            if obj_a == obj_b:
                funnel["both_clicks_same_v117_object"] += 1
                continue

            arm_obj = [_majority_object(box, sk.V[a[2]]) for a in arms]
            n_host = sum(1 for o in arm_obj if o == obj_a)
            n_add = sum(1 for o in arm_obj if o == obj_b)
            if not (n_host == 2 and n_add == 1):
                funnel["arms_not_host_host_added"] += 1
                continue
            added = int(np.flatnonzero(np.array(arm_obj) == obj_b)[0])

            arm_pts = [sk.V[a[0]] for a in arms]
            dirs = [np.asarray(p) - sk.V[v] for p in arm_pts]
            r_pf = [float(sk.rad[a[0]]) for a in arms]
            if not all(np.isfinite(r_pf)) or min(r_pf) < MIN_RADIUS_NM:
                funnel["proofread_radius_unusable"] += 1
                continue

            est = [box.radius(arm_obj[i], arm_pts[i]) for i in range(3)]
            if not all(e.usable for e in est):
                funnel["v117_caliber_unusable"] += 1
                continue
            r_v117 = [float(e.radius_nm) for e in est]

            # -- real distractors, nearest first, this cell's tissue removed --
            dense = sk.dense_points()
            near_mask = (np.abs(dense - anchor) <= DISTRACTOR_MAX_GAP_NM
                         + SELF_TISSUE_NM).all(axis=1)
            own_tree = cKDTree(dense[near_mask]) if near_mask.any() else None
            cands = box.objects_within(anchor, DISTRACTOR_MAX_GAP_NM,
                                       exclude=(obj_a, obj_b))
            range_nm = float(np.linalg.norm(arm_pts[added] - anchor))
            distractors = []
            for obj, gap, p_near in cands:
                if len(distractors) >= N_DISTRACTORS:
                    break
                if own_tree is not None and \
                        float(own_tree.query(p_near, k=1)[0]) <= SELF_TISSUE_NM:
                    funnel["distractor_is_own_tissue"] += 1
                    continue
                d_dir = p_near - anchor
                q = box.point_at_range(obj, anchor, d_dir, range_nm,
                                       cone_cos=DISTRACTOR_CONE_COS,
                                       tol_nm=DISTRACTOR_RANGE_TOL_NM)
                if q is None:
                    funnel["distractor_no_point_at_range"] += 1
                    continue
                e = box.radius(obj, q)
                if not e.usable:
                    funnel["distractor_caliber_unusable"] += 1
                    continue
                distractors.append({"object": int(obj), "gap_nm": round(gap, 1),
                                    "radius_nm": round(float(e.radius_nm), 2),
                                    "dir": (q - sk.V[v])})
            if not distractors:
                funnel["no_usable_distractor"] += 1
                continue

            l2_check = float("nan")
            if l2cal is not None and sk.lvl2 is not None \
                    and len(sk.lvl2) == len(sk.V):
                l2r = vertex_radii_from_l2(sk.lvl2[[a[0] for a in arms]], l2cal,
                                           n_vertices=3)
                l2_check = float(np.nanmedian(l2r))

            sites.append({
                "cell": cell, "op": rec["op"], "side": side, "vertex": int(v),
                "obj_host": int(obj_a), "obj_added": int(obj_b),
                "added_arm": added, "dirs": dirs, "arm_pts": arm_pts,
                "r_proofread": r_pf, "r_v117": r_v117,
                "range_nm": round(range_nm, 1),
                "arm_arc_nm": [round(float(a[1]), 1) for a in arms],
                "distractors": distractors,
                "n_candidates_in_reach": len(cands),
                "l2_median_radius_nm": l2_check,
            })
            funnel["sites_accepted"] += 1
            if len(sites) % 50 == 0:
                print(f"  {len(sites)} sites  ({time.time() - t0:.0f}s)",
                      flush=True)
            break                      # one site per merge operation

    n_sites = len(sites)
    skipped = funnel["box_skipped_size_or_bounds"]
    skip_frac = skipped / max(skipped + n_sites, 1)
    print(f"  accepted {n_sites} sites across "
          f"{len({s['cell'] for s in sites})} cells "
          f"({time.time() - t0:.0f}s)", flush=True)

    if n_sites < MIN_SITES:
        return Outcome(
            passed=False,
            observed={"auc_v117_real_distractor": None,
                      "auc_proofread_control": None,
                      "auc_v117_permuted": None,
                      "join_sites": n_sites, "min_sites": MIN_SITES,
                      "read_box_skip_fraction": round(skip_frac, 4),
                      "resolution_nm": resolution_nm},
            tables={"funnel": dict(funnel)},
            population={"cells": len({s["cell"] for s in sites})},
            note=(f"only {n_sites} join sites survived the funnel, below the "
                  f"{MIN_SITES} declared as the minimum. No area under the "
                  f"curve is reported: the funnel is the result. Largest "
                  f"losses: " + ", ".join(
                      f"{k}={v}" for k, v in sorted(
                          funnel.items(), key=lambda kv: -kv[1])[:4])))

    # -- the three scorings --------------------------------------------------
    cells = np.array([s["cell"] for s in sites], np.int64)
    fold = fold_of(cells)

    true_pf = [score_triple(s["r_proofread"], s["dirs"]) for s in sites]
    true_v = [score_triple(s["r_v117"], s["dirs"]) for s in sites]

    # Permuted distractor: the added arm's radius taken from a DIFFERENT cell's
    # join site, which is what a wrong join grafts on. Permuting within a cell
    # would compare an arbor against itself.
    perm = np.arange(n_sites)
    rng2 = np.random.default_rng(SEED + 1)
    for _ in range(64):
        perm = rng2.permutation(n_sites)
        if np.mean(cells[perm] != cells) > 0.95:
            break
    funnel["permutation_same_cell_pairs"] = int(np.sum(cells[perm] == cells))

    def swapped(s, other, key):
        r = list(s[key])
        r[s["added_arm"]] = other[key][other["added_arm"]]
        return r

    wrong_pf = [score_triple(swapped(sites[i], sites[perm[i]], "r_proofread"),
                             sites[i]["dirs"]) for i in range(n_sites)]
    wrong_v_perm = [score_triple(swapped(sites[i], sites[perm[i]], "r_v117"),
                                 sites[i]["dirs"]) for i in range(n_sites)]

    wrong_v_real, real_owner = [], []
    for i, s in enumerate(sites):
        for d in s["distractors"]:
            r = list(s["r_v117"])
            r[s["added_arm"]] = d["radius_nm"]
            dirs = list(s["dirs"])
            dirs[s["added_arm"]] = d["dir"]
            wrong_v_real.append(score_triple(r, dirs))
            real_owner.append(i)
    real_owner = np.asarray(real_owner, np.int64)

    def assemble(true_rows, wrong_rows, wrong_site_idx=None,
                 true_site_idx=None):
        """Stack one true and one wrong set into (label, fold, score) arrays.

        Both index sets are explicit: a row's fold is the fold of the SITE it
        was built at, and with several distractors per site the two sets are
        different lengths, so neither may be assumed to be ``arange(n_sites)``.
        """
        widx = (np.arange(n_sites) if wrong_site_idx is None else wrong_site_idx)
        tidx = (np.arange(n_sites) if true_site_idx is None else true_site_idx)
        assert len(tidx) == len(true_rows) and len(widx) == len(wrong_rows)
        y = np.r_[np.ones(len(true_rows)), np.zeros(len(wrong_rows))]
        f = np.r_[fold[tidx], fold[widx]]
        mur = np.r_[[t["murray"] for t in true_rows],
                    [w["murray"] for w in wrong_rows]]
        ang = np.r_[[t["angle_prior"] for t in true_rows],
                    [w["angle_prior"] for w in wrong_rows]]
        return y, f, mur, ang

    results = {}
    for name, tr, wr, wi in (
            ("A_proofread_permuted", true_pf, wrong_pf, None),
            ("B_v117_permuted", true_v, wrong_v_perm, None),
            ("C_v117_real_distractor", true_v, wrong_v_real, real_owner)):
        y, f, mur, ang = assemble(tr, wr, wi)
        feats = np.c_[mur, ang]
        results[name] = {
            "murray": heldout_auc(mur, y, f),
            "angle_prior": heldout_auc(ang, y, f),
            "combined_oof": combined_auc(feats, y, f),
            "n_true": len(tr), "n_wrong": len(wr),
            "p_true": dist_summary([t["p"] for t in tr]),
            "p_wrong": dist_summary([w["p"] for w in wr]),
            "abs_p_minus_3_true": dist_summary(
                [abs(t["p"] - 3.0) for t in tr]),
            "abs_p_minus_3_wrong": dist_summary(
                [abs(w["p"] - 3.0) for w in wr]),
        }
        if wi is None:
            results[name]["paired_murray"] = paired_rate(
                np.array([t["murray"] for t in tr]),
                np.array([w["murray"] for w in wr]))
        print(f"  {name:<26} Murray AUC "
              f"{results[name]['murray']['auc']}  angle "
              f"{results[name]['angle_prior']['auc']}  "
              f"n {len(tr)}/{len(wr)}", flush=True)

    # nearest distractor only, so arm C has a 1:1 number directly comparable
    # with the 1:1 permuted arms rather than only a pooled one.
    first = {}
    for k, i in enumerate(real_owner.tolist()):
        first.setdefault(i, k)
    sel = np.array(sorted(first.values()), np.int64)
    near_rows = [wrong_v_real[k] for k in sel]
    near_idx = real_owner[sel]
    y, f, mur, ang = assemble([true_v[i] for i in near_idx.tolist()],
                              near_rows, near_idx, near_idx)
    results["C_nearest_distractor_only"] = {
        "murray": heldout_auc(mur, y, f),
        "angle_prior": heldout_auc(ang, y, f),
        "n_true": len(near_idx), "n_wrong": len(near_rows),
        "paired_murray": paired_rate(
            np.array([true_v[i]["murray"] for i in near_idx.tolist()]),
            np.array([w["murray"] for w in near_rows])),
    }

    # -- is the v117 ruler measuring the same thing? -------------------------
    rp = np.array([r for s in sites for r in s["r_proofread"]], float)
    rv = np.array([r for s in sites for r in s["r_v117"]], float)
    m = np.isfinite(rp) & np.isfinite(rv) & (rp > 0)
    if m.sum() > 10:
        a = np.argsort(np.argsort(rp[m])).astype(float)
        b = np.argsort(np.argsort(rv[m])).astype(float)
        spearman = float(np.corrcoef(a, b)[0, 1])
        ratio = float(np.median(rv[m] / rp[m]))
    else:
        spearman, ratio = float("nan"), float("nan")
    calibration = {
        "n_arms": int(m.sum()),
        "spearman_v117_vs_proofread": round(spearman, 4),
        "median_ratio_v117_over_proofread": round(ratio, 4),
        "proofread_radius_nm": dist_summary(rp),
        "v117_radius_nm": dist_summary(rv),
        "l2_cache_cross_check_sites": int(sum(
            1 for s in sites if np.isfinite(s["l2_median_radius_nm"]))),
    }
    print(f"  caliber calibration: Spearman {spearman:.3f}, median v117/"
          f"proofread ratio {ratio:.3f} over {int(m.sum())} arms", flush=True)

    # -- verdict -------------------------------------------------------------
    auc_c = results["C_v117_real_distractor"]["murray"]["auc"]
    auc_a = results["A_proofread_permuted"]["murray"]["auc"]
    auc_b = results["B_v117_permuted"]["murray"]["auc"]
    control_ok = (auc_a is not None
                  and abs(auc_a - EXP084_AUC) <= CONTROL_TOLERANCE)
    bar_ok = auc_c is not None and auc_c >= BAR_AUC_V117
    skip_ok = skip_frac <= MAX_SKIP_FRACTION
    passed = bool(bar_ok and control_ok and skip_ok)

    fails = []
    if not bar_ok:
        fails.append(f"arm C Murray AUC {auc_c} below {BAR_AUC_V117}")
    if not control_ok:
        fails.append(f"proofread control {auc_a} outside "
                     f"{EXP084_AUC}+-{CONTROL_TOLERANCE}")
    if not skip_ok:
        fails.append(f"read-box guard skipped {skip_frac:.1%} of sites")

    if not control_ok:
        note = (f"CONTROL FAILED, so nothing below it may be attributed to the "
                f"v117 radius: this construction on proofread radii scores "
                f"{auc_a}, not EXP-084's {EXP084_AUC}+-{CONTROL_TOLERANCE}. "
                f"Join sites are not the bifurcation population EXP-084 "
                f"measured, and that difference -- not resolution -- is what "
                f"moved the number. Arm C reads {auc_c} and arm B {auc_b}, "
                f"reported but not interpretable as a resolution effect.")
    else:
        note = (f"proofread control reproduces EXP-084 ({auc_a} vs "
                f"{EXP084_AUC}), so the construction is not what moves the "
                f"number. Same distractor on v117 radii: {auc_b} (the radius "
                f"source costs {round((auc_a or 0) - (auc_b or 0), 3)}). Real "
                f"nearby wrong objects on v117 radii: {auc_c} over "
                f"{n_sites} join sites and "
                f"{len(wrong_v_real)} distractors, "
                + ("at or above " if bar_ok else "below ")
                + f"the {BAR_AUC_V117} bar. Caliber measured on v117 tracks "
                  f"the proofread radius at Spearman {spearman:.2f}, median "
                  f"ratio {ratio:.2f}.")
    if fails:
        note += " Failed clauses: " + "; ".join(fails) + "."

    return Outcome(
        passed=passed,
        observed={
            "auc_v117_real_distractor": auc_c,
            "auc_proofread_control": auc_a,
            "auc_v117_permuted": auc_b,
            "auc_v117_nearest_distractor_only":
                results["C_nearest_distractor_only"]["murray"]["auc"],
            "join_sites": n_sites,
            "distractors": len(wrong_v_real),
            "distractors_per_site": round(len(wrong_v_real) / max(n_sites, 1), 3),
            "cells": len({s["cell"] for s in sites}),
            "holdout": {"scheme": "cell-disjoint folds, a cell never split",
                        "n_folds": N_FOLDS, "seed": SEED,
                        "auc_by_fold_arm_c":
                            results["C_v117_real_distractor"]["murray"]["auc_by_fold"],
                        "fitted_component": "only the two-feature combination; "
                                            "the Murray and angle scores fit "
                                            "nothing"},
            "distractor_construction": {
                "source": "v117 objects with a voxel within "
                          f"{DISTRACTOR_MAX_GAP_NM:.0f} nm of the join site, "
                          "nearest first, this cell's own tissue removed",
                "max_gap_nm": DISTRACTOR_MAX_GAP_NM,
                "n_per_site_cap": N_DISTRACTORS,
                "own_tissue_exclusion_nm": SELF_TISSUE_NM,
                "sampled_at_same_range_as_true_arm": True,
                "cone_cos": DISTRACTOR_CONE_COS,
                "range_tolerance_nm": DISTRACTOR_RANGE_TOL_NM,
                "median_candidates_in_reach": float(np.median(
                    [s["n_candidates_in_reach"] for s in sites])),
                "median_gap_nm": float(np.median(
                    [d["gap_nm"] for s in sites for d in s["distractors"]])),
            },
            "murray_exponent": {
                "true_proofread": results["A_proofread_permuted"]["p_true"],
                "wrong_proofread": results["A_proofread_permuted"]["p_wrong"],
                "true_v117": results["B_v117_permuted"]["p_true"],
                "wrong_v117_permuted": results["B_v117_permuted"]["p_wrong"],
                "wrong_v117_real": results["C_v117_real_distractor"]["p_wrong"],
            },
            "caliber_calibration": calibration,
            "read_box_skip_fraction": round(skip_frac, 4),
            "resolution_nm": resolution_nm,
            "failed_clauses": fails,
        },
        tables={"arms": results, "funnel": dict(funnel),
                "per_site": [{k: s[k] for k in
                              ("cell", "op", "vertex", "obj_host", "obj_added",
                               "added_arm", "range_nm", "arm_arc_nm",
                               "n_candidates_in_reach")}
                             for s in sites[:200]]},
        population={"cells": len({s["cell"] for s in sites}),
                    "join_sites": n_sites,
                    "distractors": len(wrong_v_real),
                    "post_v117_merges_available":
                        funnel["post_v117_merges_two_located_endpoints"]},
        note=note,
    )


if __name__ == "__main__":
    raise SystemExit(main(SPEC, run))
