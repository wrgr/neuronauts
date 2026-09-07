"""EXP-086 -- is an "unexplained" cut surface a real split, or our own artifact?

EXP-085 classified 8,183 sampled cut surfaces with a three-way terminal grammar
-- synaptic terminal 25.8%, field boundary 30.6%, unexplained 43.6% -- and
proposed the unexplained population (~281,790 sites extrapolated over the cube)
as the negative class a stop-versus-extend decision would train on. Its own
stated limit is the reason this experiment exists: **"unexplained" has never
been verified to mean "true segmentation split."** Some share of it is our own
pipeline showing through -- the dust floor, mip-5 truncation of the centroid
clouds, a box face the boundary test did not catch, or a tip that is not a cut
surface at all. Training on a population before measuring its contamination is
how a negative class quietly becomes noise.

The question here is narrow and answerable offline: of the cut surfaces the
grammar calls *unexplained*, what fraction are genuine splits -- real cable
that continues in another object owned by the same proofread cell -- and what
fraction are artifacts of this pipeline?

Method
------
Restrict to objects carrying a trustworthy proofread owner (pure atom, owner
tier above none, from ``labels_v1822``). Labels are used for **evaluation
only**, never to select or steer a tip: the tips and their grammar class are
computed exactly as EXP-085 computed them, label-blind, and the label is
consulted afterwards to adjudicate what lies beyond.

Each tip is then adjudicated in a fixed precedence order, declared here before
any data was seen, because several causes can be true at once and a breakdown
that double-counts explains nothing:

1. ``tip_detection`` (artifact) -- the object's own cloud, read at **full
   point density rather than the 1,500-point subsample tip-finding uses**,
   continues past the tip along its outward axis. Then the "cut surface" is not
   a cut surface; it is a sampling artifact of our own tip finder.
2. ``true_split`` -- a *different* object with the *same* proofread owner lies
   within the continuation radius and inside the outward cone. Real cable that
   continues in another object of the same cell is the definition of a split.
3. ``search_exits_box`` (artifact) -- the outward search ray of one continuation
   radius leaves the cube. Nothing could have been found beyond this tip whether
   or not it continues, so it must not be scored as evidence either way.
4. ``dust_floor`` (artifact) -- an object below the repo's physical dust floor
   (0.041 um^3, synapse-carriers exempt -- EXP-072's floor) lies beyond the tip.
   The cable continues; our substrate would have thrown the continuation away.
5. ``unresolved`` -- neither. Split into ``other_object_beyond`` (some
   above-floor object does lie beyond, but it carries no trustworthy label, so
   we cannot say whose it is) and ``nothing_beyond`` (the cone is empty).
   Reported as its own bucket and never folded into either side.

Controls, on the same code path with nothing changed but the input tips:

* the **synaptic-terminal** class. If the grammar's labels mean anything, a
  bouton or spine head must show a far lower true-split rate than an
  unexplained tip. This is the clause that actually tests the grammar, and it
  is the robust one: whatever bias label coverage introduces applies to both
  classes and largely cancels in a ratio.
* the **field-boundary** class, counted for false boundaries: a tip called
  "truncated by the box" whose same-owner continuation nonetheless sits inside
  the cube was misfiled by the boundary test.

What this cannot settle, stated plainly
---------------------------------------
* **Resolution.** Tips come from mip-5 centroid clouds (256x256x160 nm voxels),
  as EXP-085's did, and the continuation test runs on the same clouds. The
  "does it continue when examined more finely" question is therefore answered
  only in the sense available at this resolution -- full point density versus
  the subsample -- not at mip 2. A mip-2 cube-wide cloud does not exist in this
  repository; when one does, cause 1 should be re-measured against it. The
  result records ``resolution_nm`` so this limit travels with the number.
* **Label coverage bounds the true-split rate from below.** A continuation into
  an unlabelled fragment of the same cell is invisible here and lands in
  ``unresolved.other_object_beyond``, never in ``true_split``. The absolute
  rate is therefore a floor, not an estimate, and the ``unresolved`` bucket
  size says how much room is left underneath it.
* Nothing here proves an unresolved tip is or is not a split. It says the test
  could not tell.

    python -m neuronauts.experiments.exp086_unexplained_split
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from neuronauts.experiments._runner import Context, Outcome, Spec, main
from neuronauts.harness.labels import TIER_NONE, load_labels
from neuronauts.harness.population import load_population

CLOUDS = "data/substrate/c100um/object_clouds_mip5.npz"
OBJECTS = "data/substrate/c100um/objects_v117_mip5.npz"
LABELS_NPZ = "data/substrate/c100um/labels_v1822.npz"
POPULATION = "data/substrate/c100um/population.npz"

# --- tip finding ------------------------------------------------------------
# Lifted deliberately from scripts/label_terminal_grammar.py::tips_of, which is
# itself EXP-081's scripts/measure_frontier_load.py::tips with the soma replaced
# by the object's own centroid (EXP-085 tips every object, not just seeded
# cells). The parameters are reproduced with their original values and given
# names here, because this experiment audits EXP-085's population and a
# different tip finder would audit a different population.
TIP_NEIGHBOUR_NM = 3000.0      # label_terminal_grammar: nbr
TIP_BEYOND_NM = 600.0          # label_terminal_grammar: beyond
TIP_DEDUP_NM = 6000.0          # claim radius: one tip per ending
TIP_MAX_POINTS = 1500          # label_terminal_grammar: maxp
MIN_POINTS_PER_OBJECT = 12     # label_terminal_grammar: per >= 12

# --- the grammar being audited (EXP-085's own thresholds, unchanged) --------
SYNAPSE_RADIUS_NM = 1500.0     # a synapse ON this object within this radius
BOUNDARY_NM = 3000.0           # a tip this close to a box face is "field boundary"

# --- the continuation test (new here) ---------------------------------------
#: 5 um is EXP-081's live-site radius ("a tip is live if a fragment of the
#: seeded target lies within 5 um"), so the bar is set at the same reach the
#: frontier framing already uses. 2 and 3 um bracket it, and are reported so
#: the rate's sensitivity to the radius is visible rather than assumed.
CONTINUATION_RADII_NM = (2000.0, 3000.0, 5000.0)
BAR_RADIUS_NM = 5000.0
#: "Roughly along the outward direction": cos >= 0.5, a 60 degree half-angle
#: cone about the tip's own outward axis. The no-cone variant (radius only) is
#: reported beside it; in tissue this dense, a bare ball finds a neighbour
#: almost anywhere, which is the mistake EXP-085 caught in its own first pass.
CONE_COS_MIN = 0.5
#: EXP-072's physical dust floor, unchanged: an object carrying no synapse in
#: the cube and smaller than this is debris our substrate drops. 0.041 um^3 is
#: 1,000 voxels at 32x32x40 nm.
MIN_VOLUME_UM3 = 0.041
#: Tips per chunk of the neighbour query. At mip-5 cloud density a 5 um ball
#: returns thousands of points per query point (EXP-072 measured ~2,400), so
#: querying every tip at once is a needless multi-GB index array.
BALL_CHUNK = 256

# --- the bar, declared before the data existed ------------------------------
BAR_TRUE_SPLIT_RATE = 0.60
BAR_CLASS_RATIO = 3.0

CLASS_TERMINAL, CLASS_BOUNDARY, CLASS_UNEXPLAINED = 0, 1, 2
CLASS_NAMES = {CLASS_TERMINAL: "synaptic_terminal",
               CLASS_BOUNDARY: "field_boundary",
               CLASS_UNEXPLAINED: "unexplained"}

OUTCOMES = ("true_split", "artifact", "unresolved")
CAUSES = ("tip_detection", "search_exits_box", "dust_floor",
          "other_object_beyond", "nothing_beyond")

SPEC = Spec(
    id="EXP-086",
    title="Is an unexplained cut surface a real split?",
    question="Of the cut surfaces EXP-085's terminal grammar calls "
             "unexplained, what fraction are genuine segmentation splits -- "
             "cable continuing in another object owned by the same proofread "
             "cell -- rather than artifacts of our own pipeline?",
    criterion=(
        f"on tips of objects carrying a trustworthy proofread owner, with tips "
        f"and their grammar class computed label-blind exactly as EXP-085 "
        f"computed them, at continuation radius "
        f"{BAR_RADIUS_NM/1000:.0f} um inside a "
        f"{math.degrees(math.acos(CONE_COS_MIN)):.0f} degree outward cone, and "
        f"under the fixed precedence tip_detection > true_split > "
        f"search_exits_box > dust_floor > unresolved: (1) at least "
        f"{BAR_TRUE_SPLIT_RATE:.0%} of UNEXPLAINED tips are true splits, "
        f"denominator every unexplained tip on a labelled object including "
        f"those the test cannot adjudicate; AND (2) the synaptic-terminal "
        f"class's true-split rate is at least {BAR_CLASS_RATIO:.0f}x lower "
        f"than the unexplained class's. BOTH must hold. If either fails, "
        f"EXP-085's unexplained population is contaminated and must not be "
        f"used as a training negative class as it stands. Clause 2 is the "
        f"bias-robust one: label coverage suppresses both classes alike and "
        f"largely cancels in the ratio, while clause 1 is a floor because a "
        f"continuation into an unlabelled fragment cannot be counted. The "
        f"field-boundary class is counted as a third, non-binding control for "
        f"false boundaries. Labels are evaluation-only"),
    requires_ran=["EXP-071", "EXP-072"],
    inputs=[CLOUDS, OBJECTS, LABELS_NPZ, POPULATION],
    params={"tip_neighbour_nm": TIP_NEIGHBOUR_NM,
            "tip_beyond_nm": TIP_BEYOND_NM,
            "tip_dedup_nm": TIP_DEDUP_NM,
            "tip_max_points": TIP_MAX_POINTS,
            "min_points_per_object": MIN_POINTS_PER_OBJECT,
            "synapse_radius_nm": SYNAPSE_RADIUS_NM,
            "boundary_nm": BOUNDARY_NM,
            "continuation_radii_nm": list(CONTINUATION_RADII_NM),
            "bar_radius_nm": BAR_RADIUS_NM,
            "cone_cos_min": CONE_COS_MIN,
            "min_volume_um3": MIN_VOLUME_UM3,
            "bar_true_split_rate": BAR_TRUE_SPLIT_RATE,
            "bar_class_ratio": BAR_CLASS_RATIO},
    flags={"synthetic_fallback": False,
           "labels_used_only_for_evaluation": True},
)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_clouds(path: str | Path):
    """``(object_id, node_ptr, pos_nm, n_voxels_per_point, meta)``.

    Same two-name tolerance as EXP-072: ``n_voxels_per_point`` is the current
    key, older clouds carry ``n_voxels`` for the same per-POINT quantity.
    """
    with np.load(Path(path), allow_pickle=False) as z:
        nvox = (z["n_voxels_per_point"] if "n_voxels_per_point" in z.files
                else z["n_voxels"])
        meta = json.loads(bytes(z["meta"]).decode()) if "meta" in z.files else {}
        return z["object_id"], z["node_ptr"], z["pos_nm"], nvox, meta


def box_bounds(meta: dict):
    """``(lo, hi)`` of the cube in nm, from the clouds' own metadata.

    EXP-085's script hardcoded ``C = [663, 591, 860] * 1000`` and a 50 um
    half-width. Reading it from the artifact instead means a cube built at a
    different centre or side cannot silently be scored against the wrong faces.
    Raises rather than guessing, because a wrong box makes every boundary call
    wrong and would look like a finding.
    """
    if "centre_um" not in meta or "side_um" not in meta:
        raise KeyError(
            "clouds meta carries no centre_um/side_um, so the box faces cannot "
            "be derived; refusing to assume the 663/591/860 um, 100 um cube")
    centre = np.asarray(meta["centre_um"], float) * 1000.0
    half = float(meta["side_um"]) * 1000.0 / 2.0
    return centre - half, centre + half


# ---------------------------------------------------------------------------
# tip finding -- lifted from scripts/label_terminal_grammar.py::tips_of
# ---------------------------------------------------------------------------

def tips_of(P, *, nbr_nm=TIP_NEIGHBOUR_NM, beyond_nm=TIP_BEYOND_NM,
            dedup_nm=TIP_DEDUP_NM, max_points=TIP_MAX_POINTS):
    """Cut surfaces of one object: points with no cable beyond them.

    Returns ``(tips [T,3], outward_dirs [T,3])``. The direction is the unit
    vector from the (subsampled) cloud centroid to the tip -- the same ``u``
    the acceptance test uses -- returned here because the continuation test
    needs the outward axis and recomputing it elsewhere would risk a different
    centroid.

    Reproduced from ``scripts/label_terminal_grammar.py`` including the
    ``linspace`` subsample (not a stride) and the order of the three tests, so
    that the tips adjudicated here are the tips EXP-085 counted.
    """
    P = np.asarray(P)
    if len(P) > max_points:
        P = P[np.linspace(0, len(P) - 1, max_points).astype(int)]
    if len(P) < 4:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
    ctr = P.mean(0)
    tr = cKDTree(P)
    claimed = np.zeros(len(P), bool)
    out, dirs = [], []
    for i in np.argsort(-np.linalg.norm(P - ctr, axis=1)):
        if claimed[i]:
            continue
        nb = tr.query_ball_point(P[i], r=nbr_nm)
        if len(nb) < 3:
            continue
        u = P[i] - ctr
        n = np.linalg.norm(u)
        if n < 1:
            continue
        u = u / n
        if np.any((P[nb] - P[i]) @ u > beyond_nm):
            continue
        out.append(P[i])
        dirs.append(u)
        for j in tr.query_ball_point(P[i], r=dedup_nm):
            claimed[j] = True
    if not out:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
    return (np.asarray(out, np.float32), np.asarray(dirs, np.float32))


def self_continues(P_full, tip, u, *, nbr_nm=TIP_NEIGHBOUR_NM,
                   beyond_nm=TIP_BEYOND_NM) -> bool:
    """Does the object's own cloud continue past this tip at FULL density?

    The same test ``tips_of`` applies, run against every point of the object
    rather than the 1,500-point subsample. True means the tip is a sampling
    artifact of our tip finder, not a cut surface -- it can only fire for
    objects larger than ``TIP_MAX_POINTS`` points, since smaller ones were
    already tipped at full density.
    """
    if not len(P_full):
        return False
    v = P_full - tip
    d2 = np.einsum("ij,ij->i", v, v)
    near = d2 <= nbr_nm * nbr_nm
    if not near.any():
        return False
    return bool(np.any(v[near] @ u > beyond_nm))


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def _rate(num: int, den: int):
    """``num/den``, or None when the denominator is empty.

    None rather than NaN or 0.0: a rate with no denominator is not a rate, it
    serialises as JSON ``null`` rather than the non-standard ``NaN`` token, and
    every comparison below has to notice it rather than silently treat it as
    zero.
    """
    return (float(num) / den) if den else None


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _class_ratio(unexplained_rate, terminal_rate):
    """``(ratio or None, a phrase saying what the ratio means here)``.

    Three cases, kept apart because collapsing them is how a division by zero
    becomes a claim. The clause-2 verdict is decided from this pair, never from
    a bare float, so "the control could not be run" can never read as a pass.
    """
    if unexplained_rate is None or terminal_rate is None:
        return None, "not computable: a class has no tips"
    if terminal_rate == 0.0:
        if unexplained_rate > 0.0:
            return None, ("terminal true-split rate is exactly zero, so the "
                          "separation is total and unbounded")
        return None, "both rates are zero; the classes are indistinguishable"
    return unexplained_rate / terminal_rate, f"{unexplained_rate / terminal_rate:.2f}x"


def _clause_2(unexplained_rate, terminal_rate, bar: float) -> bool:
    """Does the terminal control sit at least ``bar`` times below unexplained?

    A missing class denominator is a failure, not a pass: without the control
    the grammar's labels were never tested, which is the thing clause 2 exists
    to test.
    """
    if unexplained_rate is None or terminal_rate is None:
        return False
    if terminal_rate == 0.0:
        return unexplained_rate > 0.0
    return (unexplained_rate / terminal_rate) >= bar


def run(ctx: Context) -> Outcome:
    root = ctx.root
    obj_id, ptr, pos, nvox, meta = load_clouds(root / CLOUDS)
    lo, hi = box_bounds(meta)
    res_nm = np.asarray(meta.get("resolution_nm", [256, 256, 160]), float)
    vox_um3 = float(np.prod(res_nm)) / 1e9
    n_obj = len(obj_id)
    per = np.diff(ptr)
    print(f"  clouds: {n_obj:,} objects, {len(pos):,} points, voxel "
          f"{res_nm.tolist()} nm; box {lo.tolist()} -> {hi.tolist()} nm",
          flush=True)

    # --- dust floor, in physical units, synapse-carriers exempt (EXP-072) ---
    with np.load(root / OBJECTS, allow_pickle=False) as z:
        pop_ids, in_pop = z["object_id"], z["in_population"]
    obj_vox = np.add.reduceat(nvox, ptr[:-1]) if len(ptr) > 1 else nvox
    obj_um3 = obj_vox * vox_um3
    carries_syn = np.isin(obj_id, pop_ids[in_pop])
    is_dust = (~carries_syn) & (obj_um3 < MIN_VOLUME_UM3)
    print(f"  dust floor {MIN_VOLUME_UM3:g} um^3: {int(is_dust.sum()):,} of "
          f"{n_obj:,} objects below it ({int(carries_syn.sum()):,} exempt as "
          f"synapse-carrying)", flush=True)

    # --- proofread ownership, EVALUATION ONLY -------------------------------
    labels = load_labels(root / LABELS_NPZ)
    li = labels.index_of(obj_id)
    has = li >= 0
    owner_row = np.zeros(n_obj, np.int64)
    trusted = np.zeros(n_obj, bool)
    if has.any():
        j = li[has]
        own = labels.owner[j].astype(np.int64)
        ok = labels.pure[j] & (labels.owner_tier[j] > TIER_NONE) & (own > 0)
        rows = np.flatnonzero(has)[ok]
        owner_row[rows] = own[ok]
        trusted[rows] = True
    tippable = trusted & (per >= MIN_POINTS_PER_OBJECT)
    print(f"  labelled objects (pure, proofread owner): {int(trusted.sum()):,};"
          f" of those with >= {MIN_POINTS_PER_OBJECT} cloud points: "
          f"{int(tippable.sum()):,}", flush=True)
    if not tippable.any():
        return Outcome(
            passed=False,
            observed={"n_tips": 0},
            note="no object carries both a trustworthy proofread owner and "
                 "enough cloud points to have an end; nothing was measured, "
                 "and no rate is reported rather than one on an empty "
                 "denominator")

    # --- synapses owned by those objects (the grammar's STOP evidence) ------
    # EXP-085's correction, kept: a synapse must be ON the object, not merely
    # near it. With ~901k synapses in a 100 um cube the mean spacing is ~1.04
    # um, so "within 1.5 um of ANY synapse" is satisfied by chance nearly
    # everywhere and returned 99.96% explained.
    pop = load_population(root / POPULATION)
    ctr_nm = pop.syn_ctr_nm.astype(np.float32)
    side_atom = np.concatenate([pop.syn_atom_pre, pop.syn_atom_post])
    side_ctr = np.concatenate([ctr_nm, ctr_nm])
    want = obj_id[tippable]
    keep = (side_atom > 0) & np.isin(side_atom, want)
    side_atom, side_ctr = side_atom[keep], side_ctr[keep]
    order = np.argsort(side_atom, kind="stable")
    side_atom, side_ctr = side_atom[order], side_ctr[order]
    syn_of: dict[int, np.ndarray] = {}
    if len(side_atom):
        ua, starts = np.unique(side_atom, return_index=True)
        ends = np.r_[starts[1:], len(side_atom)]
        syn_of = {int(a): side_ctr[s:e]
                  for a, s, e in zip(ua.tolist(), starts.tolist(), ends.tolist())}
    print(f"  synapse sides on tippable objects: {len(side_atom):,} over "
          f"{len(syn_of):,} objects", flush=True)

    # --- pass 1: tips and their label-blind grammar class -------------------
    tip_pos, tip_dir, tip_row, tip_cls = [], [], [], []
    tip_self, tip_d_edge, tip_d_syn = [], [], []
    n_tipped_objects = 0
    for k in np.flatnonzero(tippable).tolist():
        s, e = int(ptr[k]), int(ptr[k + 1])
        P_full = pos[s:e]
        T, U = tips_of(P_full)
        if not len(T):
            continue
        n_tipped_objects += 1
        S = syn_of.get(int(obj_id[k]))
        if S is not None and len(S):
            # min over this object's own synapses; tips are few and synapses
            # per object are few, so the broadcast is cheaper than a tree
            d_syn = np.linalg.norm(T[:, None, :] - S[None, :, :], axis=2).min(1)
        else:
            d_syn = np.full(len(T), np.inf, np.float64)
        d_edge = np.minimum((T - lo).min(1), (hi - T).min(1))
        is_edge = d_edge < BOUNDARY_NM
        is_syn = (~is_edge) & (d_syn < SYNAPSE_RADIUS_NM)
        cls = np.where(is_edge, CLASS_BOUNDARY,
                       np.where(is_syn, CLASS_TERMINAL, CLASS_UNEXPLAINED))
        for t, u in zip(T, U):
            tip_self.append(self_continues(P_full, t, u))
        tip_pos.append(T)
        tip_dir.append(U)
        tip_row.append(np.full(len(T), k, np.int64))
        tip_cls.append(cls.astype(np.int8))
        tip_d_edge.append(d_edge.astype(np.float32))
        tip_d_syn.append(d_syn.astype(np.float32))

    tip_pos = np.concatenate(tip_pos) if tip_pos else np.empty((0, 3), np.float32)
    tip_dir = np.concatenate(tip_dir) if len(tip_dir) else np.empty((0, 3), np.float32)
    tip_row = np.concatenate(tip_row) if len(tip_row) else np.empty(0, np.int64)
    tip_cls = np.concatenate(tip_cls) if len(tip_cls) else np.empty(0, np.int8)
    tip_self = np.asarray(tip_self, bool)
    tip_d_edge = np.concatenate(tip_d_edge) if len(tip_d_edge) else np.empty(0, np.float32)
    tip_d_syn = np.concatenate(tip_d_syn) if len(tip_d_syn) else np.empty(0, np.float32)
    n_tips = len(tip_pos)
    grammar_counts = {CLASS_NAMES[c]: int((tip_cls == c).sum())
                      for c in CLASS_NAMES}
    print(f"  {n_tipped_objects:,} objects -> {n_tips:,} cut surfaces  "
          + "  ".join(f"{k} {v:,} ({_pct(_rate(v, n_tips))})"
                      for k, v in grammar_counts.items()), flush=True)
    if n_tips == 0:
        return Outcome(passed=False, observed={"n_tips": 0},
                       tables={"grammar_counts": grammar_counts},
                       note="no cut surface was found on any labelled object; "
                            "no rate is reported on an empty denominator")

    # --- pass 2: what lies beyond each tip ----------------------------------
    # One tree over the WHOLE cloud, dust included: the dust-floor cause is only
    # visible if sub-floor objects are in the point set. Per-point attributes
    # are carried as an int32 row index rather than a uint64 object id -- same
    # information, half the memory over ~70M points.
    pt_row = np.repeat(np.arange(n_obj, dtype=np.int32), per)
    print("  building the cloud tree ...", flush=True)
    tree = cKDTree(pos)
    print(f"  tree built over {len(pos):,} points", flush=True)

    max_r = max(CONTINUATION_RADII_NM)
    settings = [(r, cone) for r in CONTINUATION_RADII_NM for cone in (True, False)]
    # outcome[setting] -> per-tip code; cause[setting] -> per-tip cause code
    n_set = len(settings)
    out_code = np.zeros((n_set, n_tips), np.int8)      # index into OUTCOMES
    cause_code = np.full((n_set, n_tips), -1, np.int8)  # index into CAUSES
    same_owner_any = np.zeros((n_set, n_tips), bool)    # ignoring precedence

    # search_exits_box is precedence-free and depends only on the radius
    exits = {r: np.any((tip_pos + tip_dir * r < lo) | (tip_pos + tip_dir * r > hi),
                       axis=1) for r in CONTINUATION_RADII_NM}

    for s0 in range(0, n_tips, BALL_CHUNK):
        s1 = min(s0 + BALL_CHUNK, n_tips)
        balls = tree.query_ball_point(tip_pos[s0:s1], r=max_r)
        for local, idx in enumerate(balls):
            i = s0 + local
            idx = np.asarray(idx, np.int64)
            my_row = int(tip_row[i])
            my_owner = int(owner_row[my_row])
            if len(idx):
                rows = pt_row[idx]
                keep = rows != my_row
                idx, rows = idx[keep], rows[keep]
            if len(idx):
                vec = pos[idx] - tip_pos[i]
                dist = np.linalg.norm(vec, axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    cosang = (vec @ tip_dir[i]) / np.maximum(dist, 1.0)
                # a point sitting on the tip has no direction; count it as in
                # the cone rather than dropping a touching continuation
                cosang = np.where(dist < 1.0, 1.0, cosang)
            else:
                rows = np.empty(0, np.int32)
                dist = np.empty(0)
                cosang = np.empty(0)
            for si, (r, cone) in enumerate(settings):
                sel = dist <= r
                if cone:
                    sel = sel & (cosang >= CONE_COS_MIN)
                near_rows = rows[sel]
                if len(near_rows):
                    near_owner = owner_row[near_rows]
                    same = bool(my_owner > 0 and np.any(near_owner == my_owner))
                    dusty = bool(np.any(is_dust[near_rows]))
                    solid = bool(np.any(~is_dust[near_rows]))
                else:
                    same = dusty = solid = False
                same_owner_any[si, i] = same
                if tip_self[i]:
                    o, c = "artifact", "tip_detection"
                elif same:
                    o, c = "true_split", None
                elif exits[r][i]:
                    o, c = "artifact", "search_exits_box"
                elif dusty:
                    o, c = "artifact", "dust_floor"
                elif solid:
                    o, c = "unresolved", "other_object_beyond"
                else:
                    o, c = "unresolved", "nothing_beyond"
                out_code[si, i] = OUTCOMES.index(o)
                cause_code[si, i] = -1 if c is None else CAUSES.index(c)
        if (s1 // BALL_CHUNK) % 20 == 0 or s1 == n_tips:
            print(f"    adjudicated {s1:,}/{n_tips:,} tips", flush=True)

    # --- tabulate -----------------------------------------------------------
    by_setting: dict[str, dict] = {}
    for si, (r, cone) in enumerate(settings):
        key = f"r{int(r)}_{'cone' if cone else 'ball'}"
        per_class = {}
        for c, cname in CLASS_NAMES.items():
            m = tip_cls == c
            den = int(m.sum())
            oc = out_code[si, m]
            counts = {name: int((oc == OUTCOMES.index(name)).sum())
                      for name in OUTCOMES}
            causes = {name: int((cause_code[si, m] == CAUSES.index(name)).sum())
                      for name in CAUSES}
            adjudicable = den - causes["tip_detection"] - causes["search_exits_box"]
            per_class[cname] = {
                "n_tips": den,
                **{f"n_{k}": v for k, v in counts.items()},
                "true_split_rate": _rate(counts["true_split"], den),
                "n_adjudicable": adjudicable,
                "true_split_rate_adjudicable":
                    _rate(counts["true_split"], adjudicable),
                "same_owner_beyond_ignoring_precedence":
                    int(same_owner_any[si, m].sum()),
                "artifact_causes": causes,
            }
        u = per_class[CLASS_NAMES[CLASS_UNEXPLAINED]]["true_split_rate"]
        t = per_class[CLASS_NAMES[CLASS_TERMINAL]]["true_split_rate"]
        ratio, ratio_note = _class_ratio(u, t)
        by_setting[key] = {
            "continuation_radius_nm": r,
            "cone": cone,
            "cone_half_angle_deg": (round(math.degrees(math.acos(CONE_COS_MIN)), 1)
                                    if cone else None),
            "by_class": per_class,
            "unexplained_over_terminal_ratio": ratio,
            "ratio_note": ratio_note,
        }
        print(f"  {key:<14} unexplained true-split {_pct(u):>7}  terminal "
              f"{_pct(t):>7}  boundary "
              f"{_pct(per_class[CLASS_NAMES[CLASS_BOUNDARY]]['true_split_rate']):>7}"
              f"  ratio {ratio_note}", flush=True)

    # --- setting-independent distance diagnostics ---------------------------
    # The brief named "near the region bound but missed by the boundary test"
    # as a candidate cause. ``search_exits_box`` is the version of it this
    # experiment scores, because it asks whether the *search volume* left the
    # cube rather than picking a second arbitrary margin. These counts are the
    # other reading of the same question, reported so both are available and
    # neither has to be re-run: how far each class's tips actually sit from the
    # nearest face, and -- for the unexplained class -- how many sit in the
    # 3-10 um band just outside the grammar's own 3 um boundary threshold.
    dist_diag = {}
    for c, cname in CLASS_NAMES.items():
        m = tip_cls == c
        if not m.any():
            dist_diag[cname] = {"n_tips": 0}
            continue
        de, ds = tip_d_edge[m].astype(float), tip_d_syn[m].astype(float)
        finite_ds = ds[np.isfinite(ds)]
        dist_diag[cname] = {
            "n_tips": int(m.sum()),
            "d_to_nearest_face_nm": {
                "p10": float(np.percentile(de, 10)),
                "p50": float(np.percentile(de, 50)),
                "p90": float(np.percentile(de, 90))},
            "n_within_3_to_5_um_of_a_face":
                int(((de >= BOUNDARY_NM) & (de < 5000.0)).sum()),
            "n_within_5_to_10_um_of_a_face":
                int(((de >= 5000.0) & (de < 10000.0)).sum()),
            "n_tips_with_a_synapse_on_the_object": int(len(finite_ds)),
            "d_to_own_nearest_synapse_nm_p50":
                float(np.percentile(finite_ds, 50)) if len(finite_ds) else None,
        }

    bar_key = f"r{int(BAR_RADIUS_NM)}_cone"
    bar = by_setting[bar_key]
    unexp = bar["by_class"][CLASS_NAMES[CLASS_UNEXPLAINED]]
    term = bar["by_class"][CLASS_NAMES[CLASS_TERMINAL]]
    bound = bar["by_class"][CLASS_NAMES[CLASS_BOUNDARY]]
    ratio = bar["unexplained_over_terminal_ratio"]
    ratio_note = bar["ratio_note"]

    u_rate, t_rate = unexp["true_split_rate"], term["true_split_rate"]
    clause_1 = bool(u_rate is not None and u_rate >= BAR_TRUE_SPLIT_RATE)
    clause_2 = _clause_2(u_rate, t_rate, BAR_CLASS_RATIO)
    passed = clause_1 and clause_2
    fails = []
    if not clause_1:
        fails.append(f"unexplained true-split rate {_pct(u_rate)} "
                     f"< {BAR_TRUE_SPLIT_RATE:.0%}"
                     + ("" if u_rate is not None
                        else " (no unexplained tip on any labelled object)"))
    if not clause_2:
        fails.append(f"unexplained/terminal separation {ratio_note}, short of "
                     f"{BAR_CLASS_RATIO:.0f}x")

    contaminated = unexp["n_artifact"] + unexp["n_unresolved"]
    note_head = (
        f"at {BAR_RADIUS_NM/1000:.0f} um in a "
        f"{math.degrees(math.acos(CONE_COS_MIN)):.0f} degree cone, "
        f"{unexp['n_true_split']:,} of {unexp['n_tips']:,} unexplained tips on "
        f"labelled objects ({_pct(u_rate)}) have a same-owner continuation; the "
        f"synaptic-terminal control is {_pct(t_rate)} on "
        f"{term['n_tips']:,} tips (separation {ratio_note}). Artifacts and "
        f"unresolved account for {contaminated:,} of the unexplained tips "
        f"({_pct(_rate(contaminated, unexp['n_tips']))})")
    note = note_head + (
        "; both clauses hold, so EXP-085's unexplained population is a usable "
        "negative class at this resolution, with the unresolved bucket as the "
        "stated remaining uncertainty"
        if passed else
        f"; fails on {'; '.join(fails)}. EXP-085's unexplained population "
        f"cannot be used as a training negative class as it stands. The "
        f"artifact-cause breakdown says which correction it needs, and the "
        f"unresolved bucket says how much of the shortfall is label coverage "
        f"rather than contamination")

    return Outcome(
        passed=passed,
        observed={
            "unexplained_true_split_rate": u_rate,
            "terminal_true_split_rate": t_rate,
            "unexplained_over_terminal_ratio": ratio,
            "unexplained_over_terminal_note": ratio_note,
            "n_unexplained_tips": unexp["n_tips"],
            "n_unexplained_true_split": unexp["n_true_split"],
            "n_unexplained_artifact": unexp["n_artifact"],
            "n_unexplained_unresolved": unexp["n_unresolved"],
            "n_terminal_tips": term["n_tips"],
            "n_terminal_true_split": term["n_true_split"],
            "boundary_false_boundary_rate": bound["true_split_rate"],
            "unexplained_true_split_rate_adjudicable":
                unexp["true_split_rate_adjudicable"],
            "failed_clauses": fails,
        },
        population={
            "n_objects_in_clouds": n_obj,
            "n_objects_below_dust_floor": int(is_dust.sum()),
            "n_objects_labelled_trusted": int(trusted.sum()),
            "n_objects_tippable": int(tippable.sum()),
            "n_objects_with_tips": n_tipped_objects,
            "n_tips": n_tips,
            "grammar_class_counts": grammar_counts,
            "grammar_class_shares": {k: _rate(v, n_tips)
                                     for k, v in grammar_counts.items()},
            "box_lo_nm": [float(x) for x in lo],
            "box_hi_nm": [float(x) for x in hi],
            "resolution_nm": res_nm.tolist(),
        },
        tables={
            "by_setting": by_setting,
            "bar_setting": bar_key,
            "distance_diagnostics": dist_diag,
            "params": dict(ctx.spec.params),
            "clouds_meta": meta,
            "precedence": list(CAUSES),
            "resolution_limit":
                "tips and the continuation test both run on mip-5 centroid "
                "clouds (resolution_nm above), as EXP-085's did. The "
                "tip_detection cause is measured as full point density versus "
                "the 1,500-point tip-finding subsample, NOT at a finer mip; no "
                "cube-wide mip-2 cloud exists in this repository. Re-measure "
                "that cause when one does.",
            "coverage_limit":
                "a continuation into a fragment of the same cell that carries "
                "no trustworthy label cannot be counted as a true split and "
                "lands in unresolved.other_object_beyond. The true-split rate "
                "is therefore a floor; the unresolved bucket bounds what it "
                "leaves out.",
        },
        note=note,
    )


if __name__ == "__main__":
    raise SystemExit(main(SPEC, run))
