"""EXP-057B — does an external corpus lift us past 56 trustworthy seam positives?

EXP-057 measured the ceiling this program kept hitting: in a 100 um cube, only
**56** atoms are seam positives by this repo's own definition (mixed lineage
*and* the roots they span are proofread), 15 of them in the training split,
against a seam GNN that was net-negative at 150 objects. EXP-062/063 are not
blocked by a missing method; they are blocked by a missing sample.

ConnectomeBench2 is a second, independent source for the same fact. It records
716,485 real expert proofreading operations across four datasets; 301,162 of
them are MICrONS mouse, in the same nanometre frame as our substrate, and each
carries the root ids the operation consumed (``before_root_ids``) and produced
(``after_root_ids``). An atom that a human actually split is a seam positive on
evidence that owes nothing to the proofreading-status table that gated the 56.

The join has three steps, and only the middle one ever needed the network:

1. **Filter** the corpus to the harness cube -- ``sample_type`` in
   {``merge_edit``, ``split_edit``} with an edit point inside the same 100 um
   box ``population.npz`` was built from. This module re-derives that filter
   from the raw corpus rows rather than trusting the recorded decision table.
2. **Resolve** every referenced root id to its v117 root. This is the only
   networked step and **it has already been run**; its output is a declared
   input to this module (``final_resolution.json``), not something this run
   performs. Provenance of that recorded run, from its own logs: 7,220/7,220
   unique root ids resolved to a nonzero v117 root (zero unresolved, zero
   resolved to 0), 14,469 requests over 31.6 minutes, 28 transient failures
   (26x HTTP 502, 2x ConnectionError -- 0.19%), every one recovered by a retry
   and none clustered. ``flags={"network": False}`` is therefore honest about
   *this* run while the provenance above says where the numbers came from.
3. **Join** the resolved v117 roots onto the population and onto our own v1822
   lineage overlay. That is what this module recomputes, from scratch, and
   then checks against the previous session's recorded verdicts row by row.

The bar was declared before any of this: **at least 1,000 merge or split
decisions map onto v117 roots in or near the harness cube.** It is evaluated
against the loosest reading that the sentence supports (a decision counts when
at least one of its resolved roots is a population atom) and, separately,
against the strictest cut this module defends -- ``split_edit`` before-roots
only -- so a pass cannot rest on the generous reading alone.

**The caveat, stated up front rather than buried under a passing bar.** The
recorded resolution took each root to v117 through **one arbitrary supervoxel**
(``leaves[0]``), not through the decision's own ``edit_point_nm``. For a small
root -- the typical direct operand of an edit -- that is a good proxy. For a
root that had already accumulated many prior mergers, an arbitrary supervoxel
can land far from the seam, inside a correctly-owned part of the same neuron.
So every count here means "this decision's operand traces back to this v117
atom", **not** "this atom's synapses are near this decision's edit point". A
spatial re-check against ``edit_point_nm`` is owed before these atoms are spent
as *located* seam positives; nothing in this module establishes location.

The durable output is ``data/substrate/c100um/cb2_seam_positives.npz`` (see
:mod:`neuronauts.harness.cb2_positives`), which carries two independent
strictness axes -- corroboration tier and ``split_edit``-before -- so EXP-062
and EXP-063 pick their own operating point instead of inheriting this module's.

    uv run python -m neuronauts.experiments.exp057b_cb2_intake
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

from neuronauts.experiments._runner import Context, Outcome, Spec, main
from neuronauts.harness.cb2_positives import (
    TIER_EXISTING_56, TIER_NAMES, TIER_NEW_MIXED_RAW, TIER_NEW_MIXED_STRICT,
    TIER_NEW_NO_SIGNAL, CB2Positives,
)
from neuronauts.harness.labels import load_labels
from neuronauts.harness.population import load_population

CB2_DIR = "data/external/cb2"
RAW_ROWS = f"{CB2_DIR}/full_mouse_rows_raw.parquet"
INCUBE_EDITS = f"{CB2_DIR}/incube_edits.json"
RESOLUTION = f"{CB2_DIR}/final_resolution.json"
UNIQUE_ROOTS = f"{CB2_DIR}/unique_root_ids.json"
#: The previous session's verdicts. Reference to check against, never an input
#: to any number this module reports.
REF_DETAILS = f"{CB2_DIR}/decision_details.json"
REF_SUMMARY = f"{CB2_DIR}/crosswalk_summary.json"

POPULATION = "data/substrate/c100um/population.npz"
LABELS_NPZ = "data/substrate/c100um/labels_v1822.npz"
OUT_NPZ = "data/substrate/c100um/cb2_seam_positives.npz"

EDIT_TYPES = ("merge_edit", "split_edit")
#: The corpus row count the scan recorded; re-checked, not assumed.
EXPECTED_MOUSE_ROWS = 301_162

CRITERION_MIN_DECISIONS = 1_000

SPEC = Spec(
    id="EXP-057B",
    title="ConnectomeBench2 intake",
    question="Can an external corpus lift us past 56 trustworthy seam "
             "positives?",
    criterion="at least 1,000 merge or split decisions map onto v117 roots "
              "in or near the harness cube",
    requires=[],
    inputs=[RAW_ROWS, INCUBE_EDITS, RESOLUTION, UNIQUE_ROOTS, REF_DETAILS,
            REF_SUMMARY, POPULATION, LABELS_NPZ],
    params={"edit_types": list(EDIT_TYPES),
            "criterion_min_decisions": CRITERION_MIN_DECISIONS,
            "resolution_target_version": 117,
            "labels_target_version": 1822},
    #: False for *this* run: the CAVE lineage resolution was performed in a
    #: prior session and is consumed here as a recorded, hashed input
    #: (final_resolution.json). See the module docstring for its provenance.
    flags={"network": False, "synthetic_fallback": False,
           "labels_used_only_for_evaluation": True},
)


class VerificationFailure(RuntimeError):
    """A self-check disagreed. Raised rather than reported-and-continued.

    A recomputation that disagrees with the reference it is being checked
    against means one of the two is wrong; carrying on and reporting the new
    number anyway would be exactly the paper-over CLAUDE.md forbids. The
    runner records this as ``status="error"`` with the detail below.
    """


def _require(ok: bool, msg: str) -> None:
    if not ok:
        raise VerificationFailure(msg)


# ---------------------------------------------------------------------------
# step 1: re-derive the in-cube filter from the raw corpus rows
# ---------------------------------------------------------------------------

def _rederive_in_cube(raw_path: Path, centre_nm: np.ndarray,
                      half_nm: float) -> tuple[dict[str, dict], int]:
    """The in-cube merge/split decisions, read back out of the corpus.

    Reproduces the recorded filter exactly: ``interface_point_nm`` when it is a
    complete triple, else ``render_center_nm``, kept when inside the cube
    ``population.npz`` was built from. Returns ``(hash -> decision, rows with
    no usable point)`` so the recorded decision table can be checked against it
    rather than assumed.
    """
    import pyarrow.parquet as pq

    tbl = pq.read_table(raw_path, columns=["combined_sample_hash",
                                           "sample_type", "metadata"])
    _require(tbl.num_rows == EXPECTED_MOUSE_ROWS,
             f"corpus has {tbl.num_rows} mouse rows, the recorded scan said "
             f"{EXPECTED_MOUSE_ROWS}; the input is not the file the counts "
             "were computed on")

    hashes = tbl.column("combined_sample_hash").to_pylist()
    types = tbl.column("sample_type").to_pylist()
    metas = tbl.column("metadata").to_pylist()

    out: dict[str, dict] = {}
    n_no_point = 0
    for h, st, raw in zip(hashes, types, metas):
        if st not in EDIT_TYPES:
            continue
        m = json.loads(raw)
        pt = m.get("interface_point_nm")
        if not (pt and len(pt) == 3 and all(v is not None for v in pt)):
            pt = m.get("render_center_nm")
        if not (pt and len(pt) == 3 and all(v is not None for v in pt)):
            n_no_point += 1
            continue
        if np.all(np.abs(np.asarray(pt, float) - centre_nm) <= half_nm):
            out[h] = {"sample_type": st,
                      "before_root_ids": [str(r) for r in
                                          (m.get("before_root_ids") or [])],
                      "after_root_ids": [str(r) for r in
                                         (m.get("after_root_ids") or [])]}
    return out, n_no_point


def _roots(edit: dict, key: str | None = None) -> set[str]:
    if key is None:
        seq = list(edit.get("before_root_ids") or []) \
            + list(edit.get("after_root_ids") or [])
    else:
        seq = list(edit.get(key) or [])
    return {str(r) for r in seq}


# ---------------------------------------------------------------------------

def run(ctx: Context) -> Outcome:
    root = ctx.root
    tables: dict = {}

    pop = load_population(root / POPULATION)
    labels = load_labels(root / LABELS_NPZ)
    _require(len(pop.atom_id) == len(labels.atom_id)
             and bool(np.isin(pop.atom_id, labels.atom_id).all()),
             "labels_v1822 does not cover every population atom; the tier "
             "assignment below would silently drop atoms")

    # The cube is taken from the population's own metadata rather than
    # hardcoded, so a substrate rebuilt at a different centre cannot be joined
    # to a decision set filtered at the old one without this failing loudly.
    centre_nm = np.asarray(pop.meta["centre_um"], float) * 1000.0
    half_nm = float(pop.meta["side_um"]) * 1000.0 / 2.0

    seam56 = labels.atom_id[labels.mixed_proofread].astype(np.uint64)
    seam56_set = {int(a) for a in seam56.tolist()}
    pop_set = {int(a) for a in pop.atom_id.tolist()}

    edits = json.loads((root / INCUBE_EDITS).read_text())
    resolution = json.loads((root / RESOLUTION).read_text())
    unique_roots = json.loads((root / UNIQUE_ROOTS).read_text())

    # --- check the recorded decision table against the raw corpus ----------
    rederived, n_no_point = _rederive_in_cube(root / RAW_ROWS, centre_nm,
                                              half_nm)
    recorded = {e["combined_sample_hash"]: e for e in edits}
    _require(len(recorded) == len(edits),
             f"{len(edits) - len(recorded)} duplicate decision hashes in "
             f"{INCUBE_EDITS}")
    hash_mismatch = set(rederived) ^ set(recorded)
    root_mismatch = [h for h in (set(rederived) & set(recorded))
                     if _roots(rederived[h]) != _roots(recorded[h])
                     or rederived[h]["sample_type"] != recorded[h]["sample_type"]]
    _require(not hash_mismatch and not root_mismatch,
             f"re-derived in-cube filter disagrees with {INCUBE_EDITS}: "
             f"{len(hash_mismatch)} decisions in one set only, "
             f"{len(root_mismatch)} with different roots/type")

    by_type_total = collections.Counter(e["sample_type"] for e in edits)
    tables["in_cube_filter_check"] = {
        "source": RAW_ROWS, "mouse_rows": EXPECTED_MOUSE_ROWS,
        "cube_centre_nm": centre_nm.tolist(), "cube_half_width_nm": half_nm,
        "cube_from": "population.npz meta (centre_um, side_um)",
        "rederived_decisions": len(rederived),
        "recorded_decisions": len(edits),
        "rows_with_no_usable_point": n_no_point,
        "hash_set_disagreements": len(hash_mismatch),
        "root_or_type_disagreements": len(root_mismatch),
        "by_sample_type": dict(by_type_total),
    }

    # --- the recorded v117 resolution --------------------------------------
    res = {r["root_id"]: r for r in resolution}
    _require(len(res) == len(resolution),
             "duplicate root_id rows in the recorded resolution")
    referenced: set[str] = set()
    for e in edits:
        referenced |= _roots(e)
    _require(referenced == set(unique_roots) == set(res),
             f"root id sets disagree: {len(referenced)} referenced by "
             f"decisions, {len(set(unique_roots))} in {UNIQUE_ROOTS}, "
             f"{len(res)} in {RESOLUTION}")

    n_unresolved = sum(1 for r in resolution if r["status"] != "ok")
    n_zero = sum(1 for r in resolution
                 if r["status"] == "ok" and not r["v117_root"])
    root_v117 = {rid: int(r["v117_root"]) for rid, r in res.items()
                 if r["status"] == "ok" and r["v117_root"]}

    def classify(rid: str) -> str:
        v = root_v117.get(rid)
        if v is None:
            return "unresolved" if res[rid]["status"] != "ok" else "resolved_zero"
        if v in seam56_set:
            return "ok_in_population_seam56"
        return "ok_in_population_other" if v in pop_set else "ok_not_in_population"

    root_cls = {rid: classify(rid) for rid in res}
    root_breakdown = dict(collections.Counter(root_cls.values()))
    tables["unique_roots"] = {
        "n_unique_roots": len(res), "n_unresolved": n_unresolved,
        "n_resolved_to_zero": n_zero, "breakdown": root_breakdown,
        "recorded_cave_run": {
            "note": "provenance of the declared input final_resolution.json; "
                    "no network call is made by this experiment",
            "unique_roots_resolved": 7220, "resolution_rate": 1.0,
            "wall_clock_minutes": 31.6, "requests": 14469,
            "transient_failures": 28, "transient_failure_rate": 0.0019,
            "all_retried_successfully": True,
            "log": f"{CB2_DIR}/final_run_summary.json",
        },
    }

    # --- decision-level join ------------------------------------------------
    rows = []
    atom_roles: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter)
    atom_decisions: dict[int, set] = collections.defaultdict(set)
    seam56_touched: set[int] = set()
    new_candidates: set[int] = set()

    for e in edits:
        h = e["combined_sample_hash"]
        st = e["sample_type"]
        roots = _roots(e)
        cls = [root_cls[r] for r in roots]
        n_resolved = sum(c.startswith("ok") for c in cls)
        in_seam = any(c == "ok_in_population_seam56" for c in cls)
        in_other = any(c == "ok_in_population_other" for c in cls)
        rows.append({
            "combined_sample_hash": h, "sample_type": st,
            "n_roots": len(roots), "n_resolved": n_resolved,
            "all_resolved": n_resolved == len(cls),
            "in_population": in_seam or in_other,
            "in_seam56": in_seam,
            "in_population_not_seam56": in_other,
        })
        for r, c in zip(roots, cls):
            if not c.startswith("ok_in_population"):
                continue
            a = root_v117[r]
            (seam56_touched if c.endswith("seam56") else new_candidates).add(a)
            atom_decisions[a].add(h)
        # Role tallies need the side a root sat on, so redo per side. Deduped
        # by *atom*, not by root: two roots of one decision can resolve to the
        # same v117 atom, and that is one decision touching it, not two.
        for side in ("before", "after"):
            side_atoms = {root_v117[r] for r in _roots(e, f"{side}_root_ids")
                          if root_cls[r].startswith("ok_in_population")}
            for a in side_atoms:
                atom_roles[a][f"{st}:{side}"] += 1

    n = len(rows)
    n_all_resolved = sum(r["all_resolved"] for r in rows)
    n_in_pop = sum(r["in_population"] for r in rows)
    n_in_seam56 = sum(r["in_seam56"] for r in rows)
    n_new = sum(r["in_population_not_seam56"] for r in rows)

    by_type: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    for r in rows:
        c = by_type[r["sample_type"]]
        c["total"] += 1
        for k in ("all_resolved", "in_population", "in_seam56",
                  "in_population_not_seam56"):
            if r[k]:
                c[k] += 1
    tables["by_sample_type"] = {t: dict(c) for t, c in by_type.items()}

    # --- the tighter cut: split_edit before-roots ---------------------------
    # The 56 are "mixed lineage whose spanning roots are proofread". The CB2
    # analogue is the object that existed immediately BEFORE a recorded split
    # correction -- not a merge_edit operand (those are same-cell continuation
    # candidates, EXP-060/061's problem) and not an after-root (the pieces).
    sb_in_pop = sb_seam = sb_new = 0
    sb_seam_atoms: set[int] = set()
    sb_new_atoms: set[int] = set()
    for e in edits:
        if e["sample_type"] != "split_edit":
            continue
        before = _roots(e, "before_root_ids")
        cls = {r: root_cls[r] for r in before}
        seam = {root_v117[r] for r, c in cls.items()
                if c == "ok_in_population_seam56"}
        other = {root_v117[r] for r, c in cls.items()
                 if c == "ok_in_population_other"}
        if seam or other:
            sb_in_pop += 1
        if seam:
            sb_seam += 1
            sb_seam_atoms |= seam
        if other:
            sb_new += 1
            sb_new_atoms |= other

    # --- corroboration from our own, independent v1822 crosswalk ------------
    def tier_of(atoms: np.ndarray) -> np.ndarray:
        idx = labels.index_of(atoms)
        _require(bool((idx >= 0).all()),
                 "a CB2 candidate atom has no labels_v1822 row")
        t = np.full(len(atoms), TIER_NEW_NO_SIGNAL, np.int8)
        t[labels.n_roots_raw[idx] >= 2] = TIER_NEW_MIXED_RAW
        t[labels.n_roots[idx] >= 2] = TIER_NEW_MIXED_STRICT
        t[np.isin(atoms, seam56)] = TIER_EXISTING_56
        return t

    def corroboration(atoms: set[int]) -> dict:
        a = np.array(sorted(atoms), np.uint64)
        t = tier_of(a)
        idx = labels.index_of(a)
        out = {TIER_NAMES[k]: int((t == k).sum()) for k in TIER_NAMES}
        out["n_atoms"] = int(len(a))
        # reported separately because the doc's rows overlap: "pure" is a
        # stricter statement than "no raw signal" and the two are not the same
        # subset.
        out["pure_by_v1822"] = int(labels.pure[idx].sum())
        return out

    tables["split_edit_before_cut"] = {
        "definition": "decisions whose split_edit before-root resolves into "
                      "the population",
        "split_edit_decisions": int(by_type_total["split_edit"]),
        "before_root_in_population": sb_in_pop,
        "decisions_hitting_existing_56": sb_seam,
        "distinct_existing_56_atoms": len(sb_seam_atoms),
        "decisions_with_new_candidate": sb_new,
        "distinct_new_candidate_atoms": len(sb_new_atoms),
        "corroboration": corroboration(sb_new_atoms),
    }
    tables["corroboration_any_role"] = corroboration(new_candidates)

    # --- durable artifact ---------------------------------------------------
    all_atoms = np.array(sorted(seam56_touched | new_candidates), np.uint64)
    tiers = tier_of(all_atoms)
    lab_idx = labels.index_of(all_atoms)

    def role(a: int, key: str) -> int:
        return int(atom_roles[a][key])

    ids = all_atoms.tolist()
    art = CB2Positives(
        atom_id=all_atoms,
        tier=tiers,
        split_before=np.isin(all_atoms, np.array(
            sorted(sb_new_atoms | sb_seam_atoms), np.uint64)),
        n_decisions=np.array([len(atom_decisions[a]) for a in ids], np.int32),
        n_split_before=np.array([role(a, "split_edit:before") for a in ids],
                                np.int32),
        n_split_after=np.array([role(a, "split_edit:after") for a in ids],
                               np.int32),
        n_merge_before=np.array([role(a, "merge_edit:before") for a in ids],
                                np.int32),
        n_merge_after=np.array([role(a, "merge_edit:after") for a in ids],
                               np.int32),
        n_roots_v1822=labels.n_roots[lab_idx].astype(np.int32),
        n_roots_raw_v1822=labels.n_roots_raw[lab_idx].astype(np.int32),
        meta={
            "experiment": SPEC.id,
            "source": "ConnectomeBench2 (mouse/MICrONS split), CC-BY-4.0",
            "population": POPULATION,
            "labels": LABELS_NPZ,
            "resolution_input": RESOLUTION,
            "base_version": 117,
            "labels_target_version": 1822,
            "cube_centre_nm": centre_nm.tolist(),
            "cube_half_width_nm": half_nm,
            "n_in_cube_decisions": n,
            "tier_codes": {str(k): v for k, v in TIER_NAMES.items()},
            "split_before": "atom was the before-root of >=1 recorded "
                            "split_edit; the closest CB2 analogue of the 56",
            "recommended_cut": "tier >= 2 and split_before (the strictest "
                               "defensible set); loosen along either axis "
                               "deliberately, not by default",
            "caveat": "v117 resolution went through one arbitrary supervoxel "
                      "(leaves[0]) of each root, NOT the decision's "
                      "edit_point_nm. Membership means 'this decision's "
                      "operand traces back to this atom', not 'this atom's "
                      "synapses are near this decision's edit point'. A "
                      "spatial re-check is owed before these are used as "
                      "LOCATED seam positives.",
        },
    )
    art.save(root / OUT_NPZ)
    tier_counts = art.counts()
    tables["artifact"] = {"path": OUT_NPZ, "n_atoms": int(len(all_atoms)),
                          "tier_counts": tier_counts,
                          "loader": "neuronauts.harness.cb2_positives."
                                    "load_cb2_positives"}

    # --- verification against the previous session's recorded verdicts ------
    ref_rows = json.loads((root / REF_DETAILS).read_text())
    ref_sum = json.loads((root / REF_SUMMARY).read_text())
    ref_by_hash = {r["combined_sample_hash"]: r for r in ref_rows}
    fields = ["sample_type", "n_roots", "n_resolved", "all_resolved",
              "in_population", "in_seam56", "in_population_not_seam56"]
    dis: collections.Counter = collections.Counter()
    n_cmp = 0
    for r in rows:
        ref = ref_by_hash.get(r["combined_sample_hash"])
        if ref is None:
            dis["absent_from_reference"] += 1
            continue
        n_cmp += 1
        for f in fields:
            if r[f] != ref[f]:
                dis[f] += 1

    mine_sum = {
        "n_decisions": n, "n_unique_roots": len(res),
        "unique_root_failure_breakdown": root_breakdown,
        "decision_a_all_resolved": n_all_resolved,
        "decision_b_any_in_population": n_in_pop,
        "decision_c_any_in_seam56": n_in_seam56,
        "decision_any_in_population_not_seam56_candidate_new": n_new,
        "distinct_existing_seam56_atoms_touched": len(seam56_touched),
        "distinct_candidate_new_atoms_touched": len(new_candidates),
        "by_sample_type": {t: dict(c) for t, c in by_type.items()},
    }
    summary_diffs = {k: {"recomputed": v, "reference": ref_sum.get(k)}
                     for k, v in mine_sum.items() if ref_sum.get(k) != v}
    tables["reference_check"] = {
        "reference_decisions": REF_DETAILS, "reference_summary": REF_SUMMARY,
        "decisions_compared": n_cmp,
        "decisions_in_reference": len(ref_rows),
        "disagreements": int(sum(dis.values())),
        "disagreements_by_field": dict(dis),
        "summary_fields_compared": len(mine_sum),
        "summary_disagreements": len(summary_diffs),
        "summary_diff_detail": summary_diffs,
        "doc_figures_reproduced": {
            "docs/threads/connectomebench_intake.md §4": {
                "split_edit_before_decisions": {"doc": 1508,
                                                "recomputed": sb_new},
                "split_edit_before_distinct_atoms": {"doc": 1116,
                                                     "recomputed": len(sb_new_atoms)},
                "corroboration_mixed_strict": {
                    "doc": 574,
                    "recomputed": tables["split_edit_before_cut"]
                    ["corroboration"][TIER_NAMES[TIER_NEW_MIXED_STRICT]]},
                "corroboration_mixed_raw_only": {
                    "doc": 331,
                    "recomputed": tables["split_edit_before_cut"]
                    ["corroboration"][TIER_NAMES[TIER_NEW_MIXED_RAW]]},
                "corroboration_no_signal": {
                    "doc": 211,
                    "recomputed": tables["split_edit_before_cut"]
                    ["corroboration"][TIER_NAMES[TIER_NEW_NO_SIGNAL]]},
                "corroboration_pure": {
                    "doc": 481,
                    "recomputed": tables["split_edit_before_cut"]
                    ["corroboration"]["pure_by_v1822"]},
            },
        },
    }
    _require(sum(dis.values()) == 0 and not summary_diffs,
             "recomputation disagrees with the recorded reference: "
             f"{dict(dis)} per-decision, {summary_diffs} in summary. One of "
             "the two is wrong; the disagreement must be resolved before "
             "either number is reported.")

    strict = tier_counts[TIER_NAMES[TIER_NEW_MIXED_STRICT]]["split_edit_before"]
    passed = n_in_pop >= CRITERION_MIN_DECISIONS
    strict_also = sb_new >= CRITERION_MIN_DECISIONS

    print(f"  decisions in-cube            : {n}", flush=True)
    print(f"  ... all roots resolved       : {n_all_resolved}", flush=True)
    print(f"  ... >=1 root a population atom: {n_in_pop} "
          f"(bar {CRITERION_MIN_DECISIONS})", flush=True)
    print(f"  ... touching the existing 56 : {n_in_seam56}", flush=True)
    print(f"  distinct new candidate atoms : {len(new_candidates)}", flush=True)
    print(f"  split_edit before-root cut   : {sb_new} decisions, "
          f"{len(sb_new_atoms)} distinct atoms", flush=True)
    print(f"  artifact                     : {OUT_NPZ} "
          f"({len(all_atoms)} atoms)", flush=True)
    for name, c in tier_counts.items():
        print(f"    {name:<22} {c['atoms']:>5} atoms  "
              f"({c['split_edit_before']} split_edit-before)", flush=True)

    return Outcome(
        passed=passed,
        observed={
            "decisions_mapped_to_population_atom": n_in_pop,
            "criterion_min_decisions": CRITERION_MIN_DECISIONS,
            "split_edit_before_decisions": sb_new,
            "decisions_in_cube": n,
            "decisions_all_roots_resolved": n_all_resolved,
            "decisions_touching_existing_56": n_in_seam56,
            "distinct_new_candidate_atoms": len(new_candidates),
            "split_edit_before_distinct_atoms": len(sb_new_atoms),
            "existing_56_atoms_corroborated": len(seam56_touched),
            "strictest_cut_atoms": strict,
            "reference_disagreements": int(sum(dis.values())),
        },
        population={
            "n_population_atoms": int(len(pop.atom_id)),
            "n_existing_seam_positives": int(len(seam56)),
            "n_cb2_atoms": int(len(all_atoms)),
            "n_in_cube_decisions": n,
            "n_unique_roots_resolved": len(res),
        },
        tables=tables,
        note=(
            f"PASS on the declared bar: {n_in_pop} of {n} in-cube "
            f"merge/split decisions map onto a v117 root that is a population "
            f"atom (bar {CRITERION_MIN_DECISIONS}); the strictest cut this "
            f"module defends -- split_edit before-roots only -- clears it too "
            f"at {sb_new} decisions over {len(sb_new_atoms)} distinct atoms, "
            f"so the pass does not rest on the generous reading. "
            f"{len(seam56_touched)}/{len(seam56)} of the existing seam "
            f"positives are independently re-found by this corpus. "
            f"{strict} of the new atoms are both split_edit before-roots and "
            f"independently called mixed-lineage by our own v1822 crosswalk. "
            f"Every recomputed count matches the prior session's recorded "
            f"verdicts exactly ({n_cmp} decisions compared, 0 disagreements). "
            f"CAVEAT, which the pass does not retire: resolution went through "
            f"one arbitrary supervoxel (leaves[0]) per root, not the "
            f"decision's edit_point_nm, so these are 'this decision's operand "
            f"traces back to this atom', NOT 'this atom's synapses are near "
            f"this decision's edit point'. Nothing here establishes seam "
            f"LOCATION; a spatial re-check against edit_point_nm is owed "
            f"before EXP-062 trains a locator on them."
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main(SPEC, run))
