"""EXP-057 — how much of the substrate carries usable ground truth, and where?

Everything downstream is gated on this. A candidate panel can only be scored
where a pair's label is known; a seam locator can only be trained where a false
merge is certified. So the first question is not "does the method work" but
"how much trustworthy label is there, and is it enough".

The bar, declared before looking: **at least 30% of the region's synapse mass
must sit on single-lineage atoms owned by a proofread cell.** Below that, a
metric computed on the labelled subset would not describe the region, and the
right move is to widen the tier rather than to report a flattering number.

Two things are measured that the plan did not originally ask for, because both
change what the next experiments can attempt:

*Positives for seam location.* An atom is a usable seam positive only when it
is mixed-lineage **and** the roots it spans are proofread — otherwise "this
object spans two cells" rests on unproofread lineage. That count is reported
separately from the unrestricted mixed count, and never pooled with it. This
repo's own seam GNN was net-negative at 150 objects and first cleared zero at
513, so the number is compared against those two thresholds directly.

*The spatial split.* Learned scorers must be fitted on tissue they are not
scored on. Splitting along one axis with a buffer wider than the candidate
radius means no candidate pair can straddle the seam, which is the leak a
random split over atoms would introduce.

    uv run python -m neuronauts.experiments.exp057_gt_overlay
"""

from __future__ import annotations

import numpy as np

from neuronauts.experiments._runner import Context, Outcome, Spec, main
from neuronauts.harness.labels import TIER_GOLD, TIER_NONE, load_labels, summarize
from neuronauts.harness.population import load_population
from neuronauts.harness.spatial_split import assign_split, describe

POPULATION = "data/substrate/c100um/population.npz"
LABELS_NPZ = "data/substrate/c100um/labels_v1822.npz"

#: Wider than the widest candidate radius considered in EXP-060 (10 um), so a
#: candidate pair cannot straddle the seam.
SPLIT_BUFFER_NM = 20_000.0

#: The two sample sizes at which this repo's seam GNN was measured.
SEAM_NET_NEGATIVE_AT = 150
SEAM_CLEARED_ZERO_AT = 513

SPEC = Spec(
    id="EXP-057",
    title="GT overlay and spatial split",
    question="What fraction of atoms and synapse mass carry unambiguous ground "
             "truth, and where?",
    criterion="at least 30% of synapse mass on single-lineage atoms with a "
              "proofread owner; else widen the tier before proceeding",
    requires=[],
    inputs=[POPULATION, LABELS_NPZ],
    params={"split_buffer_nm": SPLIT_BUFFER_NM, "split_axis": 0,
            "target_version": 1822},
    flags={"synthetic_fallback": False,
           "labels_used_only_for_evaluation": True},
)

MASS_BAR = 0.30


def run(ctx: Context) -> Outcome:
    root = ctx.root
    pop = load_population(root / POPULATION)
    labels = load_labels(root / LABELS_NPZ)

    # Align labels to the population; an atom with no label row is unknown, not
    # negative, and is counted as such.
    idx = labels.index_of(pop.atom_id)
    has_row = idx >= 0
    n_syn = pop.n_synapses.astype(np.int64)
    total_mass = int(n_syn.sum())

    pure = np.zeros(len(pop.atom_id), bool)
    mixed = np.zeros(len(pop.atom_id), bool)
    mixed_pr = np.zeros(len(pop.atom_id), bool)
    tier = np.full(len(pop.atom_id), TIER_NONE, np.int8)
    pure[has_row] = labels.pure[idx[has_row]]
    mixed[has_row] = labels.mixed[idx[has_row]]
    mixed_pr[has_row] = labels.mixed_proofread[idx[has_row]]
    tier[has_row] = labels.owner_tier[idx[has_row]]

    pure_proofread = pure & (tier > TIER_NONE)
    pure_gold = pure & (tier == TIER_GOLD)

    mass = {
        "total": total_mass,
        "pure_proofread": int(n_syn[pure_proofread].sum()),
        "pure_gold": int(n_syn[pure_gold].sum()),
        "mixed": int(n_syn[mixed].sum()),
        "mixed_proofread": int(n_syn[mixed_pr].sum()),
        "unlabelled": int(n_syn[~has_row].sum()),
    }
    frac = {k: (v / total_mass if total_mass else 0.0)
            for k, v in mass.items() if k != "total"}

    # --- spatial split, over the labelled population -------------------------
    centre_nm = float(np.median(pop.centroid_nm[:, 0]))
    split = assign_split(pop.centroid_nm, axis=0, centre_nm=centre_nm,
                         buffer_nm=SPLIT_BUFFER_NM)
    split_all = describe(split)
    split_labelled = describe(split[pure_proofread])
    split_seam = describe(split[mixed_pr])

    n_seam = int(mixed_pr.sum())
    passed = frac["pure_proofread"] >= MASS_BAR

    note = (f"{frac['pure_proofread']:.1%} of synapse mass on proofread-owned "
            f"single-lineage atoms (bar {MASS_BAR:.0%}); "
            f"{n_seam} usable seam positives")
    if n_seam < SEAM_NET_NEGATIVE_AT:
        note += (f" -- below the {SEAM_NET_NEGATIVE_AT} at which this repo's "
                 f"seam GNN was net-negative, so EXP-062/063 are data-starved "
                 f"until EXP-057B lands")

    return Outcome(
        passed=passed,
        observed={
            "mass_frac_pure_proofread": round(frac["pure_proofread"], 6),
            "mass_frac_pure_gold": round(frac["pure_gold"], 6),
            "n_seam_positives": n_seam,
            "n_pure_proofread_atoms": int(pure_proofread.sum()),
        },
        population={
            "n_atoms": int(len(pop.atom_id)),
            "n_atoms_with_label_row": int(has_row.sum()),
            "n_synapse_sides": total_mass,
            "target_version": 1822,
            **summarize(labels),
        },
        tables={
            "synapse_mass": {
                k: {"synapses": mass[k], "fraction": round(frac[k], 6)}
                for k in ("pure_proofread", "pure_gold", "mixed",
                          "mixed_proofread", "unlabelled")},
            "spatial_split": {
                "all_atoms": split_all,
                "pure_proofread_atoms": split_labelled,
                "seam_positives": split_seam},
            "seam_sample_size": {
                "observed": {"n": n_seam},
                "seam_gnn_net_negative_at": {"n": SEAM_NET_NEGATIVE_AT},
                "seam_gnn_cleared_zero_at": {"n": SEAM_CLEARED_ZERO_AT}},
        },
        note=note,
    )


if __name__ == "__main__":
    raise SystemExit(main(SPEC, run))
