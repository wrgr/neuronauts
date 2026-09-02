"""ConnectomeBench2 seam positives: external proofreading decisions, per atom.

:mod:`neuronauts.harness.labels` derives seam positives from *this* repo's own
synapse-side tally: an atom is a positive when its sides land in two or more
v1822 roots and those roots are proofread. That definition is strict enough to
be trustworthy and, at 56 atoms in a 100 um cube, small enough to starve
EXP-062/063.

This module carries the second, independent source. ConnectomeBench2 records
real proofreading operations (a human split a false merge, a human joined a
false split) with the root ids each operation consumed and produced. EXP-057B
resolves those root ids to v117 and joins them onto the population, so an atom
can be a seam positive because *someone actually cut it*, with no dependence on
the proofreading-status table that gated the 56.

Two independent axes of strictness are stored, and a consumer picks its own
point on both rather than inheriting one:

``tier``          how much our own v1822 crosswalk corroborates the atom, from
                  :data:`TIER_EXISTING_56` (it is already one of the 56) down
                  to :data:`TIER_NEW_NO_SIGNAL` (CB2 alone says so).
``split_before``  whether the atom was the *before*-root of a ``split_edit`` --
                  the single object that existed immediately before a recorded
                  split correction. That is the closest CB2 analogue of "this
                  object spans two cells"; a ``merge_edit`` operand or an
                  after-root is weaker evidence of a seam and is kept, flagged,
                  rather than silently pooled with it.

Read the ``caveat`` field of :attr:`CB2Positives.meta` before spending these as
located seam positives: the v117 resolution went through one arbitrary
supervoxel of each root, not the decision's edit point, so membership means
"this decision's operand traces back to this atom", not "this atom's synapses
are near this decision's edit point".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neuronauts.harness.labels import lookup_index

#: Corroboration tiers, strongest first. Ordered so ``tier >= t`` is a
#: meaningful strictness filter.
TIER_EXISTING_56 = 3       # already a v1822 mixed-and-proofread seam positive
TIER_NEW_MIXED_STRICT = 2  # new; our own v1822 tally calls it mixed (n_roots>=2)
TIER_NEW_MIXED_RAW = 1     # new; mixed only at the loosest raw threshold
TIER_NEW_NO_SIGNAL = 0     # new; our own tally sees no mixed-lineage signal

TIER_NAMES = {
    TIER_EXISTING_56: "existing_56",
    TIER_NEW_MIXED_STRICT: "new_mixed_strict",
    TIER_NEW_MIXED_RAW: "new_mixed_raw_only",
    TIER_NEW_NO_SIGNAL: "new_no_v1822_signal",
}


@dataclass
class CB2Positives:
    """Per-atom ConnectomeBench2 seam evidence over the harness population."""

    atom_id: np.ndarray           # [N] uint64, v117 root; always a population atom
    tier: np.ndarray              # [N] int8, one of the TIER_* codes above
    split_before: np.ndarray      # [N] bool, was a split_edit before-root
    n_decisions: np.ndarray       # [N] int32, distinct in-cube decisions touching it
    n_split_before: np.ndarray    # [N] int32, as a split_edit before-root
    n_split_after: np.ndarray     # [N] int32, as a split_edit after-root
    n_merge_before: np.ndarray    # [N] int32, as a merge_edit before-root
    n_merge_after: np.ndarray     # [N] int32, as a merge_edit after-root
    n_roots_v1822: np.ndarray     # [N] int32, robust v1822 roots (labels_v1822)
    n_roots_raw_v1822: np.ndarray  # [N] int32, v1822 roots at any support
    meta: dict

    def select(self, *, min_tier: int = TIER_NEW_MIXED_STRICT,
               split_before_only: bool = True) -> np.ndarray:
        """Atom ids at a chosen strictness on both axes.

        The default is the cut EXP-057B recommends: a recorded ``split_edit``
        before-root that our own v1822 tally independently calls mixed.
        """
        m = self.tier >= min_tier
        if split_before_only:
            m &= self.split_before
        return self.atom_id[m]

    def counts(self) -> dict:
        """Atoms per tier, split by the ``split_before`` axis."""
        out: dict = {}
        for t, name in TIER_NAMES.items():
            m = self.tier == t
            out[name] = {"atoms": int(m.sum()),
                         "split_edit_before": int((m & self.split_before).sum())}
        out["total"] = {"atoms": int(len(self.atom_id)),
                        "split_edit_before": int(self.split_before.sum())}
        return out

    def index_of(self, atoms: np.ndarray) -> np.ndarray:
        """Row index of each atom id, or -1 when the atom has no row."""
        return lookup_index(self.atom_id, atoms)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, atom_id=self.atom_id, tier=self.tier,
            split_before=self.split_before, n_decisions=self.n_decisions,
            n_split_before=self.n_split_before, n_split_after=self.n_split_after,
            n_merge_before=self.n_merge_before, n_merge_after=self.n_merge_after,
            n_roots_v1822=self.n_roots_v1822,
            n_roots_raw_v1822=self.n_roots_raw_v1822,
            meta=np.frombuffer(json.dumps(self.meta).encode(), np.uint8))


def load_cb2_positives(path: str | Path) -> CB2Positives:
    with np.load(Path(path), allow_pickle=False) as z:
        return CB2Positives(
            atom_id=z["atom_id"], tier=z["tier"],
            split_before=z["split_before"].astype(bool),
            n_decisions=z["n_decisions"], n_split_before=z["n_split_before"],
            n_split_after=z["n_split_after"],
            n_merge_before=z["n_merge_before"],
            n_merge_after=z["n_merge_after"],
            n_roots_v1822=z["n_roots_v1822"],
            n_roots_raw_v1822=z["n_roots_raw_v1822"],
            meta=json.loads(bytes(z["meta"]).decode()) if "meta" in z else {})
