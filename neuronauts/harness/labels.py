"""Ground-truth overlay for the label-blind atom population (evaluation only).

The population is enumerated without labels. Labels are attached afterwards,
per atom, from two offline tables:

  * the supervoxel -> target-version root map for every supervoxel the
    population's synapses touch (``sv_v<target>.npz``, one batched
    ``roots_at`` call, cached), and
  * the ``proofreading_status_and_strategy`` table at the same version, which
    says which of those roots a human has verified and how far.

For each atom we tally, over its synapse *sides* (a synapse contributes its
pre side to the pre atom and its post side to the post atom), which target
roots those sides land in. An atom whose sides all land in one root is *pure*;
that root is its owner. An atom whose sides land in two or more roots (each
with enough support to rule out a stray synapse) is *mixed* -- a real v117
false merge that proofreading later cut.

Ownership by a *proofread* root is what makes a label trustworthy. A gold
cell (dendrite extended, axon fully extended) is complete, so an atom that is
not in it is genuinely not part of it; a silver cell may still be missing
axon, so an unproofread atom near it is not a safe negative. The pair-label
rules in :func:`pair_labels` encode exactly that, and everything else is
"unknown" rather than guessed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

TIER_NONE, TIER_SILVER, TIER_GOLD = 0, 1, 2
LABEL_UNKNOWN, LABEL_NEG, LABEL_POS = -1, 0, 1


@dataclass
class AtomLabels:
    """Per-atom ground truth at a target materialization."""

    atom_id: np.ndarray        # [A] uint64
    owner: np.ndarray          # [A] uint64 dominant target root (0 = none)
    owner_frac: np.ndarray     # [A] float32 share of sides on the owner
    owner_tier: np.ndarray     # [A] int8 proofread tier of the owner
    n_sides: np.ndarray        # [A] int32 synapse sides tallied
    n_roots_raw: np.ndarray    # [A] int32 distinct target roots, any support
    n_roots: np.ndarray        # [A] int32 distinct target roots, robust
    n_roots_proofread: np.ndarray  # [A] int32 robust roots that are proofread
    meta: dict

    @property
    def pure(self) -> np.ndarray:
        return (self.n_roots == 1) & (self.owner_frac >= self.meta.get(
            "pure_min_owner_frac", 0.9))

    @property
    def mixed(self) -> np.ndarray:
        return self.n_roots >= 2

    @property
    def mixed_proofread(self) -> np.ndarray:
        return self.n_roots_proofread >= 2

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, atom_id=self.atom_id, owner=self.owner,
            owner_frac=self.owner_frac, owner_tier=self.owner_tier,
            n_sides=self.n_sides, n_roots_raw=self.n_roots_raw,
            n_roots=self.n_roots, n_roots_proofread=self.n_roots_proofread,
            meta=np.frombuffer(json.dumps(self.meta).encode(), np.uint8))

    def index_of(self, atoms: np.ndarray) -> np.ndarray:
        """Row index of each atom id, or -1 when the atom has no row."""
        return lookup_index(self.atom_id, atoms)


def load_labels(path: str | Path) -> AtomLabels:
    with np.load(Path(path), allow_pickle=False) as z:
        return AtomLabels(
            atom_id=z["atom_id"], owner=z["owner"], owner_frac=z["owner_frac"],
            owner_tier=z["owner_tier"], n_sides=z["n_sides"],
            n_roots_raw=z["n_roots_raw"], n_roots=z["n_roots"],
            n_roots_proofread=z["n_roots_proofread"],
            meta=json.loads(bytes(z["meta"]).decode()) if "meta" in z else {})


def lookup_index(keys: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Index into ``keys`` (unsorted, unique) for each query, -1 if absent."""
    keys = np.asarray(keys)
    queries = np.asarray(queries, dtype=keys.dtype)
    if len(keys) == 0:
        return np.full(len(queries), -1, np.int64)
    order = np.argsort(keys, kind="stable")
    srt = keys[order]
    j = np.searchsorted(srt, queries)
    jc = np.clip(j, 0, len(srt) - 1)
    ok = srt[jc] == queries
    out = np.where(ok, order[jc], -1).astype(np.int64)
    return out


# ---------------------------------------------------------------------------
# proofread tiers
# ---------------------------------------------------------------------------

def proofread_tiers(df) -> dict[int, int]:
    """Root id -> tier from a ``proofreading_status_and_strategy`` frame.

    Gold: dendrite extended *and* axon fully extended -- the cell is complete.
    Silver: any other row whose dendrite status is set -- verified, but the
    axon may be partial, so absence from the cell is not evidence.
    """
    tiers: dict[int, int] = {}
    valid = df["valid"].astype(str).str.lower().isin(["true", "t", "1"]) \
        if "valid" in df.columns else np.ones(len(df), bool)
    for row, ok in zip(df.itertuples(index=False), np.asarray(valid)):
        if not ok:
            continue
        root = int(row.pt_root_id)
        dend = bool(row.status_dendrite)
        axon = bool(row.status_axon)
        gold = (dend and axon
                and str(row.strategy_dendrite) == "dendrite_extended"
                and str(row.strategy_axon) == "axon_fully_extended")
        tier = TIER_GOLD if gold else (TIER_SILVER if dend else TIER_NONE)
        tiers[root] = max(tiers.get(root, TIER_NONE), tier)
    return tiers


# ---------------------------------------------------------------------------
# per-atom tally
# ---------------------------------------------------------------------------

def tally_atom_targets(atom_of_side: np.ndarray, target_of_side: np.ndarray,
                       *, tiers: Optional[dict[int, int]] = None,
                       min_side_count: int = 2, min_side_frac: float = 0.05,
                       pure_min_owner_frac: float = 0.9,
                       meta: Optional[dict] = None) -> AtomLabels:
    """Aggregate (atom, target root) side counts into :class:`AtomLabels`.

    A target root counts as *robust* support for an atom when it holds at
    least ``min_side_count`` sides and at least ``min_side_frac`` of the atom's
    sides; a single stray synapse must not turn a clean atom into a
    frankenmerge.
    """
    atom_of_side = np.asarray(atom_of_side, np.uint64)
    target_of_side = np.asarray(target_of_side, np.uint64)
    keep = (atom_of_side > 0) & (target_of_side > 0)
    a, t = atom_of_side[keep], target_of_side[keep]

    atoms, atom_inv = np.unique(a, return_inverse=True)
    pair_key = atom_inv.astype(np.int64) * (int(t.max()) + 1 if len(t) else 1)
    # (atom, target) pairs and their counts; a composite key over uint64
    # targets can overflow, so pair through a second unique on the targets.
    _, t_inv = np.unique(t, return_inverse=True)
    n_t = int(t_inv.max()) + 1 if len(t_inv) else 1
    key = atom_inv.astype(np.int64) * n_t + t_inv.astype(np.int64)
    ukey, cnt = np.unique(key, return_counts=True)
    pair_atom = (ukey // n_t).astype(np.int64)
    pair_t_idx = (ukey % n_t).astype(np.int64)
    targets_sorted = np.unique(t)
    pair_target = targets_sorted[pair_t_idx]

    n_atoms = len(atoms)
    n_sides = np.bincount(atom_inv, minlength=n_atoms).astype(np.int32)
    n_roots_raw = np.bincount(pair_atom, minlength=n_atoms).astype(np.int32)

    robust = (cnt >= min_side_count) & (cnt >= min_side_frac * n_sides[pair_atom])
    # keep the dominant root even if the atom is tiny
    order = np.lexsort((-cnt, pair_atom))
    first = np.ones(len(order), bool)
    first[1:] = pair_atom[order][1:] != pair_atom[order][:-1]
    dominant = np.zeros(len(order), bool)
    dominant[order[first]] = True
    robust |= dominant

    n_roots = np.bincount(pair_atom[robust], minlength=n_atoms).astype(np.int32)
    owner = np.zeros(n_atoms, np.uint64)
    owner_cnt = np.zeros(n_atoms, np.int64)
    owner[pair_atom[order[first]]] = pair_target[order[first]]
    owner_cnt[pair_atom[order[first]]] = cnt[order[first]]
    owner_frac = (owner_cnt / np.maximum(n_sides, 1)).astype(np.float32)

    tiers = tiers or {}
    tier_of = np.fromiter((tiers.get(int(r), TIER_NONE) for r in pair_target),
                          np.int8, len(pair_target))
    owner_tier = np.zeros(n_atoms, np.int8)
    owner_tier[pair_atom[order[first]]] = tier_of[order[first]]
    n_roots_pr = np.bincount(pair_atom[robust & (tier_of > 0)],
                             minlength=n_atoms).astype(np.int32)

    m = dict(meta or {})
    m.update({"min_side_count": int(min_side_count),
              "min_side_frac": float(min_side_frac),
              "pure_min_owner_frac": float(pure_min_owner_frac),
              "n_atoms": int(n_atoms), "n_sides": int(len(a))})
    return AtomLabels(atom_id=atoms, owner=owner, owner_frac=owner_frac,
                      owner_tier=owner_tier, n_sides=n_sides,
                      n_roots_raw=n_roots_raw, n_roots=n_roots,
                      n_roots_proofread=n_roots_pr, meta=m)


# ---------------------------------------------------------------------------
# pair labels
# ---------------------------------------------------------------------------

def pair_labels(labels: AtomLabels, atom_a: np.ndarray, atom_b: np.ndarray,
                *, mode: str = "strict") -> np.ndarray:
    """Merge label for atom pairs: 1 same neuron, 0 different, -1 unknown.

    strict (paper ground truth):
      * both atoms pure;
      * positive when they share a *proofread* owner;
      * negative when owners differ and either one is gold, or both are
        proofread. A gold cell is complete, so anything outside it is not it;
        two different verified cells are two different cells.
      * everything else unknown: mixed atoms, unverified owners, an
        unproofread atom next to a silver cell with a partial axon.

    lenient (segmentation-consistent):
      * both atoms pure; same owner -> 1, different owner -> 0, whatever the
        proofread status. Useful for scale, not for claims.
    """
    ia = labels.index_of(np.asarray(atom_a, np.uint64))
    ib = labels.index_of(np.asarray(atom_b, np.uint64))
    out = np.full(len(ia), LABEL_UNKNOWN, np.int8)
    have = (ia >= 0) & (ib >= 0)
    if not have.any():
        return out
    ja, jb = ia[have], ib[have]
    pure = labels.pure
    both_pure = pure[ja] & pure[jb]
    oa, ob = labels.owner[ja], labels.owner[jb]
    ta, tb = labels.owner_tier[ja], labels.owner_tier[jb]
    same = oa == ob
    lab = np.full(len(ja), LABEL_UNKNOWN, np.int8)
    if mode == "lenient":
        lab[both_pure & same] = LABEL_POS
        lab[both_pure & ~same] = LABEL_NEG
    elif mode == "strict":
        pos = both_pure & same & (ta > TIER_NONE)
        neg = both_pure & ~same & ((ta == TIER_GOLD) | (tb == TIER_GOLD)
                                   | ((ta > TIER_NONE) & (tb > TIER_NONE)))
        lab[pos] = LABEL_POS
        lab[neg] = LABEL_NEG
    else:
        raise ValueError(f"unknown mode {mode!r}")
    out[have] = lab
    return out


def summarize(labels: AtomLabels, subset: Optional[np.ndarray] = None) -> dict:
    """Counts a report needs: how much of the population carries GT."""
    m = np.ones(len(labels.atom_id), bool) if subset is None else subset
    pure, mixed = labels.pure & m, labels.mixed & m
    tier = labels.owner_tier
    return {
        "n_atoms": int(m.sum()),
        "n_pure": int(pure.sum()),
        "n_mixed": int(mixed.sum()),
        "n_mixed_proofread": int((labels.mixed_proofread & m).sum()),
        "n_pure_gold": int((pure & (tier == TIER_GOLD)).sum()),
        "n_pure_silver": int((pure & (tier == TIER_SILVER)).sum()),
        "n_pure_unproofread": int((pure & (tier == TIER_NONE)).sum()),
        "n_owner_roots_gold": int(len(np.unique(
            labels.owner[pure & (tier == TIER_GOLD)]))),
        "n_owner_roots_silver": int(len(np.unique(
            labels.owner[pure & (tier == TIER_SILVER)]))),
        "n_owner_roots_proofread": int(len(np.unique(
            labels.owner[pure & (tier > TIER_NONE)]))),
    }
