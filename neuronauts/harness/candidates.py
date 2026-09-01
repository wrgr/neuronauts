"""Label-blind candidate merge panel from the contracted endpoint table.

A false split shows up as two atom endpoints facing each other across a gap.
The panel enumerates those geometrically, with no ground truth: every
endpoint that survives a leaf-length / caliber filter is matched to its ``k``
nearest endpoints of *other* atoms within ``radius_nm``. Endpoint pairs are
then reduced to one row per atom pair (the closest endpoint pair), which is
the unit every scorer ranks and every assembler joins.

Per row the panel carries what a geometric scorer needs -- gap, how squarely
the two tips face each other, how well each tip points at the other, tip
caliber and leaf length -- and the two atom ids, so atom-level context
(synapse counts, polarity, cable) can be joined without recomputation.

The filter thresholds are inputs, not facts. The atom topology report shows
that degree-1 L2 nodes are mostly spines, so *some* filter is needed for the
panel to be finite; which one is right is measured against the labelled
positives, not assumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

PAIR_COLS = ["gap_nm", "facing", "align_a", "align_b", "caliber_a", "caliber_b",
             "leaf_len_a", "leaf_len_b"]


@dataclass
class CandidatePanel:
    atom_a: np.ndarray        # [P] uint64, atom_a < atom_b
    atom_b: np.ndarray        # [P] uint64
    ep_a: np.ndarray          # [P] int64 row into the endpoint table
    ep_b: np.ndarray          # [P] int64
    feat: np.ndarray          # [P, len(PAIR_COLS)] float32
    meta: dict

    def __len__(self) -> int:
        return len(self.atom_a)

    def col(self, name: str) -> np.ndarray:
        return self.feat[:, PAIR_COLS.index(name)]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, atom_a=self.atom_a, atom_b=self.atom_b, ep_a=self.ep_a,
            ep_b=self.ep_b, feat=self.feat,
            meta=np.frombuffer(json.dumps(self.meta).encode(), np.uint8))

    def subset(self, mask: np.ndarray) -> "CandidatePanel":
        return CandidatePanel(self.atom_a[mask], self.atom_b[mask],
                              self.ep_a[mask], self.ep_b[mask],
                              self.feat[mask], dict(self.meta))


def load_panel(path: str | Path) -> CandidatePanel:
    with np.load(Path(path), allow_pickle=False) as z:
        return CandidatePanel(
            atom_a=z["atom_a"], atom_b=z["atom_b"], ep_a=z["ep_a"],
            ep_b=z["ep_b"], feat=z["feat"],
            meta=json.loads(bytes(z["meta"]).decode()) if "meta" in z else {})


def endpoint_pair_features(pos_a, tan_a, pos_b, tan_b) -> tuple[np.ndarray, ...]:
    """Gap and the three cosines for endpoint pairs.

    ``facing``  = -t_a . t_b   (1 when the tips point at each other)
    ``align_a`` =  t_a . u      (1 when a's tip points at b), u = (p_b-p_a)/|.|
    ``align_b`` = -t_b . u      (1 when b's tip points at a)
    """
    d = np.asarray(pos_b, np.float64) - np.asarray(pos_a, np.float64)
    gap = np.linalg.norm(d, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = d / gap[:, None]
    u[~np.isfinite(u).all(axis=1)] = 0.0
    ta = np.asarray(tan_a, np.float64)
    tb = np.asarray(tan_b, np.float64)
    facing = -(ta * tb).sum(axis=1)
    align_a = (ta * u).sum(axis=1)
    align_b = -(tb * u).sum(axis=1)
    return gap, facing, align_a, align_b


def build_candidate_panel(ep_atom: np.ndarray, ep_pos: np.ndarray,
                          ep_tan: np.ndarray, ep_leaf_len: np.ndarray,
                          ep_caliber: np.ndarray, *,
                          min_leaf_nm: float = 1000.0,
                          min_caliber_nm: float = 30.0,
                          radius_nm: float = 5000.0, k: int = 8,
                          atom_subset: Optional[np.ndarray] = None,
                          meta: Optional[dict] = None) -> CandidatePanel:
    """Enumerate cross-atom endpoint pairs and reduce them to atom pairs."""
    from scipy.spatial import cKDTree

    ep_atom = np.asarray(ep_atom, np.uint64)
    n_ep = len(ep_atom)
    ok = (np.isfinite(ep_pos).all(axis=1) & np.isfinite(ep_tan).all(axis=1)
          & (ep_leaf_len >= min_leaf_nm) & (ep_caliber >= min_caliber_nm))
    if atom_subset is not None:
        ok &= np.isin(ep_atom, np.asarray(atom_subset, np.uint64))
    idx = np.flatnonzero(ok)
    empty = CandidatePanel(np.zeros(0, np.uint64), np.zeros(0, np.uint64),
                           np.zeros(0, np.int64), np.zeros(0, np.int64),
                           np.zeros((0, len(PAIR_COLS)), np.float32), {})
    if len(idx) < 2:
        empty.meta = _panel_meta(min_leaf_nm, min_caliber_nm, radius_nm, k,
                                 n_ep, len(idx), 0, 0, meta)
        return empty

    pos = ep_pos[idx].astype(np.float64)
    tree = cKDTree(pos)
    kk = min(k + 1, len(idx))
    dist, nbr = tree.query(pos, k=kk, distance_upper_bound=radius_nm)
    dist = dist[:, 1:] if kk > 1 else dist[:, :0]
    nbr = nbr[:, 1:] if kk > 1 else nbr[:, :0]
    src = np.repeat(np.arange(len(idx)), nbr.shape[1])
    dst = nbr.reshape(-1)
    hit = np.isfinite(dist.reshape(-1)) & (dst < len(idx))
    src, dst = src[hit], dst[hit]
    a_atom, b_atom = ep_atom[idx[src]], ep_atom[idx[dst]]
    cross = a_atom != b_atom
    src, dst = src[cross], dst[cross]
    n_ep_pairs = int(len(src))
    if n_ep_pairs == 0:
        empty.meta = _panel_meta(min_leaf_nm, min_caliber_nm, radius_nm, k,
                                 n_ep, len(idx), 0, 0, meta)
        return empty

    # orient every endpoint pair so atom_a < atom_b, then keep the closest
    # endpoint pair per atom pair
    ea, eb = idx[src], idx[dst]
    swap = ep_atom[ea] > ep_atom[eb]
    ea, eb = np.where(swap, eb, ea), np.where(swap, ea, eb)
    gap = np.linalg.norm(ep_pos[ea].astype(np.float64)
                         - ep_pos[eb].astype(np.float64), axis=1)
    key = np.stack([ep_atom[ea], ep_atom[eb]], axis=1)
    order = np.lexsort((gap, key[:, 1], key[:, 0]))
    key_o = key[order]
    first = np.ones(len(order), bool)
    first[1:] = np.any(key_o[1:] != key_o[:-1], axis=1)
    sel = order[first]
    ea, eb = ea[sel], eb[sel]

    gap, facing, al_a, al_b = endpoint_pair_features(
        ep_pos[ea], ep_tan[ea], ep_pos[eb], ep_tan[eb])
    feat = np.stack([gap, facing, al_a, al_b, ep_caliber[ea], ep_caliber[eb],
                     ep_leaf_len[ea], ep_leaf_len[eb]], axis=1).astype(np.float32)
    return CandidatePanel(
        atom_a=ep_atom[ea], atom_b=ep_atom[eb],
        ep_a=ea.astype(np.int64), ep_b=eb.astype(np.int64), feat=feat,
        meta=_panel_meta(min_leaf_nm, min_caliber_nm, radius_nm, k, n_ep,
                         len(idx), n_ep_pairs, len(ea), meta))


def _panel_meta(min_leaf_nm, min_caliber_nm, radius_nm, k, n_ep, n_kept,
                n_ep_pairs, n_pairs, extra) -> dict:
    m = dict(extra or {})
    m.update({"min_leaf_nm": float(min_leaf_nm),
              "min_caliber_nm": float(min_caliber_nm),
              "radius_nm": float(radius_nm), "k": int(k),
              "n_endpoints": int(n_ep), "n_endpoints_kept": int(n_kept),
              "n_endpoint_pairs": int(n_ep_pairs), "n_atom_pairs": int(n_pairs),
              "selection": "label-blind: k nearest cross-atom endpoints within "
                           "radius, closest endpoint pair per atom pair"})
    return m
