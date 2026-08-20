"""Level-1 seam stitching: assemble per-tile partitions into global neurons.

This implements the first coarsening level of the hierarchical tree-assembly
design (``docs/tree_assembly_algorithm.md``).  The unit of computation is the
**super-fragment**: one predicted cluster from one tile's partition, carrying

  - its member atoms (v117 roots / fragment ids),
  - a merged tree skeleton with endpoints (the stitch handles),
  - pooled DNA (observation-count-weighted mean of member embeddings),
  - observation keys (stable ids shared across tiles via halo overlap),
  - a soma flag and observation count (for cannot-link constraints).

Two evidence channels connect super-fragments across tiles:

1. **Shared observations** (halo overlap) — two tiles that both processed the
   same observation and clustered it agree on identity.  Mutual-best-overlap
   pairs are *forced* merges: no model involved.
2. **Candidate stitch edges** — cross-tile endpoint pairs within a radius,
   scored by endpoint proximity × pooled-DNA compatibility (the geometry-only
   scorer; a learned scorer can replace ``score`` per candidate later).

Inference is a **constrained maximum-weight forest** (Kruskal): candidates are
accepted in score order subject to (a) union-find cycle rejection, (b) each
skeleton endpoint used at most once, (c) cannot-link constraints — at most one
soma per assembled neuron, optional observation-count cap.  Merging is
monotone: under-merge is recoverable at the next level, over-merge is not, so
every constraint errs conservative.

Everything here is numpy-only — no torch dependency.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from neuronauts.schemas import Fragment
from treestitch.assemble import merge_fragment_skeletons


# ---------------------------------------------------------------------------
# Super-fragment
# ---------------------------------------------------------------------------

@dataclass
class SuperFragment:
    """One predicted cluster from one tile, summarised for seam stitching."""

    tile_id: str
    cluster_id: int                 # tile-local cluster label
    atom_ids: frozenset             # member fragment ids (v117 roots)
    skeleton: Fragment              # merged member skeletons (tree/forest)
    dna: np.ndarray | None          # pooled member embedding, L2-normalised
    n_obs: int                      # observations in this cluster
    obs_keys: np.ndarray            # [n_obs] stable observation ids
    has_soma: bool = False
    majority_label: int = 0         # ground-truth majority object id (eval only)


def build_super_fragments(
    tile_id: str,
    fragments: list[Fragment],
    pred_labels: np.ndarray,
    fragment_id_per_obs: np.ndarray,
    obs_keys: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    soma_atoms: set[int] | None = None,
    stitch_radius_nm: float = 5_000.0,
) -> list[SuperFragment]:
    """Summarise one tile's partition as a list of SuperFragments.

    Parameters
    ----------
    fragments:
        The tile's Fragment objects (dna optional).  Looked up by
        ``base_root_id``.
    pred_labels:
        [N] tile-local cluster id per observation.  Negative = abstained
        (skipped).
    fragment_id_per_obs:
        [N] atom (fragment) id per observation — ``graph.fragment_id``.
    obs_keys:
        [N] stable observation ids (e.g. CAVE synapse ids, or global indices).
        These are the halo join keys: the same physical observation must carry
        the same key in every tile that sees it.
    labels:
        [N] ground-truth object ids, used only to fill ``majority_label`` for
        evaluation.  ``None`` leaves majority_label = 0.
    soma_atoms:
        Atom ids known to contain a soma (e.g. from the nucleus table).
    stitch_radius_nm:
        Forwarded to ``merge_fragment_skeletons`` for the within-cluster merge.
    """
    pred_labels = np.asarray(pred_labels, dtype=np.int64)
    fragment_id_per_obs = np.asarray(fragment_id_per_obs, dtype=np.int64)
    obs_keys = np.asarray(obs_keys, dtype=np.int64)
    root_to_frag = {int(f.base_root_id): f for f in fragments}
    soma_atoms = soma_atoms or set()

    supers: list[SuperFragment] = []
    for cid in np.unique(pred_labels):
        if cid < 0:
            continue
        mask = pred_labels == cid
        atoms = frozenset(int(a) for a in np.unique(fragment_id_per_obs[mask]))
        member_frags = [root_to_frag[a] for a in atoms if a in root_to_frag]
        if not member_frags:
            continue
        skeleton = merge_fragment_skeletons(
            member_frags, stitch_radius_nm=stitch_radius_nm)

        # Pooled DNA: observation-count-weighted mean of member embeddings.
        dna = None
        weighted: list[np.ndarray] = []
        for f in member_frags:
            if f.dna is None:
                continue
            w = int((fragment_id_per_obs[mask] == f.base_root_id).sum())
            weighted.append(np.asarray(f.dna, dtype=np.float64) * max(w, 1))
        if weighted:
            pooled = np.sum(weighted, axis=0)
            norm = np.linalg.norm(pooled)
            if norm > 1e-9:
                dna = (pooled / norm).astype(np.float32)

        majority = 0
        if labels is not None:
            lab = np.asarray(labels, dtype=np.int64)[mask]
            lab = lab[lab != 0]
            if len(lab):
                vals, counts = np.unique(lab, return_counts=True)
                majority = int(vals[np.argmax(counts)])

        supers.append(SuperFragment(
            tile_id=tile_id,
            cluster_id=int(cid),
            atom_ids=atoms,
            skeleton=skeleton,
            dna=dna,
            n_obs=int(mask.sum()),
            obs_keys=obs_keys[mask].copy(),
            has_soma=any(a in soma_atoms for a in atoms),
            majority_label=majority,
        ))
    return supers


# ---------------------------------------------------------------------------
# Forced merges: halo-overlap identity
# ---------------------------------------------------------------------------

def link_shared_observations(
    supers: list[SuperFragment],
    *,
    min_shared: int = 3,
    mutual_best: bool = True,
) -> list[tuple[int, int]]:
    """Cross-tile super-fragment pairs identified by shared observations.

    Two super-fragments from *different* tiles that both contain the same
    observation keys (halo overlap) are the same emerging object — provided the
    overlap is substantial and unambiguous.  With ``mutual_best=True`` a pair
    is linked only when each side's largest cross-tile overlap is the other
    (this keeps a frankenmerge that two tiles split differently from gluing the
    two halves back together through a sliver of shared observations).

    Returns undirected index pairs ``(i, j)`` with ``i < j``.
    """
    key_owner: dict[int, list[int]] = defaultdict(list)
    for si, s in enumerate(supers):
        for k in s.obs_keys.tolist():
            key_owner[int(k)].append(si)
    return _link_by_shared_keys(supers, key_owner, min_shared=min_shared,
                                mutual_best=mutual_best)


def link_shared_atoms(
    supers: list[SuperFragment],
    *,
    min_shared: int = 1,
    mutual_best: bool = True,
) -> list[tuple[int, int]]:
    """Cross-tile super-fragment pairs that share a member atom.

    An atom (v117 root / fragment) that straddles a tile seam appears in both
    tiles' fragment sets, so the two clusters that own it are the same object —
    the second exact halo-identity channel (works even when the tiles share no
    observations, e.g. after independent subsampling or with zero halo).

    Caveat: a *frankenmerge* atom spans two true objects, so an atom link can
    glue the two halves back together when the tiles split it differently.
    ``mutual_best=True`` (each side's largest shared-atom partner must be the
    other) limits the damage; downstream, this channel should be treated like
    a forced merge only when frankenmerge pressure is low, or fed through the
    cannot-link constraints instead.
    """
    key_owner: dict[int, list[int]] = defaultdict(list)
    for si, s in enumerate(supers):
        for a in s.atom_ids:
            key_owner[int(a)].append(si)
    return _link_by_shared_keys(supers, key_owner, min_shared=min_shared,
                                mutual_best=mutual_best)


def _link_by_shared_keys(
    supers: list[SuperFragment],
    key_owner: dict[int, list[int]],
    *,
    min_shared: int,
    mutual_best: bool,
) -> list[tuple[int, int]]:
    overlap: dict[tuple[int, int], int] = defaultdict(int)
    for owners in key_owner.values():
        if len(owners) < 2:
            continue
        for a in range(len(owners)):
            for b in range(a + 1, len(owners)):
                i, j = owners[a], owners[b]
                if supers[i].tile_id == supers[j].tile_id:
                    continue
                overlap[(min(i, j), max(i, j))] += 1

    pairs = [(p, n) for p, n in overlap.items() if n >= min_shared]
    if not mutual_best:
        return [p for p, _ in pairs]

    # best cross-tile partner per (super, other-tile)
    best: dict[tuple[int, str], tuple[int, int]] = {}
    for (i, j), n in pairs:
        for a, b in ((i, j), (j, i)):
            key = (a, supers[b].tile_id)
            if key not in best or n > best[key][1]:
                best[key] = (b, n)

    linked: list[tuple[int, int]] = []
    for (i, j), n in pairs:
        if (best.get((i, supers[j].tile_id), (None,))[0] == j
                and best.get((j, supers[i].tile_id), (None,))[0] == i):
            linked.append((i, j))
    return linked


# ---------------------------------------------------------------------------
# Candidate stitch edges: cross-tile endpoint pairs
# ---------------------------------------------------------------------------

@dataclass
class StitchCandidate:
    score: float
    i: int                 # super-fragment index
    j: int
    ep_i: int              # endpoint index within supers[i].skeleton.endpoints_nm
    ep_j: int
    gap_nm: float
    dna_cos: float


def candidate_stitch_edges(
    supers: list[SuperFragment],
    *,
    endpoint_radius_nm: float = 10_000.0,
    max_edges_per_super: int = 8,
) -> list[StitchCandidate]:
    """Cross-tile endpoint pairs within radius, scored like ``score_edge``.

    score = (1 − gap/radius) × (dna_cosine + 1)/2, with the DNA factor = 1
    when either side has no embedding.  One candidate per super pair (the
    closest endpoint pair), degree-capped per super by score.
    """
    if not supers:
        return []

    ep_pts: list[np.ndarray] = []
    ep_super: list[int] = []
    ep_local: list[int] = []
    for si, s in enumerate(supers):
        eps = np.asarray(s.skeleton.endpoints_nm, dtype=np.float64)
        for li in range(len(eps)):
            ep_pts.append(eps[li])
            ep_super.append(si)
            ep_local.append(li)
    if not ep_pts:
        return []
    ep_arr = np.stack(ep_pts, axis=0)

    try:
        from scipy.spatial import KDTree
        pairs = KDTree(ep_arr).query_pairs(r=endpoint_radius_nm,
                                           output_type="ndarray")
    except ImportError:
        pairs = _brute_pairs(ep_arr, endpoint_radius_nm)

    # closest endpoint pair per super pair
    best: dict[tuple[int, int], tuple[float, int, int]] = {}
    for pi, pj in pairs:
        si, sj = ep_super[pi], ep_super[pj]
        if si == sj or supers[si].tile_id == supers[sj].tile_id:
            continue
        key = (min(si, sj), max(si, sj))
        if key != (si, sj):
            pi, pj = pj, pi
        gap = float(np.linalg.norm(ep_arr[pi] - ep_arr[pj]))
        if key not in best or gap < best[key][0]:
            best[key] = (gap, ep_local[pi], ep_local[pj])

    cands: list[StitchCandidate] = []
    for (si, sj), (gap, li, lj) in best.items():
        proximity = max(0.0, 1.0 - gap / endpoint_radius_nm)
        dna_cos = 0.0
        dna_compat = 1.0
        a, b = supers[si].dna, supers[sj].dna
        if a is not None and b is not None:
            dna_cos = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
            dna_compat = (dna_cos + 1.0) / 2.0
        cands.append(StitchCandidate(
            score=proximity * dna_compat, i=si, j=sj,
            ep_i=li, ep_j=lj, gap_nm=gap, dna_cos=dna_cos))

    if max_edges_per_super and max_edges_per_super > 0:
        # hard per-super degree bound: greedy by score, both sides must have
        # capacity left (keeps the Kruskal candidate list O(S · cap))
        degree: dict[int, int] = defaultdict(int)
        retained: list[StitchCandidate] = []
        for c in sorted(cands, key=lambda c: -c.score):
            if degree[c.i] < max_edges_per_super and degree[c.j] < max_edges_per_super:
                retained.append(c)
                degree[c.i] += 1
                degree[c.j] += 1
        cands = retained
    return cands


def _brute_pairs(pts: np.ndarray, radius: float) -> np.ndarray:
    n = len(pts)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if float(np.linalg.norm(pts[i] - pts[j])) <= radius:
                out.append([i, j])
    return (np.array(out, dtype=np.int64).reshape(-1, 2)
            if out else np.empty((0, 2), dtype=np.int64))


# ---------------------------------------------------------------------------
# Constrained maximum-weight forest (Kruskal)
# ---------------------------------------------------------------------------

class _UF:
    def __init__(self, n: int) -> None:
        self._p = list(range(n))

    def find(self, x: int) -> int:
        while self._p[x] != x:
            self._p[x] = self._p[self._p[x]]
            x = self._p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self._p[ra] = rb
        return True


@dataclass
class StitchResult:
    super_cluster: np.ndarray            # [S] global cluster id per super
    accepted: list[StitchCandidate]
    forced_pairs: list[tuple[int, int]]
    rejected: dict = field(default_factory=dict)  # reason → count
    soma_conflicts: list[tuple[int, int]] = field(default_factory=list)


def stitch_super_fragments(
    supers: list[SuperFragment],
    *,
    endpoint_radius_nm: float = 10_000.0,
    min_score: float = 0.05,
    forced_pairs: list[tuple[int, int]] | None = None,
    candidates: list[StitchCandidate] | None = None,
    enforce_single_soma: bool = True,
    max_obs_per_cluster: int = 0,
    max_edges_per_super: int = 8,
) -> StitchResult:
    """Assemble super-fragments into global clusters.

    Order of evidence:

    1. ``forced_pairs`` (default: ``link_shared_observations``) are applied
       unconditionally — shared-observation identity is exact evidence.  A
       forced merge that joins two somata is recorded in ``soma_conflicts``
       (it flags a level-0 over-merge for review) but still applied.
    2. Candidate stitch edges (default: ``candidate_stitch_edges``) are
       accepted greedily by score, subject to: score ≥ ``min_score``, the two
       sides are in different clusters (cycle rejection), each skeleton
       endpoint is used at most once, the merged cluster keeps ≤ 1 soma
       (``enforce_single_soma``) and ≤ ``max_obs_per_cluster`` observations
       (0 disables the cap).

    Rejection counts per reason are returned for diagnostics.
    """
    S = len(supers)
    uf = _UF(S)
    soma_count = [1 if s.has_soma else 0 for s in supers]
    obs_count = [s.n_obs for s in supers]
    rejected: dict[str, int] = defaultdict(int)
    soma_conflicts: list[tuple[int, int]] = []

    if forced_pairs is None:
        forced_pairs = link_shared_observations(supers)
    for i, j in forced_pairs:
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        if soma_count[ri] + soma_count[rj] > 1:
            soma_conflicts.append((i, j))
        uf.union(ri, rj)
        r = uf.find(ri)
        soma_count[r] = soma_count[ri] + soma_count[rj]
        obs_count[r] = obs_count[ri] + obs_count[rj]

    if candidates is None:
        candidates = candidate_stitch_edges(
            supers, endpoint_radius_nm=endpoint_radius_nm,
            max_edges_per_super=max_edges_per_super)

    used_endpoints: set[tuple[int, int]] = set()
    accepted: list[StitchCandidate] = []
    for c in sorted(candidates, key=lambda c: -c.score):
        if c.score < min_score:
            rejected["below_min_score"] += 1
            continue
        ri, rj = uf.find(c.i), uf.find(c.j)
        if ri == rj:
            rejected["cycle"] += 1
            continue
        if (c.i, c.ep_i) in used_endpoints or (c.j, c.ep_j) in used_endpoints:
            rejected["endpoint_used"] += 1
            continue
        if enforce_single_soma and soma_count[ri] + soma_count[rj] > 1:
            rejected["soma_cannot_link"] += 1
            continue
        if max_obs_per_cluster and obs_count[ri] + obs_count[rj] > max_obs_per_cluster:
            rejected["obs_cap"] += 1
            continue
        uf.union(ri, rj)
        r = uf.find(ri)
        soma_count[r] = soma_count[ri] + soma_count[rj]
        obs_count[r] = obs_count[ri] + obs_count[rj]
        used_endpoints.add((c.i, c.ep_i))
        used_endpoints.add((c.j, c.ep_j))
        accepted.append(c)

    # canonical contiguous cluster ids
    canon: dict[int, int] = {}
    out = np.empty(S, dtype=np.int64)
    for si in range(S):
        r = uf.find(si)
        if r not in canon:
            canon[r] = len(canon)
        out[si] = canon[r]

    return StitchResult(
        super_cluster=out,
        accepted=accepted,
        forced_pairs=list(forced_pairs),
        rejected=dict(rejected),
        soma_conflicts=soma_conflicts,
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def pairwise_merge_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    ignore_pred: int = -1,
    ignore_true: int = 0,
) -> dict:
    """Pairwise co-membership precision/recall from the contingency table.

    A pair of observations is a *predicted merge* when both share a predicted
    cluster; a *true merge* when both share a ground-truth object.  Computed in
    closed form (Σ C(n,2) over contingency cells) — no pair sampling.
    """
    pred = np.asarray(pred, dtype=np.int64)
    true = np.asarray(true, dtype=np.int64)
    keep = (pred != ignore_pred) & (true != ignore_true)
    pred, true = pred[keep], true[keep]
    if len(pred) == 0:
        return {"merge_precision": float("nan"), "merge_recall": float("nan"),
                "merge_f1": float("nan"), "n_obs": 0}

    def _pairs(x: np.ndarray) -> float:
        _, c = np.unique(x, return_counts=True)
        return float((c * (c - 1) // 2).sum())

    # joint cells via a combined key — remap to compact indices first so the
    # multiplication cannot overflow int64 (real root ids are ~1e18)
    _, pred_c = np.unique(pred, return_inverse=True)
    _, true_c = np.unique(true, return_inverse=True)
    pred_c = pred_c.astype(np.int64)
    true_c = true_c.astype(np.int64)
    joint = pred_c * (int(true_c.max()) + 1) + true_c
    tp = _pairs(joint)
    pred_pos = _pairs(pred)
    true_pos = _pairs(true)
    prec = tp / pred_pos if pred_pos > 0 else float("nan")
    rec = tp / true_pos if true_pos > 0 else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if prec == prec and rec == rec and (prec + rec) > 0 else float("nan"))
    return {"merge_precision": prec, "merge_recall": rec, "merge_f1": f1,
            "n_obs": int(len(pred))}


def stitch_edge_precision(
    supers: list[SuperFragment],
    accepted: list[StitchCandidate],
) -> dict:
    """Fraction of accepted stitch edges whose sides share a majority label.

    Requires ``majority_label`` filled at super-fragment build time (labels
    passed in).  Edges touching an unlabelled side (majority 0) are skipped.
    """
    n_ok = n_bad = 0
    for c in accepted:
        a, b = supers[c.i].majority_label, supers[c.j].majority_label
        if a == 0 or b == 0:
            continue
        if a == b:
            n_ok += 1
        else:
            n_bad += 1
    total = n_ok + n_bad
    return {"stitch_precision": (n_ok / total if total else float("nan")),
            "n_correct": n_ok, "n_wrong": n_bad, "n_scored": total}


__all__ = [
    "SuperFragment",
    "StitchCandidate",
    "StitchResult",
    "build_super_fragments",
    "link_shared_observations",
    "link_shared_atoms",
    "candidate_stitch_edges",
    "stitch_super_fragments",
    "pairwise_merge_metrics",
    "stitch_edge_precision",
]
