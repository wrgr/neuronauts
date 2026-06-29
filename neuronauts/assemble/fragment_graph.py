"""Phase 2 global fragment graph assembler.

Nodes = Fragments (with DNA embeddings).
Edges = spatially-proximal endpoints + DNA-cosine compatibility.
No box boundary — fragments from any region share one coordinate frame.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from neuronauts.schemas import Fragment, NeuronHypothesis


# ---------------------------------------------------------------------------
# Edge scoring
# ---------------------------------------------------------------------------

def score_edge(
    frag_a: Fragment,
    frag_b: Fragment,
    *,
    endpoint_radius_nm: float,
) -> float:
    """Score the candidate edge between two fragments in [0, 1].

    Combines two signals:
    - **Endpoint proximity**: 1 - (min_endpoint_gap / endpoint_radius_nm),
      clamped to [0, 1].  Zero if the closest endpoint pair exceeds the radius.
    - **DNA cosine similarity**: (cosine + 1) / 2, mapped to [0, 1].
      Only applied when both fragments have a DNA embedding; otherwise 1.0.

    Final score = proximity × dna_compat.
    """
    # -- endpoint proximity ---------------------------------------------------
    ep_a = frag_a.endpoints_nm  # [Ta, 3]
    ep_b = frag_b.endpoints_nm  # [Tb, 3]

    # pairwise distance between all endpoint pairs
    # [Ta, 1, 3] - [1, Tb, 3] → [Ta, Tb]
    diff = ep_a[:, None, :] - ep_b[None, :, :]
    dist2 = (diff ** 2).sum(axis=2)
    min_dist = float(np.sqrt(dist2.min()))

    if min_dist >= endpoint_radius_nm:
        return 0.0

    proximity = 1.0 - min_dist / endpoint_radius_nm  # (0, 1]

    # -- DNA cosine similarity ------------------------------------------------
    dna_compat = 1.0
    if frag_a.dna is not None and frag_b.dna is not None:
        a = frag_a.dna.astype(np.float64)
        b = frag_b.dna.astype(np.float64)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a > 1e-9 and norm_b > 1e-9:
            cosine = float(np.dot(a, b) / (norm_a * norm_b))
            dna_compat = (cosine + 1.0) / 2.0

    return proximity * dna_compat


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_fragment_graph(
    fragments: list[Fragment],
    *,
    endpoint_radius_nm: float = 5_000.0,
    max_edges_per_fragment: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a global fragment adjacency graph.

    For each fragment, find all other fragments whose closest endpoint pair is
    within ``endpoint_radius_nm``, score the edge, and keep at most
    ``max_edges_per_fragment`` highest-scoring neighbours.

    Edges are undirected and deduplicated (i < j).

    Parameters
    ----------
    fragments:
        Fragments with DNA filled (or None — edges will use proximity only).
    endpoint_radius_nm:
        Maximum endpoint gap for a candidate edge.
    max_edges_per_fragment:
        Degree cap: keep only the top-k edges per fragment by score.

    Returns
    -------
    edge_src : [E] int64 — fragment indices (i ≤ j enforced)
    edge_dst : [E] int64
    edge_score : [E] float32
    """
    n = len(fragments)
    if n == 0:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32))

    # Stack all endpoints with fragment membership for fast radius search.
    all_ep: list[np.ndarray] = []
    ep_frag_idx: list[int] = []
    for fi, frag in enumerate(fragments):
        ep = np.asarray(frag.endpoints_nm, dtype=np.float64)
        all_ep.append(ep)
        ep_frag_idx.extend([fi] * len(ep))

    all_ep_arr = np.concatenate(all_ep, axis=0)  # [P, 3]
    ep_frag_arr = np.array(ep_frag_idx, dtype=np.int64)  # [P]

    # KD-tree query for radius neighbours.
    try:
        from scipy.spatial import KDTree
        tree = KDTree(all_ep_arr)
        pairs_raw = tree.query_pairs(r=endpoint_radius_nm, output_type="ndarray")
        # pairs_raw: [K, 2] — pairs of endpoint indices
    except ImportError:
        # Fallback O(P²) for environments without scipy.
        pairs_raw = _brute_pairs(all_ep_arr, endpoint_radius_nm)

    # Map endpoint pairs → fragment pairs (deduplicate same-fragment hits).
    cand: dict[tuple[int, int], float] = {}
    for pi, pj in pairs_raw:
        fi = int(ep_frag_arr[pi])
        fj = int(ep_frag_arr[pj])
        if fi == fj:
            continue
        key = (min(fi, fj), max(fi, fj))
        if key not in cand:
            cand[key] = score_edge(
                fragments[key[0]], fragments[key[1]],
                endpoint_radius_nm=endpoint_radius_nm,
            )

    # Enforce per-fragment degree cap (keep top-k by score).
    if max_edges_per_fragment is not None and max_edges_per_fragment > 0:
        # Collect neighbour scores per fragment.
        nbr_scores: dict[int, list[tuple[float, tuple[int, int]]]] = defaultdict(list)
        for (fi, fj), sc in cand.items():
            nbr_scores[fi].append((sc, (fi, fj)))
            nbr_scores[fj].append((sc, (fi, fj)))

        # Build the set of retained edges.
        retained: set[tuple[int, int]] = set()
        for fi, nbr_list in nbr_scores.items():
            nbr_list.sort(reverse=True)
            for _, key in nbr_list[:max_edges_per_fragment]:
                retained.add(key)
        cand = {k: v for k, v in cand.items() if k in retained}

    if not cand:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32))

    keys = list(cand.keys())
    edge_src = np.array([k[0] for k in keys], dtype=np.int64)
    edge_dst = np.array([k[1] for k in keys], dtype=np.int64)
    edge_score = np.array([cand[k] for k in keys], dtype=np.float32)

    return edge_src, edge_dst, edge_score


def _brute_pairs(pts: np.ndarray, radius: float) -> np.ndarray:
    """O(P²) fallback for query_pairs when scipy is unavailable."""
    n = len(pts)
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if float(np.linalg.norm(pts[i] - pts[j])) < radius:
                pairs.append((i, j))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(pairs, dtype=np.int64)


# ---------------------------------------------------------------------------
# Clustering: union-find → NeuronHypotheses
# ---------------------------------------------------------------------------

@dataclass
class _UF:
    parent: list[int]
    rank: list[int]

    @classmethod
    def make(cls, n: int) -> "_UF":
        return cls(parent=list(range(n)), rank=[0] * n)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def assemble_fragments(
    fragments: list[Fragment],
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_score: np.ndarray,
    *,
    score_threshold: float = 0.3,
    synapse_region_synapse_indices: list[np.ndarray] | None = None,
) -> list[NeuronHypothesis]:
    """Cluster fragments into NeuronHypotheses via union-find.

    Merges fragment pairs whose ``edge_score >= score_threshold`` (after
    the graph is constructed with :func:`build_fragment_graph`).

    Parameters
    ----------
    fragments:
        Fragment list (same ordering as used to build the graph).
    edge_src, edge_dst:
        Fragment index arrays returned by :func:`build_fragment_graph`.
    edge_score:
        Per-edge scores in [0, 1].
    score_threshold:
        Minimum score to merge two fragments.
    synapse_region_synapse_indices:
        Optional list (one entry per fragment) of global synapse index arrays.
        If provided, NeuronHypothesis.synapse_indices is the union of its
        fragments' synapse_indices.  Otherwise falls back to
        ``Fragment.synapse_indices``.

    Returns
    -------
    List of :class:`~neuronauts.schemas.NeuronHypothesis`, sorted by descending
    fragment count (largest neuron hypothesis first).
    """
    n = len(fragments)
    uf = _UF.make(n)

    for i in range(len(edge_src)):
        if float(edge_score[i]) >= score_threshold:
            uf.union(int(edge_src[i]), int(edge_dst[i]))

    # Group fragment indices by component.
    components: dict[int, list[int]] = defaultdict(list)
    for fi in range(n):
        components[uf.find(fi)].append(fi)

    neuron_id_counter = 1
    hypotheses: list[NeuronHypothesis] = []

    for _, frag_indices in sorted(components.items(),
                                  key=lambda kv: -len(kv[1])):
        frags = [fragments[i] for i in frag_indices]

        # Pool DNA (mean of non-None embeddings).
        dnas = [f.dna for f in frags if f.dna is not None]
        pooled_dna = (
            np.mean(np.stack(dnas, axis=0), axis=0).astype(np.float32)
            if dnas else None
        )

        # Collect global synapse indices.
        if synapse_region_synapse_indices is not None:
            syn_chunks = [synapse_region_synapse_indices[i] for i in frag_indices]
        else:
            syn_chunks = [f.synapse_indices for f in frags]
        syn_indices = (
            np.unique(np.concatenate(syn_chunks)).astype(np.int64)
            if syn_chunks and any(len(c) for c in syn_chunks)
            else np.empty(0, dtype=np.int64)
        )

        # Record which regions the fragments span.
        spans = sorted({f.region_id for f in frags})

        nh = NeuronHypothesis(
            neuron_id=neuron_id_counter,
            fragment_ids=[f.fragment_id for f in frags],
            synapse_indices=syn_indices,
            pooled_dna=pooled_dna,
            spans_regions=spans,
        ).validate()
        hypotheses.append(nh)
        neuron_id_counter += 1

    return hypotheses
