"""Learned skeleton-synapse grammar for neuron merge detection.

Architecture
------------
The grammar has two levels:

  Level 1 — inter-synapse path encoder:
    For each consecutive synapse pair (ordered by skeleton path distance from
    soma), extract the skeleton path between them and encode it with a small
    Transformer (PathEdgeEncoder).  Output: one 16-dim token per synapse gap.

  Level 2 — partition encoder:
    The sequence of level-1 tokens is encoded by a second Transformer
    (TorchPathEncoder).  Output: one 32-dim embedding per half-partition.

  Scorer:
    MLP over concat(emb_A, emb_B, |emb_A - emb_B|, log_dist) -> merge prob.

Why this matters
----------------
Hard-coded F/B/L/R bigrams fix the vocabulary and the feature type before
seeing any data.  This model learns the vocabulary and the temporal pattern
jointly from the v117->v1718 merge supervision.

Skeletons are cheap (cached) and provide the morphological ordering and path
geometry that synapse positions alone cannot.  Each inter-synapse path through
the skeleton is a richer token than a single PCA-aligned direction vector.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Skeleton path helpers  (pure numpy, no torch required here)
# ---------------------------------------------------------------------------

def _build_adj(edges: np.ndarray, n: int) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for row in edges.tolist():
        u, v = int(row[0]), int(row[1])
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _bfs_from_root(adj: list, root: int) -> tuple[list[int], list[int]]:
    """BFS from root; returns (dist_array, parent_array)."""
    n = len(adj)
    dist = [-1] * n
    parent = [-1] * n
    dist[root] = 0
    q = deque([root])
    while q:
        v = q.popleft()
        for u in adj[v]:
            if dist[u] == -1:
                dist[u] = dist[v] + 1
                parent[u] = v
                q.append(u)
    return dist, parent


def _tree_path(adj: list, v1: int, v2: int) -> list[int]:
    """Exact path between v1 and v2 in a tree via BFS."""
    if v1 == v2:
        return [v1]
    par: dict[int, int] = {v1: -1}
    q = deque([v1])
    found = False
    while q and not found:
        v = q.popleft()
        for u in adj[v]:
            if u not in par:
                par[u] = v
                if u == v2:
                    found = True
                    break
                q.append(u)
    if not found:
        return [v1, v2]   # disconnected component — fallback
    path = [v2]
    while path[-1] != v1:
        path.append(par[path[-1]])
    return path[::-1]


def _find_soma(verts: np.ndarray, radius) -> int:
    if radius is not None and len(radius) == len(verts) and float(radius.max()) > 0:
        return int(np.argmax(radius))
    return int(np.argmin(np.linalg.norm(verts - verts.mean(axis=0), axis=1)))


def _label_components(adj: list, n: int) -> list[int]:
    """Assign each vertex a component ID (0 = largest component)."""
    comp = [-1] * n
    cid = 0
    comp_sizes: list[tuple[int, int]] = []
    for start in range(n):
        if comp[start] != -1:
            continue
        q = deque([start])
        comp[start] = cid
        size = 0
        while q:
            v = q.popleft()
            size += 1
            for u in adj[v]:
                if comp[u] == -1:
                    comp[u] = cid
                    q.append(u)
        comp_sizes.append((cid, size))
        cid += 1
    # Remap so that component 0 is the largest (most likely the main axon/dendrite trunk)
    if len(comp_sizes) > 1:
        largest = max(comp_sizes, key=lambda x: x[1])[0]
        remap = {largest: 0}
        next_id = 1
        for c, _ in sorted(comp_sizes, key=lambda x: -x[1]):
            if c not in remap:
                remap[c] = next_id
                next_id += 1
        comp = [remap[c] for c in comp]
    return comp


def _fragment_neuron(
    skel_verts: np.ndarray,
    skel_edges: np.ndarray,
    syn_pts: np.ndarray,
    *,
    min_chunk: int,
    max_pieces: int,
    rng,
    mode: str = "realistic",
    radius=None,
    gap_nm: float = 1500.0,
    thin_bias: float = 2.0,
):
    """Realistic over-segmentation of a neuron into fragments, with break gaps.

    Returns ``(fragments, adj_pairs)``:
      fragments : list of synapse-index arrays, each with >= min_chunk synapses
      adj_pairs : list of [i, j] index pairs into ``fragments`` that were directly
                  connected across a single break (the true "reconnect-the-break"
                  merge candidates).

    Break model (mode="realistic"): skeleton edges are cut with probability
    weighted toward **thin** necks (small radius, ``(1/r)**thin_bias``) and
    **branch** points (degree >= 3) — the places real segmentation actually
    fragments (thin axons, spine necks, branches). At each break a short
    ``gap_nm`` of skeleton around the break is removed and its synapses dropped,
    simulating a lost connector piece: the two true fragments are then NOT
    perfectly adjacent, so closest-point distance is informative-but-not-decisive.
    mode="branch"/"axis" keep the simpler cut rules.
    """
    from scipy.spatial import cKDTree
    from .pcfg_partitions import _random_piece_sizes

    n_v = len(skel_verts)
    n_syn = len(syn_pts)
    if n_syn < 2 * min_chunk or n_v < 2:
        return [], []
    adj = _build_adj(skel_edges, n_v)
    deg = [len(a) for a in adj]
    nearest = cKDTree(skel_verts).query(syn_pts, k=1)[1].tolist()
    edges = [(int(u), int(v)) for u, v in skel_edges.tolist()]
    if not edges:
        whole = np.arange(n_syn)
        return ([whole], []) if n_syn >= min_chunk else ([], [])

    # ----- axis mode: contiguous pieces along path order; consecutive=adjacent --
    if mode == "axis":
        soma = _find_soma(skel_verts, radius)
        dist, _ = _bfs_from_root(adj, soma)
        big = 10 ** 9
        order = sorted(range(n_syn), key=lambda i: dist[nearest[i]] if dist[nearest[i]] >= 0 else big)
        sizes = _random_piece_sizes(n_syn, min_chunk, max_pieces, rng)
        if sizes is None:
            return [], []
        rad_arr = np.asarray(radius) if (radius is not None and len(radius) == n_v) else None
        frags, start = [], 0
        for sz in sizes:
            sidx = np.array(order[start:start + sz], dtype=np.int64)
            start += sz
            # axis pieces don't sever the skeleton cleanly; use the full skeleton
            frags.append((sidx, skel_verts, skel_edges, rad_arr))
        adj_pairs = [[i, i + 1] for i in range(len(frags) - 1)]
        return frags, adj_pairs

    # ----- branch / realistic: cut edges, then group by component ---------------
    def _edge_weight(u, v):
        w = 1.0
        if mode == "realistic" and radius is not None and len(radius) == n_v:
            r = 0.5 * (float(radius[u]) + float(radius[v])) + 1e-3
            w = (1.0 / r) ** thin_bias            # thin necks break preferentially
        if deg[u] >= 3 or deg[v] >= 3:
            w *= 3.0                               # branches also break
        return w

    if mode == "branch":
        cand = [e for e in edges if deg[e[0]] >= 3 or deg[e[1]] >= 3] or edges
    else:  # realistic
        cand = edges
    w = np.array([_edge_weight(u, v) for (u, v) in cand], dtype=float)
    w = w / w.sum() if w.sum() > 0 else None
    n_cuts = min(max_pieces - 1, len(cand))
    if n_cuts < 1:
        return [], []
    chosen = rng.choice(len(cand), size=n_cuts, replace=False, p=w)
    cut = {(min(*cand[c]), max(*cand[c])) for c in chosen}

    # component labels after removing the cut edges
    adj2: list[list[int]] = [[] for _ in range(n_v)]
    for u, v in edges:
        if (min(u, v), max(u, v)) in cut:
            continue
        adj2[u].append(v)
        adj2[v].append(u)
    comp = _label_components(adj2, n_v)

    # gap: drop synapses within gap_nm of a break location (the lost connector).
    # A small physical gap (~1-2um), not a hop neighborhood — keeps both fragments
    # mostly intact while making them no-longer-touching at the break.
    dropped: set = set()
    if gap_nm > 0 and cut:
        cut_pts = np.array([0.5 * (skel_verts[u] + skel_verts[v]) for (u, v) in cut],
                           dtype=np.float64)
        dmin, _ = cKDTree(cut_pts).query(np.asarray(syn_pts, dtype=np.float64), k=1)
        dropped = {int(i) for i in np.flatnonzero(dmin < gap_nm)}

    groups: dict = {}
    for si, vtx in enumerate(nearest):
        if si in dropped:
            continue                               # lost in the break gap
        groups.setdefault(comp[vtx], []).append(si)

    # vertices per component, for building each fragment's SEVERED sub-skeleton
    comp_verts: dict = {}
    for v in range(n_v):
        comp_verts.setdefault(comp[v], []).append(v)
    has_rad = radius is not None and len(radius) == n_v
    rad_arr = np.asarray(radius) if has_rad else None

    comp_to_frag: dict = {}
    fragments: list = []   # each: (syn_idx, sub_verts, sub_edges, sub_radius)
    for c, sidx in groups.items():
        if len(sidx) < min_chunk:
            continue
        vids = comp_verts[c]
        remap = {old: i for i, old in enumerate(vids)}
        sub_verts = skel_verts[vids]
        sub_edges = np.array(
            [[remap[u], remap[v]] for (u, v) in edges if comp[u] == c and comp[v] == c],
            dtype=np.int64,
        )
        if sub_edges.size == 0:
            sub_edges = np.zeros((0, 2), dtype=np.int64)
        sub_rad = rad_arr[vids] if has_rad else None
        comp_to_frag[c] = len(fragments)
        fragments.append((np.array(sidx, dtype=np.int64), sub_verts, sub_edges, sub_rad))

    adj_pairs_set: set = set()
    for (u, v) in cut:
        cu, cv = comp[u], comp[v]
        if cu != cv and cu in comp_to_frag and cv in comp_to_frag:
            a, b = comp_to_frag[cu], comp_to_frag[cv]
            adj_pairs_set.add((min(a, b), max(a, b)))
    return fragments, [list(p) for p in adj_pairs_set]


def _split_synapses_into_fragments(
    skel_verts: np.ndarray,
    skel_edges: np.ndarray,
    syn_pts: np.ndarray,
    *,
    min_chunk: int,
    max_pieces: int,
    rng,
    mode: str = "branch",
    radius=None,
) -> list[np.ndarray]:
    """Over-segment a neuron into fragments (lists of synapse indices), each with
    >= ``min_chunk`` synapses; fragments below that floor are dropped (low-degree).

    mode="branch": cut up to ``max_pieces - 1`` skeleton edges that are incident to
      a branch vertex (degree >= 3), then group synapses by the resulting connected
      component — fragments are sub-trees, like real over-segmentation at branches.
    mode="axis": order synapses by skeleton-path distance from the soma and cut into
      contiguous pieces.
    Branch mode falls back to axis when it cannot produce >= 2 valid fragments.
    """
    from scipy.spatial import cKDTree
    from .pcfg_partitions import _random_piece_sizes

    n_v = len(skel_verts)
    n_syn = len(syn_pts)
    if n_syn < 2 * min_chunk or n_v < 2:
        return []
    adj = _build_adj(skel_edges, n_v)
    nearest = cKDTree(skel_verts).query(syn_pts, k=1)[1].tolist()

    if mode == "branch":
        deg = [len(a) for a in adj]
        edges = [(int(u), int(v)) for u, v in skel_edges.tolist()]
        branch_edges = [(u, v) for (u, v) in edges if deg[u] >= 3 or deg[v] >= 3]
        if branch_edges:
            perm = rng.permutation(len(branch_edges))
            cut = set()
            for pi in perm:
                if len(cut) >= max_pieces - 1:
                    break
                u, v = branch_edges[int(pi)]
                cut.add((min(u, v), max(u, v)))
            adj2: list[list[int]] = [[] for _ in range(n_v)]
            for u, v in edges:
                if (min(u, v), max(u, v)) in cut:
                    continue
                adj2[u].append(v)
                adj2[v].append(u)
            comp = _label_components(adj2, n_v)
            groups: dict[int, list[int]] = {}
            for si, vtx in enumerate(nearest):
                groups.setdefault(comp[vtx], []).append(si)
            frags = [np.array(g, dtype=np.int64) for g in groups.values() if len(g) >= min_chunk]
            if len(frags) >= 2:
                return frags
        # fall through to axis if branch cutting failed to yield >= 2 fragments

    # axis mode: order by path distance from soma, contiguous random-size pieces
    soma = _find_soma(skel_verts, radius)
    dist, _ = _bfs_from_root(adj, soma)
    big = 10 ** 9
    order = sorted(range(n_syn), key=lambda i: dist[nearest[i]] if dist[nearest[i]] >= 0 else big)
    sizes = _random_piece_sizes(n_syn, min_chunk, max_pieces, rng)
    if sizes is None:
        return []
    frags, start = [], 0
    for sz in sizes:
        frags.append(np.array(order[start:start + sz], dtype=np.int64))
        start += sz
    return frags


# ---------------------------------------------------------------------------
# Feature extraction: synapse paths through skeleton
# ---------------------------------------------------------------------------

def compute_intersynapse_paths(
    skel_verts: np.ndarray,
    skel_edges: np.ndarray,
    syn_pts: np.ndarray,
    radius=None,
    mode: str = "raw_delta3+skeleton",
    min_path_steps: int = 1,
) -> tuple[np.ndarray, list[np.ndarray | None]]:
    """Order synapses by skeleton path distance from soma; extract inter-synapse paths.

    Noisy/disconnected skeleton fragments are handled gracefully:
    - Synapses are grouped by connected component, ordered by (component_id, dist_within_component)
      so synapses in the same fragment stay together in the sequence
    - Gaps between disconnected fragments are returned as None rather than degenerate 2-step paths
    - prepare_partition_input() marks None entries with has_path=False so PathEdgeEncoder
      zeroes them out rather than treating garbage as real morphology

    Parameters
    ----------
    skel_verts  : (V, 3) float skeleton vertex positions in nm
    skel_edges  : (E, 2) int edge index pairs
    syn_pts     : (N, 3) float synapse positions in nm
    radius      : (V,) float per-vertex radii, or None
    mode        : featurize_path_points mode (see neuronauts/grammar.py)

    Returns
    -------
    ordered_syn_pts : (N, 3) synapse positions ordered by (component, dist_from_soma)
    path_feats      : list of N-1 entries; each is either:
                        np.ndarray [T_i, D] for a valid intra-component path, or
                        None for a cross-component gap (declined merge)
    """
    from neuronauts.grammar import featurize_path_points
    from scipy.spatial import cKDTree

    verts = skel_verts.astype(np.float64)
    n = len(verts)
    adj = _build_adj(skel_edges, n)

    # Find connected components — component 0 = largest (main trunk)
    comp = _label_components(adj, n)

    soma = _find_soma(verts, radius)
    # BFS per component to get intra-component path distances
    # Start BFS from the soma vertex for the main component; for satellite fragments,
    # start from the vertex with maximum radius (or centroid) within that component
    n_comps = max(comp) + 1
    comp_roots = [soma] + [-1] * (n_comps - 1)
    if n_comps > 1:
        comp_verts: list[list[int]] = [[] for _ in range(n_comps)]
        for v, c in enumerate(comp):
            comp_verts[c].append(v)
        for c in range(1, n_comps):
            cvs = comp_verts[c]
            if radius is not None and len(radius) == n:
                comp_roots[c] = max(cvs, key=lambda v: float(radius[v]))
            else:
                centroid = verts[cvs].mean(axis=0)
                comp_roots[c] = min(cvs, key=lambda v: float(np.linalg.norm(verts[v] - centroid)))

    # BFS distances within each component
    dist_in_comp = [-1] * n
    for c in range(n_comps):
        r = comp_roots[c]
        d, _ = _bfs_from_root(adj, r)
        for v in range(n):
            if comp[v] == c:
                dist_in_comp[v] = max(0, d[v])

    # Map each synapse to nearest skeleton vertex
    tree = cKDTree(verts)
    _, nearest = tree.query(syn_pts, k=1)
    nearest = nearest.tolist()

    # Sort synapses by (component_id, dist_within_component) — keep fragments grouped
    syn_comp = [comp[v] for v in nearest]
    syn_dist = [dist_in_comp[v] for v in nearest]
    order = sorted(range(len(syn_pts)), key=lambda i: (syn_comp[i], syn_dist[i]))
    ordered_verts = [nearest[i] for i in order]
    ordered_comps = [syn_comp[i] for i in order]

    # Extract inter-synapse paths; return None for cross-component gaps
    path_feats: list[np.ndarray | None] = []
    D = 6 if "skeleton" in mode else 3
    for k in range(len(order) - 1):
        if ordered_comps[k] != ordered_comps[k + 1]:
            # Cross-component gap: decline to merge, mark as missing
            path_feats.append(None)
            continue
        v1, v2 = ordered_verts[k], ordered_verts[k + 1]
        path_idx = _tree_path(adj, v1, v2)
        path_pts = verts[path_idx]
        if len(path_pts) >= 2:
            feats = featurize_path_points(path_pts, mode=mode)  # [T, D]
        else:
            feats = np.zeros((1, D), dtype=np.float32)
        path_feats.append(feats.astype(np.float32))

    ordered_syn = syn_pts[order]
    return ordered_syn.astype(np.float64), path_feats


# ---------------------------------------------------------------------------
# Torch model
# ---------------------------------------------------------------------------

def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError("pip install torch (or pip install -e .[topology])")


@dataclass
class ModelConfig:
    """Hyper-parameters for SkeletonSynapseNet."""
    path_input_dim: int = 6       # featurize_path_points output dim
    path_d_model:   int = 32      # PathEdgeEncoder internal dim
    path_output_dim: int = 16     # PathEdgeEncoder output (= token dim for level 2)
    path_max_len:   int = 64      # max skeleton steps per inter-synapse segment
    seq_d_model:    int = 64      # TorchPathEncoder internal dim
    seq_output_dim: int = 32      # partition embedding dim
    seq_max_len:    int = 256     # max synapses per partition
    scorer_hidden:  int = 64
    dropout:        float = 0.1
    n_heads:        int = 4
    n_layers:       int = 2


def build_model(cfg: ModelConfig | None = None):
    """Build SkeletonSynapseNet from config. Returns an nn.Module."""
    torch = _require_torch()
    import torch.nn as nn
    from neuronauts.grammar import TorchPathEncoder
    from neuronauts.path_edge_encoder import PathEdgeEncoder

    if cfg is None:
        cfg = ModelConfig()

    class SkeletonSynapseNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Level 1: encode each inter-synapse skeleton path -> token
            self.path_enc = PathEdgeEncoder(
                input_dim=cfg.path_input_dim,
                d_model=cfg.path_d_model,
                output_dim=cfg.path_output_dim,
                max_len=cfg.path_max_len,
                n_heads=2,
                n_layers=2,
                dropout=cfg.dropout,
            )
            # Level 2: encode sequence of tokens -> partition embedding
            self.seq_enc = TorchPathEncoder(
                input_dim=cfg.path_output_dim,
                d_model=cfg.seq_d_model,
                output_dim=cfg.seq_output_dim,
                max_len=cfg.seq_max_len,
                n_heads=cfg.n_heads,
                n_layers=cfg.n_layers,
                dropout=cfg.dropout,
            )
            # Pairwise scorer
            in_dim = cfg.seq_output_dim * 3 + 1  # emb_A, emb_B, |diff|, log_dist
            self.scorer = nn.Sequential(
                nn.Linear(in_dim, cfg.scorer_hidden),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.scorer_hidden, 1),
            )

        def encode_partition(self, path_seqs, path_masks, has_path, seq_mask):
            """Encode one partition.

            path_seqs  : [N_gaps, T_max, path_input_dim]
            path_masks : [N_gaps, T_max] bool (True = pad)
            has_path   : [N_gaps] bool (True if non-degenerate path)
            seq_mask   : [1, N_gaps] bool (True = pad) for level-2 encoder

            Returns [1, seq_output_dim] embedding.
            """
            # Level 1: [N_gaps, path_output_dim]
            tok = self.path_enc(path_seqs, path_masks, has_path)
            # Add batch dim for level-2 encoder: [1, N_gaps, path_output_dim]
            tok = tok.unsqueeze(0)
            # Level 2: [1, seq_output_dim]
            return self.seq_enc(tok, seq_mask)

        def forward(self, batch_a, batch_b, log_dists):
            """Score a batch of partition pairs.

            batch_a/b: list of (path_seqs, path_masks, has_path, seq_mask) tuples
            log_dists: [B] float tensor
            Returns:   [B] logits
            """
            emb_a = torch.cat([self.encode_partition(*x) for x in batch_a], dim=0)
            emb_b = torch.cat([self.encode_partition(*x) for x in batch_b], dim=0)
            diff  = (emb_a - emb_b).abs()
            x = torch.cat([emb_a, emb_b, diff, log_dists.unsqueeze(1)], dim=1)
            return self.scorer(x).squeeze(1)

    return SkeletonSynapseNet()


# ---------------------------------------------------------------------------
# Batch preparation
# ---------------------------------------------------------------------------

def prepare_partition_input(path_feats: list, max_path_len: int = 64, max_seq: int = 256):
    """Convert inter-synapse path features to padded tensors for PathEdgeEncoder.

    path_feats: list of N_gaps entries, each either:
      - np.ndarray [T_i, D]  — valid intra-component skeleton path
      - None                 — cross-component gap (declined merge); has_path=False

    Returns (path_seqs, path_masks, has_path, seq_mask) — all torch tensors.
    has_path[i]=False tells PathEdgeEncoder to zero out that slot.

    ``max_seq`` caps the number of inter-synapse gaps (level-2 sequence length) to
    the encoder's positional-encoding size; a very large fragment (e.g. a soma with
    hundreds of synapses) is truncated to its first ``max_seq`` gaps.
    """
    torch = _require_torch()

    if path_feats and len(path_feats) > max_seq:
        path_feats = path_feats[:max_seq]

    if not path_feats:
        D = 6
        return (
            torch.zeros(1, 1, D),
            torch.ones(1, 1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            torch.ones(1, 1, dtype=torch.bool),
        )

    N = len(path_feats)
    valid = [f for f in path_feats if f is not None]
    D = valid[0].shape[1] if valid and valid[0].ndim == 2 else 6
    T_max = min(max_path_len, max((f.shape[0] for f in valid), default=1))

    seqs  = np.zeros((N, T_max, D), dtype=np.float32)
    masks = np.ones((N, T_max), dtype=bool)   # True = pad
    has   = np.zeros(N, dtype=bool)

    for i, f in enumerate(path_feats):
        if f is None:
            continue  # gap: has[i] stays False, zero seq
        T = min(f.shape[0], T_max)
        if T > 0:
            seqs[i, :T] = f[:T]
            masks[i, :T] = False
            has[i] = True

    seq_mask = np.zeros((1, N), dtype=bool)

    return (
        torch.tensor(seqs),
        torch.tensor(masks),
        torch.tensor(has),
        torch.tensor(seq_mask),
    )


# ---------------------------------------------------------------------------
# Dataset and training loop
# ---------------------------------------------------------------------------

def build_training_data(
    sk_partitions,   # list of SkeletonPartition
    radii: dict | None = None,
    max_neg_ratio: float = 3.0,
    rng=None,
    mode: str = "raw_delta3+skeleton",
    match_distance: bool = False,
    min_chunk_synapses: int = 4,
    max_pieces: int = 6,
    split_mode: str = "realistic",
    use_distance: bool = True,
    gap_nm: float = 1500.0,
    thin_bias: float = 2.0,
    singleton_negatives: bool = False,
):
    """Compute inter-synapse path features and labeled pairs.

    Returns:
        part_inputs : list of (path_seqs, path_masks, has_path, seq_mask) per partition
        pairs       : list of (i, j, log_dist, label) index tuples

    When real positive pairs (GT merges with both sides having skeletons) are
    fewer than 2, falls back to artificial positives: each skeleton partition
    with ≥ 4 valid inter-synapse gaps is split at its midpoint and the two
    halves become a positive pair.  This mirrors _artificial_positives() in
    pcfg_partitions.py and gives the learned model a training signal even when
    skeleton coverage of split neurons is low.
    """
    from collections import defaultdict
    from itertools import combinations
    from .pcfg_partitions import closest_point_dist

    if rng is None:
        rng = np.random.default_rng(42)

    _radii = radii or {}

    # Pre-compute raw path_feats AND prepared inputs for every partition
    all_raw_path_feats = []
    all_ordered_syn_pts = []
    part_inputs = []
    centroids = []
    for p in sk_partitions:
        rad = _radii.get(p.root_id)
        ordered_pts, path_feats = compute_intersynapse_paths(
            p.skel_verts, p.skel_edges, p.pts,
            radius=rad, mode=mode,
        )
        all_raw_path_feats.append(path_feats)
        all_ordered_syn_pts.append(ordered_pts)
        part_inputs.append(prepare_partition_input(path_feats))
        centroids.append(p.pts.mean(axis=0))

    centroids = np.array(centroids, dtype=np.float64)
    v18xx  = [p.v18xx_root for p in sk_partitions]
    rids   = [p.root_id   for p in sk_partitions]
    n_real = len(sk_partitions)

    # Real positive pairs: different v117 roots mapping to the same v18xx root
    pos_pairs = []
    by_v18 = defaultdict(list)
    for i, v in enumerate(v18xx):
        by_v18[v].append(i)
    for group in by_v18.values():
        if len(group) < 2:
            continue
        for i, j in combinations(group, 2):
            if rids[i] == rids[j]:
                continue
            d = closest_point_dist(sk_partitions[i].pts, sk_partitions[j].pts)
            pos_pairs.append((i, j, np.log1p(d), 1))

    n_v18xx_groups_with2 = sum(1 for g in by_v18.values() if len(g) >= 2)
    print(f"  v18xx groups >=2 members (skel subset): {n_v18xx_groups_with2}  real pos pairs: {len(pos_pairs)}")

    # Fallback: SYNTHETIC split->merge positives (and matched negatives).
    # When there are no real false-split positives, cut each well-skeletonised
    # partition in half along its synapse path order: the two halves are the same
    # neuron (positive). NEGATIVES are also built from halves of DIFFERENT neurons,
    # so both members of every pair are half-fragments — no fragment-size confound
    # (a half has ~half the path tokens; pairing halves-with-wholes would let the
    # model cheat on token count rather than learn the grammar).
    art_used = False
    half_idx: list[int] = []     # part_inputs index per fragment
    half_parent: list[int] = []  # source partition (neuron) index
    half_cent: list = []         # fragment centroid (nm)
    half_pts: list = []          # fragment synapse points (for closest-point dist)
    n_parents_used = 0
    n_singletons = 0

    def _add_fragment(src_idx, frag_pts, sub_verts, sub_edges, sub_rad):
        """Append one fragment (on its SEVERED sub-skeleton) to the pool."""
        _, frag_pf = compute_intersynapse_paths(
            sub_verts, sub_edges, frag_pts, radius=sub_rad, mode=mode,
        )
        fo = len(half_idx)
        half_idx.append(len(part_inputs))
        part_inputs.append(prepare_partition_input(frag_pf))
        half_parent.append(src_idx)
        half_cent.append(frag_pts.mean(axis=0))
        half_pts.append(frag_pts)
        return fo

    if len(pos_pairs) < 2:
        for src_idx, p in enumerate(sk_partitions):
            rad = _radii.get(p.root_id)
            frags, adj_pairs = _fragment_neuron(
                p.skel_verts, p.skel_edges, p.pts,
                min_chunk=min_chunk_synapses, max_pieces=max_pieces,
                rng=rng, mode=split_mode, radius=rad,
                gap_nm=gap_nm, thin_bias=thin_bias,
            )
            if len(frags) < 2 or not adj_pairs:
                # Too small to cut. Optionally keep it as a single fragment for the
                # negative pool (off by default — see --small-neuron-negatives). Uses
                # the full (already volume-truncated) skeleton.
                if singleton_negatives and len(p.pts) >= min_chunk_synapses:
                    _add_fragment(src_idx, p.pts, p.skel_verts, p.skel_edges, rad)
                    n_singletons += 1
                continue
            local = [_add_fragment(src_idx, p.pts[syn_idx], sv, se, sr)
                     for (syn_idx, sv, se, sr) in frags]
            # positives: only fragments ADJACENT across a break (a real
            # "reconnect-the-break" candidate), not every same-neuron pair
            for (a, b) in adj_pairs:
                fa, fb = local[a], local[b]
                d = closest_point_dist(half_pts[fa], half_pts[fb])
                pos_pairs.append((half_idx[fa], half_idx[fb], np.log1p(d), 1))
            n_parents_used += 1
        art_used = len(half_idx) > 0
        if art_used:
            print(f"  [synthetic split->merge ({split_mode}, gap={gap_nm:.0f}nm): "
                  f"{len(pos_pairs)} adjacent-break positive pairs from "
                  f"{len(half_idx)} fragments / {n_parents_used} cut neurons "
                  f"(+{n_singletons} small single-fragment neurons for negatives)]")

    # Negative pairs (KD-tree nearest, prefer spatially close different-neuron pairs)
    n_neg = max(1, int(len(pos_pairs) * max_neg_ratio))
    neg_pairs = []
    try:
        from scipy.spatial import cKDTree
        if art_used:
            # half-vs-half, different parent neuron (no size confound)
            cent_arr = np.array(half_cent, dtype=np.float64)
            n_h = len(half_idx)
            k = min(51, n_h)
            dists_nn, idx_nn = cKDTree(cent_arr).query(cent_arr, k=k, workers=-1)
            seen = set()
            for a in rng.permutation(n_h):
                for slot in range(1, k):
                    b = int(idx_nn[a, slot])
                    if half_parent[a] == half_parent[b]:
                        continue
                    pair = (min(a, b), max(a, b))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    d = closest_point_dist(half_pts[a], half_pts[b])
                    neg_pairs.append((half_idx[a], half_idx[b], np.log1p(d), 0))
                    if len(neg_pairs) >= n_neg * 5:
                        break
                if len(neg_pairs) >= n_neg * 5:
                    break
        else:
            k = min(51, len(part_inputs))
            tree = cKDTree(centroids)
            dists_nn, idx_nn = tree.query(centroids, k=k, workers=-1)
            seen = set()
            # Only use real partitions (0..n_real-1) as anchors for negative search
            anchor_perm = rng.permutation(n_real)
            for i in anchor_perm:
                for slot in range(1, k):
                    j = int(idx_nn[i, slot])
                    pair = (min(i, j), max(i, j))
                    if pair in seen or rids[i] == rids[j] or v18xx[i] == v18xx[j]:
                        continue
                    seen.add(pair)
                    d = closest_point_dist(sk_partitions[i].pts, sk_partitions[j].pts)
                    neg_pairs.append((i, j, np.log1p(d), 0))
                    if len(neg_pairs) >= n_neg * 5:
                        break
                if len(neg_pairs) >= n_neg * 5:
                    break
    except ImportError:
        pass

    neg_pairs.sort(key=lambda r: r[2])

    if match_distance and pos_pairs and neg_pairs:
        pos_log_dists = np.array([r[2] for r in pos_pairs])
        neg_log_dists = np.array([r[2] for r in neg_pairs])
        pos_mean = float(pos_log_dists.mean())
        pos_std  = float(pos_log_dists.std() + 1e-6)
        scores = -np.abs(neg_log_dists - pos_mean) / pos_std
        probs  = np.exp(scores - scores.max())
        probs /= probs.sum()
        k = min(n_neg, len(neg_pairs))
        n_support = int((probs > 1e-10).sum())
        chosen = rng.choice(len(neg_pairs), size=k, replace=(k > n_support), p=probs)
        neg_pairs = [neg_pairs[c] for c in chosen]
    else:
        neg_pairs = neg_pairs[:n_neg]

    all_pairs = pos_pairs + neg_pairs
    if not use_distance:
        # zero the distance input so the scorer decides purely from the learned
        # morphology embeddings (a constant input carries no discriminative signal)
        all_pairs = [(a, b, 0.0, lbl) for (a, b, _d, lbl) in all_pairs]
    perm = rng.permutation(len(all_pairs))
    all_pairs = [all_pairs[k] for k in perm]

    return part_inputs, all_pairs


def train_and_eval(
    sk_partitions,
    radii: dict | None = None,
    *,
    n_folds: int = 5,
    seed: int = 42,
    n_epochs: int = 30,
    lr: float = 3e-4,
    batch_size: int = 32,
    max_neg_ratio: float = 3.0,
    cfg: ModelConfig | None = None,
    verbose: bool = True,
    min_chunk_synapses: int = 4,
    max_pieces: int = 6,
    split_mode: str = "realistic",
    run_honest: bool = True,
    use_distance: bool = True,
    gap_nm: float = 1500.0,
    thin_bias: float = 2.0,
    singleton_negatives: bool = False,
) -> float:
    """Stratified k-fold CV; returns mean ROC-AUC.

    Trains SkeletonSynapseNet on labeled merge pairs derived from v117->v1718
    ground truth embedded in the SkeletonPartition.v18xx_root field.
    """
    torch = _require_torch()
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(seed)

    if verbose:
        print(f"  Computing synapse paths through skeletons ({len(sk_partitions)} partitions)...")
    part_inputs, pairs = build_training_data(
        sk_partitions, radii=radii, max_neg_ratio=max_neg_ratio, rng=rng,
        min_chunk_synapses=min_chunk_synapses, max_pieces=max_pieces, split_mode=split_mode,
        use_distance=use_distance, gap_nm=gap_nm, thin_bias=thin_bias,
        singleton_negatives=singleton_negatives,
    )

    if len(pairs) == 0:
        print("  No pairs -- skipping learned grammar.")
        return float("nan")

    y = np.array([p[3] for p in pairs], dtype=np.int64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if verbose:
        print(f"  Pairs: {len(pairs):,}  ({n_pos} pos / {n_neg} neg)")
    if n_pos < n_folds or n_neg < n_folds:
        print(f"  Too few pairs for {n_folds}-fold CV -- skipping.")
        return float("nan")

    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(y), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(pairs, y)):
        if verbose:
            print(f"  Fold {fold + 1}/{n_folds} — training {len(tr_idx)} / val {len(va_idx)} pairs ...")

        model = build_model(cfg)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        tr_pairs = [pairs[i] for i in tr_idx]
        tr_y     = y[tr_idx]

        for epoch in range(n_epochs):
            perm = rng.permutation(len(tr_pairs))
            epoch_loss = 0.0
            n_batches = 0
            for b_start in range(0, len(tr_pairs), batch_size):
                b_idx = perm[b_start: b_start + batch_size]
                batch = [tr_pairs[k] for k in b_idx]
                b_y   = torch.tensor(tr_y[b_idx], dtype=torch.float32)

                batch_a = [part_inputs[p[0]] for p in batch]
                batch_b = [part_inputs[p[1]] for p in batch]
                log_d   = torch.tensor([p[2] for p in batch], dtype=torch.float32)

                optimizer.zero_grad()
                logits = model(batch_a, batch_b, log_d)
                loss = criterion(logits, b_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.detach().item()
                n_batches += 1
            if verbose:
                print(f"    [std] fold {fold + 1}/{n_folds} epoch {epoch + 1}/{n_epochs}  "
                      f"loss={epoch_loss / max(1, n_batches):.4f}", flush=True)

        # Validation
        model.eval()
        va_pairs = [pairs[i] for i in va_idx]
        probs = []
        with torch.no_grad():
            for b_start in range(0, len(va_pairs), batch_size):
                batch = va_pairs[b_start: b_start + batch_size]
                batch_a = [part_inputs[p[0]] for p in batch]
                batch_b = [part_inputs[p[1]] for p in batch]
                log_d   = torch.tensor([p[2] for p in batch], dtype=torch.float32)
                logits  = model(batch_a, batch_b, log_d)
                probs.extend(torch.sigmoid(logits).tolist())

        oof_probs[va_idx] = probs

    auc_std = float(roc_auc_score(y, oof_probs))
    if verbose:
        print(f"  Standard CV AUC = {auc_std:.3f}")

    if not run_honest:
        if verbose:
            print(f"  CV AUC = {auc_std:.3f}  (standard; distance-matched pass skipped)")
        return auc_std

    # -------------------------------------------------------------------
    # Honest (distance-matched) evaluation — mirrors bigram grammar
    # -------------------------------------------------------------------
    if verbose:
        print(f"\n  [Honest] Building distance-matched pairs ...")
    rng_h = np.random.default_rng(seed + 7)
    part_inputs_h, pairs_h = build_training_data(
        sk_partitions, radii=radii, max_neg_ratio=max_neg_ratio,
        rng=rng_h, match_distance=True, min_chunk_synapses=min_chunk_synapses,
        max_pieces=max_pieces, split_mode=split_mode, use_distance=use_distance,
        gap_nm=gap_nm, thin_bias=thin_bias, singleton_negatives=singleton_negatives,
    )
    y_h = np.array([p[3] for p in pairs_h], dtype=np.int64)
    n_pos_h = int(y_h.sum())
    n_neg_h = len(y_h) - n_pos_h
    if verbose:
        print(f"  [Honest] Pairs: {len(pairs_h):,}  ({n_pos_h} pos / {n_neg_h} neg)")

    auc_h = float("nan")
    if n_pos_h >= n_folds and n_neg_h >= n_folds:
        pos_weight_h = torch.tensor([n_neg_h / max(1, n_pos_h)], dtype=torch.float32)
        skf_h = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + 7)
        oof_probs_h = np.zeros(len(y_h), dtype=np.float64)

        for fold, (tr_idx_h, va_idx_h) in enumerate(skf_h.split(pairs_h, y_h)):
            if verbose:
                print(f"  [Honest] Fold {fold + 1}/{n_folds} — training {len(tr_idx_h)} / val {len(va_idx_h)} ...")

            model_h = build_model(cfg)
            model_h.train()
            optimizer_h = torch.optim.Adam(model_h.parameters(), lr=lr)
            criterion_h = nn.BCEWithLogitsLoss(pos_weight=pos_weight_h)

            tr_pairs_h = [pairs_h[i] for i in tr_idx_h]
            tr_y_h     = y_h[tr_idx_h]

            for epoch in range(n_epochs):
                perm_h = rng_h.permutation(len(tr_pairs_h))
                ep_loss_h, nb_h = 0.0, 0
                for b_start in range(0, len(tr_pairs_h), batch_size):
                    b_idx_h   = perm_h[b_start: b_start + batch_size]
                    batch_h   = [tr_pairs_h[k] for k in b_idx_h]
                    b_y_h     = torch.tensor(tr_y_h[b_idx_h], dtype=torch.float32)
                    batch_a_h = [part_inputs_h[p[0]] for p in batch_h]
                    batch_b_h = [part_inputs_h[p[1]] for p in batch_h]
                    log_d_h   = torch.tensor([p[2] for p in batch_h], dtype=torch.float32)
                    optimizer_h.zero_grad()
                    logits_h = model_h(batch_a_h, batch_b_h, log_d_h)
                    loss_h   = criterion_h(logits_h, b_y_h)
                    loss_h.backward()
                    torch.nn.utils.clip_grad_norm_(model_h.parameters(), 1.0)
                    optimizer_h.step()
                    ep_loss_h += loss_h.detach().item()
                    nb_h += 1
                if verbose:
                    print(f"    [honest] fold {fold + 1}/{n_folds} epoch {epoch + 1}/{n_epochs}  "
                          f"loss={ep_loss_h / max(1, nb_h):.4f}", flush=True)

            model_h.eval()
            va_pairs_h = [pairs_h[i] for i in va_idx_h]
            probs_h = []
            with torch.no_grad():
                for b_start in range(0, len(va_pairs_h), batch_size):
                    batch_h   = va_pairs_h[b_start: b_start + batch_size]
                    batch_a_h = [part_inputs_h[p[0]] for p in batch_h]
                    batch_b_h = [part_inputs_h[p[1]] for p in batch_h]
                    log_d_h   = torch.tensor([p[2] for p in batch_h], dtype=torch.float32)
                    logits_h  = model_h(batch_a_h, batch_b_h, log_d_h)
                    probs_h.extend(torch.sigmoid(logits_h).tolist())
            oof_probs_h[va_idx_h] = probs_h

        auc_h = float(roc_auc_score(y_h, oof_probs_h))
        if verbose:
            print(f"  Learned grammar (honest) CV AUC = {auc_h:.3f}")
    else:
        if verbose:
            print(f"  [Honest] Too few pairs for CV — skipping.")

    if verbose:
        print(f"  CV AUC = {auc_h:.3f}  (honest/distance-matched)  |  standard = {auc_std:.3f}")
    return auc_h


def _score_pairs(model, pairs, part_inputs, batch_size, torch):
    """Run a trained model over labeled pairs -> probability array (model set to eval)."""
    model.eval()
    probs = []
    with torch.no_grad():
        for b_start in range(0, len(pairs), batch_size):
            batch = pairs[b_start: b_start + batch_size]
            batch_a = [part_inputs[p[0]] for p in batch]
            batch_b = [part_inputs[p[1]] for p in batch]
            log_d   = torch.tensor([p[2] for p in batch], dtype=torch.float32)
            logits  = model(batch_a, batch_b, log_d)
            probs.extend(torch.sigmoid(logits).tolist())
    return np.array(probs, dtype=np.float64)


def _bootstrap_auc_ci(y, probs, *, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap 95% CI for ROC-AUC (resample pairs with replacement).

    With few positives the held-out AUC is noisy; the CI makes that explicit
    instead of reporting a single point estimate that looks more precise than it
    is. Returns (lo, hi) or (nan, nan) if a class is missing.
    """
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); probs = np.asarray(probs)
    n = len(y)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():   # degenerate resample (one class) -> skip
            continue
        aucs.append(roc_auc_score(yb, probs[idx]))
    if not aucs:
        return float("nan"), float("nan")
    lo = float(np.percentile(aucs, 100 * alpha / 2))
    hi = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return lo, hi


def _merge_decision(y, probs, thr):
    """Turn scores into merge/no-merge calls at ``thr`` and score the MERGE class.

    Positives are true across-break fragment pairs that should be merged. The
    "do nothing" baseline merges nothing -> recall 0, F1 0. Returns precision /
    recall / F1 over the merges actually proposed.
    """
    y = np.asarray(y); probs = np.asarray(probs)
    pred = probs >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 1.0      # no merges -> vacuously precise
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc  = float((pred == (y == 1)).mean())
    return {"thr": float(thr), "tp": tp, "fp": fp, "fn": fn, "n_merges": tp + fp,
            "precision": prec, "recall": rec, "f1": f1, "accuracy": acc}


def _best_f1_threshold(y, probs):
    """Threshold (from the score grid) maximising MERGE-class F1. Returns (thr, f1)."""
    y = np.asarray(y); probs = np.asarray(probs)
    if len(np.unique(y)) < 2:
        return 0.5, float("nan")
    best_thr, best_f1 = 0.5, -1.0
    for t in np.unique(probs):
        f1 = _merge_decision(y, probs, t)["f1"]
        if f1 > best_f1:
            best_thr, best_f1 = float(t), f1
    return best_thr, best_f1


def train_and_eval_holdout(
    train_partitions,
    eval_partitions,
    train_radii: dict | None = None,
    eval_radii: dict | None = None,
    val_partitions=None,
    val_radii: dict | None = None,
    *,
    seed: int = 42,
    n_epochs: int = 30,
    lr: float = 3e-4,
    batch_size: int = 32,
    max_neg_ratio: float = 3.0,
    cfg: ModelConfig | None = None,
    verbose: bool = True,
    min_chunk_synapses: int = 5,
    max_pieces: int = 6,
    split_mode: str = "realistic",
    use_distance: bool = True,
    gap_nm: float = 1500.0,
    thin_bias: float = 2.0,
    singleton_negatives: bool = False,
    val_frac: float = 0.2,
    checkpoint_path: str | None = None,
) -> dict:
    """Train ONE model on ``train_partitions``; evaluate on a DISJOINT ``eval_partitions``.

    This is the honest generalisation estimate. Unlike k-fold CV
    (:func:`train_and_eval`), the train and eval fragments come from spatially
    separated boxes, so:

    * **No fragment leaks.** In within-box CV, two pieces of the same cut neuron
      can land one in the train fold and one in the val fold — the model sees the
      partner of a held-out fragment during training. Across disjoint regions
      that is impossible.
    * **No biggest-neuron subset.** The eval region is built the same
      representative way as deploy ("a mess of fragments"), so the estimate
      reflects what you would actually get, not a curated easy subset.

    Epoch selection never touches the eval region. If a third disjoint
    ``val_partitions`` (region C) is given it is used — the cleanest choice,
    because a split held out from the SAME region as training overfits together
    with it (its AUC keeps rising while true cross-region AUC falls), so it
    cannot detect region-specific overfit; an independent region can. Otherwise a
    split of the TRAINING pairs (``val_frac``) is used, else last-epoch. The
    headline ``auc`` is the eval AUC at the selected epoch, so it neither peeks
    at the test set nor pays the last-epoch overfit tax. Diagnostics
    ``auc_final`` (last epoch) and ``eval_peak`` (best eval epoch — peeks, not
    reported) bracket it. Returns the AUC + 95% bootstrap CI, the selection
    method, pair counts, and both the eval and selection per-epoch curves.
    """
    torch = _require_torch()
    import torch.nn as nn
    from dataclasses import asdict
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)   # reproducible model init + dropout across runs
    cfg = cfg or ModelConfig()

    # --- build train + eval datasets (independent rngs; disjoint regions) ---
    rng_tr = np.random.default_rng(seed)
    if verbose:
        print(f"  [holdout] building TRAIN data from {len(train_partitions)} partitions ...")
    tr_inputs, tr_pairs = build_training_data(
        train_partitions, radii=train_radii, max_neg_ratio=max_neg_ratio, rng=rng_tr,
        min_chunk_synapses=min_chunk_synapses, max_pieces=max_pieces, split_mode=split_mode,
        use_distance=use_distance, gap_nm=gap_nm, thin_bias=thin_bias,
        singleton_negatives=singleton_negatives,
    )
    rng_ev = np.random.default_rng(seed + 1009)
    if verbose:
        print(f"  [holdout] building EVAL data from {len(eval_partitions)} partitions ...")
    ev_inputs, ev_pairs = build_training_data(
        eval_partitions, radii=eval_radii, max_neg_ratio=max_neg_ratio, rng=rng_ev,
        min_chunk_synapses=min_chunk_synapses, max_pieces=max_pieces, split_mode=split_mode,
        use_distance=use_distance, gap_nm=gap_nm, thin_bias=thin_bias,
        singleton_negatives=singleton_negatives,
    )

    def _empty(n_tr, n_ev):
        return {"auc": float("nan"), "ci": (float("nan"), float("nan")),
                "auc_final": float("nan"), "eval_peak": float("nan"), "best_epoch": -1,
                "val_method": "n/a", "merge_decision": None, "do_nothing_f1": float("nan"),
                "n_train_pairs": n_tr, "n_eval_pairs": n_ev,
                "n_eval_pos": 0, "n_eval_neg": 0, "curve": [], "val_curve": []}

    if len(tr_pairs) == 0 or len(ev_pairs) == 0:
        print("  [holdout] no pairs on one side -- skipping.")
        return _empty(len(tr_pairs), len(ev_pairs))

    y_tr = np.array([p[3] for p in tr_pairs], dtype=np.int64)
    y_ev = np.array([p[3] for p in ev_pairs], dtype=np.int64)
    n_pos_tr, n_neg_tr = int(y_tr.sum()), int((y_tr == 0).sum())
    n_pos_ev, n_neg_ev = int(y_ev.sum()), int((y_ev == 0).sum())

    if verbose:
        print(f"  [holdout] train pairs: {len(tr_pairs):,} ({n_pos_tr} pos / {n_neg_tr} neg)")
        print(f"  [holdout] eval  pairs: {len(ev_pairs):,} ({n_pos_ev} pos / {n_neg_ev} neg)")
        ev_pos_d = np.expm1(np.array([p[2] for p in ev_pairs if p[3] == 1], dtype=np.float64))
        ev_neg_d = np.expm1(np.array([p[2] for p in ev_pairs if p[3] == 0], dtype=np.float64))
        if len(ev_pos_d) and len(ev_neg_d):
            print(f"  [holdout] eval closest-pt dist: pos med={np.median(ev_pos_d)/1e3:.1f}um  "
                  f"neg med={np.median(ev_neg_d)/1e3:.1f}um")

    if n_pos_ev < 1 or n_neg_ev < 1:
        print("  [holdout] eval set has only one class -- cannot compute AUC.")
        return _empty(len(tr_pairs), len(ev_pairs))

    # --- choose the epoch-SELECTION set (NEVER the eval/test region) --------
    # Priority:
    #   1. An independent disjoint VALIDATION region C (best). A same-region split
    #      overfits together with training, so its AUC keeps rising while true
    #      cross-region AUC falls — it cannot detect region-specific overfit. A
    #      third disjoint region can, and selecting on it never peeks at eval.
    #   2. Else a split held out from the TRAINING pairs (val').
    #   3. Else fall back to last-epoch.
    from sklearn.model_selection import StratifiedKFold
    fit_pairs, fit_y = tr_pairs, y_tr
    sel_pairs = sel_inputs = sel_y = None
    sel_label = "last-epoch"

    if val_partitions is not None:
        rng_vr = np.random.default_rng(seed + 2017)
        if verbose:
            print(f"  [holdout] building VAL-region data from {len(val_partitions)} partitions ...")
        vr_inputs, vr_pairs = build_training_data(
            val_partitions, radii=val_radii, max_neg_ratio=max_neg_ratio, rng=rng_vr,
            min_chunk_synapses=min_chunk_synapses, max_pieces=max_pieces, split_mode=split_mode,
            use_distance=use_distance, gap_nm=gap_nm, thin_bias=thin_bias,
            singleton_negatives=singleton_negatives,
        )
        y_vr = np.array([p[3] for p in vr_pairs], dtype=np.int64)
        if len(vr_pairs) and y_vr.min() != y_vr.max():
            sel_pairs, sel_inputs, sel_y = vr_pairs, vr_inputs, y_vr
            sel_label = "val-region C (disjoint)"
            if verbose:
                print(f"  [holdout] val-region pairs: {len(vr_pairs):,} "
                      f"({int(y_vr.sum())} pos / {int((y_vr == 0).sum())} neg)  "
                      f"[disjoint from BOTH train and eval]")
        elif verbose:
            print("  [holdout] val-region unusable (one class) -- falling back to train-internal split")

    if sel_pairs is None:   # no val region: hold a split out of the TRAIN pairs
        use_es = val_frac and 0.0 < val_frac < 0.5 and n_pos_tr >= 5 and n_neg_tr >= 5
        if use_es:
            n_val_folds = max(2, int(round(1.0 / val_frac)))
            skf_es = StratifiedKFold(n_splits=n_val_folds, shuffle=True, random_state=seed)
            tr_idx, val_idx = next(skf_es.split(tr_pairs, y_tr))
            fit_pairs = [tr_pairs[i] for i in tr_idx]; fit_y = y_tr[tr_idx]
            sel_pairs = [tr_pairs[i] for i in val_idx]; sel_inputs = tr_inputs
            sel_y = y_tr[val_idx]; sel_label = "val' (train-internal)"
            if verbose:
                print(f"  [holdout] early-stopping split: fit {len(fit_pairs)} "
                      f"({int(fit_y.sum())} pos) / val {len(sel_pairs)} ({int(sel_y.sum())} pos)")

    n_pos_fit, n_neg_fit = int(fit_y.sum()), int((fit_y == 0).sum())

    pos_weight = torch.tensor([n_neg_fit / max(1, n_pos_fit)], dtype=torch.float32)
    model = build_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    curve: list[float] = []          # eval AUC per epoch (disjoint test region)
    val_curve: list[float] = []      # selection AUC per epoch (NEVER the eval region)
    best_sel = float("-inf")         # selection criterion
    best_epoch = -1
    best_state = None
    peak_eval = float("-inf")        # diagnostic only (peeks at eval)
    for epoch in range(n_epochs):
        model.train()
        perm = rng_tr.permutation(len(fit_pairs))
        epoch_loss, n_batches = 0.0, 0
        for b_start in range(0, len(fit_pairs), batch_size):
            b_idx = perm[b_start: b_start + batch_size]
            batch = [fit_pairs[k] for k in b_idx]
            b_y   = torch.tensor(fit_y[b_idx], dtype=torch.float32)
            batch_a = [tr_inputs[p[0]] for p in batch]
            batch_b = [tr_inputs[p[1]] for p in batch]
            log_d   = torch.tensor([p[2] for p in batch], dtype=torch.float32)
            optimizer.zero_grad()
            logits = model(batch_a, batch_b, log_d)
            loss = criterion(logits, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.detach().item()
            n_batches += 1

        # eval AUC (disjoint TEST region) — the learning curve we report
        probs_ev = _score_pairs(model, ev_pairs, ev_inputs, batch_size, torch)
        auc_ev = float(roc_auc_score(y_ev, probs_ev))
        curve.append(auc_ev)
        peak_eval = max(peak_eval, auc_ev)

        # selection AUC (val region C or val' split; never the eval region)
        if sel_pairs is not None:
            probs_sel = _score_pairs(model, sel_pairs, sel_inputs, batch_size, torch)
            auc_selep = (float(roc_auc_score(sel_y, probs_sel))
                         if sel_y.min() != sel_y.max() else float("nan"))
        else:
            auc_selep = auc_ev  # last-epoch fallback
        val_curve.append(auc_selep)
        if not np.isnan(auc_selep) and auc_selep > best_sel:
            best_sel = auc_selep
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"    [holdout] epoch {epoch + 1}/{n_epochs}  loss={epoch_loss / max(1, n_batches):.4f}"
                  f"  sel-AUC={auc_selep:.3f}  eval-AUC={auc_ev:.3f}", flush=True)

    # Last-epoch eval AUC (overfit-prone with few pairs) — diagnostic.
    auc_final = curve[-1] if curve else float("nan")
    # Headline = eval AUC at the epoch chosen by the selection set (no eval peeking).
    if best_state is not None:
        model.load_state_dict({k: v for k, v in best_state.items()})
    probs_sel = _score_pairs(model, ev_pairs, ev_inputs, batch_size, torch)
    auc_sel = float(roc_auc_score(y_ev, probs_sel))
    ci_lo, ci_hi = _bootstrap_auc_ci(y_ev, probs_sel, seed=seed)
    if verbose:
        print(f"  [holdout] HELD-OUT AUC = {auc_sel:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]  "
              f"(epoch {best_epoch} selected on {sel_label}; {n_pos_ev} pos / {n_neg_ev} neg)")
        print(f"  [holdout]   diagnostics: last-epoch {auc_final:.3f}  |  "
              f"eval-peak {peak_eval:.3f} (peeks at eval, not reported)")

    # --- merge decision vs DO NOTHING --------------------------------------
    # AUC is threshold-free; the deployable question is "at a real threshold, how
    # many true cross-region merges does the grammar recover vs leaving every
    # fragment split (do nothing)?" Threshold is picked on the SELECTION set
    # (region C / val'), never on eval -> the operating point doesn't peek either.
    if sel_pairs is not None:
        probs_selset = _score_pairs(model, sel_pairs, sel_inputs, batch_size, torch)
        thr, _ = _best_f1_threshold(sel_y, probs_selset)
    else:
        thr = 0.5
    md = _merge_decision(y_ev, probs_sel, thr)
    do_nothing = _merge_decision(y_ev, probs_sel, 1.1)  # threshold above 1 -> merge nothing
    if verbose:
        print(f"  [holdout] merge decision @thr={thr:.3f} (chosen on {sel_label}):")
        print(f"  [holdout]   GRAMMAR:    P={md['precision']:.2f} R={md['recall']:.2f} "
              f"F1={md['f1']:.2f}  acc={md['accuracy']:.2f}  "
              f"({md['tp']}/{n_pos_ev} true merges recovered, {md['fp']} wrong)")
        print(f"  [holdout]   DO NOTHING: P={do_nothing['precision']:.2f} R=0.00 F1=0.00  "
              f"acc={do_nothing['accuracy']:.2f}  (0 merges)  ->  grammar dF1 = +{md['f1']:.2f}")

    if checkpoint_path:
        torch.save({
            "state_dict": best_state if best_state is not None else model.state_dict(),
            "cfg": asdict(cfg),
            "meta": {
                "auc": auc_sel, "ci": (ci_lo, ci_hi),
                "auc_final": auc_final, "eval_peak": peak_eval,
                "best_epoch": best_epoch, "n_epochs": n_epochs, "val_method": sel_label,
                "n_train_pairs": len(tr_pairs), "n_eval_pairs": len(ev_pairs),
                "n_eval_pos": n_pos_ev, "n_eval_neg": n_neg_ev,
                "min_chunk_synapses": min_chunk_synapses, "max_pieces": max_pieces,
                "split_mode": split_mode, "use_distance": use_distance,
                "gap_nm": gap_nm, "thin_bias": thin_bias,
                "merge_decision": md, "do_nothing_f1": do_nothing["f1"],
            },
        }, checkpoint_path)
        if verbose:
            print(f"  [holdout] checkpoint saved (early-stopped model) -> {checkpoint_path}")

    return {
        "auc": auc_sel,              # headline: epoch selected on the selection set
        "ci": (ci_lo, ci_hi),
        "auc_final": auc_final,      # last epoch (overfit-prone)
        "eval_peak": peak_eval,      # diagnostic only (peeks at eval)
        "best_epoch": best_epoch,
        "val_method": sel_label,
        "merge_decision": md,        # P/R/F1/acc at thr (chosen on selection set)
        "do_nothing_f1": do_nothing["f1"],
        "n_train_pairs": len(tr_pairs),
        "n_eval_pairs": len(ev_pairs),
        "n_eval_pos": n_pos_ev,
        "n_eval_neg": n_neg_ev,
        "curve": curve,
        "val_curve": val_curve,
    }
