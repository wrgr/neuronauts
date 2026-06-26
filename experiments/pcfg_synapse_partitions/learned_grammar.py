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

def prepare_partition_input(path_feats: list, max_path_len: int = 64):
    """Convert inter-synapse path features to padded tensors for PathEdgeEncoder.

    path_feats: list of N_gaps entries, each either:
      - np.ndarray [T_i, D]  — valid intra-component skeleton path
      - None                 — cross-component gap (declined merge); has_path=False

    Returns (path_seqs, path_masks, has_path, seq_mask) — all torch tensors.
    has_path[i]=False tells PathEdgeEncoder to zero out that slot.
    """
    torch = _require_torch()

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
            d = float(np.linalg.norm(centroids[i] - centroids[j]))
            pos_pairs.append((i, j, np.log1p(d), 1))

    n_v18xx_groups_with2 = sum(1 for g in by_v18.values() if len(g) >= 2)
    print(f"  v18xx groups >=2 members (skel subset): {n_v18xx_groups_with2}  real pos pairs: {len(pos_pairs)}")

    # Fallback: artificial positives via midpoint split of path-feature sequences.
    # When split neurons (the positive pairs we care about) are small fragments
    # with poor skeleton coverage, real positives can be 0.  Splitting each
    # well-skeletonized partition in half gives a valid same-neuron training signal.
    art_used = False
    if len(pos_pairs) < 2:
        for src_idx, (pf, ordered_pts) in enumerate(
            zip(all_raw_path_feats, all_ordered_syn_pts)
        ):
            valid = [f for f in pf if f is not None]
            if len(valid) < 4:
                continue
            mid = len(valid) // 2
            left_pf = valid[:mid]
            right_pf = valid[mid:]
            # Accurate per-half centroids from ordered synapse positions
            n_syn = len(ordered_pts)
            syn_mid = max(1, n_syn // 2)
            left_cent  = ordered_pts[:syn_mid].mean(axis=0)
            right_cent = ordered_pts[syn_mid:].mean(axis=0)
            i_left  = len(part_inputs)
            i_right = len(part_inputs) + 1
            part_inputs.append(prepare_partition_input(left_pf))
            part_inputs.append(prepare_partition_input(right_pf))
            centroids = np.vstack([centroids, left_cent[None], right_cent[None]])
            v18xx.append(v18xx[src_idx])
            v18xx.append(v18xx[src_idx])
            rids.append(rids[src_idx])
            rids.append(rids[src_idx])
            d = float(np.linalg.norm(left_cent - right_cent))
            pos_pairs.append((i_left, i_right, np.log1p(d), 1))
        art_used = len(pos_pairs) > 0

    if art_used:
        print(f"  [artificial positives: {len(pos_pairs)} pairs from {len(sk_partitions)} partitions]")

    # Negative pairs (KD-tree nearest, prefer spatially close different-neuron pairs)
    n_neg = max(1, int(len(pos_pairs) * max_neg_ratio))
    neg_pairs = []
    try:
        from scipy.spatial import cKDTree
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
                d = float(dists_nn[i, slot])
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

    # -------------------------------------------------------------------
    # Honest (distance-matched) evaluation — mirrors bigram grammar
    # -------------------------------------------------------------------
    if verbose:
        print(f"\n  [Honest] Building distance-matched pairs ...")
    rng_h = np.random.default_rng(seed + 7)
    part_inputs_h, pairs_h = build_training_data(
        sk_partitions, radii=radii, max_neg_ratio=max_neg_ratio,
        rng=rng_h, match_distance=True,
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
