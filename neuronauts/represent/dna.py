"""Tree-DNA encoder: seg-root skeleton tree → learned morphological embedding.

The encoder samples K leaf-to-leaf paths through the kimimaro skeleton tree,
encodes each path with a shared Transformer (``TorchPathEncoder`` from
``grammar.py``), and mean-pools across paths to produce one ``[D]`` embedding
per seg root.  Using paths preserves the sequential/morphological structure
that makes tree-DNA a useful identity signal (branch caliber, tortuosity,
tangent flow) while handling variable-topology trees without a fixed ordering.

Training uses triplet contrastive loss:
- Positives:  two Fragments with the same ``base_root_id`` (confirmed clean).
- Negatives:  Fragments from a different ``base_root_id``.
- Contaminated roots (``base_root_id`` maps to >1 ``label_version`` roots) are
  masked out of both positive and negative sets so the encoder never learns a
  coherent identity for a false-merged seg root.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from typing import Sequence

import numpy as np

from ..grammar import TorchPathEncoder, featurize_path_points, path_feature_dim
from ..schemas import Fragment


# ---------------------------------------------------------------------------
# Tree path sampling
# ---------------------------------------------------------------------------

def _build_adj(edges: np.ndarray, n_verts: int) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = {i: [] for i in range(n_verts)}
    for row in edges:
        u, v = int(row[0]), int(row[1])
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _bfs_path(adj: dict[int, list[int]], src: int, dst: int) -> list[int] | None:
    """Return the unique vertex-index path from src to dst, or None."""
    if src == dst:
        return [src]
    parent: dict[int, int | None] = {src: None}
    queue: deque[int] = deque([src])
    while queue:
        v = queue.popleft()
        for u in adj[v]:
            if u not in parent:
                parent[u] = v
                if u == dst:
                    path: list[int] = []
                    curr: int | None = dst
                    while curr is not None:
                        path.append(curr)
                        curr = parent[curr]
                    return path[::-1]
                queue.append(u)
    return None


def sample_tree_paths(
    vertices_nm: np.ndarray,
    edges: np.ndarray,
    n_paths: int = 16,
    *,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Sample up to *n_paths* leaf-to-leaf paths through the skeleton tree.

    Returns a list of float32 ``[T, 3]`` ordered vertex coordinate arrays,
    one per sampled path.

    - For a chain (exactly 2 leaves) returns the single path.
    - For a single vertex or empty graph returns ``[vertices_nm]``.
    - For a tree with ≥3 leaves, samples random leaf pairs.
    """
    verts = np.asarray(vertices_nm, dtype=np.float32)
    n = len(verts)
    if n == 0:
        return []
    if len(edges) == 0 or n == 1:
        return [verts]

    eds = np.asarray(edges, dtype=np.int64)
    if eds.ndim == 1:
        eds = eds.reshape(-1, 2)

    adj = _build_adj(eds, n)
    degree = {v: len(nbrs) for v, nbrs in adj.items()}
    leaves = [v for v, d in degree.items() if d <= 1]

    if len(leaves) == 0:
        leaves = [0, min(1, n - 1)]
    elif len(leaves) == 1:
        # degenerate: use the same vertex twice → single-step path
        leaves = [leaves[0], leaves[0]]

    if len(leaves) == 2:
        path = _bfs_path(adj, leaves[0], leaves[1])
        return [verts[path]] if path else [verts]

    # Tree with ≥3 leaves: sample random leaf pairs
    _rng = rng or np.random.default_rng()
    paths: list[np.ndarray] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    while len(paths) < n_paths and attempts < n_paths * 4:
        attempts += 1
        i, j = _rng.choice(len(leaves), size=2, replace=False)
        a, b = leaves[int(i)], leaves[int(j)]
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        path = _bfs_path(adj, a, b)
        if path:
            paths.append(verts[path])

    if not paths:
        path = _bfs_path(adj, leaves[0], leaves[1])
        paths = [verts[path]] if path else [verts]

    return paths


# ---------------------------------------------------------------------------
# Fragment featurization
# ---------------------------------------------------------------------------

def featurize_fragment(
    fragment: Fragment,
    *,
    n_paths: int = 16,
    feature_mode: str = "raw_delta3+skeleton",
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Sample paths and compute per-step features for each.

    Returns a list of float32 ``[T_k, D]`` arrays (one per sampled path) where
    ``D = path_feature_dim(feature_mode)`` (6 for ``"raw_delta3+skeleton"``).
    ``T_k = path_length - 1``.

    A single-vertex fragment returns ``[zeros([1, D])]`` (no crash).
    """
    D = path_feature_dim(feature_mode)
    verts = np.asarray(fragment.vertices_nm, dtype=np.float32)

    if len(verts) <= 1:
        return [np.zeros((1, D), dtype=np.float32)]

    paths = sample_tree_paths(verts, fragment.edges, n_paths, rng=rng)

    result: list[np.ndarray] = []
    for path_verts in paths:
        if len(path_verts) < 2:
            result.append(np.zeros((1, D), dtype=np.float32))
        else:
            feat = featurize_path_points(path_verts, mode=feature_mode)
            result.append(feat if feat.shape[0] > 0 else np.zeros((1, D), dtype=np.float32))

    return result or [np.zeros((1, D), dtype=np.float32)]


# ---------------------------------------------------------------------------
# TreeDNAEncoder
# ---------------------------------------------------------------------------

def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "torch is required for TreeDNAEncoder.  pip install torch"
        ) from exc
    return torch, nn


class TreeDNAEncoder:
    """Factory for a tree-skeleton DNA encoder.

    Architecture
    ------------
    1. ``featurize_fragment`` samples K leaf-to-leaf paths → K ``[T_k, D]`` arrays.
    2. All paths across the batch are padded and fed through a shared
       ``TorchPathEncoder`` (Transformer + CLS pooling) → ``[N_paths, output_dim]``.
    3. Path embeddings are mean-pooled per fragment → ``[B, output_dim]``.

    Parameters
    ----------
    n_paths:
        Number of paths sampled per fragment during encoding/training.
    feature_mode:
        Path featurization mode passed to ``featurize_path_points``.
    d_model, n_heads, n_layers, ffn_dim, dropout:
        ``TorchPathEncoder`` Transformer hyper-parameters.
    output_dim:
        Dimension of the final embedding.
    max_path_len:
        Maximum path length for sinusoidal positional encoding.
    """

    def __new__(
        cls,
        n_paths: int = 16,
        feature_mode: str = "raw_delta3+skeleton",
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        output_dim: int = 64,
        max_path_len: int = 512,
    ):
        torch, nn = _require_torch()
        _input_dim = path_feature_dim(feature_mode)
        _path_encoder = TorchPathEncoder(
            input_dim=_input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            output_dim=output_dim,
            max_len=max_path_len,
        )

        class _TreeDNAEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_paths = n_paths
                self.feature_mode = feature_mode
                self.output_dim = output_dim
                self._init_kwargs = dict(
                    n_paths=n_paths,
                    feature_mode=feature_mode,
                    d_model=d_model,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    output_dim=output_dim,
                    max_path_len=max_path_len,
                )
                self.encoder = _path_encoder

            def forward(
                self,
                path_features_batch: list[list[np.ndarray]],
            ) -> "torch.Tensor":
                """Encode a batch of fragments.

                Parameters
                ----------
                path_features_batch:
                    List of B items; each item is the list of ``[T_k, D]`` numpy
                    arrays returned by ``featurize_fragment`` for one fragment.

                Returns
                -------
                torch.Tensor
                    Shape ``[B, output_dim]``.
                """
                # Flatten all paths from all fragments
                all_arrays: list[np.ndarray] = []
                path_counts: list[int] = []
                for paths in path_features_batch:
                    all_arrays.extend(paths)
                    path_counts.append(len(paths))

                if not all_arrays:
                    dev = next(self.encoder.parameters()).device
                    return torch.zeros(len(path_features_batch), self.output_dim, device=dev)

                # Truncate to encoder's max_path_len so positional encoding fits.
                max_path_len: int = self.encoder.max_len  # type: ignore[attr-defined]
                all_arrays = [a[:max_path_len] for a in all_arrays]

                D = all_arrays[0].shape[1]
                N = len(all_arrays)
                dev = next(self.encoder.parameters()).device

                # Encode in chunks to bound memory: each chunk processes at most
                # _PATH_CHUNK paths through the Transformer (attention scales O(T²·N)).
                _PATH_CHUNK = 32
                all_embs_list: list[torch.Tensor] = []
                for chunk_start in range(0, N, _PATH_CHUNK):
                    chunk = all_arrays[chunk_start : chunk_start + _PATH_CHUNK]
                    max_T = max(a.shape[0] for a in chunk)
                    nc = len(chunk)
                    padded = torch.zeros(nc, max_T, D, device=dev)
                    pad_mask = torch.ones(nc, max_T, dtype=torch.bool, device=dev)
                    for i, arr in enumerate(chunk):
                        T = arr.shape[0]
                        padded[i, :T] = torch.from_numpy(arr).to(dev)
                        pad_mask[i, :T] = False
                    all_embs_list.append(self.encoder(padded, pad_mask))
                path_embs = torch.cat(all_embs_list, dim=0)

                # Mean-pool per fragment
                offset = 0
                result: list[torch.Tensor] = []
                for count in path_counts:
                    if count > 0:
                        result.append(path_embs[offset : offset + count].mean(dim=0))
                    else:
                        result.append(torch.zeros(self.output_dim, device=dev))
                    offset += count

                return torch.stack(result, dim=0)

        return _TreeDNAEncoder()


# ---------------------------------------------------------------------------
# Batch encoding
# ---------------------------------------------------------------------------

def encode_fragments(
    encoder: "torch.nn.Module",
    fragments: list[Fragment],
    *,
    device: str = "cpu",
    batch_size: int = 64,
    n_paths: int = 16,
) -> list[Fragment]:
    """Run *encoder* over *fragments*; return copies with ``dna`` filled.

    The original ``Fragment`` objects are not mutated.
    """
    import torch

    encoder = encoder.to(device).eval()
    results: list[Fragment] = list(fragments)

    with torch.no_grad():
        for start in range(0, len(fragments), batch_size):
            batch = fragments[start : start + batch_size]
            feats = [
                featurize_fragment(f, n_paths=n_paths, feature_mode=encoder.feature_mode)
                for f in batch
            ]
            embs = encoder(feats)
            for i, emb in enumerate(embs):
                dna = emb.cpu().numpy().astype(np.float32)
                results[start + i] = dataclasses.replace(batch[i], dna=dna)

    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_dna_encoder(
    encoder: "torch.nn.Module",
    fragment_lists: list[list[Fragment]],
    *,
    n_epochs: int = 20,
    lr: float = 1e-3,
    margin: float = 1.0,
    batch_size: int = 256,
    device: str = "cpu",
    root_label_map: "dict[int, set[int]] | None" = None,
    n_paths: int = 16,
) -> dict:
    """Triplet contrastive training for ``TreeDNAEncoder``.

    Parameters
    ----------
    encoder:
        A ``TreeDNAEncoder`` instance.
    fragment_lists:
        One list of ``Fragment`` objects per region.
    root_label_map:
        Optional mapping ``base_root_id → set[label_version root_ids]``.
        Fragments whose root maps to more than one label root are contaminated
        (false-merge survivors) and are excluded from both positive and negative
        sets.  If ``None``, all fragments are used with ``base_root_id`` as the
        identity proxy (noisier but workable for small experiments).
    n_paths:
        Number of paths sampled per fragment during training.

    Returns
    -------
    dict
        ``{"loss": [...], "pos_cosine": [...], "neg_cosine": [...]}`` per epoch.
    """
    import torch
    import torch.nn.functional as F

    encoder = encoder.to(device).train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    triplet_loss_fn = torch.nn.TripletMarginLoss(margin=margin, p=2)

    # Build index.
    # When root_label_map is provided: group fragments by label_root (same-neuron
    # = positive).  When absent: group by base_root_id (backward-compat, requires
    # multiple fragments per seg root).
    group_to_frags: dict[int, list[Fragment]] = {}
    all_frags_flat: list[Fragment] = [f for fl in fragment_lists for f in fl]
    for frag in all_frags_flat:
        rid = frag.base_root_id
        if root_label_map is not None:
            labels = root_label_map.get(rid, set())
            if len(labels) != 1:
                continue  # contaminated or unknown — skip
            group_key = next(iter(labels))  # label_root: same-neuron = positive
        else:
            group_key = rid  # fall back to seg-root identity
        group_to_frags.setdefault(group_key, []).append(frag)

    # Groups with ≥2 fragments can form exact positive pairs.
    # Single-fragment groups use path augmentation: the same fragment is
    # featurized twice with different random seeds, producing two distinct
    # views of the same skeleton.  Negatives always come from a different group.
    valid_groups = [gid for gid, frags in group_to_frags.items() if len(frags) >= 1]
    all_groups = list(group_to_frags.keys())

    if len(valid_groups) < 2:
        raise ValueError(
            f"Need ≥2 neuron groups with ≥1 clean fragment each; got {len(valid_groups)}. "
            "Check root_label_map or use more regions."
        )

    rng = np.random.default_rng(42)
    history: dict[str, list[float]] = {"loss": [], "pos_cosine": [], "neg_cosine": []}

    # All unique fragments used in training
    all_train_frags: list[Fragment] = [f for frags in group_to_frags.values() for f in frags]
    frag_to_idx: dict[int, int] = {id(f): i for i, f in enumerate(all_train_frags)}
    fm = encoder.feature_mode

    for epoch in range(n_epochs):
        epoch_losses: list[float] = []
        pos_cosines: list[float] = []
        neg_cosines: list[float] = []

        # Pre-cache path features for every fragment once per epoch (avoids
        # rebuilding adjacency on every featurize_fragment call in the inner loop).
        epoch_seed = int(rng.integers(2**31))
        cached_feats: list[list[np.ndarray]] = [
            featurize_fragment(f, n_paths=n_paths, feature_mode=fm,
                               rng=np.random.default_rng(epoch_seed + i))
            for i, f in enumerate(all_train_frags)
        ]
        # For path augmentation (positives): cache a second independent view.
        aug_seed = int(rng.integers(2**31))
        aug_feats: list[list[np.ndarray]] = [
            featurize_fragment(f, n_paths=n_paths, feature_mode=fm,
                               rng=np.random.default_rng(aug_seed + i))
            for i, f in enumerate(all_train_frags)
        ]

        n_triplets = max(batch_size, len(valid_groups) * 4)
        a_indices: list[int] = []
        p_indices: list[int] = []
        p_use_aug: list[bool] = []
        n_indices: list[int] = []

        for step_i in range(n_triplets):
            anchor_group = valid_groups[int(rng.integers(len(valid_groups)))]
            frags = group_to_frags[anchor_group]
            if len(frags) >= 2:
                ia, ip = rng.choice(len(frags), size=2, replace=False)
                a_indices.append(frag_to_idx[id(frags[int(ia)])])
                p_indices.append(frag_to_idx[id(frags[int(ip)])])
                p_use_aug.append(False)
            else:
                # Path augmentation: anchor uses cached_feats, positive uses aug_feats
                idx = frag_to_idx[id(frags[0])]
                a_indices.append(idx)
                p_indices.append(idx)
                p_use_aug.append(True)

            neg_group = anchor_group
            while neg_group == anchor_group:
                neg_group = all_groups[int(rng.integers(len(all_groups)))]
            neg_frags = group_to_frags[neg_group]
            n_indices.append(frag_to_idx[id(neg_frags[int(rng.integers(len(neg_frags)))])])

        for start in range(0, n_triplets, batch_size):
            ai = a_indices[start : start + batch_size]
            pi = p_indices[start : start + batch_size]
            ni = n_indices[start : start + batch_size]
            use_aug = p_use_aug[start : start + batch_size]

            a_emb = F.normalize(encoder([cached_feats[i] for i in ai]), dim=-1)
            p_emb = F.normalize(encoder([aug_feats[i] if u else cached_feats[i]
                                         for i, u in zip(pi, use_aug)]), dim=-1)
            n_emb = F.normalize(encoder([cached_feats[i] for i in ni]), dim=-1)

            loss = triplet_loss_fn(a_emb, p_emb, n_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            pos_cosines.append((a_emb * p_emb).sum(dim=-1).mean().detach().item())
            neg_cosines.append((a_emb * n_emb).sum(dim=-1).mean().detach().item())

        history["loss"].append(float(np.mean(epoch_losses)))
        history["pos_cosine"].append(float(np.mean(pos_cosines)))
        history["neg_cosine"].append(float(np.mean(neg_cosines)))

    return history
