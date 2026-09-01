"""DNA enrichment: map per-seg-root embeddings onto synapses, evaluate ablation.

This module bridges the represent/ stage (Fragment.dna) and the global synapse
graph (Phase 2).  It provides:

1. ``build_synapse_dna_matrix`` — assigns each synapse in a Region a DNA vector
   derived from its owning Fragment.

2. ``synapse_pair_dna_scores`` — computes cosine-similarity scores + same-neuron
   labels for a random sample of synapse pairs (the pair AUC task).

3. ``evaluate_dna_auc`` — wraps the above into the ablation metric reported in
   STATUS.md: "DNA-only AUC vs spatial-proximity baseline on same-neuron
   synapse-pair classification."

Ground truth: ``Region.pre_root_id`` (``label_version`` root IDs).
DNA features: ``Fragment.dna`` keyed by ``Fragment.synapse_indices`` (which rows
of the Region belong to that seg root).

A synapse with no matching Fragment (or a Fragment with ``dna=None``) gets a
zero vector and is excluded from AUC computation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..schemas import Fragment, Region


def build_synapse_dna_matrix(
    region: Region,
    fragments: Sequence[Fragment],
) -> np.ndarray:
    """Assign each synapse a DNA vector from its owning Fragment.

    Parameters
    ----------
    region:
        The owning Region (used only for ``n_synapses`` sizing).
    fragments:
        Fragments extracted from ``region``'s skeleton archive, with
        ``dna`` filled by the ``represent/`` stage.

    Returns
    -------
    np.ndarray
        Shape ``[N_synapses, D]`` float32.  Rows without a matching
        fragment DNA are zero-vectors.  ``D`` is inferred from the first
        fragment with a non-None dna; returns ``[N, 0]`` if no fragment
        carries a DNA.
    """
    # Determine embedding dimension.
    D = 0
    for frag in fragments:
        if frag.dna is not None:
            D = len(frag.dna)
            break
    if D == 0:
        return np.zeros((region.n_synapses, 0), dtype=np.float32)

    mat = np.zeros((region.n_synapses, D), dtype=np.float32)
    for frag in fragments:
        if frag.dna is None:
            continue
        for si in frag.synapse_indices:
            idx = int(si)
            if 0 <= idx < region.n_synapses:
                mat[idx] = frag.dna
    return mat


def synapse_pair_dna_scores(
    region: Region,
    fragments: Sequence[Fragment],
    *,
    max_pairs: int = 10_000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample synapse pairs and compute DNA cosine similarity + same-neuron labels.

    Parameters
    ----------
    region:
        Region with ``pre_root_id`` set (``label_version`` ground truth).
    fragments:
        Fragments from this region with ``dna`` filled.
    max_pairs:
        Maximum number of pairs to evaluate (half positive, half negative).
    rng:
        Random generator for reproducibility.

    Returns
    -------
    scores : np.ndarray
        Shape ``[P]`` float32 — cosine similarity in ``[-1, 1]``.
    labels : np.ndarray
        Shape ``[P]`` int8 — 1 for same-neuron pairs, 0 for different-neuron.

    Raises
    ------
    ValueError
        If ``region.pre_root_id`` is None or no fragment carries a DNA.
    """
    if region.pre_root_id is None:
        raise ValueError("region.pre_root_id must be set (label_version root IDs required)")

    dna_mat = build_synapse_dna_matrix(region, fragments)
    if dna_mat.shape[1] == 0:
        raise ValueError("No fragment carries a DNA embedding — run encode_fragments first")

    rng = rng or np.random.default_rng(0)
    root_ids = region.pre_root_id  # [N] label_version root IDs

    # Valid synapses: root_id > 0 AND have a non-zero DNA vector.
    has_dna = np.any(dna_mat != 0, axis=1)  # [N] bool
    valid_mask = (root_ids > 0) & has_dna
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) < 2:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int8)

    # Group valid indices by root_id for positive sampling.
    root_to_indices: dict[int, list[int]] = {}
    for idx in valid_idx:
        rid = int(root_ids[idx])
        root_to_indices.setdefault(rid, []).append(idx)

    # Roots that have ≥2 synapses can yield positive pairs.
    pos_roots = [rid for rid, idxs in root_to_indices.items() if len(idxs) >= 2]
    all_valid_roots = list(root_to_indices.keys())

    n_each = max_pairs // 2

    scores_list: list[float] = []
    labels_list: list[int] = []

    # --- Positive pairs (same root) ---
    pos_sampled = 0
    attempts = 0
    while pos_sampled < n_each and attempts < n_each * 4 and pos_roots:
        attempts += 1
        rid = pos_roots[int(rng.integers(len(pos_roots)))]
        idxs = root_to_indices[rid]
        ia, ib = rng.choice(len(idxs), size=2, replace=False)
        i, j = idxs[int(ia)], idxs[int(ib)]
        sim = _cosine(dna_mat[i], dna_mat[j])
        scores_list.append(sim)
        labels_list.append(1)
        pos_sampled += 1

    # --- Negative pairs (different roots) ---
    neg_sampled = 0
    attempts = 0
    while neg_sampled < n_each and attempts < n_each * 4 and len(all_valid_roots) >= 2:
        attempts += 1
        ra, rb = rng.choice(len(all_valid_roots), size=2, replace=False)
        rid_a = all_valid_roots[int(ra)]
        rid_b = all_valid_roots[int(rb)]
        if rid_a == rid_b:
            continue
        idxs_a = root_to_indices[rid_a]
        idxs_b = root_to_indices[rid_b]
        ia = int(rng.integers(len(idxs_a)))
        ib = int(rng.integers(len(idxs_b)))
        i, j = idxs_a[ia], idxs_b[ib]
        sim = _cosine(dna_mat[i], dna_mat[j])
        scores_list.append(sim)
        labels_list.append(0)
        neg_sampled += 1

    return (
        np.array(scores_list, dtype=np.float32),
        np.array(labels_list, dtype=np.int8),
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def spatial_proximity_scores(
    region: Region,
    *,
    max_pairs: int = 10_000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Baseline: spatial proximity (1 / (1 + distance_nm)) as a same-neuron score.

    Provides a geometric reference point for the DNA AUC ablation.
    Uses the same pair-sampling logic as ``synapse_pair_dna_scores``.
    """
    if region.pre_root_id is None:
        raise ValueError("region.pre_root_id must be set")

    rng = rng or np.random.default_rng(0)
    root_ids = region.pre_root_id
    pts = (region.pre_pt_nm + region.post_pt_nm) / 2.0  # [N, 3] synapse centroids

    valid_mask = root_ids > 0
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 2:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int8)

    root_to_indices: dict[int, list[int]] = {}
    for idx in valid_idx:
        rid = int(root_ids[idx])
        root_to_indices.setdefault(rid, []).append(idx)

    pos_roots = [rid for rid, idxs in root_to_indices.items() if len(idxs) >= 2]
    all_valid_roots = list(root_to_indices.keys())
    n_each = max_pairs // 2

    scores_list: list[float] = []
    labels_list: list[int] = []

    for label, root_pool, n_target in [
        (1, pos_roots, n_each),
        (0, None, n_each),
    ]:
        attempts = 0
        sampled = 0
        while sampled < n_target and attempts < n_target * 4:
            attempts += 1
            if label == 1:
                if not root_pool:
                    break
                rid = root_pool[int(rng.integers(len(root_pool)))]
                idxs = root_to_indices[rid]
                ia, ib = rng.choice(len(idxs), size=2, replace=False)
                i, j = idxs[int(ia)], idxs[int(ib)]
            else:
                if len(all_valid_roots) < 2:
                    break
                ra, rb = rng.choice(len(all_valid_roots), size=2, replace=False)
                rid_a, rid_b = all_valid_roots[int(ra)], all_valid_roots[int(rb)]
                if rid_a == rid_b:
                    continue
                idxs_a, idxs_b = root_to_indices[rid_a], root_to_indices[rid_b]
                i = idxs_a[int(rng.integers(len(idxs_a)))]
                j = idxs_b[int(rng.integers(len(idxs_b)))]
            dist = float(np.linalg.norm(pts[i] - pts[j]))
            scores_list.append(1.0 / (1.0 + dist))
            labels_list.append(label)
            sampled += 1

    return (
        np.array(scores_list, dtype=np.float32),
        np.array(labels_list, dtype=np.int8),
    )


def evaluate_dna_auc(
    region: Region,
    fragments: Sequence[Fragment],
    *,
    max_pairs: int = 10_000,
    rng: np.random.Generator | None = None,
    include_baseline: bool = True,
) -> dict:
    """Compute AUC of DNA cosine similarity for same-neuron synapse-pair prediction.

    Also computes a spatial-proximity baseline AUC for comparison.

    Parameters
    ----------
    region:
        Region with ``pre_root_id`` (label_version ground truth).
    fragments:
        Fragments with ``dna`` filled by ``encode_fragments``.
    max_pairs:
        Number of pairs to sample (half positive, half negative).
    include_baseline:
        If True, also compute spatial-proximity baseline AUC.

    Returns
    -------
    dict with keys:
        ``dna_auc``, ``n_pos``, ``n_neg``, ``n_no_dna``
        and optionally ``baseline_auc`` (spatial proximity).
    """
    def roc_auc_score(y_true, y_score):
        try:
            from sklearn.metrics import roc_auc_score as _sklearn_auc
            return float(_sklearn_auc(y_true, y_score))
        except ImportError:
            from scipy.stats import rankdata
            y_true = np.asarray(y_true)
            y_score = np.asarray(y_score)
            ranks = rankdata(y_score)
            n_pos = int(np.sum(y_true == 1))
            n_neg = int(np.sum(y_true == 0))
            if n_pos == 0 or n_neg == 0:
                return 0.5
            u = np.sum(ranks[y_true == 1]) - n_pos * (n_pos + 1) / 2.0
            return float(u / (n_pos * n_neg))

    _rng = rng or np.random.default_rng(0)

    dna_mat = build_synapse_dna_matrix(region, fragments)
    has_dna = np.any(dna_mat != 0, axis=1)
    n_no_dna = int((~has_dna).sum())

    dna_scores, labels = synapse_pair_dna_scores(
        region, fragments, max_pairs=max_pairs, rng=_rng
    )

    result: dict = {"n_no_dna": n_no_dna}

    if len(labels) < 2 or len(np.unique(labels)) < 2:
        result.update({"dna_auc": float("nan"), "n_pos": 0, "n_neg": 0})
        return result

    result["n_pos"] = int((labels == 1).sum())
    result["n_neg"] = int((labels == 0).sum())
    result["dna_auc"] = float(roc_auc_score(labels, dna_scores))

    if include_baseline:
        _rng2 = np.random.default_rng(0)
        base_scores, base_labels = spatial_proximity_scores(
            region, max_pairs=max_pairs, rng=_rng2
        )
        if len(base_labels) >= 2 and len(np.unique(base_labels)) >= 2:
            result["baseline_auc"] = float(roc_auc_score(base_labels, base_scores))

    return result
