"""Path discrimination dataset and training for PathEdgeEncoder.

Instead of training on spatial boxes (which are dominated by singletons),
we train PathEdgeEncoder to discriminate *valid path continuations* from
*splice negatives* — fake paths created by concatenating chain segments
from two different cells.

Positive: K consecutive synapses from the same cell (real arbor fragment).
Hard negative: first K/2 synapses from cell A, then K/2 from cell B where
  B's start synapse is spatially nearest to A's end synapse (looks plausible).
Easy negative: random splice between two unrelated cells (obvious break).

This formulation:
- Works on cells with as few as window_size synapses across all boxes
- Produces near-infinite negatives (O(N²) possible cell pairs)
- Avoids boundary/singleton artifacts entirely
- Teaches multi-hop path coherence, not just single-edge grammar
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .dataset_builder import BoxCache

_MIP2_VOX = np.array([32.0, 32.0, 40.0], dtype=np.float32)
# Scale absolute nm → µm for PathEdgeEncoder input stability
_NM_TO_UM = np.array([1e-3, 1e-3, 1e-3], dtype=np.float32)


def extract_cell_chains(
    cache: "BoxCache",
    *,
    role: str = "pre",
    min_synapses_per_cell: int = 5,
) -> dict[int, np.ndarray]:
    """Load all boxes and extract sorted per-cell synapse position chains.

    Converts voxel coordinates to absolute nm, aggregates across boxes, then
    sorts each cell's synapses along the PCA principal axis to approximate
    the axon/dendrite ordering.

    Returns
    -------
    dict[int, np.ndarray]
        root_id → float32 array of shape [N, 3] in absolute nm, sorted along
        the principal component.  Only cells with ≥ min_synapses_per_cell
        visible synapses (across all boxes combined) are included.
    """
    from collections import defaultdict
    from .fetch import make_cube_bbox_nm

    root_positions: dict[int, list[np.ndarray]] = defaultdict(list)

    for record in cache.iter_records():
        try:
            _, synapses = cache.load(record, load_volume=False)
        except Exception:
            continue
        bbox = make_cube_bbox_nm(tuple(record.center_nm), record.side_um)
        box_origin = np.array(bbox[0], dtype=np.float32)

        if role == "pre":
            pts_vox = synapses.pre_pt.astype(np.float32)
            root_ids = synapses.pre_root_id
        else:
            pts_vox = synapses.post_pt.astype(np.float32)
            root_ids = synapses.post_root_id

        pts_nm = pts_vox * _MIP2_VOX + box_origin  # absolute nm [N, 3]

        for i, rid in enumerate(root_ids):
            if int(rid) != 0:
                root_positions[int(rid)].append(pts_nm[i])

    chains: dict[int, np.ndarray] = {}
    for rid, pos_list in root_positions.items():
        if len(pos_list) < min_synapses_per_cell:
            continue
        pts = np.array(pos_list, dtype=np.float32)
        centred = pts - pts.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(centred, full_matrices=False)
            order = np.argsort(centred @ Vt[0])
        except np.linalg.LinAlgError:
            order = np.arange(len(pts))
        chains[rid] = pts[order]

    return chains


def _featurize_window(pts_nm: np.ndarray) -> "np.ndarray | None":
    """Featurize a [W, 3] position window into [W-1, 6] step features (µm)."""
    from .grammar import featurize_path_points

    pts_um = pts_nm * _NM_TO_UM
    feat = featurize_path_points(pts_um, iso_scale=np.ones(3, dtype=np.float32))
    return feat if feat.shape[0] > 0 else None


def _chain_directions(pts: np.ndarray) -> np.ndarray:
    """Unit direction vectors at each synapse using centered differences."""
    if len(pts) < 2:
        return np.zeros((len(pts), 3), dtype=np.float32)
    dirs = np.zeros_like(pts)
    dirs[1:-1] = pts[2:] - pts[:-2]
    dirs[0] = pts[1] - pts[0]
    dirs[-1] = pts[-1] - pts[-2]
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs / np.maximum(norms, 1e-6)


def build_spatial_index(chains: dict[int, np.ndarray]) -> dict:
    """Pre-build the KD-tree and global arrays from chains.

    Call once before the training epoch loop and pass the result as
    ``spatial_index=`` to ``generate_path_examples`` to avoid rebuilding
    the KD-tree every epoch (which accounts for most of the per-epoch wall
    time on CPU).

    Returns a dict with keys: tree, all_pos_arr, all_dir_arr, all_ci_arr,
    all_si_arr, chain_start, chain_list.
    """
    from scipy.spatial import KDTree

    chain_ids = list(chains.keys())
    chain_list = [chains[k] for k in chain_ids]
    chain_dirs_list = [_chain_directions(pts) for pts in chain_list]

    all_pos: list[np.ndarray] = []
    all_dir: list[np.ndarray] = []
    all_ci: list[int] = []
    all_si: list[int] = []
    chain_start = np.zeros(len(chain_list) + 1, dtype=np.int64)
    for ci, pts in enumerate(chain_list):
        dirs = chain_dirs_list[ci]
        for si, (pos, d) in enumerate(zip(pts, dirs)):
            all_pos.append(pos)
            all_dir.append(d)
            all_ci.append(ci)
            all_si.append(si)
        chain_start[ci + 1] = chain_start[ci] + len(pts)

    all_pos_arr = np.array(all_pos, dtype=np.float32)
    all_dir_arr = np.array(all_dir, dtype=np.float32)
    all_ci_arr = np.array(all_ci, dtype=np.int32)
    all_si_arr = np.array(all_si, dtype=np.int32)
    return {
        "tree": KDTree(all_pos_arr),
        "all_pos_arr": all_pos_arr,
        "all_dir_arr": all_dir_arr,
        "all_ci_arr": all_ci_arr,
        "all_si_arr": all_si_arr,
        "chain_start": chain_start,
        "chain_list": chain_list,
        "chain_dirs_list": chain_dirs_list,
    }


def generate_path_examples(
    chains: dict[int, np.ndarray],
    *,
    window_size: int = 8,
    stride: int | None = None,
    neg_per_pos: int = 4,
    hard_neg_fraction: float = 0.5,
    insert_delete_fraction: float = 0.3,
    parallel_cell_fraction: float = 0.15,
    rng: np.random.Generator | None = None,
    max_examples: int | None = None,
    spatial_index: dict | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Build (feature_array, label) training pairs.

    Negative example types
    ----------------------
    splice-easy    : random two-cell concat (obvious position jump)
    splice-hard    : KD-tree nearest-neighbor splice (spatially proximate)
    insert         : one foreign synapse injected into the middle of a valid chain.
                     The window stays window_size synapses long (W-1 real + 1 foreign).
                     Teaches detection of a single wrong segment merged in.
    delete         : one synapse removed, creating an anomalous gap.
                     The window takes W+1 synapses then drops one from the middle.
                     Teaches detection of a chain that needs to be split.
    parallel-cell  : splice at a point where cell B runs in the SAME direction as
                     cell A (cos_sim ≥ 0.8 between local directions).  This is the
                     hardest case — the step at the junction looks completely normal
                     in direction and distance; only longer-range trajectory divergence
                     distinguishes them.

    Hard negatives share hard_neg_fraction of the total budget.
    Within that:
      - insert_delete_fraction → insert/delete (split equally)
      - parallel_cell_fraction → parallel-cell splices
      - remainder              → KD-tree nearest-neighbor splices

    Edit-history examples (real CAVE proofreading merge/split errors) can be added via
    ``add_edit_history_examples()`` — these represent the ground-truth hard cases
    where the connectome boundary is genuinely ambiguous.

    Returns
    -------
    features : list of float32 arrays, each [window_size-1, 6]
    labels   : float32 array [N], 1 = valid path, 0 = negative
    """
    if rng is None:
        rng = np.random.default_rng()
    if stride is None:
        stride = max(1, window_size // 2)

    half = window_size // 2

    if spatial_index is not None:
        tree = spatial_index["tree"]
        all_pos_arr = spatial_index["all_pos_arr"]
        all_dir_arr = spatial_index["all_dir_arr"]
        all_ci_arr = spatial_index["all_ci_arr"]
        all_si_arr = spatial_index["all_si_arr"]
        chain_start = spatial_index["chain_start"]
        chain_list = spatial_index["chain_list"]
        chain_dirs_list = spatial_index["chain_dirs_list"]
    else:
        idx = build_spatial_index(chains)
        tree = idx["tree"]
        all_pos_arr = idx["all_pos_arr"]
        all_dir_arr = idx["all_dir_arr"]
        all_ci_arr = idx["all_ci_arr"]
        all_si_arr = idx["all_si_arr"]
        chain_start = idx["chain_start"]
        chain_list = idx["chain_list"]
        chain_dirs_list = idx["chain_dirs_list"]

    # --- Positive examples ---
    features: list[np.ndarray] = []
    labels: list[float] = []
    source_ci: list[int] = []

    # Target budget: with neg_per_pos negatives per positive, we need at most
    # max_examples / (1 + neg_per_pos) positives.  Shuffle chains so that with
    # early stopping we sample a random subset rather than always the same cells.
    n_pos_target = (
        max(1, max_examples // (1 + neg_per_pos)) if max_examples is not None else None
    )
    ci_order = rng.permutation(len(chain_list))
    for ci in ci_order:
        if n_pos_target is not None and len(features) >= n_pos_target:
            break
        pts = chain_list[ci]
        N = len(pts)
        if N < window_size:
            continue
        start = 0
        while start + window_size <= N:
            feat = _featurize_window(pts[start: start + window_size])
            if feat is not None:
                features.append(feat)
                labels.append(1.0)
                source_ci.append(ci)
            start += stride

    n_pos = len(features)
    n_neg_total = n_pos * neg_per_pos
    n_hard_total = int(n_neg_total * hard_neg_fraction)
    n_easy = n_neg_total - n_hard_total

    # Hard budget: split between insert/delete, parallel-cell, and KD-tree splices
    n_ins_del = int(n_hard_total * insert_delete_fraction)
    n_insert = n_ins_del // 2
    n_delete = n_ins_del - n_insert
    n_parallel = int(n_hard_total * parallel_cell_fraction)
    n_kd_splice = n_hard_total - n_ins_del - n_parallel

    # Helper: find nearest different-cell synapse to a query position
    def _nearest_foreign(pos_nm: np.ndarray, exclude_ci: int, k: int = 32):
        _, nn_idx = tree.query(pos_nm, k=k)
        for ni in nn_idx:
            if int(all_ci_arr[ni]) != exclude_ci:
                return int(all_ci_arr[ni]), int(all_si_arr[ni])
        return None, None

    # Helper: find nearest synapse from a PARALLEL-RUNNING different cell.
    # "Parallel" means the local direction of the candidate synapse has
    # cos_sim ≥ min_cos with the query direction.  This creates the hardest
    # possible splice — the junction step looks normal in both distance and
    # direction; only longer-range trajectory divergence distinguishes it.
    def _nearest_parallel_foreign(
        pos_nm: np.ndarray,
        dir_nm: np.ndarray,
        exclude_ci: int,
        k: int = 64,
        min_cos: float = 0.75,
    ):
        _, nn_idx = tree.query(pos_nm, k=k)
        # Vectorized direction check: batch dot product instead of Python loop
        candidate_dirs = all_dir_arr[nn_idx]       # [k, 3]
        candidate_ci = all_ci_arr[nn_idx]           # [k]
        candidate_si = all_si_arr[nn_idx]           # [k]
        cos_sims = candidate_dirs @ dir_nm           # [k]
        for j in range(len(nn_idx)):
            if candidate_ci[j] == exclude_ci:
                continue
            if cos_sims[j] >= min_cos:
                return int(candidate_ci[j]), int(candidate_si[j])
        return None, None

    # --- Insert negatives ---
    # Take window_size-1 real synapses, inject one foreign synapse at position M.
    # Result: [s0..s_{M-1}, X, s_M..s_{W-2}] → window_size positions.
    insert_added = 0
    pos_idx = rng.permutation(n_pos)
    for pi in pos_idx:
        if insert_added >= n_insert:
            break
        ci_a = source_ci[int(pi)]
        pts_a = chain_list[ci_a]
        N_a = len(pts_a)
        if N_a < window_size:
            continue
        start_a = int(rng.integers(0, max(1, N_a - (window_size - 1) + 1)))
        seg = pts_a[start_a: start_a + window_size - 1]  # W-1 real synapses
        if len(seg) < window_size - 1:
            continue
        insert_pos = int(rng.integers(1, window_size - 1))  # inject inside, not at ends
        junction = seg[insert_pos - 1]  # position just before injection
        ci_b, si_b = _nearest_foreign(junction, ci_a)
        if ci_b is None:
            continue
        # O(1) lookup via precomputed chain_start index
        gidx = int(chain_start[ci_b]) + si_b
        foreign = all_pos_arr[gidx: gidx + 1]
        injected = np.concatenate([seg[:insert_pos], foreign, seg[insert_pos:]], axis=0)
        feat = _featurize_window(injected)
        if feat is not None:
            features.append(feat)
            labels.append(0.0)
            insert_added += 1

    # --- Delete negatives ---
    # Take window_size+1 real synapses, remove one from the middle.
    # Result: window_size positions with an anomalous gap at the deletion site.
    delete_added = 0
    pos_idx = rng.permutation(n_pos)
    for pi in pos_idx:
        if delete_added >= n_delete:
            break
        ci_a = source_ci[int(pi)]
        pts_a = chain_list[ci_a]
        N_a = len(pts_a)
        if N_a < window_size + 1:
            continue
        start_a = int(rng.integers(0, max(1, N_a - (window_size + 1) + 1)))
        seg = pts_a[start_a: start_a + window_size + 1]  # W+1 real synapses
        if len(seg) < window_size + 1:
            continue
        # Delete from the middle third to make the gap clearly anomalous
        del_lo = window_size // 3
        del_hi = 2 * window_size // 3
        del_pos = int(rng.integers(del_lo, del_hi + 1))
        gapped = np.concatenate([seg[:del_pos], seg[del_pos + 1:]], axis=0)
        feat = _featurize_window(gapped)
        if feat is not None:
            features.append(feat)
            labels.append(0.0)
            delete_added += 1

    # --- KD-tree splice negatives (hard) ---
    kd_added = 0
    pos_idx = rng.permutation(n_pos)
    for pi in pos_idx:
        if kd_added >= n_kd_splice:
            break
        ci_a = source_ci[int(pi)]
        pts_a = chain_list[ci_a]
        N_a = len(pts_a)
        if N_a < half:
            continue
        start_a = int(rng.integers(0, max(1, N_a - window_size + 1)))
        junction_pos = pts_a[start_a + half - 1]
        ci_b, si_b = _nearest_foreign(junction_pos, ci_a)
        if ci_b is None:
            continue
        pts_b = chain_list[ci_b]
        n_needed = window_size - half
        si_b = min(si_b, max(0, len(pts_b) - n_needed))
        seg_b = pts_b[si_b: si_b + n_needed]
        if len(seg_b) < n_needed:
            continue
        splice = np.concatenate([pts_a[start_a: start_a + half], seg_b], axis=0)
        feat = _featurize_window(splice)
        if feat is not None:
            features.append(feat)
            labels.append(0.0)
            kd_added += 1

    # --- Parallel-cell negatives ---
    # Cell B runs in the same direction as cell A at the junction.
    # The splice step looks normal — only long-range trajectory divergence
    # distinguishes the two cells.  These are the hardest possible negatives.
    parallel_added = 0
    pos_idx = rng.permutation(n_pos)
    for pi in pos_idx:
        if parallel_added >= n_parallel:
            break
        ci_a = source_ci[int(pi)]
        pts_a = chain_list[ci_a]
        dirs_a = chain_dirs_list[ci_a]  # precomputed — no per-iteration recompute
        N_a = len(pts_a)
        if N_a < half:
            continue
        start_a = int(rng.integers(0, max(1, N_a - window_size + 1)))
        junction_idx = start_a + half - 1
        junction_pos = pts_a[junction_idx]
        junction_dir = dirs_a[junction_idx]
        ci_b, si_b = _nearest_parallel_foreign(junction_pos, junction_dir, ci_a)
        if ci_b is None:
            continue
        pts_b = chain_list[ci_b]
        n_needed = window_size - half
        si_b = min(si_b, max(0, len(pts_b) - n_needed))
        seg_b = pts_b[si_b: si_b + n_needed]
        if len(seg_b) < n_needed:
            continue
        splice = np.concatenate([pts_a[start_a: start_a + half], seg_b], axis=0)
        feat = _featurize_window(splice)
        if feat is not None:
            features.append(feat)
            labels.append(0.0)
            parallel_added += 1

    # --- Easy negatives: random cell pair ---
    easy_added = 0
    attempts = 0
    while easy_added < n_easy and attempts < n_easy * 10:
        attempts += 1
        i_a, i_b = rng.choice(len(chain_list), size=2, replace=False)
        pts_a = chain_list[i_a]
        pts_b = chain_list[i_b]
        if len(pts_a) < half or len(pts_b) < (window_size - half):
            continue
        start_a = int(rng.integers(0, max(1, len(pts_a) - half + 1)))
        start_b = int(rng.integers(0, max(1, len(pts_b) - (window_size - half) + 1)))
        splice = np.concatenate([
            pts_a[start_a: start_a + half],
            pts_b[start_b: start_b + (window_size - half)],
        ], axis=0)
        feat = _featurize_window(splice)
        if feat is not None:
            features.append(feat)
            labels.append(0.0)
            easy_added += 1

    labels_arr = np.array(labels, dtype=np.float32)

    # Shuffle
    idx = rng.permutation(len(features))
    features = [features[i] for i in idx]
    labels_arr = labels_arr[idx]

    return features, labels_arr


def add_edit_history_examples(
    features: list[np.ndarray],
    labels: list[float],
    edit_pairs_tsv: str,
    chains: dict[int, np.ndarray],
    *,
    window_size: int = 8,
    rng: np.random.Generator | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Augment a feature/label list with edit-history–derived examples.

    Reads a TSV of (root_id_before, root_id_after, operation) rows produced
    by CAVE proofreading history.  Each row corresponds to a real merge or
    split correction:

    - merge (two roots became one): the junction between the two pre-merge
      chains is a ground-truth hard splice negative.  The model should learn
      "this crossing point is the kind of thing that gets merged by mistake."

    - split (one root became two): the split point is a hard positive boundary —
      a chain that was incorrectly cut.  Treated as a positive example here
      (the path across the split point IS valid).

    Edit-history examples are the hardest training signal because they represent
    the exact confusion boundary where the data is genuinely ambiguous — these
    are the cases that human proofreaders had to fix.

    Parameters
    ----------
    features, labels : mutable lists to append into
    edit_pairs_tsv : path to TSV with columns root_id_a, root_id_b, operation
    chains : output of extract_cell_chains() keyed by root_id
    window_size : synapse window length (same as generate_path_examples)
    rng : random generator

    Returns
    -------
    features_arr : list (extended in place and returned)
    labels_arr   : np.ndarray
    """
    if rng is None:
        rng = np.random.default_rng()

    import csv
    added = 0
    with open(edit_pairs_tsv, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            root_a = int(row.get("root_id_a", row.get("root_id_before", 0)))
            root_b = int(row.get("root_id_b", row.get("root_id_after", 0)))
            op = row.get("operation", "merge").strip().lower()

            if root_a not in chains or root_b not in chains:
                continue

            pts_a = chains[root_a]
            pts_b = chains[root_b]
            half = window_size // 2

            if op == "merge":
                # The merge joined pts_a and pts_b incorrectly — splice negative.
                # Allow short half-chains (even 1 synapse = isolate) by taking
                # as many synapses as available from each side (min 1 each) then
                # padding the window with repeated leading points if total < window_size.
                n_from_a = min(window_size - 1, max(1, len(pts_a)))
                n_from_b = min(window_size - n_from_a, max(1, len(pts_b)))
                if n_from_a < 1 or n_from_b < 1:
                    continue
                seg = np.concatenate([pts_a[-n_from_a:], pts_b[:n_from_b]], axis=0)
                if len(seg) < window_size:
                    pad = np.tile(seg[:1], (window_size - len(seg), 1))
                    seg = np.concatenate([pad, seg], axis=0)
                feat = _featurize_window(seg)
                if feat is not None:
                    features.append(feat)
                    labels.append(0.0)
                    added += 1

            elif op == "split":
                # The split broke what should be one chain — valid junction positive.
                # Same short-chain handling as the merge case above.
                n_from_a = min(window_size - 1, max(1, len(pts_a)))
                n_from_b = min(window_size - n_from_a, max(1, len(pts_b)))
                if n_from_a < 1 or n_from_b < 1:
                    continue
                seg = np.concatenate([pts_a[-n_from_a:], pts_b[:n_from_b]], axis=0)
                if len(seg) < window_size:
                    pad = np.tile(seg[:1], (window_size - len(seg), 1))
                    seg = np.concatenate([pad, seg], axis=0)
                feat = _featurize_window(seg)
                if feat is not None:
                    features.append(feat)
                    labels.append(1.0)
                    added += 1

    return features, np.array(labels, dtype=np.float32)


def train_path_encoder(
    chains: dict[int, np.ndarray],
    *,
    d_model: int = 32,
    n_heads: int = 2,
    n_layers: int = 3,
    output_dim: int = 16,
    window_size: int = 8,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 512,
    neg_per_pos: int = 4,
    hard_neg_fraction: float = 0.65,
    insert_delete_fraction: float = 0.3,
    parallel_cell_fraction: float = 0.15,
    checkpoint_path: str = "models/path_encoder.pt",
    checkpoint_every: int = 5,
    rng_seed: int = 42,
    max_examples_per_epoch: int | None = None,
    edit_pairs_tsv: "str | None" = None,
    edit_chains: "dict[int, np.ndarray] | None" = None,
    pool_mode: str = "cls",
) -> object:
    """Train PathEdgeEncoder on path discrimination and save checkpoint.

    The encoder is pre-trained standalone here; it can then be loaded and
    fine-tuned jointly with CellGNN.

    Returns the trained PathEdgeEncoder module.
    """
    from .grammar import _require_torch
    from .path_edge_encoder import PathEdgeEncoder, pad_path_sequences, PATH_STEP_FEAT_DIM

    torch, nn = _require_torch()
    import torch.nn.functional as F

    rng = np.random.default_rng(rng_seed)

    encoder = PathEdgeEncoder(
        input_dim=PATH_STEP_FEAT_DIM,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        output_dim=output_dim,
        pool_mode=pool_mode,
    )
    head = nn.Linear(output_dim, 1)

    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Merge CAVE edit chains into the main chains dict for training
    all_chains = dict(chains)
    if edit_chains:
        all_chains.update(edit_chains)

    n_cells = len(all_chains)
    chain_list = list(all_chains.values())
    print(
        f"PathEncoder: d={d_model} layers={n_layers} heads={n_heads} out={output_dim}"
    )
    print(
        f"Training: epochs={epochs} lr={lr} window={window_size} "
        f"neg_per_pos={neg_per_pos} hard_frac={hard_neg_fraction:.1f}"
    )
    print(f"Chains: {n_cells} cells  "
          f"(median_len={int(np.median([len(c) for c in chain_list]))})")

    # Build spatial index once — reused every epoch to avoid per-epoch KD-tree rebuild
    print("Building spatial index… ", end="", flush=True)
    _t_idx = time.monotonic()
    _spatial_index = build_spatial_index(all_chains)
    print(f"{time.monotonic() - _t_idx:.1f}s")

    # Pre-generate edit history examples (static — same pairs each epoch)
    edit_feats: list = []
    edit_lbls = np.empty(0, dtype=np.float32)
    if edit_pairs_tsv is not None:
        import os
        if os.path.exists(edit_pairs_tsv):
            edit_feats_list: list = []
            edit_lbls_list: list = []
            add_edit_history_examples(
                edit_feats_list,
                edit_lbls_list,
                edit_pairs_tsv,
                all_chains,
                window_size=window_size,
                rng=np.random.default_rng(rng_seed + 9999),
            )
            edit_feats = edit_feats_list
            edit_lbls = np.array(edit_lbls_list, dtype=np.float32)
            print(f"Edit history: {len(edit_feats)} examples "
                  f"({int(edit_lbls.sum())} pos, "
                  f"{len(edit_lbls) - int(edit_lbls.sum())} neg) from {edit_pairs_tsv}")
        else:
            print(f"[warn] edit_pairs_tsv not found: {edit_pairs_tsv}")

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        t0 = time.monotonic()

        features, labels = generate_path_examples(
            all_chains,
            window_size=window_size,
            neg_per_pos=neg_per_pos,
            hard_neg_fraction=hard_neg_fraction,
            insert_delete_fraction=insert_delete_fraction,
            parallel_cell_fraction=parallel_cell_fraction,
            rng=rng,
            max_examples=max_examples_per_epoch,
            spatial_index=_spatial_index,
        )

        # Append edit history examples (ground-truth hard cases from real corrections)
        if edit_feats:
            features = features + edit_feats
            labels = np.concatenate([labels, edit_lbls])

        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos

        encoder.train()
        head.train()

        total_loss = 0.0
        total_correct = 0
        n_batches = max(1, (len(features) + batch_size - 1) // batch_size)

        for b in range(n_batches):
            batch_f = features[b * batch_size: (b + 1) * batch_size]
            batch_l = labels[b * batch_size: (b + 1) * batch_size]

            path_seq, path_mask, has_path = pad_path_sequences(
                batch_f, max_len=window_size - 1, feat_dim=PATH_STEP_FEAT_DIM
            )
            path_seq_t = torch.from_numpy(path_seq).float()
            path_mask_t = torch.from_numpy(path_mask)
            has_path_t = torch.from_numpy(has_path)
            lbl_t = torch.from_numpy(batch_l).float()

            emb = encoder(path_seq_t, path_mask_t, has_path_t)
            logits = head(emb).squeeze(-1)

            # Use actual observed neg:pos ratio as pos_weight rather than
            # the target neg_per_pos, because hard-negative generation fails
            # for short chains, yielding fewer negatives than requested.
            actual_ratio = n_neg / max(n_pos, 1)
            pos_w = torch.tensor(float(actual_ratio), dtype=torch.float32)
            loss = F.binary_cross_entropy_with_logits(logits, lbl_t, pos_weight=pos_w)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch_f)
            preds = (logits.detach() > 0).float()
            total_correct += (preds == lbl_t).sum().item()

        scheduler.step()

        avg_loss = total_loss / max(len(features), 1)
        acc = total_correct / max(len(features), 1)
        wall = time.monotonic() - t0

        print(
            f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  acc={acc:.3f}"
            f"  n_pos={n_pos}  n_neg={n_neg}  wall={wall:.0f}s"
        )

        if epoch % checkpoint_every == 0 or epoch == epochs:
            _save_path_encoder(encoder, head, checkpoint_path.replace(".pt", f"_ep{epoch}.pt"))

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            _save_path_encoder(encoder, head, checkpoint_path.replace(".pt", "_best.pt"))

    _save_path_encoder(encoder, head, checkpoint_path)
    print(f"Saved path encoder → {checkpoint_path}  (best acc={best_acc:.3f} @ ep{best_epoch})")
    return encoder


def _save_path_encoder(encoder, head, path: str) -> None:
    from .grammar import _require_torch
    torch, _ = _require_torch()
    torch.save(
        {
            "encoder_state": encoder.state_dict(),
            "head_state": head.state_dict(),
            "init_kwargs": encoder._init_kwargs,
        },
        path,
    )


def load_path_encoder(path: str):
    """Load a saved PathEdgeEncoder from checkpoint.

    Returns (encoder, head) tuple — the encoder is the part used at inference.
    """
    from .grammar import _require_torch
    from .path_edge_encoder import PathEdgeEncoder

    torch, nn = _require_torch()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    encoder = PathEdgeEncoder(**ckpt["init_kwargs"])
    encoder.load_state_dict(ckpt["encoder_state"])
    head = nn.Linear(ckpt["init_kwargs"]["output_dim"], 1)
    head.load_state_dict(ckpt["head_state"])
    return encoder, head


# ---------------------------------------------------------------------------
# CAVE edit-history chain fetcher
# ---------------------------------------------------------------------------
#
# NOTE (Minnie65 v1412 limitation):
# The cells in the box cache at root_id_version=1412 are *already fully
# proofread* neurons.  All proofreading merge/split operations happened
# between tiny supervoxel fragments whose synapses (if any) are not
# accessible through the current materialization table.
# Querying before-roots at v117 (2021-06-11) and after-roots at current
# version both return 0 synapses because:
# - Split after-roots are stubs/debris with no pre-synapses
# - Merge before-roots were tiny fragments from the initial over-segmentation
# - Merge after-roots have been further split/merged hundreds of times and
#   are no longer valid root IDs by v943+ (2024+)
#
# The proofread neurons already *are* the ground-truth positive data.
# Their synapse chains (used in generate_path_examples) directly encode what
# correct arbor trajectories look like.  No separate edit-history augmentation
# is needed for the Minnie65 dataset at v1412.
#
# This module is retained for datasets where proofreading is less complete
# and intermediate states are accessible through versioned materialization.

_CAVE_MIP2_VOXEL_NM = np.array([8.0, 8.0, 40.0], dtype=np.float32)
# Minnie65 synapses are stored in 8-8-40 nm voxels


def _build_chain_from_positions(positions_nm: np.ndarray) -> "np.ndarray | None":
    """Sort [N,3] absolute-nm positions along their PCA axis.  Returns None if N<2."""
    if len(positions_nm) < 2:
        return None
    pts = positions_nm.astype(np.float32)
    centred = pts - pts.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
        order = np.argsort(centred @ Vt[0])
    except np.linalg.LinAlgError:
        order = np.arange(len(pts))
    return pts[order]


def fetch_cave_edit_history(
    cave_token: str,
    datastack: str = "minnie65_phase3_v1",
    *,
    n_ops: int = 500,
    min_synapses: int = 5,
    old_mat_version: int = 117,
    rng_seed: int = 0,
) -> "tuple[dict[int, np.ndarray], list[tuple[int, int, str]]]":
    """Fetch real merge/split corrections from CAVE and build training chains.

    For each **split** correction (proofreader fixed a false merge):
    - Queries pre-synapses of each after-root at the current materialization.
    - Records the pair as ``operation='merge'`` (junction was a false merge).

    For each **merge** correction (proofreader fixed a false split):
    - Queries pre-synapses of each before-root at the oldest available
      materialization version (``old_mat_version``), where those leaf-node
      roots still exist.
    - Records the pair as ``operation='split'`` (junction was a false split).

    Parameters
    ----------
    cave_token:
        CAVE authentication token.
    datastack:
        CAVE datastack name (default: Minnie65 phase 3).
    n_ops:
        Maximum number of distinct operation IDs to sample (splits and merges
        combined).
    min_synapses:
        Skip roots with fewer than this many pre-synapses.
    old_mat_version:
        Materialization version used to retrieve before-roots for merge ops.
    rng_seed:
        Random seed for sampling operations.

    Returns
    -------
    chains : dict[int, np.ndarray]
        root_id -> float32 [N, 3] sorted positions in absolute nm.
        Can be merged into the output of :func:`extract_cell_chains`.
    edit_pairs : list of (root_id_a, root_id_b, operation) tuples
        Operation is ``'merge'`` or ``'split'``.  Only pairs where both
        roots are present in the returned ``chains`` dict are included.
    """
    try:
        import caveclient
        import datetime
    except ImportError as exc:
        raise ImportError("caveclient is required: pip install caveclient") from exc

    rng = np.random.default_rng(rng_seed)
    client = caveclient.CAVEclient(datastack, auth_token=cave_token)

    # Use a 2-year sliding window ending today to bound the delta-roots response size.
    # The full Minnie65 history (2020-2026) returns ~2.7M roots and takes ~2 minutes.
    # A 2-year window returns ~400K roots in ~15 seconds.
    end = datetime.datetime.now(tz=datetime.timezone.utc)
    start = end - datetime.timedelta(days=730)
    print(f"[CAVE] fetching delta roots {start.date()} -> {end.date()} ...")
    _, new_roots = client.chunkedgraph.get_delta_roots(
        timestamp_past=start, timestamp_future=end
    )
    print(f"[CAVE] {len(new_roots):,} new roots found")

    # Sample roots and retrieve their change logs
    sample_size = min(n_ops * 20, len(new_roots))
    idx = rng.choice(len(new_roots), sample_size, replace=False)
    sample_roots = new_roots[idx].tolist()

    import pandas as pd
    all_op_rows = []
    batch = 100
    for i in range(0, len(sample_roots), batch):
        logs = client.chunkedgraph.get_tabular_change_log(sample_roots[i: i + batch])
        for df in logs.values():
            if len(df) > 0:
                all_op_rows.append(df)
        if i % 2000 == 0 and i > 0:
            collected = sum(len(d) for d in all_op_rows)
            print(f"[CAVE] scanned {i}/{len(sample_roots)} roots, "
                  f"{collected} ops so far ...")

    if not all_op_rows:
        print("[CAVE] no operations found; returning empty")
        return {}, []

    all_ops = pd.concat(all_op_rows, ignore_index=True).drop_duplicates("operation_id")
    merges = all_ops[all_ops["is_merge"]].head(n_ops // 2)
    splits = all_ops[~all_ops["is_merge"]].head(n_ops // 2)
    print(f"[CAVE] {len(merges)} merge ops, {len(splits)} split ops sampled")

    def _batch_query_synapses(root_ids, mat_version=None, batch_size=200):
        """Batch-query synapses for a list of root IDs using filter_in_dict.

        Returns root_id -> [N,3] absolute-nm positions for roots with >= min_synapses.
        Uses one query per batch of root_ids rather than one query per root.
        """
        result: dict[int, np.ndarray] = {}
        root_ids = [int(r) for r in root_ids]
        for i in range(0, len(root_ids), batch_size):
            batch = root_ids[i: i + batch_size]
            try:
                kwargs: dict = dict(
                    filter_in_dict={"pre_pt_root_id": batch},
                    select_columns=["pre_pt_root_id", "pre_pt_position"],
                )
                if mat_version is not None:
                    kwargs["materialization_version"] = mat_version
                df = client.materialize.query_table("synapses_pni_2", **kwargs)
                if len(df) == 0:
                    continue
                for rid, grp in df.groupby("pre_pt_root_id"):
                    if len(grp) < min_synapses:
                        continue
                    pos_vox = np.array(grp["pre_pt_position"].tolist(), dtype=np.float32)
                    pos_nm = pos_vox * _CAVE_MIP2_VOXEL_NM
                    chain = _build_chain_from_positions(pos_nm)
                    if chain is not None:
                        result[int(rid)] = chain
            except Exception:
                pass
        return result

    chains: dict[int, np.ndarray] = {}
    edit_pairs: list[tuple[int, int, str]] = []

    # Split ops: proofreader fixed a false merge -> after-roots are a negative pair
    split_pairs = []
    for _, row in splits.iterrows():
        after = [int(r) for r in row["after_root_ids"]]
        if len(after) >= 2:
            split_pairs.append((after[0], after[1]))

    if split_pairs:
        all_split_roots = list({r for pair in split_pairs for r in pair})
        print(f"[CAVE] querying synapses for {len(all_split_roots)} split-op roots ...")
        split_chains = _batch_query_synapses(all_split_roots)
        n_split_added = 0
        for root_a, root_b in split_pairs:
            if root_a in split_chains and root_b in split_chains:
                chains[root_a] = split_chains[root_a]
                chains[root_b] = split_chains[root_b]
                edit_pairs.append((root_a, root_b, "merge"))
                n_split_added += 1
        print(f"[CAVE] split ops -> {n_split_added} negative pairs added")

    # Merge ops: proofreader fixed a false split -> before-roots are a positive pair
    merge_pairs = []
    for _, row in merges.iterrows():
        before = [int(r) for r in row["before_root_ids"]]
        if len(before) >= 2:
            merge_pairs.append((before[0], before[1]))

    if merge_pairs:
        all_merge_roots = list({r for pair in merge_pairs for r in pair})
        print(f"[CAVE] querying synapses for {len(all_merge_roots)} merge-op roots "
              f"(mat v{old_mat_version}) ...")
        merge_chains = _batch_query_synapses(all_merge_roots, mat_version=old_mat_version)
        n_merge_added = 0
        for root_a, root_b in merge_pairs:
            if root_a in merge_chains and root_b in merge_chains:
                chains[root_a] = merge_chains[root_a]
                chains[root_b] = merge_chains[root_b]
                edit_pairs.append((root_a, root_b, "split"))
                n_merge_added += 1
        print(f"[CAVE] merge ops -> {n_merge_added} positive pairs added")
    print(f"[CAVE] total: {len(chains)} chains, {len(edit_pairs)} edit pairs")
    return chains, edit_pairs


def fetch_cave_false_merge_chains(
    cave_token: str,
    datastack: str = "minnie65_phase3_v1",
    *,
    past_timestamp: str = "2021-06-11",
    n_sample_old_roots: int = 10000,
    max_false_merges: int = 500,
    svid_batch_size: int = 2000,
    min_synapses_per_root: int = 8,
    rng_seed: int = 0,
) -> "tuple[dict[int, np.ndarray], list[tuple[int, int, str]]]":
    """Build real CV-error training pairs from CAVE delta root history.

    This implements the pre/post proofreading training approach: the path
    encoder learns the transfer function from CV output (past_timestamp) to
    proofread ground truth (current).

    A single pass of supervoxel → current-root lookups yields **both** error
    types simultaneously, at no extra API cost:

    **False merge** (hard negative, label=0):
        One past root whose supervoxels now map to 2+ current roots.
        The CV incorrectly merged two biological cells; proofreaders split them.
        The junction between the two half-chains is a real cell boundary the
        model must learn to detect.  Single-synapse isolates are included —
        even a 1-synapse foreign segment creates a real "insert" pattern.

    **False split** (hard positive, label=1):
        Two or more past roots whose supervoxels all map to the *same* current
        root.  The CV incorrectly split one biological cell; proofreaders
        merged the fragments.  The junction between their chains IS a valid
        same-cell path — the model must learn NOT to flag it.

    Training requires **both** types to calibrate correctly.  Training only on
    false merges teaches the model to distrust all junctions; the false-split
    positives force it to rely on actual path features (trajectory, step
    distances, curvature) rather than junction presence alone.

    Complex edit histories (interleaved merge+split sequences) are handled
    correctly because all labels are conditioned on the final v1412 ground
    truth, not on intermediate edit operations.  A supervoxel that passed
    through multiple intermediate roots still gets its correct final label.

    Parameters
    ----------
    cave_token : CAVE auth token.
    past_timestamp : ISO date of the raw/early state (v117 ≈ 2021-06-11;
        use the latest timestamp *before* manual proofreading began for the
        cleanest CV-output signal).
    n_sample_old_roots : delta roots to probe (more → richer signal, slower).
    max_false_merges : soft cap on merge pairs (split pairs are uncapped).
    svid_batch_size : supervoxels per get_roots call.
    min_synapses_per_root : drop v117 roots with fewer than this many synapses
        before the svid lookup.  Very short roots produce half-chains that can't
        fill a training window and are silently skipped by
        ``add_edit_history_examples``; filtering here avoids wasting API budget.
    rng_seed : RNG seed.

    Returns
    -------
    chains : dict[int → np.ndarray [N, 3] float32]
        Synthetic (negative) root_id → nm positions sorted along PCA axis.
        Merge into the main chains dict before calling
        :func:`add_edit_history_examples`.
    pairs : list of (chain_id_a, chain_id_b, operation)
        ``operation`` is ``'merge'`` (hard negative) or ``'split'``
        (hard positive).  Both chain IDs are keys in ``chains``.
    """
    try:
        import caveclient
        import datetime
    except ImportError as exc:
        raise ImportError("caveclient is required: pip install caveclient") from exc

    from collections import defaultdict

    past_dt = datetime.datetime.fromisoformat(past_timestamp).replace(
        tzinfo=datetime.timezone.utc
    )
    rng = np.random.default_rng(rng_seed)
    client = caveclient.CAVEclient(datastack, auth_token=cave_token)

    # ---- Step 1: sample old roots (v117 roots that changed since past_dt) ----
    print(f"[CAVE false-merge] fetching delta roots since {past_dt.date()} ...", flush=True)
    old_roots, _ = client.chunkedgraph.get_delta_roots(timestamp_past=past_dt)
    sample_size = min(n_sample_old_roots, len(old_roots))
    sample = rng.choice(old_roots, sample_size, replace=False).tolist()
    print(f"[CAVE false-merge] sampled {sample_size:,} of {len(old_roots):,} old roots", flush=True)

    # ---- Step 2: query v117 synapse positions + supervoxel IDs ----
    print(f"[CAVE false-merge] querying v117 synapses ...", flush=True)
    df = client.materialize.query_table(
        "synapses_pni_2",
        filter_in_dict={"pre_pt_root_id": sample},
        select_columns=[
            "pre_pt_root_id",
            "pre_pt_supervoxel_id",
            "pre_pt_position",
        ],
        materialization_version=117,
    )
    print(f"[CAVE false-merge] got {len(df):,} synapses from {df['pre_pt_root_id'].nunique()} roots", flush=True)

    if len(df) == 0:
        print("[CAVE false-merge] no synapses found — try an earlier past_timestamp")
        return {}, []

    # Build root → [(svid, position_nm)] mapping, then drop roots whose
    # total synapse count is below min_synapses_per_root.  Pairs from very
    # short roots produce half-chains that can't fill a training window and
    # are silently skipped by add_edit_history_examples; filtering here avoids
    # wasting svid lookup budget on useless entries.
    _vox = np.array([8.0, 8.0, 40.0], dtype=np.float32)  # Minnie65 voxel size nm
    root_to_entries_raw: dict[int, list] = defaultdict(list)
    for row in df.itertuples(index=False):
        pos_vox = np.asarray(row.pre_pt_position, dtype=np.float32)
        pos_nm = pos_vox * _vox
        svid = int(row.pre_pt_supervoxel_id)
        root_to_entries_raw[int(row.pre_pt_root_id)].append((svid, pos_nm))

    root_to_entries = {
        r: entries
        for r, entries in root_to_entries_raw.items()
        if len(entries) >= min_synapses_per_root
    }
    all_svids: list[int] = [e[0] for entries in root_to_entries.values() for e in entries]
    print(
        f"[CAVE false-merge] {len(root_to_entries)} roots with "
        f">={min_synapses_per_root} synapses ({len(all_svids):,} svids to resolve)",
        flush=True,
    )

    # ---- Step 3: ONE batched get_roots call for all supervoxels at once ----
    # Sending 2000 individual calls (one per root) would take 30-60 min.
    # Sending all svids in large batches takes seconds.
    print(
        f"[CAVE false-merge] looking up current roots for {len(all_svids):,} supervoxels "
        f"in batches of {svid_batch_size} ...",
        flush=True,
    )
    all_cur_roots: list[int] = []
    for i in range(0, len(all_svids), svid_batch_size):
        batch = all_svids[i : i + svid_batch_size]
        all_cur_roots.extend(int(r) for r in client.chunkedgraph.get_roots(batch))
        if (i // svid_batch_size) % 50 == 0 and i > 0:
            print(f"  {i:,}/{len(all_svids):,} svids resolved", flush=True)
    print(f"[CAVE false-merge] svid lookup complete", flush=True)

    # Map svid → current root for O(1) lookup
    svid_to_cur_root = dict(zip(all_svids, all_cur_roots))

    # ---- Step 4: classify both error types ----
    # false merge : one v117 root → 2+ current roots  → hard negative (merge)
    # false split : 2+ v117 roots → same current root → hard positive  (split)
    chains: dict[int, np.ndarray] = {}
    pairs: list[tuple[int, int, str]] = []
    synthetic_id = -1  # negative IDs don't clash with real root IDs

    cur_root_to_sids: dict[int, list[int]] = defaultdict(list)

    n_checked = 0
    for v117_root, entries in root_to_entries.items():
        # Process ALL roots for split detection — only cap merge pair accumulation.
        # Breaking early here would leave cur_root_to_sids incomplete and produce 0
        # split pairs (different v117 roots mapping to the same current root can only
        # be detected if we walk the full root list).
        merge_cap_hit = len(pairs) >= max_false_merges * 2

        positions = [e[1] for e in entries]
        cur_roots_list = [svid_to_cur_root[e[0]] for e in entries]

        # Group positions by current root; build a chain per group
        cur_root_to_positions: dict[int, list] = defaultdict(list)
        for cur_root, pos in zip(cur_roots_list, positions):
            cur_root_to_positions[cur_root].append(pos)

        half_ids: list[int] = []
        for cur_root, pos_list in cur_root_to_positions.items():
            pos_arr = np.stack(pos_list, axis=0).astype(np.float32)
            chain = _build_chain_from_positions(pos_arr)
            if chain is None:
                chain = pos_arr  # single point — still usable as isolate
            sid = synthetic_id
            synthetic_id -= 1
            chains[sid] = chain
            half_ids.append(sid)
            cur_root_to_sids[cur_root].append(sid)

        # False merge: this v117 root spanned 2+ biological cells
        if not merge_cap_hit and len(half_ids) >= 2:
            for i in range(len(half_ids)):
                for j in range(i + 1, len(half_ids)):
                    pairs.append((half_ids[i], half_ids[j], "merge"))

        n_checked += 1

    # False splits: same current root absorbing chains from 2+ distinct v117 roots.
    # Those v117 roots were incorrectly split; their junction is a hard positive.
    split_pairs: list[tuple[int, int, str]] = []
    for cur_root, sids in cur_root_to_sids.items():
        if len(sids) < 2:
            continue
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                split_pairs.append((sids[i], sids[j], "split"))

    all_pairs = pairs + split_pairs
    rng.shuffle(all_pairs)

    n_merge = sum(1 for _, _, op in all_pairs if op == "merge")
    n_split = len(all_pairs) - n_merge
    print(
        f"[CAVE false-merge] done: {len(chains)} chains, "
        f"{n_merge} merge pairs (hard-neg), {n_split} split pairs (hard-pos) "
        f"from {n_checked} checked roots",
        flush=True,
    )
    return chains, all_pairs


def fetch_cave_edits_spatially_stratified(
    cave_token: str,
    datastack: str = "minnie65_phase3_v1",
    *,
    past_timestamp: str = "2021-06-11",
    volume_bounds_vox: "tuple[list[int], list[int]] | None" = None,
    spatial_bins: "tuple[int, int, int]" = (4, 4, 2),
    roots_per_bin: int = 300,
    svid_batch_size: int = 2000,
    min_synapses_per_root: int = 8,
    rng_seed: int = 0,
) -> "tuple[dict[int, np.ndarray], list[tuple[int, int, str]]]":
    """Spatially stratified CAVE edit pair fetch.

    Divides the EM volume into a ``spatial_bins`` grid and samples
    ``roots_per_bin`` delta roots from each cell, guaranteeing even coverage
    across the volume.  This is critical for deployment: a model trained on
    spatially biased edits will have uneven performance across the volume.

    Within each spatial bin the same false-merge / false-split logic as
    :func:`fetch_cave_false_merge_chains` applies — supervoxels are resolved
    backward in time to their pre-proofreading root IDs to identify both error
    types in a single pass.

    Parameters
    ----------
    volume_bounds_vox : ``([xmin, ymin, zmin], [xmax, ymax, zmax])`` in MIP-0
        voxels (8×8×40 nm for Minnie65).  Defaults to the Minnie65 extent.
    spatial_bins : grid dimensions (nx, ny, nz).
    roots_per_bin : delta roots to probe per spatial bin.
    svid_batch_size, min_synapses_per_root, rng_seed : same as
        :func:`fetch_cave_false_merge_chains`.

    Returns
    -------
    chains, pairs : same format as :func:`fetch_cave_false_merge_chains`.
    """
    try:
        import caveclient
        import datetime
    except ImportError as exc:
        raise ImportError("caveclient is required: pip install caveclient") from exc

    from collections import defaultdict

    # Minnie65 MIP-0 extent (voxels at 8×8×40 nm)
    _DEFAULT_BOUNDS = ([17000, 17000, 0], [300000, 200000, 26000])
    lo, hi = volume_bounds_vox or _DEFAULT_BOUNDS

    past_dt = datetime.datetime.fromisoformat(past_timestamp).replace(
        tzinfo=datetime.timezone.utc
    )
    rng = np.random.default_rng(rng_seed)
    client = caveclient.CAVEclient(datastack, auth_token=cave_token)
    _vox = np.array([8.0, 8.0, 40.0], dtype=np.float32)

    nx, ny, nz = spatial_bins
    xs = np.linspace(lo[0], hi[0], nx + 1, dtype=int)
    ys = np.linspace(lo[1], hi[1], ny + 1, dtype=int)
    zs = np.linspace(lo[2], hi[2], nz + 1, dtype=int)
    n_bins = nx * ny * nz
    print(
        f"[strat-edit] {n_bins} spatial bins ({nx}×{ny}×{nz}), "
        f"{roots_per_bin} roots/bin → up to {n_bins * roots_per_bin:,} roots",
        flush=True,
    )

    # Accumulate across all bins
    root_to_entries: dict[int, list] = defaultdict(list)

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                bin_lo = [int(xs[ix]), int(ys[iy]), int(zs[iz])]
                bin_hi = [int(xs[ix + 1]), int(ys[iy + 1]), int(zs[iz + 1])]
                bin_label = f"bin({ix},{iy},{iz})"
                try:
                    df = client.materialize.query_table(
                        "synapses_pni_2",
                        filter_spatial_dict={
                            "pre_pt_position": [bin_lo, bin_hi]
                        },
                        select_columns=[
                            "pre_pt_root_id",
                            "pre_pt_supervoxel_id",
                            "pre_pt_position",
                        ],
                        materialization_version=117,
                    )
                except Exception as exc:
                    print(f"[strat-edit] {bin_label} query failed: {exc}", flush=True)
                    continue

                if len(df) == 0:
                    continue

                # Sample roots_per_bin unique roots from this bin
                unique_roots = df["pre_pt_root_id"].unique().tolist()
                sample = rng.choice(
                    unique_roots,
                    min(roots_per_bin, len(unique_roots)),
                    replace=False,
                ).tolist()
                sample_set = set(sample)

                for row in df.itertuples(index=False):
                    rid = int(row.pre_pt_root_id)
                    if rid not in sample_set:
                        continue
                    pos_vox = np.asarray(row.pre_pt_position, dtype=np.float32)
                    svid = int(row.pre_pt_supervoxel_id)
                    root_to_entries[rid].append((svid, pos_vox * _vox))

                print(
                    f"[strat-edit] {bin_label}: {len(df):,} synapses, "
                    f"{len(unique_roots)} roots, sampled {len(sample)}",
                    flush=True,
                )

    # Filter short roots
    root_to_entries = {
        r: entries
        for r, entries in root_to_entries.items()
        if len(entries) >= min_synapses_per_root
    }
    all_svids = [e[0] for entries in root_to_entries.values() for e in entries]
    print(
        f"[strat-edit] {len(root_to_entries):,} roots with "
        f">={min_synapses_per_root} synapses ({len(all_svids):,} svids to resolve)",
        flush=True,
    )

    if not all_svids:
        return {}, []

    # Batch resolve svids → past root IDs
    print(
        f"[strat-edit] resolving {len(all_svids):,} svids in batches of {svid_batch_size} ...",
        flush=True,
    )
    all_cur_roots: list[int] = []
    for i in range(0, len(all_svids), svid_batch_size):
        batch = all_svids[i : i + svid_batch_size]
        all_cur_roots.extend(int(r) for r in client.chunkedgraph.get_roots(batch))
        if (i // svid_batch_size) % 100 == 0 and i > 0:
            print(f"  {i:,}/{len(all_svids):,} resolved", flush=True)
    svid_to_cur = dict(zip(all_svids, all_cur_roots))
    print("[strat-edit] svid lookup complete", flush=True)

    # Classify error types (same logic as fetch_cave_false_merge_chains)
    chains: dict[int, np.ndarray] = {}
    pairs: list[tuple[int, int, str]] = []
    split_pairs: list[tuple[int, int, str]] = []
    synthetic_id = -1
    cur_root_to_sids: dict[int, list[int]] = defaultdict(list)

    for v117_root, entries in root_to_entries.items():
        cur_root_to_pos: dict[int, list] = defaultdict(list)
        for svid, pos in entries:
            cur_root_to_pos[svid_to_cur[svid]].append(pos)

        half_ids: list[int] = []
        for cur_root, pos_list in cur_root_to_pos.items():
            pos_arr = np.stack(pos_list).astype(np.float32)
            chain = _build_chain_from_positions(pos_arr) or pos_arr
            sid = synthetic_id
            synthetic_id -= 1
            chains[sid] = chain
            half_ids.append(sid)
            cur_root_to_sids[cur_root].append(sid)

        if len(half_ids) >= 2:
            for i in range(len(half_ids)):
                for j in range(i + 1, len(half_ids)):
                    pairs.append((half_ids[i], half_ids[j], "merge"))

    for cur_root, sids in cur_root_to_sids.items():
        if len(sids) < 2:
            continue
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                split_pairs.append((sids[i], sids[j], "split"))

    all_pairs = pairs + split_pairs
    rng.shuffle(all_pairs)

    n_merge = len(pairs)
    n_split = len(split_pairs)
    print(
        f"[strat-edit] done: {len(chains):,} chains, "
        f"{n_merge:,} merge pairs (hard-neg), {n_split:,} split pairs (hard-pos)",
        flush=True,
    )
    return chains, all_pairs


def fetch_cave_edit_pairs_from_cache(
    cave_token: str,
    cache_dir: str,
    datastack: str = "minnie65_phase3_v1",
    *,
    past_timestamp: str = "2021-06-11",
    svid_batch_size: int = 2000,
    min_synapses_per_root: int = 8,
    role: str = "pre",
    rng_seed: int = 0,
) -> "tuple[dict[int, np.ndarray], list[tuple[int, int, str]]]":
    """Build real CV-error training pairs from the box cache supervoxel IDs.

    This is the preferred approach over :func:`fetch_cave_false_merge_chains`
    because the box cache is already spatially stratified across the volume and
    already contains supervoxel IDs (``pre_seg_id`` / ``post_seg_id``), so no
    CAVE materialization query is needed.  The only network call is a batched
    ``get_roots(svids, timestamp=past_timestamp)`` to resolve each supervoxel
    backward to its pre-proofreading root.

    Starting from the **current** proofread roots avoids the 500K-row query cap
    that limits :func:`fetch_cave_false_merge_chains` to ~5K roots per call.
    The full box cache (~50K–100K unique roots across 252 boxes) is used.

    Both error types emerge from a single pass:

    **False split** (hard positive, label=1):
        One current root whose supervoxels span 2+ past roots.  The CV split one
        biological cell; proofreaders merged it.  The junction across the two
        sub-chains is a valid same-cell path — model must NOT flag it.

    **False merge** (hard negative, label=0):
        The same past root appears in 2+ current roots.  The CV merged two cells;
        proofreaders split them.  The junction between their chains is a real
        boundary — model must flag it.

    Parameters
    ----------
    cave_token : CAVE auth token.
    cache_dir : path to box cache directory (contains ``*.npz`` files with
        ``pre_seg_id``, ``pre_root_id``, ``pre_pt`` arrays).
    past_timestamp : ISO date of the raw CV state (v117 ≈ 2021-06-11).
    svid_batch_size : supervoxels per ``get_roots`` call.
    min_synapses_per_root : drop current roots with fewer synapses (too short
        to form useful training windows).
    role : ``'pre'``, ``'post'``, or ``'both'`` — which synapse side to use.
    rng_seed : RNG seed for shuffling output pairs.

    Returns
    -------
    chains : dict[int → np.ndarray [N, 3] float32]
        Synthetic root_id → nm positions sorted along PCA axis.
        Negative IDs for false-merge half-chains; positive current root IDs
        for false-split sub-chains (partitioned by their past root group).
    pairs : list of (chain_id_a, chain_id_b, operation)
        ``operation`` is ``'merge'`` (hard negative) or ``'split'``
        (hard positive).
    """
    try:
        import caveclient
        import datetime
    except ImportError as exc:
        raise ImportError("caveclient is required: pip install caveclient") from exc

    import os
    from collections import defaultdict

    past_dt = datetime.datetime.fromisoformat(past_timestamp).replace(
        tzinfo=datetime.timezone.utc
    )
    rng = np.random.default_rng(rng_seed)
    client = caveclient.CAVEclient(datastack, auth_token=cave_token)

    _vox = np.array([8.0, 8.0, 40.0], dtype=np.float32)  # Minnie65 MIP0 voxel nm

    # ---- Step 1: load supervoxel IDs + positions from box cache ----
    roles = ["pre", "post"] if role == "both" else [role]
    root_to_entries: dict[int, list] = defaultdict(list)

    npz_files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".npz"))
    print(
        f"[cache-edit] loading {len(npz_files)} boxes from {cache_dir} (role={role}) ...",
        flush=True,
    )
    n_skipped = 0
    for fname in npz_files:
        try:
            d = np.load(os.path.join(cache_dir, fname), allow_pickle=True)
        except Exception:
            n_skipped += 1
            continue
        for r in roles:
            seg_key = f"{r}_seg_id"
            root_key = f"{r}_root_id"
            pt_key = f"{r}_pt"
            if seg_key not in d or root_key not in d or pt_key not in d:
                continue
            for svid, root_id, pt_vox in zip(
                d[seg_key].tolist(), d[root_key].tolist(), d[pt_key].tolist()
            ):
                pos_nm = np.asarray(pt_vox, dtype=np.float32) * _vox
                root_to_entries[int(root_id)].append((int(svid), pos_nm))

    if n_skipped:
        print(f"[cache-edit] skipped {n_skipped} corrupt NPZ files", flush=True)

    # Filter short roots before the svid lookup
    root_to_entries = {
        r: entries
        for r, entries in root_to_entries.items()
        if len(entries) >= min_synapses_per_root
    }
    all_svids = [e[0] for entries in root_to_entries.values() for e in entries]
    print(
        f"[cache-edit] {len(root_to_entries):,} current roots with "
        f">={min_synapses_per_root} synapses ({len(all_svids):,} svids to resolve)",
        flush=True,
    )

    # ---- Step 2: batch resolve all svids → past root IDs ----
    # Socket-level cap so a dropped connection cannot stall the whole step.
    # Without this, get_roots inherits urllib3's default (no timeout) and a
    # CLOSE_WAIT TCP socket hangs the resolution loop indefinitely.
    import socket as _socket
    import time as _time

    _socket.setdefaulttimeout(180)

    print(
        f"[cache-edit] resolving {len(all_svids):,} svids to v117 roots "
        f"(batches of {svid_batch_size}) ...",
        flush=True,
    )
    all_past_roots: list[int] = []
    for i in range(0, len(all_svids), svid_batch_size):
        batch = all_svids[i : i + svid_batch_size]
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                all_past_roots.extend(
                    int(r) for r in client.chunkedgraph.get_roots(batch, timestamp=past_dt)
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                print(
                    f"  [retry] batch starting at {i} failed "
                    f"({type(exc).__name__}: {exc}); sleeping {wait}s "
                    f"(attempt {attempt + 1}/5)",
                    flush=True,
                )
                _time.sleep(wait)
        if last_exc is not None:
            raise last_exc
        if (i // svid_batch_size) % 100 == 0 and i > 0:
            print(f"  {i:,}/{len(all_svids):,} svids resolved", flush=True)
    svid_to_past = dict(zip(all_svids, all_past_roots))
    print("[cache-edit] svid lookup complete", flush=True)

    # ---- Step 3: classify both error types ----
    # false split : cur_root has 2+ past roots → positives at their junction
    # false merge : same past root in 2+ cur roots → negatives at their junction
    chains: dict[int, np.ndarray] = {}
    merge_pairs: list[tuple[int, int, str]] = []
    split_pairs: list[tuple[int, int, str]] = []
    synthetic_id = -1

    # Track which chains came from each past root (for merge detection)
    past_root_to_chain_ids: dict[int, list[int]] = defaultdict(list)

    n_splits = 0
    for cur_root, entries in root_to_entries.items():
        svids = [e[0] for e in entries]
        positions = [e[1] for e in entries]
        past_roots = [svid_to_past[s] for s in svids]

        # Group positions by past root
        past_root_to_pos: dict[int, list] = defaultdict(list)
        for past_root, pos in zip(past_roots, positions):
            past_root_to_pos[past_root].append(pos)

        # Build one chain per (cur_root, past_root) group
        group_chain_ids: list[int] = []
        for past_root, pos_list in past_root_to_pos.items():
            pos_arr = np.stack(pos_list).astype(np.float32)
            chain = _build_chain_from_positions(pos_arr)
            if chain is None:
                chain = pos_arr
            sid = synthetic_id
            synthetic_id -= 1
            chains[sid] = chain
            group_chain_ids.append(sid)
            past_root_to_chain_ids[past_root].append(sid)

        # False split: this current root descended from 2+ past roots
        if len(group_chain_ids) >= 2:
            for i in range(len(group_chain_ids)):
                for j in range(i + 1, len(group_chain_ids)):
                    split_pairs.append((group_chain_ids[i], group_chain_ids[j], "split"))
            n_splits += 1

    # False merge: same past root contributed to 2+ current roots
    for past_root, chain_ids in past_root_to_chain_ids.items():
        if len(chain_ids) < 2:
            continue
        for i in range(len(chain_ids)):
            for j in range(i + 1, len(chain_ids)):
                merge_pairs.append((chain_ids[i], chain_ids[j], "merge"))

    all_pairs = merge_pairs + split_pairs
    rng.shuffle(all_pairs)

    n_merge = len(merge_pairs)
    n_split = len(split_pairs)
    print(
        f"[cache-edit] done: {len(chains):,} chains, "
        f"{n_merge:,} merge pairs (hard-neg), {n_split:,} split pairs (hard-pos) "
        f"from {len(root_to_entries):,} current roots ({n_splits} had false splits)",
        flush=True,
    )
    return chains, all_pairs


def fetch_cave_lineage_pairs(
    cave_token: str,
    current_root_ids: "list[int]",
    datastack: str = "minnie65_phase3_v1",
    *,
    past_timestamp: str = "2021-06-11",
    batch_size: int = 500,
    max_pairs: int = 2000,
    rng_seed: int = 0,
) -> "list[tuple[int, int, str]]":
    """Find real false-merge and false-split pairs via CAVE chunkedgraph lineage.

    This implements the correct pre/post training data approach:

    * The **past** state (``past_timestamp``, default v117 2021-06-11) is the raw
      automated-segmentation output before human proofreading — the CV output.
    * The **current** state (v1412+) is the proofread ground truth.

    We are learning the transfer function from CV output → proofread output.
    The path encoder's job is to detect where the CV got it wrong.

    Two current v1412 roots that share the same v117 ancestor were incorrectly
    **merged** in the raw output (false merge → hard negative at their junction).

    One current v1412 root that descends from multiple v117 roots means the raw
    output had them incorrectly **split** (false split → hard positive across
    the split boundary).  These chains are not fetched here; the caller must
    supply v117 chains separately to use split pairs.

    Both root IDs in each returned pair are v1412 IDs drawn from
    ``current_root_ids``, so their chains are already available from
    :func:`extract_cell_chains` on the box cache — no extra CAVE synapse
    queries are needed for the merge pairs.

    Parameters
    ----------
    cave_token : CAVE auth token.
    current_root_ids : v1412 root IDs from the box cache.
    past_timestamp : ISO date string for the raw/early version (v117 ≈ 2021-06-11).
    batch_size : roots per CAVE API call (500 is safe under rate limits).
    max_pairs : cap on returned pairs (shuffled before truncation).
    rng_seed : RNG seed for shuffle.

    Returns
    -------
    list of (root_id_a, root_id_b, operation) tuples where operation is
    ``'merge'`` (false merge → hard negative) or ``'split'`` (false split
    → hard positive, only if both v117 roots are in ``current_root_ids``).
    """
    try:
        import caveclient
        import datetime
    except ImportError as exc:
        raise ImportError("caveclient is required: pip install caveclient") from exc

    from collections import defaultdict

    past_dt = datetime.datetime.fromisoformat(past_timestamp).replace(
        tzinfo=datetime.timezone.utc
    )

    rng = np.random.default_rng(rng_seed)
    client = caveclient.CAVEclient(datastack, auth_token=cave_token)

    root_ids = list(current_root_ids)
    rng.shuffle(root_ids)

    print(
        f"[CAVE lineage] querying {len(root_ids)} roots' ancestors at {past_dt.date()} ..."
    )

    current_set = set(current_root_ids)
    current_to_past: dict[int, list[int]] = {}
    n_batches = (len(root_ids) + batch_size - 1) // batch_size
    for i in range(0, len(root_ids), batch_size):
        batch = root_ids[i : i + batch_size]
        result = client.chunkedgraph.get_past_ids(
            root_ids=batch,
            timestamp_past=past_dt,
        )
        past_map = result["past_id_map"]
        for cid, pids in past_map.items():
            current_to_past[int(cid)] = [int(p) for p in pids]
        bn = i // batch_size + 1
        if bn % 10 == 0 or bn == n_batches:
            print(f"  batch {bn}/{n_batches}")

    # False merges: two current roots sharing the same past root.
    past_to_current: dict[int, list[int]] = defaultdict(list)
    for cid, pids in current_to_past.items():
        for pid in pids:
            past_to_current[pid].append(cid)

    merge_pairs: list[tuple[int, int, str]] = []
    for _past_root, cur_roots in past_to_current.items():
        if len(cur_roots) < 2:
            continue
        unique = sorted(set(cur_roots))
        for ii in range(len(unique)):
            for jj in range(ii + 1, len(unique)):
                merge_pairs.append((unique[ii], unique[jj], "merge"))

    # False splits: one current root descended from multiple past roots.
    # Only include if BOTH past roots happen to be in current_root_ids (rare
    # but possible when a cell was incorrectly over-split then later merged).
    split_pairs: list[tuple[int, int, str]] = []
    for cid, pids in current_to_past.items():
        if len(pids) < 2:
            continue
        pid_set = [p for p in pids if p in current_set]
        for ii in range(len(pid_set)):
            for jj in range(ii + 1, len(pid_set)):
                split_pairs.append((pid_set[ii], pid_set[jj], "split"))

    all_pairs = merge_pairs + split_pairs
    rng.shuffle(all_pairs)
    all_pairs = all_pairs[:max_pairs]

    n_merge = sum(1 for _, _, op in all_pairs if op == "merge")
    n_split = len(all_pairs) - n_merge
    print(
        f"[CAVE lineage] {len(all_pairs)} pairs "
        f"({n_merge} merge/hard-neg, {n_split} split/hard-pos)"
    )
    return all_pairs


def save_edit_pairs_tsv(
    edit_pairs: "list[tuple[int, int, str]]",
    tsv_path: str,
) -> None:
    """Write edit pairs to a TSV for :func:`add_edit_history_examples`."""
    import csv
    import os
    os.makedirs(os.path.dirname(os.path.abspath(tsv_path)), exist_ok=True)
    with open(tsv_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["root_id_a", "root_id_b", "operation"])
        for root_a, root_b, op in edit_pairs:
            writer.writerow([root_a, root_b, op])
    print(f"[CAVE] saved {len(edit_pairs)} edit pairs -> {tsv_path}")
