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
    chain_ids = list(chains.keys())
    chain_list = [chains[k] for k in chain_ids]

    # --- Build global KD-tree, direction index, and O(1) position lookup ---
    # Precompute chain directions once (not inside negative-generation loops).
    chain_dirs_list = [_chain_directions(pts) for pts in chain_list]

    all_pos: list[np.ndarray] = []
    all_dir: list[np.ndarray] = []
    all_ci: list[int] = []
    all_si: list[int] = []
    # chain_start[ci] = first global index for chain ci; allows O(1) lookup:
    #   global_idx(ci, si) = chain_start[ci] + si
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

    from scipy.spatial import KDTree
    tree = KDTree(all_pos_arr)

    # --- Positive examples ---
    features: list[np.ndarray] = []
    labels: list[float] = []
    source_ci: list[int] = []

    for ci, pts in enumerate(chain_list):
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

    # Shuffle and optionally cap
    idx = rng.permutation(len(features))
    if max_examples is not None and len(features) > max_examples:
        idx = idx[:max_examples]
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
                # The merge joined pts_a and pts_b incorrectly.
                # Generate a splice negative at their natural junction.
                if len(pts_a) < half or len(pts_b) < (window_size - half):
                    continue
                seg = np.concatenate([pts_a[-half:], pts_b[: window_size - half]], axis=0)
                feat = _featurize_window(seg)
                if feat is not None:
                    features.append(feat)
                    labels.append(0.0)
                    added += 1

            elif op == "split":
                # The split broke what should be one chain.
                # Generate a positive that crosses the split boundary.
                half_a = min(half, len(pts_a))
                half_b = min(window_size - half_a, len(pts_b))
                if half_a + half_b < window_size:
                    continue
                seg = np.concatenate([pts_a[-half_a:], pts_b[:half_b]], axis=0)
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
    )
    head = nn.Linear(output_dim, 1)

    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_cells = len(chains)
    chain_list = list(chains.values())
    print(
        f"PathEncoder: d={d_model} layers={n_layers} heads={n_heads} out={output_dim}"
    )
    print(
        f"Training: epochs={epochs} lr={lr} window={window_size} "
        f"neg_per_pos={neg_per_pos} hard_frac={hard_neg_fraction:.1f}"
    )
    print(f"Chains: {n_cells} cells  "
          f"(median_len={int(np.median([len(c) for c in chain_list]))})")

    for epoch in range(1, epochs + 1):
        t0 = time.monotonic()

        features, labels = generate_path_examples(
            chains,
            window_size=window_size,
            neg_per_pos=neg_per_pos,
            hard_neg_fraction=hard_neg_fraction,
            insert_delete_fraction=insert_delete_fraction,
            parallel_cell_fraction=parallel_cell_fraction,
            rng=rng,
            max_examples=max_examples_per_epoch,
        )
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

            # pos_weight to handle class imbalance (neg_per_pos:1 negatives)
            pos_w = torch.tensor(float(neg_per_pos), dtype=torch.float32)
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

    _save_path_encoder(encoder, head, checkpoint_path)
    print(f"Saved path encoder → {checkpoint_path}")
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

    # Collect operations over the full proofreading window
    start = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)
    end = datetime.datetime.now(tz=datetime.timezone.utc)
    print(f"[CAVE] fetching delta roots {start.date()} -> today ...")
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

    def _query_synapses(root_ids, mat_version=None):
        """Return root_id -> [N,3] absolute-nm positions for roots with >= min_synapses."""
        result: dict[int, np.ndarray] = {}
        for rid in root_ids:
            try:
                kwargs: dict = dict(
                    filter_equal_dict={"pre_pt_root_id": int(rid)},
                    select_columns=["pre_pt_root_id", "pre_pt_position"],
                )
                if mat_version is not None:
                    kwargs["materialization_version"] = mat_version
                df = client.materialize.query_table("synapses_pni_2", **kwargs)
                if len(df) < min_synapses:
                    continue
                pos_vox = np.array(df["pre_pt_position"].tolist(), dtype=np.float32)
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
    n_split_added = 0
    for _, row in splits.iterrows():
        after = [int(r) for r in row["after_root_ids"]]
        if len(after) < 2:
            continue
        root_a, root_b = after[0], after[1]
        new_chains = _query_synapses([root_a, root_b])
        if root_a in new_chains and root_b in new_chains:
            chains.update(new_chains)
            edit_pairs.append((root_a, root_b, "merge"))
            n_split_added += 1

    print(f"[CAVE] split ops -> {n_split_added} negative pairs added")

    # Merge ops: proofreader fixed a false split -> before-roots are a positive pair
    n_merge_added = 0
    for _, row in merges.iterrows():
        before = [int(r) for r in row["before_root_ids"]]
        if len(before) < 2:
            continue
        root_a, root_b = before[0], before[1]
        new_chains = _query_synapses([root_a, root_b], mat_version=old_mat_version)
        if root_a in new_chains and root_b in new_chains:
            chains.update(new_chains)
            edit_pairs.append((root_a, root_b, "split"))
            n_merge_added += 1

    print(f"[CAVE] merge ops -> {n_merge_added} positive pairs added")
    print(f"[CAVE] total: {len(chains)} chains, {len(edit_pairs)} edit pairs")
    return chains, edit_pairs


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
