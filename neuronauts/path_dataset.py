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


def generate_path_examples(
    chains: dict[int, np.ndarray],
    *,
    window_size: int = 8,
    stride: int | None = None,
    neg_per_pos: int = 4,
    hard_neg_fraction: float = 0.5,
    rng: np.random.Generator | None = None,
    max_examples: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Build (feature_array, label) training pairs.

    Positive examples are sliding windows over real chains.  Negative
    examples are splices: the first half from cell A, the second half from
    cell B, where hard negatives pick B's synapse nearest to A's endpoint
    using a KD-tree.

    Returns
    -------
    features : list of float32 arrays, each [window_size-1, 6]
    labels   : float32 array [N], 1 = valid path, 0 = splice
    """
    if rng is None:
        rng = np.random.default_rng()
    if stride is None:
        stride = max(1, window_size // 2)

    half = window_size // 2

    chain_ids = list(chains.keys())
    chain_list = [chains[k] for k in chain_ids]

    # --- Build global position index for hard-negative lookup ---
    # Each entry: (chain_idx, syn_idx_within_chain, position_nm)
    all_pos: list[np.ndarray] = []
    all_ci: list[int] = []
    all_si: list[int] = []
    for ci, pts in enumerate(chain_list):
        for si, pos in enumerate(pts):
            all_pos.append(pos)
            all_ci.append(ci)
            all_si.append(si)

    all_pos_arr = np.array(all_pos, dtype=np.float32)
    all_ci_arr = np.array(all_ci, dtype=np.int32)
    all_si_arr = np.array(all_si, dtype=np.int32)

    from scipy.spatial import KDTree
    tree = KDTree(all_pos_arr)

    # --- Positive examples ---
    features: list[np.ndarray] = []
    labels: list[float] = []
    source_ci: list[int] = []  # for hard-negative pairing

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
    n_neg_target = n_pos * neg_per_pos
    n_hard = int(n_neg_target * hard_neg_fraction)
    n_easy = n_neg_target - n_hard

    # --- Hard negatives: B's first synapse is nearest neighbour of A's junction point ---
    hard_added = 0
    attempts = 0
    pos_indices = list(range(n_pos))
    rng.shuffle(pos_indices)

    for pi in pos_indices:
        if hard_added >= n_hard:
            break
        attempts += 1
        if attempts > n_hard * 10:
            break

        ci_a = source_ci[pi]
        pts_a = chain_list[ci_a]
        N_a = len(pts_a)
        if N_a < half:
            continue

        # Pick a random start within chain A; use the half-point as junction
        start_a = int(rng.integers(0, max(1, N_a - window_size + 1)))
        junction_pos = pts_a[start_a + half - 1]  # last pos of A's portion

        # Find nearest synapses from a DIFFERENT chain
        k_query = 32
        dists, nn_indices = tree.query(junction_pos, k=k_query)
        for ni in nn_indices:
            ci_b = int(all_ci_arr[ni])
            si_b = int(all_si_arr[ni])
            if ci_b == ci_a:
                continue
            pts_b = chain_list[ci_b]
            n_needed = window_size - half
            if si_b + n_needed > len(pts_b):
                si_b = max(0, len(pts_b) - n_needed)
            seg_b = pts_b[si_b: si_b + n_needed]
            if len(seg_b) < n_needed:
                continue
            seg_a = pts_a[start_a: start_a + half]
            splice = np.concatenate([seg_a, seg_b], axis=0)
            feat = _featurize_window(splice)
            if feat is not None:
                features.append(feat)
                labels.append(0.0)
                hard_added += 1
            break

    # --- Easy negatives: random chain pair ---
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
    hard_neg_fraction: float = 0.5,
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
