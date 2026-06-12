"""End-to-end pipeline and hyperparameter optimizer.

run_pipeline chains the full sequence:
  1. train_fragment_encoder — teach fragments from the same object to embed
     close together
  2. encode_fragments — fill fragment.dna with the learned embedding
  3. build_observation_graph — typed-edge graph with same-fragment, spatial,
     and (optionally) endpoint-adjacent edges
  4. train_partition — teach the PartitionGNN to cluster observations by
     parent object
  5. partition_observations + evaluate_partition — measure ARI

optimize performs random search over the key hyperparameters, returning the
best configuration and its ARI.

Usage
-----
    from treestitch.pipeline import run_pipeline, optimize
    from treestitch.data import load_minnie65_world

    fragments, region, label_map = load_minnie65_world(n_objects=20, n_pieces=3)

    result = run_pipeline(
        fragments, region, label_map,
        embed_epochs=40, partition_epochs=40,
        endpoint_radius_nm=10_000, threshold=0.87,
    )
    print(f"ARI: {result['ari_before']:.3f} → {result['ari_after']:.3f}")

    best = optimize(fragments, region, label_map, n_trials=20)
    print(best)
"""

from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np


def run_pipeline(
    fragments: list,
    region: Any,
    root_label_map: dict,
    *,
    # Fragment encoder
    embed_epochs: int = 40,
    embed_lr: float = 1e-3,
    embed_d_model: int = 64,
    embed_output_dim: int = 32,
    embed_margin: float = 1.0,
    # Observation graph
    side: str = "pre",
    k_spatial: int = 8,
    pos_scale_nm: float = 50_000.0,
    endpoint_radius_nm: Optional[float] = 10_000.0,
    # Partition GNN
    partition_epochs: int = 40,
    partition_lr: float = 1e-3,
    partition_margin: float = 0.5,
    partition_max_pairs: int = 800,
    threshold: float = 0.87,
    # Misc
    device: str = "cpu",
    seed: int = 42,
    log_every: int = 10,
    verbose: bool = True,
) -> dict:
    """Run the full tree-stitching pipeline and return metrics.

    Parameters
    ----------
    fragments:
        List of Fragment objects (dna=None on entry).
    region:
        Observation container from ``load_minnie65_world`` or equivalent.
    root_label_map:
        ``{fragment.base_root_id: {object_id}}`` — supervision for the
        fragment encoder.
    embed_epochs:
        SkeletonGNN training epochs.  0 = use random-init encoder.
    embed_output_dim:
        Fragment embedding dimension (= DNA dimension in node features).
    endpoint_radius_nm:
        Radius for endpoint-adjacent edges.  ``None`` disables them.
    threshold:
        Cosine similarity threshold for union-find merging at inference.

    Returns
    -------
    dict with keys:
        ari_before, ari_after, delta_ari,
        homogeneity, completeness, v_measure,
        n_clusters_pred, n_clusters_true,
        embed_history, partition_history,
        config  (the full hyperparameter dict used)
    """
    from treestitch.embed import FragmentEncoder, encode_fragments, train_fragment_encoder
    from treestitch.graph import build_observation_graph
    from treestitch.partition import (
        PartitionGNN,
        evaluate_partition,
        partition_observations,
        train_partition,
    )

    config = dict(
        embed_epochs=embed_epochs, embed_lr=embed_lr,
        embed_d_model=embed_d_model, embed_output_dim=embed_output_dim,
        embed_margin=embed_margin,
        side=side, k_spatial=k_spatial, pos_scale_nm=pos_scale_nm,
        endpoint_radius_nm=endpoint_radius_nm,
        partition_epochs=partition_epochs, partition_lr=partition_lr,
        partition_margin=partition_margin, partition_max_pairs=partition_max_pairs,
        threshold=threshold, device=device, seed=seed,
    )

    # 1. Fragment encoder
    encoder = FragmentEncoder(node_input_dim=4, d_model=embed_d_model,
                              output_dim=embed_output_dim)
    embed_history: dict = {"loss": [], "pos_cos": [], "neg_cos": []}
    if embed_epochs > 0:
        if verbose:
            print(f"Training FragmentEncoder ({embed_epochs} epochs) …")
        embed_history = train_fragment_encoder(
            encoder, [fragments],
            n_epochs=embed_epochs, lr=embed_lr, margin=embed_margin,
            device=device, root_label_map=root_label_map, log_every=log_every,
        )
    elif verbose:
        print("Using random-init FragmentEncoder (embed_epochs=0) …")

    # 2. Encode fragments
    frags_enc = encode_fragments(encoder, fragments, device=device)

    # 3. Build observation graph
    graph = build_observation_graph(
        region, frags_enc, side=side, k_spatial=k_spatial,
        pos_scale_nm=pos_scale_nm, endpoint_radius_nm=endpoint_radius_nm,
    )

    n_types = int(graph.edge_type.max()) + 1 if len(graph.edge_type) > 0 else 2
    n_same = int((graph.edge_type == 0).sum())
    n_sp = int((graph.edge_type == 1).sum())
    n_ep = int((graph.edge_type == 2).sum())
    n_true = int(len(np.unique(graph.labels[graph.labels != 0])))
    if verbose:
        ep_str = f", {n_ep} endpoint-adj" if n_ep else ""
        print(f"ObservationGraph: {graph.n_nodes} nodes | {graph.n_edges} edges"
              f" ({n_same} same-frag, {n_sp} spatial{ep_str})")
        print(f"  node_dim={graph.node_dim}  true objects={n_true}")

    # 4. Baseline partition (random init)
    gnn_init = PartitionGNN(input_dim=graph.node_dim, d_model=embed_d_model,
                            output_dim=embed_output_dim, n_edge_types=n_types)
    pred_init = partition_observations(gnn_init, graph, threshold=threshold, device=device)
    r_before = evaluate_partition(pred_init, graph.labels)
    if verbose:
        print(f"ARI before training: {r_before['ari']:.4f}"
              f"  clusters={r_before['n_clusters_pred']}/{r_before['n_clusters_true']}")

    # 5. Train partition GNN
    if verbose:
        print(f"\nTraining PartitionGNN ({partition_epochs} epochs) …")
    gnn_trained, part_history = train_partition(
        graph, n_epochs=partition_epochs, lr=partition_lr,
        margin=partition_margin, max_pairs=partition_max_pairs,
        device=device, seed=seed, log_every=log_every,
    )

    # 6. Evaluate
    pred_trained = partition_observations(gnn_trained, graph, threshold=threshold,
                                          device=device)
    r_after = evaluate_partition(pred_trained, graph.labels)
    delta = r_after["ari"] - r_before["ari"]
    if verbose:
        print(f"ARI after training:  {r_after['ari']:.4f}  Δ={delta:+.4f}"
              f"  clusters={r_after['n_clusters_pred']}/{r_after['n_clusters_true']}")

    return {
        "ari_before": r_before["ari"],
        "ari_after": r_after["ari"],
        "delta_ari": delta,
        "homogeneity": r_after["homogeneity"],
        "completeness": r_after["completeness"],
        "v_measure": r_after["v_measure"],
        "n_clusters_pred": r_after["n_clusters_pred"],
        "n_clusters_true": r_after["n_clusters_true"],
        "embed_history": embed_history,
        "partition_history": part_history,
        "config": config,
    }


def optimize(
    fragments: list,
    region: Any,
    root_label_map: dict,
    *,
    n_trials: int = 20,
    objective: str = "ari_after",
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = True,
    # Fixed values (not searched)
    n_objects: Optional[int] = None,
    side: str = "pre",
    log_every: int = 0,
) -> dict:
    """Random hyperparameter search over the tree-stitching pipeline.

    Searches the following space:
        embed_d_model:         [32, 64, 128]
        embed_output_dim:      [16, 32, 64]
        embed_epochs:          [20, 40, 60, 80]
        embed_margin:          [0.5, 1.0, 2.0]
        k_spatial:             [4, 6, 8, 12]
        endpoint_radius_nm:    [None, 5_000, 10_000, 20_000]
        partition_epochs:      [20, 40, 60, 80]
        partition_margin:      [0.3, 0.5, 0.7]
        threshold:             [0.75, 0.80, 0.83, 0.85, 0.87, 0.90]
        pos_scale_nm:          [30_000, 50_000, 100_000]

    Parameters
    ----------
    n_trials:
        Number of random configurations to evaluate.
    objective:
        Key from the run_pipeline result dict to maximise.
        Default ``"ari_after"``.
    seed:
        RNG seed for the search (not for training — each trial uses its own
        seed derived from the trial index).

    Returns
    -------
    dict with keys:
        best_config — hyperparameter dict of the best trial
        best_score  — value of ``objective`` for the best trial
        all_results — list of (config, score) for all trials, sorted descending
    """
    rng = random.Random(seed)

    SPACE = {
        "embed_d_model":       [32, 64, 128],
        "embed_output_dim":    [16, 32, 64],
        "embed_epochs":        [20, 40, 60, 80],
        "embed_margin":        [0.5, 1.0, 2.0],
        "k_spatial":           [4, 6, 8, 12],
        "endpoint_radius_nm":  [None, 5_000, 10_000, 20_000],
        "partition_epochs":    [20, 40, 60, 80],
        "partition_margin":    [0.3, 0.5, 0.7],
        "threshold":           [0.75, 0.80, 0.83, 0.85, 0.87, 0.90],
        "pos_scale_nm":        [30_000.0, 50_000.0, 100_000.0],
    }

    all_results: list[tuple[dict, float]] = []
    best_score = -np.inf
    best_config: dict = {}

    for trial in range(n_trials):
        cfg = {k: rng.choice(v) for k, v in SPACE.items()}
        trial_seed = seed * 10000 + trial

        if verbose:
            print(f"\n{'='*60}")
            print(f"Trial {trial + 1}/{n_trials}:")
            print(f"  d_model={cfg['embed_d_model']}  out_dim={cfg['embed_output_dim']}"
                  f"  embed_ep={cfg['embed_epochs']}  margin={cfg['embed_margin']}")
            print(f"  k_spatial={cfg['k_spatial']}  ep_radius={cfg['endpoint_radius_nm']}"
                  f"  part_ep={cfg['partition_epochs']}  threshold={cfg['threshold']}")

        try:
            result = run_pipeline(
                fragments, region, root_label_map,
                embed_epochs=cfg["embed_epochs"],
                embed_d_model=cfg["embed_d_model"],
                embed_output_dim=cfg["embed_output_dim"],
                embed_margin=cfg["embed_margin"],
                side=side,
                k_spatial=cfg["k_spatial"],
                pos_scale_nm=cfg["pos_scale_nm"],
                endpoint_radius_nm=cfg["endpoint_radius_nm"],
                partition_epochs=cfg["partition_epochs"],
                partition_margin=cfg["partition_margin"],
                threshold=cfg["threshold"],
                device=device,
                seed=trial_seed,
                log_every=log_every,
                verbose=verbose,
            )
            score = float(result[objective])
        except Exception as exc:
            if verbose:
                print(f"  Trial failed: {exc}")
            score = -1.0

        all_results.append((cfg, score))
        if score > best_score:
            best_score = score
            best_config = cfg

        if verbose:
            print(f"  → {objective} = {score:.4f}  (best so far: {best_score:.4f})")

    all_results.sort(key=lambda x: -x[1])

    return {
        "best_config": best_config,
        "best_score": best_score,
        "all_results": all_results,
    }


__all__ = ["run_pipeline", "optimize"]
