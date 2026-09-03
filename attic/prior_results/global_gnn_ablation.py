#!/usr/bin/env python3
"""Phase 2 end-to-end ablation: DNA encoder → global synapse graph → GNN.

Pipeline
--------
1. Bisect real minnie65 skeletons (same hard-split approach as Phase 1).
2. Train TreeDNAEncoder on bisected fragments.
3. Encode all fragments → Fragment.dna.
4. Build global synapse graph (k-NN with DNA node features).
5. Train CellGNN on the global graph (contrastive loss on pre_root_id labels).
6. Evaluate pair AUC: raw DNA vs GNN-refined embeddings.

Usage
-----
  python scripts/global_gnn_ablation.py --n-neurons 40 --dna-epochs 60 --gnn-epochs 50
  python scripts/global_gnn_ablation.py --synthetic   # fast offline test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def run_gnn_ablation(
    region,
    fragments,
    root_label_map: dict,
    *,
    dna_epochs: int = 60,
    gnn_epochs: int = 50,
    dna_lr: float = 1e-3,
    gnn_lr: float = 1e-3,
    d_model: int = 64,
    output_dim: int = 32,
    gnn_d_model: int = 64,
    gnn_embedding_dim: int = 32,
    n_paths: int = 6,
    k_neighbors: int = 8,
    max_pairs: int = 500,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """Full Phase 2 ablation: DNA encoder → GNN → evaluate."""
    from sklearn.metrics import roc_auc_score

    from neuronauts.assemble import build_global_synapse_graph, run_global_gnn, train_global_gnn
    from neuronauts.represent.dna import TreeDNAEncoder, encode_fragments, train_dna_encoder
    from neuronauts.represent.enrich import evaluate_dna_auc, synapse_pair_dna_scores
    from neuronauts.schemas import Fragment

    rng = np.random.default_rng(seed)

    # --- Step 1: train DNA encoder ---
    n_neurons = len(set(v for vals in root_label_map.values() for v in vals))
    print(f"\n{'='*60}")
    print(f"Region: {region.n_synapses} synapses, {len(fragments)} fragments")
    print(f"Neurons: {n_neurons}")
    print(f"{'='*60}\n")

    encoder = TreeDNAEncoder(
        n_paths=n_paths,
        d_model=d_model,
        output_dim=output_dim,
        max_path_len=128,
    )

    print("Evaluating raw DNA AUC (random init) …")
    frags_rand = encode_fragments(encoder, fragments, device=device)
    metrics_rand = evaluate_dna_auc(region, frags_rand, max_pairs=max_pairs,
                                     rng=np.random.default_rng(seed))
    print(f"  DNA AUC (random init):   {metrics_rand['dna_auc']:.4f}")
    print(f"  Spatial baseline AUC:    {metrics_rand['baseline_auc']:.4f}")

    print(f"\nTraining TreeDNAEncoder for {dna_epochs} epochs …")
    train_dna_encoder(
        encoder, [fragments],
        n_epochs=dna_epochs,
        lr=dna_lr,
        device=device,
        root_label_map=root_label_map,
    )

    frags_trained = encode_fragments(encoder, fragments, device=device)
    metrics_dna = evaluate_dna_auc(region, frags_trained, max_pairs=max_pairs,
                                    rng=np.random.default_rng(seed))
    print(f"\n  DNA AUC (trained):       {metrics_dna['dna_auc']:.4f}")
    print(f"  Spatial baseline AUC:    {metrics_dna['baseline_auc']:.4f}")

    # --- Step 2: build global synapse graph ---
    print(f"\nBuilding global synapse graph (k={k_neighbors}) …")
    graph = build_global_synapse_graph(region, frags_trained, k_neighbors=k_neighbors)
    print(f"  {graph.n_synapses} nodes, {graph.n_edges} edges, DNA dim={graph.dna_dim}")

    # --- Step 3: train GNN ---
    print(f"\nTraining CellGNN for {gnn_epochs} epochs …")
    gnn, gnn_history = train_global_gnn(
        graph,
        n_epochs=gnn_epochs,
        lr=gnn_lr,
        d_model=gnn_d_model,
        embedding_dim=gnn_embedding_dim,
        max_pairs=max_pairs,
        device=device,
        seed=seed,
        log_every=10,
    )

    # --- Step 4: evaluate GNN embeddings ---
    gnn_emb = run_global_gnn(graph, gnn, device=device)

    # Assign mean GNN embedding to each fragment → evaluate via synapse_pair_dna_scores
    gnn_frags = []
    for frag in frags_trained:
        sub_emb = gnn_emb[frag.synapse_indices]
        mean_emb = sub_emb.mean(axis=0).astype(np.float32)
        gnn_frags.append(Fragment(
            fragment_id=frag.fragment_id,
            region_id=frag.region_id,
            base_root_id=frag.base_root_id,
            vertices_nm=frag.vertices_nm,
            edges=frag.edges,
            endpoints_nm=frag.endpoints_nm,
            radius_nm=frag.radius_nm,
            synapse_indices=frag.synapse_indices,
            dna=mean_emb,
        ).validate())

    scores_gnn, labels_gnn = synapse_pair_dna_scores(
        region, gnn_frags, max_pairs=max_pairs, rng=np.random.default_rng(seed)
    )
    auc_gnn = float(roc_auc_score(labels_gnn, scores_gnn)) if len(np.unique(labels_gnn)) > 1 else float("nan")

    print(f"\n  GNN AUC:                 {auc_gnn:.4f}")
    print(f"\n{'='*60}")
    print(f"  DNA AUC (random init):   {metrics_rand['dna_auc']:.4f}")
    print(f"  DNA AUC (trained):       {metrics_dna['dna_auc']:.4f}")
    print(f"  GNN AUC (trained DNA):   {auc_gnn:.4f}")
    print(f"  Spatial baseline:        {metrics_rand['baseline_auc']:.4f}")
    gnn_vs_dna = auc_gnn - metrics_dna["dna_auc"]
    sign = "+" if gnn_vs_dna >= 0 else ""
    print(f"  GNN vs DNA improvement:  {sign}{gnn_vs_dna:.4f}")
    print(f"{'='*60}\n")

    return {
        "dna_auc_random": metrics_rand["dna_auc"],
        "dna_auc_trained": metrics_dna["dna_auc"],
        "gnn_auc": auc_gnn,
        "baseline_auc": metrics_rand["baseline_auc"],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic data (no network access required)")
    p.add_argument("--n-neurons", type=int, default=40)
    p.add_argument("--dna-epochs", type=int, default=60)
    p.add_argument("--gnn-epochs", type=int, default=50)
    p.add_argument("--dna-lr", type=float, default=1e-3)
    p.add_argument("--gnn-lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--output-dim", type=int, default=32)
    p.add_argument("--gnn-d-model", type=int, default=64)
    p.add_argument("--gnn-embedding-dim", type=int, default=32)
    p.add_argument("--n-paths", type=int, default=6)
    p.add_argument("--k-neighbors", type=int, default=8)
    p.add_argument("--max-pairs", type=int, default=500)
    p.add_argument("--synapses-per-half", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.synthetic:
        sys.path.insert(0, str(_ROOT / "scripts"))
        from ablate_dna import _make_synthetic_world
        region, fragments, root_label_map = _make_synthetic_world(
            n_neurons=args.n_neurons,
            roots_per_neuron=2,
            synapses_per_root=args.synapses_per_half,
            rng=rng,
        )
    else:
        import gzip
        import io
        import requests
        sys.path.insert(0, str(_ROOT / "scripts"))
        from half_split_ablation import (
            NUCLEUS_URL_V1412,
            build_split_world,
        )
        print("Fetching v1412 nucleus CSV …")
        resp = requests.get(NUCLEUS_URL_V1412, timeout=60)
        resp.raise_for_status()
        root_ids: list[int] = []
        with gzip.open(io.BytesIO(resp.content)) as f:
            for line in f:
                parts = line.decode().strip().split(",")
                if len(parts) >= 4:
                    try:
                        r = int(parts[3])
                        if r != 0:
                            root_ids.append(r)
                    except ValueError:
                        pass
        print(f"  {len(root_ids)} proofread neurons at v1412")
        candidates = rng.choice(
            root_ids, size=min(args.n_neurons * 6, len(root_ids)), replace=False
        ).tolist()
        region, fragments, root_label_map = build_split_world(
            candidates,
            n_target=args.n_neurons,
            synapses_per_half=args.synapses_per_half,
            rng=rng,
        )

    run_gnn_ablation(
        region, fragments, root_label_map,
        dna_epochs=args.dna_epochs,
        gnn_epochs=args.gnn_epochs,
        dna_lr=args.dna_lr,
        gnn_lr=args.gnn_lr,
        d_model=args.d_model,
        output_dim=args.output_dim,
        gnn_d_model=args.gnn_d_model,
        gnn_embedding_dim=args.gnn_embedding_dim,
        n_paths=args.n_paths,
        k_neighbors=args.k_neighbors,
        max_pairs=args.max_pairs,
        device=args.device,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
