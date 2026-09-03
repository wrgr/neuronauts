#!/usr/bin/env python3
"""DNA-AUC ablation: same-neuron synapse-pair classification.

Evaluates how well the learned tree-DNA embedding separates same-neuron
synapse pairs vs a spatial-proximity baseline.

Two modes:
  --synthetic   Generate a realistic multi-neuron world on the fly (no data needed).
  --archive     Run on a real kimimaro skeleton archive + Region npz.

Usage
-----
  # Synthetic demo (runs anywhere, no data required)
  python attic/prior_results/ablate_dna.py --synthetic --neurons 6 --roots-per-neuron 4

  # Real data
  python attic/prior_results/ablate_dna.py --archive data/skeleton_archive.npz \
      --region data/region.npz \
      --epochs 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path when running as a script.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _make_synthetic_world(
    n_neurons: int,
    roots_per_neuron: int,
    synapses_per_root: int,
    rng: np.random.Generator,
    region_nm: float = 100_000.0,
    vertices_per_root: int = 40,
) -> tuple:
    """Build a synthetic Region + list of Fragments for ablation demo.

    Each neuron has a distinct morphological type (branch factor, step size,
    radius profile) shared across all its seg roots.  Synapses are placed
    uniformly at random (no spatial clustering) so the spatial baseline stays
    near chance while DNA can discriminate by morphology.

    Returns (region, fragments, root_label_map).
    """
    from neuronauts.schemas import Fragment, Region

    all_pre_pts: list[np.ndarray] = []
    all_post_pts: list[np.ndarray] = []
    all_pre_roots: list[int] = []
    all_post_roots: list[int] = []
    all_syn_ids: list[int] = []
    fragments: list[Fragment] = []
    root_label_map: dict[int, set[int]] = {}

    syn_idx = 0
    root_id = 100  # unique across all seg roots

    # One morphological "type" per neuron — defines its skeleton character.
    neuron_types = []
    for ni in range(n_neurons):
        trng = np.random.default_rng(ni * 17 + 3)
        neuron_types.append({
            "branch_factor": int(trng.integers(2, 6)),     # 2–5 branches
            "step_nm": float(trng.uniform(1000, 8000)),    # branch arm length
            "taper": float(trng.uniform(0.3, 1.0)),        # radius taper ratio
            "n_leaves": int(trng.integers(3, 10)),         # leaves per branch
        })

    for neuron_idx in range(n_neurons):
        label_root = neuron_idx + 1
        ntype = neuron_types[neuron_idx]

        for r in range(roots_per_neuron):
            rid = root_id
            root_label_map[rid] = {label_root}
            root_id += 1

            # Anchor anywhere in the region volume (no spatial clustering)
            root_pt = rng.uniform(0, region_nm, 3).astype(np.float32)
            bf = ntype["branch_factor"]
            step = ntype["step_nm"]

            # Tree: root → bf branches (layer 1) → n_leaves leaves each
            verts = [root_pt]
            edges_list = []
            radii_list = [300.0]
            layer1_indices = []

            for b in range(bf):
                angle = 2 * np.pi * b / bf + rng.uniform(-0.1, 0.1)
                dip = rng.uniform(-0.3, 0.3)
                branch_dir = np.array([np.cos(angle), np.sin(angle), dip], dtype=np.float32)
                branch_dir /= np.linalg.norm(branch_dir)
                branch_pt = root_pt + branch_dir * step
                layer1_indices.append(len(verts))
                verts.append(branch_pt)
                edges_list.append([0, len(verts) - 1])
                radii_list.append(300.0 * ntype["taper"])

                for lf in range(ntype["n_leaves"]):
                    leaf_dir = branch_dir + rng.normal(0, 0.4, 3)
                    leaf_dir /= np.linalg.norm(leaf_dir)
                    leaf_pt = branch_pt + leaf_dir * step * 0.6
                    verts.append(leaf_pt.astype(np.float32))
                    edges_list.append([layer1_indices[-1], len(verts) - 1])
                    radii_list.append(150.0 * ntype["taper"])

            verts_nm = np.array(verts, dtype=np.float32)
            edges_arr = np.array(edges_list, dtype=np.int64)
            radii_nm = np.array(radii_list, dtype=np.float32)

            # Synapses: UNIFORM RANDOM placement (not near skeleton)
            # → spatial proximity baseline stays near chance
            pre_pts = rng.uniform(0, region_nm, (synapses_per_root, 3)).astype(np.float32)
            post_pts = pre_pts + rng.normal(0, 200, (synapses_per_root, 3)).astype(np.float32)

            syn_indices = list(range(syn_idx, syn_idx + synapses_per_root))
            all_pre_pts.append(pre_pts)
            all_post_pts.append(post_pts)
            all_pre_roots.extend([label_root] * synapses_per_root)
            all_post_roots.extend([0] * synapses_per_root)
            all_syn_ids.extend(syn_indices)
            syn_idx += synapses_per_root

            # Find leaf vertices (degree ≤ 1)
            from collections import Counter
            deg = Counter()
            for u, v in edges_arr:
                deg[u] += 1
                deg[v] += 1
            leaves = [i for i in range(len(verts)) if deg.get(i, 0) <= 1]
            endpoints_nm = verts_nm[leaves]

            frag = Fragment(
                fragment_id=rid,
                region_id="synthetic",
                base_root_id=rid,
                vertices_nm=verts_nm,
                edges=edges_arr,
                endpoints_nm=endpoints_nm,
                radius_nm=radii_nm,
                synapse_indices=np.array(syn_indices, dtype=np.int64),
                dna=None,
            ).validate()
            fragments.append(frag)

    n_syn = syn_idx
    region = Region(
        region_id="synthetic",
        bbox_nm=((0.0, 0.0, 0.0), (region_nm, region_nm, region_nm)),
        voxel_size_nm=(8.0, 8.0, 40.0),
        seg_version=117,
        label_version=1412,
        pre_pt_nm=np.concatenate(all_pre_pts).astype(np.float32),
        post_pt_nm=np.concatenate(all_post_pts).astype(np.float32),
        pre_root_id=np.array(all_pre_roots, dtype=np.int64),
        post_root_id=np.array(all_post_roots, dtype=np.int64),
        synapse_id=np.array(all_syn_ids, dtype=np.int64),
    )
    # Sanity: branch_factor removed from signature, update docstring reference
    return region, fragments, root_label_map


def run_ablation(
    region,
    fragments,
    root_label_map,
    *,
    n_epochs: int = 30,
    lr: float = 1e-3,
    d_model: int = 64,
    output_dim: int = 32,
    n_paths: int = 8,
    max_pairs: int = 2000,
    device: str = "cpu",
    seed: int = 42,
    encoder_type: str = "path",
) -> dict:
    """Train a DNA encoder and report AUC before / after training.

    encoder_type : "path"  — TreeDNAEncoder (path-sampling, hand-crafted 6-D features)
                   "gnn"   — SkeletonGNN (graph-level, raw (x,y,z,r) node features)
    """
    from neuronauts.represent.enrich import evaluate_dna_auc

    rng = np.random.default_rng(seed)

    n_neurons = len(set(
        next(iter(root_label_map[f.base_root_id]))
        for f in fragments
        if f.base_root_id in root_label_map
    )) if root_label_map else "?"

    print(f"\n{'='*60}")
    print(f"Encoder: {encoder_type}")
    print(f"Region: {region.n_synapses} synapses, {len(fragments)} seg roots")
    print(f"Neurons: {n_neurons}")
    print(f"{'='*60}\n")

    if encoder_type == "vicreg":
        from neuronauts.global_merge.represent.vicreg_gnn import (
            VICRegSkeletonModel,
            train_vicreg_skeleton_gnn,
        )
        from neuronauts.represent.skeleton_gnn import fragment_to_tensors
        import torch

        model = VICRegSkeletonModel(in_dim=4, emb_dim=output_dim, proj_dim=output_dim * 2)

        def encode_frags_vicreg(m, frags):
            m.eval()
            encoded = []
            with torch.no_grad():
                for f in frags:
                    nf, es, ed, _ = fragment_to_tensors(f, device="cpu")
                    if nf.size(0) == 0:
                        dna = np.zeros(output_dim, dtype=np.float32)
                    else:
                        e_tens = torch.stack([es, ed], dim=1) if es.size(0) > 0 else torch.empty((0, 2), dtype=torch.long)
                        h = m.backbone(nf[:, :4], e_tens)
                        h = torch.nn.functional.normalize(h, p=2, dim=-1)
                        dna = h.cpu().numpy()
                    import dataclasses
                    encoded.append(dataclasses.replace(f, dna=dna))
            return encoded

        print("Evaluating random-init VICReg DNA AUC (before training)...")
        frags_init = encode_frags_vicreg(model, fragments)
        result_before = evaluate_dna_auc(
            region, frags_init, max_pairs=max_pairs,
            rng=np.random.default_rng(seed), include_baseline=True,
        )
        _print_before(result_before)

        # Build positive pairs using ground-truth root_label_map
        from collections import defaultdict
        gt_to_indices = defaultdict(list)
        for idx, f in enumerate(fragments):
            if root_label_map and f.base_root_id in root_label_map:
                gt = next(iter(root_label_map[f.base_root_id]))
            else:
                # If no map, assume pairs are consecutive bisected halves (idx // 2)
                gt = idx // 2
            gt_to_indices[gt].append(idx)

        pos_pairs = []
        for gt, idxs in gt_to_indices.items():
            if len(idxs) >= 2:
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        pos_pairs.append((idxs[i], idxs[j]))

        print(f"\nTraining VICRegSkeletonModel ({len(pos_pairs)} positive pairs) for {n_epochs} epochs...")
        history = train_vicreg_skeleton_gnn(
            model, fragments, pos_pairs,
            n_epochs=n_epochs, lr=lr, device=device, log_every=10,
        )

        print("\nEvaluating trained VICReg DNA AUC...")
        frags_trained = encode_frags_vicreg(model, fragments)

    elif encoder_type == "gnn":
        from neuronauts.represent.skeleton_gnn import (
            SkeletonGNN,
            encode_fragments_gnn,
            train_skeleton_gnn,
        )
        encoder = SkeletonGNN(d_model=d_model, output_dim=output_dim, n_layers=3)

        print("Evaluating random-init DNA AUC (before training)...")
        frags_init = encode_fragments_gnn(encoder, fragments, device=device)
        result_before = evaluate_dna_auc(
            region, frags_init, max_pairs=max_pairs,
            rng=np.random.default_rng(seed), include_baseline=True,
        )
        _print_before(result_before)

        print(f"\nTraining SkeletonGNN for {n_epochs} epochs...")
        history = train_skeleton_gnn(
            encoder, [fragments],
            n_epochs=n_epochs, lr=lr, margin=1.0,
            device=device, root_label_map=root_label_map, log_every=10,
        )
        _print_history(history, key_loss="loss", key_pos="pos_cos", key_neg="neg_cos")

        print("\nEvaluating trained DNA AUC...")
        frags_trained = encode_fragments_gnn(encoder, fragments, device=device)

    else:  # "path"
        from neuronauts.represent.dna import TreeDNAEncoder, encode_fragments, train_dna_encoder

        encoder = TreeDNAEncoder(
            d_model=d_model, n_heads=4, n_layers=2,
            output_dim=output_dim, n_paths=n_paths,
            max_path_len=128,
        )

        print("Evaluating random-init DNA AUC (before training)...")
        frags_init = encode_fragments(encoder, fragments, device=device, n_paths=n_paths)
        result_before = evaluate_dna_auc(
            region, frags_init, max_pairs=max_pairs,
            rng=np.random.default_rng(seed), include_baseline=True,
        )
        _print_before(result_before)

        print(f"\nTraining TreeDNAEncoder for {n_epochs} epochs...")
        history = train_dna_encoder(
            encoder, [fragments],
            n_epochs=n_epochs, lr=lr, margin=0.5, batch_size=32,
            device=device, root_label_map=root_label_map, n_paths=n_paths,
        )
        _print_history(history, key_loss="loss", key_pos="pos_cosine", key_neg="neg_cosine")

        print("\nEvaluating trained DNA AUC...")
        frags_trained = encode_fragments(encoder, fragments, device=device, n_paths=n_paths)

    result_after = evaluate_dna_auc(
        region, frags_trained, max_pairs=max_pairs,
        rng=np.random.default_rng(seed), include_baseline=True,
    )
    print(f"  DNA AUC (trained):       {result_after['dna_auc']:.4f}")
    print(f"  Spatial baseline AUC:    {result_after['baseline_auc']:.4f}")

    delta = result_after["dna_auc"] - result_before["dna_auc"]
    baseline_auc = result_after.get("baseline_auc", float("nan"))
    trained_auc = result_after["dna_auc"]

    print(f"\n{'='*60}")
    print(f"  AUC improvement:  {delta:+.4f}  (random → trained)")
    if trained_auc > baseline_auc:
        print(f"  vs spatial baseline: +{trained_auc - baseline_auc:.4f}  ✓  DNA beats proximity")
    else:
        print(f"  vs spatial baseline: -{baseline_auc - trained_auc:.4f}  (needs more training / data)")
    print(f"{'='*60}\n")

    return {
        "before": result_before,
        "after": result_after,
        "history": history,
        "delta_auc": delta,
    }


def _print_before(result: dict) -> None:
    print(f"  DNA AUC (random init):   {result['dna_auc']:.4f}")
    if "baseline_auc" in result:
        print(f"  Spatial baseline AUC:    {result['baseline_auc']:.4f}")
    print(f"  Pairs: {result['n_pos']} pos / {result['n_neg']} neg")
    print(f"  Synapses without DNA:    {result['n_no_dna']}")


def _print_history(history: dict, *, key_loss: str, key_pos: str, key_neg: str) -> None:
    losses = history.get(key_loss, [])
    pos_cos = history.get(key_pos, [])
    neg_cos = history.get(key_neg, [])
    for i, (loss, pc, nc) in enumerate(zip(losses, pos_cos, neg_cos)):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  epoch {i+1:3d}: loss={loss:.4f}  pos_cos={pc:.3f}  neg_cos={nc:.3f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true",
                      help="Generate synthetic multi-neuron world (no data needed)")
    mode.add_argument("--archive", type=Path, metavar="NPZ",
                      help="Path to kimimaro skeleton archive (.npz)")

    # Synthetic options
    p.add_argument("--neurons", type=int, default=6, help="Number of neurons (synthetic)")
    p.add_argument("--roots-per-neuron", type=int, default=4, help="Seg roots per neuron (synthetic)")
    p.add_argument("--synapses-per-root", type=int, default=8, help="Synapses per seg root (synthetic)")

    # Real data options
    p.add_argument("--region", type=Path, help="Region .npz (required with --archive)")

    # Training options
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--output-dim", type=int, default=32)
    p.add_argument("--n-paths", type=int, default=8)
    p.add_argument("--max-pairs", type=int, default=2000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--encoder", choices=["path", "gnn", "vicreg"], default="path",
                   help="path=TreeDNAEncoder (hand-crafted features), gnn=SkeletonGNN (data-driven)")
    args = p.parse_args()

    if args.synthetic:
        rng = np.random.default_rng(args.seed)
        print(f"Generating synthetic world: {args.neurons} neurons × "
              f"{args.roots_per_neuron} roots × {args.synapses_per_root} synapses/root")
        region, fragments, root_label_map = _make_synthetic_world(
            n_neurons=args.neurons,
            roots_per_neuron=args.roots_per_neuron,
            synapses_per_root=args.synapses_per_root,
            rng=rng,
        )
    else:
        if args.region is None:
            p.error("--region is required with --archive")
        from neuronauts.data.fragments import extract_fragments_for_region
        from neuronauts.schemas import Region
        region = Region.load(args.region)
        fragments = extract_fragments_for_region(region, str(args.archive))
        # No contamination map available without CAVE, treat all roots as clean.
        root_label_map = None
        print(f"Loaded {len(fragments)} fragments from archive")

    result = run_ablation(
        region, fragments, root_label_map,
        n_epochs=args.epochs,
        lr=args.lr,
        d_model=args.d_model,
        output_dim=args.output_dim,
        n_paths=args.n_paths,
        max_pairs=args.max_pairs,
        device=args.device,
        seed=args.seed,
        encoder_type=args.encoder,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
