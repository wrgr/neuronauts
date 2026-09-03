#!/usr/bin/env python3
"""Test conservative low-res scaffolding on synthetic connectome."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from attic.incubating_threads.low_res_segmentation.conservative_scaffolding import ConservativeScaffoldingPipeline
from attic.incubating_threads.low_res_segmentation.evaluate_neuron_purity import create_synthetic_connectome, evaluate_neuron_purity


def main():
    """Test scaffolding on synthetic data."""
    print("=" * 70)
    print("CONSERVATIVE LOW-RES SCAFFOLDING TEST")
    print("=" * 70)

    # Create synthetic connectome
    print("\n1. Creating synthetic connectome (5 neurons, 150 synapses)...")
    data = create_synthetic_connectome(n_neurons=5, synapses_per_neuron=30)
    vol = data["volume"]
    mem = data["membrane"]

    print(f"   ✓ Volume: {vol.shape}, range: {vol.min()}-{vol.max()}")
    print(f"   ✓ True neurons: {len(data['neurons'])}")
    print(f"   ✓ Synapses: {len(data['synapses']['pre_pt'])}")

    # Run scaffolding
    print("\n2. Running conservative scaffolding...")
    scaffolder = ConservativeScaffoldingPipeline(
        cell_body_threshold=70,    # Top 30%
        arbor_threshold=45,        # Intermediate brightness
        confidence_threshold=0.5,
        min_scaffold_size=20,
        max_merge_iterations=10,
    )

    result = scaffolder.scaffold_volume(vol, membrane_field=mem)

    # Print report
    print("\n" + scaffolder.report(result))

    # Evaluate purity
    print("\n3. Evaluating neuron purity...")
    purity = evaluate_neuron_purity(
        result.labels,
        data["synapses"]["pre_pt"],
        data["synapses"]["post_pt"],
        data["synapses"]["pre_root_id"],
        data["synapses"]["post_root_id"],
    )

    print(f"   Separation rate: {purity['separation_rate']:.1%}")
    print(f"   Correctly separated: {purity['correctly_separated']}/{purity['total_synapses']}")
    print(f"   False negatives (merges): {purity['false_negatives']}")
    print(f"   False positives (splits): {purity['false_positives']}")

    # Test filtering
    print("\n4. Testing confidence filtering (threshold=0.65)...")
    filtered = scaffolder.filter_low_confidence(result, threshold=0.65)
    print(f"   After filtering: {filtered.num_neurons} scaffolds (was {result.num_neurons})")

    # Test fragment merging
    print("\n5. Testing fragment merging...")
    merged = scaffolder.merge_fragments(result, max_distance_voxels=10)
    print(f"   After merge attempt: {merged.num_neurons} scaffolds")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Conservative scaffolding achieves:")
    print(f"  • {result.num_neurons} cell bodies with major arbors")
    print(f"  • {purity['separation_rate']:.1%} pre/post separation rate")
    print(f"  • {purity['false_positives']} false positives (excellent!)")
    print(f"  • Stopped at iteration {result.merging_stopped_at} (stopped early!)")
    print(f"  • Mean confidence: {np.mean([s.confidence for s in result.scaffolds]):.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
