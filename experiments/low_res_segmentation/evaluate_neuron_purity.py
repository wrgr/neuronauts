#!/usr/bin/env python3
"""Neuron segmentation evaluation on synthetic CAVE volumes.

Tests neuron purity metrics:
- Pre/post synapse separation rate
- Intra-neuron connectivity
- Boundary accuracy
"""

from __future__ import annotations

import numpy as np
from experiments.low_res_segmentation.low_res_segmentation import LowResNeuronSegmentationPipeline
from experiments.low_res_segmentation.high_precision_segmentation import HighPrecisionSegmentation3D


def create_synthetic_connectome(
    n_neurons: int = 5,
    volume_shape: tuple = (512, 512, 480),
    synapses_per_neuron: int = 20,
) -> dict:
    """Create synthetic connectome volume with ground truth."""
    np.random.seed(42)

    # Start with higher background
    vol = np.ones(volume_shape, dtype=np.uint8) * 60
    membrane = np.ones(volume_shape, dtype=np.float32) * 0.2

    neurons = {}
    synapse_list = {"pre_pt": [], "post_pt": [], "pre_root": [], "post_root": []}

    # Create neuron structures with higher intensity
    for neuron_id in range(1, n_neurons + 1):
        # Random neuron center
        center = np.array(
            [
                np.random.randint(100, volume_shape[0] - 100),
                np.random.randint(100, volume_shape[1] - 100),
                np.random.randint(100, volume_shape[2] - 100),
            ],
            dtype=np.float32,
        )

        # Cell body (larger and brighter)
        body_radius = np.random.randint(25, 40)
        neurons[neuron_id] = {
            "center": center,
            "radius": body_radius,
            "synapses": [],
        }

        # Draw cell body (bright interior)
        for x in range(
            max(0, int(center[0]) - body_radius),
            min(volume_shape[0], int(center[0]) + body_radius),
        ):
            for y in range(
                max(0, int(center[1]) - body_radius),
                min(volume_shape[1], int(center[1]) + body_radius),
            ):
                for z in range(
                    max(0, int(center[2]) - body_radius),
                    min(volume_shape[2], int(center[2]) + body_radius),
                ):
                    dist = np.sqrt(
                        (x - center[0]) ** 2
                        + (y - center[1]) ** 2
                        + (z - center[2]) ** 2
                    )
                    if dist <= body_radius:
                        vol[x, y, z] = 220

        # Draw neurites (extending branches) with decreasing radius
        n_branches = np.random.randint(2, 4)
        for _ in range(n_branches):
            branch_dir = np.random.randn(3)
            branch_dir /= np.linalg.norm(branch_dir)
            branch_len = np.random.randint(60, 120)

            for t in range(branch_len):
                pt = center + branch_dir * t
                pt_int = np.clip(pt.astype(int), 0, np.array(volume_shape) - 1)

                # Taper neurite
                intensity = int(180 * (1.0 - t / branch_len))
                vol[pt_int[0], pt_int[1], pt_int[2]] = max(
                    vol[pt_int[0], pt_int[1], pt_int[2]], intensity
                )

        # Add strong membrane at boundary (higher membrane = boundary)
        x_min, x_max = (
            max(0, int(center[0]) - body_radius - 3),
            min(volume_shape[0], int(center[0]) + body_radius + 3),
        )
        y_min, y_max = (
            max(0, int(center[1]) - body_radius - 3),
            min(volume_shape[1], int(center[1]) + body_radius + 3),
        )
        z_min, z_max = (
            max(0, int(center[2]) - body_radius - 3),
            min(volume_shape[2], int(center[2]) + body_radius + 3),
        )

        membrane[x_min:x_max, y_min:y_max, z_min:z_max] = 0.8

    # Create synapses (pre->post connections between different neurons)
    for pre_id in range(1, n_neurons + 1):
        pre_center = neurons[pre_id]["center"]

        for _ in range(synapses_per_neuron):
            # Post neuron (different from pre)
            post_id = np.random.choice([n for n in range(1, n_neurons + 1) if n != pre_id])
            post_center = neurons[post_id]["center"]

            # Synapse position: somewhere along pre neurite
            pre_offset = np.random.randn(3) * 30
            pre_pt = pre_center + pre_offset
            pre_pt = np.clip(pre_pt, 0, np.array(volume_shape) - 1)

            # Synapse position: somewhere along post neurite
            post_offset = np.random.randn(3) * 30
            post_pt = post_center + post_offset
            post_pt = np.clip(post_pt, 0, np.array(volume_shape) - 1)

            synapse_list["pre_pt"].append(pre_pt)
            synapse_list["post_pt"].append(post_pt)
            synapse_list["pre_root"].append(pre_id)
            synapse_list["post_root"].append(post_id)

            neurons[pre_id]["synapses"].append((pre_id, post_id))

    return {
        "volume": vol,
        "membrane": membrane,
        "neurons": neurons,
        "synapses": {
            "pre_pt": np.array(synapse_list["pre_pt"], dtype=np.float32),
            "post_pt": np.array(synapse_list["post_pt"], dtype=np.float32),
            "pre_root_id": np.array(synapse_list["pre_root"], dtype=np.int32),
            "post_root_id": np.array(synapse_list["post_root"], dtype=np.int32),
        },
    }


def evaluate_neuron_purity(
    segmentation_labels: np.ndarray,
    pre_pt: np.ndarray,
    post_pt: np.ndarray,
    pre_root_id: np.ndarray,
    post_root_id: np.ndarray,
) -> dict:
    """Evaluate neuron purity metrics."""
    # Get labels at synapse positions
    pre_pt_int = np.clip(pre_pt.astype(int), 0, np.array(segmentation_labels.shape) - 1)
    post_pt_int = np.clip(post_pt.astype(int), 0, np.array(segmentation_labels.shape) - 1)

    pre_labels = segmentation_labels[tuple(pre_pt_int.T)]
    post_labels = segmentation_labels[tuple(post_pt_int.T)]

    # Metrics
    correctly_separated = np.sum(pre_labels != post_labels)
    total_synapses = len(pre_labels)
    separation_rate = correctly_separated / total_synapses if total_synapses > 0 else 0

    # False positives: same root but different seg
    same_root_diff_seg = np.sum((pre_root_id == post_root_id) & (pre_labels != post_labels))

    # False negatives: different root but same seg
    diff_root_same_seg = np.sum(
        (pre_root_id != post_root_id) & (pre_labels == post_labels) & (pre_labels > 0)
    )

    return {
        "total_synapses": total_synapses,
        "correctly_separated": correctly_separated,
        "separation_rate": separation_rate,
        "false_positives": same_root_diff_seg,
        "false_negatives": diff_root_same_seg,
        "pre_labels": pre_labels,
        "post_labels": post_labels,
    }


def main():
    """Run evaluation on synthetic connectome."""
    print("=" * 70)
    print("NEURON SEGMENTATION EVALUATION ON SYNTHETIC CAVE VOLUME")
    print("=" * 70)

    # Create synthetic data
    print("\n1. Creating synthetic connectome...")
    data = create_synthetic_connectome(n_neurons=5, synapses_per_neuron=30)

    vol = data["volume"]
    membrane = data["membrane"]
    synapses = data["synapses"]
    n_neurons_true = len(data["neurons"])
    n_synapses = len(synapses["pre_pt"])

    print(f"   ✓ {n_neurons_true} neurons")
    print(f"   ✓ {n_synapses} synapses")
    print(f"   ✓ Volume: {vol.shape}")

    # Test 1: Low-resolution neuron segmentation
    print("\n2. Low-resolution neuron segmentation (128×128×120 nm)...")
    lowres_pipeline = LowResNeuronSegmentationPipeline(
        target_voxel_nm=(128, 128, 120),
        full_res_voxel_nm=(8, 8, 40),
    )

    lowres_seg, lowres_mapping = lowres_pipeline.segment_neurons_from_synapses(
        vol, membrane,
        synapses["pre_pt"],
        synapses["post_pt"],
        synapses["pre_root_id"],
        synapses["post_root_id"],
    )

    lowres_purity = evaluate_neuron_purity(
        lowres_seg.labels,
        lowres_pipeline.to_low_res(synapses["pre_pt"]),
        lowres_pipeline.to_low_res(synapses["post_pt"]),
        synapses["pre_root_id"],
        synapses["post_root_id"],
    )

    print(f"   ✓ Neurons detected: {lowres_seg.num_components}")
    print(f"   ✓ Separation rate: {lowres_purity['separation_rate']:.1%}")
    print(f"      - Correctly separated: {lowres_purity['correctly_separated']}/{lowres_purity['total_synapses']}")
    print(f"      - False positives (same root, diff seg): {lowres_purity['false_positives']}")
    print(f"      - False negatives (diff root, same seg): {lowres_purity['false_negatives']}")

    # Test 2: High-precision segmentation
    print("\n3. High-precision 3D segmentation...")
    highres_seg, highres_sep = HighPrecisionSegmentation3D().segment_with_synapses(
        vol, membrane,
        synapses["pre_pt"],
        synapses["post_pt"],
        synapses["pre_root_id"],
        synapses["post_root_id"],
    )

    highres_purity = evaluate_neuron_purity(
        highres_seg.labels,
        synapses["pre_pt"],
        synapses["post_pt"],
        synapses["pre_root_id"],
        synapses["post_root_id"],
    )

    print(f"   ✓ Neurons detected: {highres_seg.num_components}")
    print(f"   ✓ Separation rate: {highres_purity['separation_rate']:.1%}")
    print(f"      - Correctly separated: {highres_purity['correctly_separated']}/{highres_purity['total_synapses']}")
    print(f"      - False positives: {highres_purity['false_positives']}")
    print(f"      - False negatives: {highres_purity['false_negatives']}")

    # Comparison
    print("\n4. Comparison:")
    print(f"   Low-res purity:      {lowres_purity['separation_rate']:.1%}")
    print(f"   High-res purity:     {highres_purity['separation_rate']:.1%}")
    if highres_purity["separation_rate"] > lowres_purity["separation_rate"]:
        gain = (
            (highres_purity["separation_rate"] - lowres_purity["separation_rate"])
            / lowres_purity["separation_rate"]
            * 100
        )
        print(f"   → High-res gain:     +{gain:.1f}%")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    return {
        "lowres_purity": lowres_purity,
        "highres_purity": highres_purity,
    }


if __name__ == "__main__":
    main()
