#!/usr/bin/env python3
"""Fetch a real Minnie65 box from CAVE and run conservative scaffolding.

Pulls a small EM volume + synapses, runs the scaffolding pipeline, and
evaluates neuron purity using ground-truth root IDs from CAVE.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from attic.incubating_threads.low_res_segmentation.conservative_scaffolding import ConservativeScaffoldingPipeline
from attic.incubating_threads.low_res_segmentation.evaluate_neuron_purity import evaluate_neuron_purity


# MICrONS Minnie65
CAVE_SERVER = "https://global.daf-apis.com"
DATASTACK = "minnie65_public"
SYNAPSE_VOX_NM = np.array([4.0, 4.0, 40.0])  # synapse table voxel size

# Pick a small box near the center of the proofread volume
BOX_CENTER_NM = np.array([213_000, 413_000, 803_000], dtype=np.float64)
BOX_SIDE_UM = 12.0  # 12 micron cube


def fetch_box():
    """Fetch synapses + EM for a small Minnie65 box."""
    from caveclient import CAVEclient

    token = os.environ.get("CAVE_TOKEN")

    print(f"Connecting to CAVE ({DATASTACK})...")
    client = CAVEclient(DATASTACK, server_address=CAVE_SERVER, auth_token=token)

    # Box bounds in nm
    half = BOX_SIDE_UM * 1000.0 / 2.0
    bbox_nm = np.array(
        [BOX_CENTER_NM - half, BOX_CENTER_NM + half],
        dtype=np.float64,
    )
    print(f"Box: center={BOX_CENTER_NM} nm, side={BOX_SIDE_UM} um")
    print(f"Bounds (nm): {bbox_nm[0]} to {bbox_nm[1]}")

    # Bounds in synapse-table voxels
    bbox_vox = bbox_nm / SYNAPSE_VOX_NM[None, :]

    # Query synapses inside the box
    print("\nQuerying synapses...")
    df = client.materialize.query_table(
        "synapses_pni_2",
        filter_in_dict={},
        filter_spatial_dict={
            "pre_pt_position": bbox_vox.tolist(),
        },
        limit=2000,
    )
    print(f"  Got {len(df)} synapses")

    if len(df) == 0:
        print("No synapses in this box, trying a different center...")
        return None

    # Extract pre/post positions and root IDs
    pre_pt_vox = np.array(df["pre_pt_position"].tolist(), dtype=np.float64)
    post_pt_vox = np.array(df["post_pt_position"].tolist(), dtype=np.float64)
    pre_root_id = df["pre_pt_root_id"].values
    post_root_id = df["post_pt_root_id"].values

    # Filter to those with both endpoints in the box
    pre_nm = pre_pt_vox * SYNAPSE_VOX_NM
    post_nm = post_pt_vox * SYNAPSE_VOX_NM
    in_box = (
        np.all(pre_nm >= bbox_nm[0], axis=1)
        & np.all(pre_nm < bbox_nm[1], axis=1)
        & np.all(post_nm >= bbox_nm[0], axis=1)
        & np.all(post_nm < bbox_nm[1], axis=1)
    )

    pre_nm = pre_nm[in_box]
    post_nm = post_nm[in_box]
    pre_root_id = pre_root_id[in_box]
    post_root_id = post_root_id[in_box]

    print(f"  {len(pre_nm)} synapses fully inside box")
    print(f"  Unique pre_root_ids:  {len(np.unique(pre_root_id[pre_root_id > 0]))}")
    print(f"  Unique post_root_ids: {len(np.unique(post_root_id[post_root_id > 0]))}")

    # Fetch a low-res EM volume via cloud-volume
    print("\nFetching EM volume (low-res)...")
    from cloudvolume import CloudVolume

    em_url = "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em"
    cv = CloudVolume(em_url, mip=4, use_https=True, fill_missing=True, progress=False)
    em_vox_nm = np.array(cv.resolution)
    print(f"  EM voxel size at mip=4: {em_vox_nm} nm")

    bbox_em = (bbox_nm / em_vox_nm[None, :]).astype(int)
    print(f"  EM bounds (vox): {bbox_em[0]} to {bbox_em[1]}")

    em = cv[
        bbox_em[0, 0] : bbox_em[1, 0],
        bbox_em[0, 1] : bbox_em[1, 1],
        bbox_em[0, 2] : bbox_em[1, 2],
    ]
    em = np.squeeze(em)
    print(f"  EM shape: {em.shape}, dtype={em.dtype}, range=[{em.min()},{em.max()}]")

    # Convert synapse positions to box-local EM voxel coords
    pre_local = ((pre_nm - bbox_nm[0]) / em_vox_nm).astype(np.float32)
    post_local = ((post_nm - bbox_nm[0]) / em_vox_nm).astype(np.float32)

    return {
        "volume": em.astype(np.uint8),
        "pre_pt": pre_local,
        "post_pt": post_local,
        "pre_root_id": pre_root_id,
        "post_root_id": post_root_id,
        "em_vox_nm": em_vox_nm,
        "bbox_nm": bbox_nm,
    }


def main():
    print("=" * 70)
    print("REAL CAVE BOX SCAFFOLDING TEST")
    print("=" * 70)

    data = fetch_box()
    if data is None:
        print("Failed to fetch box.")
        return

    vol = data["volume"]

    # EM in MICrONS: low intensity = membrane (dark), high = cytoplasm (bright)
    # We need to invert convention: bright = neurite interior is already correct,
    # but EM membranes are DARK. So we treat low-intensity as boundaries.
    # Compute a quick membrane proxy from local intensity gradient
    from scipy import ndimage
    mem_proxy = 1.0 - (vol.astype(np.float32) / 255.0)
    mem_proxy = ndimage.gaussian_filter(mem_proxy, sigma=1.0)

    print("\n" + "=" * 70)
    print("RUNNING CONSERVATIVE SCAFFOLDING")
    print("=" * 70)

    scaffolder = ConservativeScaffoldingPipeline(
        cell_body_threshold=70,
        arbor_threshold=50,
        confidence_threshold=0.5,
        min_scaffold_size=50,
        max_merge_iterations=10,
    )

    result = scaffolder.scaffold_volume(vol, membrane_field=mem_proxy)
    print(scaffolder.report(result))

    # Evaluate purity
    print("\n" + "=" * 70)
    print("NEURON PURITY ON REAL DATA")
    print("=" * 70)

    purity = evaluate_neuron_purity(
        result.labels,
        data["pre_pt"],
        data["post_pt"],
        data["pre_root_id"],
        data["post_root_id"],
    )

    print(f"  Total synapses:      {purity['total_synapses']}")
    print(f"  Separation rate:     {purity['separation_rate']:.1%}")
    print(f"  Correctly separated: {purity['correctly_separated']}")
    print(f"  False positives:     {purity['false_positives']}")
    print(f"  False negatives:     {purity['false_negatives']}")


if __name__ == "__main__":
    main()
