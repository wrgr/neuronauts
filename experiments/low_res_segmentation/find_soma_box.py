#!/usr/bin/env python3
"""Find a Minnie65 box centered on a real soma, fetch EM, run scaffolding."""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.low_res_segmentation.conservative_scaffolding import ConservativeScaffoldingPipeline
from experiments.low_res_segmentation.evaluate_neuron_purity import evaluate_neuron_purity

DATASTACK = "minnie65_public"
SERVER = "https://global.daf-apis.com"
SYN_VOX_NM = np.array([4.0, 4.0, 40.0])
BOX_UM = 24.0  # bigger box to capture soma + arbors


def main():
    from caveclient import CAVEclient
    from cloudvolume import CloudVolume

    token = os.environ.get("CAVE_TOKEN")
    client = CAVEclient(DATASTACK, server_address=SERVER, auth_token=token)

    # Find nucleus tables
    print("Available tables (nucleus-related):")
    tables = client.materialize.get_tables()
    nuc_tables = [t for t in tables if "nucleus" in t.lower() or "soma" in t.lower()]
    print(" ", nuc_tables)

    nuc_table = nuc_tables[0] if nuc_tables else "nucleus_detection_v0"
    print(f"\nQuerying {nuc_table}...")
    nuc = client.materialize.query_table(nuc_table, limit=20)
    print(f"  cols: {list(nuc.columns)[:10]}")
    print(f"  got {len(nuc)} nuclei")

    # Pick first nucleus with valid root id
    pos_col = "pt_position" if "pt_position" in nuc.columns else nuc.columns[1]
    root_col = "pt_root_id" if "pt_root_id" in nuc.columns else None

    if root_col:
        nuc = nuc[nuc[root_col] > 0]
    pos_vox = np.array(nuc.iloc[0][pos_col], dtype=np.float64)
    center_nm = pos_vox * SYN_VOX_NM
    print(f"\nPicked soma at {center_nm} nm (root={nuc.iloc[0].get(root_col, '?')})")

    half = BOX_UM * 1000.0 / 2.0
    bbox_nm = np.array([center_nm - half, center_nm + half])

    # Fetch synapses in box
    bbox_vox = bbox_nm / SYN_VOX_NM[None, :]
    df = client.materialize.query_table(
        "synapses_pni_2",
        filter_spatial_dict={"pre_pt_position": bbox_vox.tolist()},
        limit=5000,
    )
    print(f"Synapses in box: {len(df)}")

    pre_nm = np.array(df["pre_pt_position"].tolist(), float) * SYN_VOX_NM
    post_nm = np.array(df["post_pt_position"].tolist(), float) * SYN_VOX_NM
    pre_root = df["pre_pt_root_id"].values
    post_root = df["post_pt_root_id"].values
    keep = (
        np.all(pre_nm >= bbox_nm[0], axis=1) & np.all(pre_nm < bbox_nm[1], axis=1)
        & np.all(post_nm >= bbox_nm[0], axis=1) & np.all(post_nm < bbox_nm[1], axis=1)
    )
    pre_nm, post_nm, pre_root, post_root = pre_nm[keep], post_nm[keep], pre_root[keep], post_root[keep]
    print(f"  fully inside: {len(pre_nm)}")

    # EM
    em_url = "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em"
    cv = CloudVolume(em_url, mip=3, use_https=True, fill_missing=True, progress=False)
    em_vox_nm = np.array(cv.resolution)
    print(f"EM voxel: {em_vox_nm} nm")
    bem = (bbox_nm / em_vox_nm[None, :]).astype(int)
    em = np.squeeze(cv[bem[0,0]:bem[1,0], bem[0,1]:bem[1,1], bem[0,2]:bem[1,2]])
    print(f"EM shape: {em.shape}, range [{em.min()},{em.max()}]")

    pre_local = ((pre_nm - bbox_nm[0]) / em_vox_nm).astype(np.float32)
    post_local = ((post_nm - bbox_nm[0]) / em_vox_nm).astype(np.float32)

    from scipy import ndimage
    vol = em.astype(np.uint8)
    mem = ndimage.gaussian_filter(1.0 - vol.astype(np.float32) / 255.0, sigma=1.0)

    print("\n=== SCAFFOLDING ===")
    sc = ConservativeScaffoldingPipeline(
        cell_body_threshold=85, arbor_threshold=40,
        confidence_threshold=0.55, min_scaffold_size=200, max_merge_iterations=20,
    )
    res = sc.scaffold_volume(vol, membrane_field=mem)
    print(sc.report(res))

    # Save slice + segmentation overlay
    try:
        from PIL import Image
        out = Path("data/cave_slices"); out.mkdir(parents=True, exist_ok=True)
        z = vol.shape[2] // 2
        gray = vol[:, :, z]
        Image.fromarray(gray).save(out / "mip3_slice.png")

        lab = res.labels[:, :, z]
        rng = np.random.default_rng(0)
        palette = rng.integers(40, 255, size=(int(res.labels.max()) + 1, 3), dtype=np.uint8)
        palette[0] = 0
        rgb = palette[lab]
        base = np.stack([gray]*3, axis=-1)
        overlay = np.where(lab[..., None] > 0, (0.5*base + 0.5*rgb).astype(np.uint8), base)
        Image.fromarray(overlay).save(out / "mip3_seg_overlay.png")
        Image.fromarray(rgb).save(out / "mip3_seg.png")
        print(f"Saved {out}/mip3_slice.png, mip3_seg.png, mip3_seg_overlay.png")
    except Exception as e:
        print(f"viz failed: {e}")

    if len(pre_local):
        purity = evaluate_neuron_purity(res.labels, pre_local, post_local, pre_root, post_root)
        print(f"\nSeparation rate: {purity['separation_rate']:.1%}  ({purity['correctly_separated']}/{purity['total_synapses']})")
        print(f"FP: {purity['false_positives']}  FN: {purity['false_negatives']}")


if __name__ == "__main__":
    main()
