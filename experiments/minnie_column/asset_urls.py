"""Canonical MICrONS / BossDB URLs for attaching EM, seg, meshes, skeletons, synapses.

See ``docs/minnie_column_downloads.md`` for when to use each product.
"""

from __future__ import annotations

# --- Electron microscopy (CloudVolume precomputed) ---
EM_PRECOMPUTED_HTTPS = (
    "precomputed://https://bossdb-open-data.s3.amazonaws.com/"
    "iarpa_microns/minnie/minnie65/em"
)
EM_PRECOMPUTED_GCS = (
    "precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/em"
)

# --- Dynamic segmentation (CloudVolume graphene; meshes via mesh table) ---
GRAPHENE_MINNIE65_PUBLIC = (
    "graphene://https://minnie.microns-daf.com/segmentation/table/minnie65_public"
)

# --- Flat proofread segmentation (versioned; pick one matching your analysis) ---
# v1300 is listed on MICrONS static repositories; align root IDs to same materialization.
SEG_FLAT_M1300 = (
    "precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg_m1300"
)

# --- Static synapse graph (roots pinned to v117 in docs; use for historical edges) ---
SYNAPSE_CSV_V117_BOSSDB = (
    "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/"
    "synapse_graph/synapses_pni_2.csv"
)

# --- Materialization DB (archived synapse tables by version; gzipped CSV + header) ---
def mat_dbs_synapse_gz(version: int) -> str:
    return (
        f"https://storage.googleapis.com/mat_dbs/public/minnie65_phase3_v1/"
        f"{version}/synapses_pni_2_v1_filtered_view.csv.gz"
    )


def mat_dbs_synapse_header(version: int) -> str:
    return (
        f"https://storage.googleapis.com/mat_dbs/public/minnie65_phase3_v1/"
        f"{version}/synapses_pni_2_v1_filtered_view_header.csv"
    )


# --- Static SWC skeletons (many files; naming is root-based — list prefix to enumerate) ---
SKELETON_SWC_PROOFREAD_PREFIX = (
    "https://storage.googleapis.com/microns-static-links/skel/swc/proofread/"
)
SKELETON_SWC_DENDRITE_PREFIX = (
    "https://storage.googleapis.com/microns-static-links/skel/swc/dendrite/"
)

# --- Nucleus detection (BossDB; static small CSV) ---
NUCLEUS_DETECTION_V0_CSV = (
    "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/"
    "nucleus_detection/nucleus_detection_v0.csv"
)


def skeleton_swc_url(root_id: int, *, proofread: bool = True) -> str:
    """Return a **candidate** URL for a static SWC file (may 404 if root not in set)."""
    base = SKELETON_SWC_PROOFREAD_PREFIX if proofread else SKELETON_SWC_DENDRITE_PREFIX
    return f"{base}{int(root_id)}.swc"
