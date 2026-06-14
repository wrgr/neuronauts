"""Lightweight Neuroglancer state builder for NeuronautS outputs.

Generates shareable Neuroglancer JSON states / URLs without requiring the
neuroglancer Python package or nglui's full dependency stack. Neuroglancer
states are plain JSON and the viewer is served from neuroglancer-demo.appspot.com
or any CAVE-hosted viewer.

Outputs
-------
- Segmentation layer pointing at the MICrONS Minnie65 graphene source
- Synapse annotation layer: pre→post pairs as line annotations, coloured by
  predicted cluster and optionally by per-observation uncertainty (entropy)
- Skeleton annotation layer: assembled neuron skeletons as line annotations,
  coloured by cluster ID

Usage
-----
    from treestitch.ngl_export import build_neuroglancer_state, state_to_url

    state = build_neuroglancer_state(
        synapse_pre_nm=region.pre_pt_nm,
        synapse_post_nm=region.post_pt_nm,
        pred_labels=pred_te,
        soft=soft_te,               # optional — enables uncertainty colouring
        shapes=shapes,              # optional — adds skeleton layer
        center_nm=None,             # auto-computed from synapse positions
    )
    url = state_to_url(state)
    print(url)

Coordinate system
-----------------
All inputs are in nanometres (float32/float64 [N, 3]). The MICrONS dataset
uses 4×4×40 nm voxels; the graphene viewer expects voxel coordinates. We
convert nm → voxel by dividing by [4, 4, 40].
"""

from __future__ import annotations

import json
import urllib.parse
from typing import Any

import numpy as np


# Minnie65 public graphene segmentation source
_MINNIE65_SEG_SOURCE = (
    "graphene://https://minnie.microns-daf.com/segmentation/table/minnie65_public"
)
_MINNIE65_IMAGE_SOURCE = (
    "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em"
)
_VOXEL_SIZE_NM = np.array([4.0, 4.0, 40.0], dtype=np.float64)
_VIEWER_URL = "https://neuroglancer-demo.appspot.com/#!"


def _nm_to_vox(pts_nm: np.ndarray) -> np.ndarray:
    """Convert nm coordinates to Minnie65 voxel coordinates."""
    return (np.asarray(pts_nm, dtype=np.float64) / _VOXEL_SIZE_NM).tolist()


def _cluster_color(cluster_id: int, n_clusters: int, alpha: float = 1.0) -> str:
    """HSL colour for a cluster ID — evenly spaced hues."""
    if cluster_id < 0:
        return f"rgba(128,128,128,{alpha})"
    hue = (cluster_id * 137.508) % 360  # golden-angle spacing
    return f"hsla({hue:.0f},70%,55%,{alpha})"


def _uncertainty_color(entropy: float, max_entropy: float, alpha: float = 1.0) -> str:
    """Red→yellow gradient by normalised entropy (0=confident, 1=uncertain)."""
    t = min(1.0, entropy / max(max_entropy, 1e-6))
    r = int(255 * t)
    g = int(255 * (1 - t))
    return f"rgba({r},{g},0,{alpha})"


def build_neuroglancer_state(
    synapse_pre_nm: np.ndarray,
    synapse_post_nm: np.ndarray,
    pred_labels: np.ndarray,
    *,
    soft: dict | None = None,
    shapes: dict | None = None,
    center_nm: np.ndarray | None = None,
    seg_source: str = _MINNIE65_SEG_SOURCE,
    image_source: str = _MINNIE65_IMAGE_SOURCE,
    max_synapses_shown: int = 2000,
    max_skeleton_verts: int = 5000,
) -> dict:
    """Build a Neuroglancer JSON state dict.

    Parameters
    ----------
    synapse_pre_nm:
        [N, 3] float — pre-synaptic positions in nm.
    synapse_post_nm:
        [N, 3] float — post-synaptic positions in nm (same length as pre).
    pred_labels:
        [N] int64 — predicted cluster ID per synapse (-ve = abstained).
    soft:
        Optional return value of `partition_observations_soft`. When provided,
        an uncertainty layer is added: each synapse is coloured red→yellow by
        per-observation entropy.
    shapes:
        Optional dict[cluster_id → Fragment] from `assemble_partition_shapes`.
        When provided, a skeleton layer is added with one polyline per edge.
    center_nm:
        3D centre of the view in nm. Auto-computed from synapse centroid if None.
    max_synapses_shown:
        Downsample to this many synapse pairs for readability.
    max_skeleton_verts:
        Truncate skeleton annotations beyond this vertex count (viewer limit).

    Returns
    -------
    dict — Neuroglancer JSON state ready for json.dumps().
    """
    pre = np.asarray(synapse_pre_nm, dtype=np.float64)
    post = np.asarray(synapse_post_nm, dtype=np.float64)
    labels = np.asarray(pred_labels, dtype=np.int64)
    N = len(labels)

    if center_nm is None:
        center_nm = pre.mean(axis=0)
    center_vox = (np.asarray(center_nm) / _VOXEL_SIZE_NM).tolist()

    # Downsample if too many synapses
    if N > max_synapses_shown:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, max_synapses_shown, replace=False)
        pre, post, labels = pre[idx], post[idx], labels[idx]
        entropy_arr = soft["entropy"][idx] if soft is not None else None
    else:
        idx = np.arange(N)
        entropy_arr = soft["entropy"] if soft is not None else None

    unique_clusters = np.unique(labels[labels >= 0])
    n_clusters = max(int(unique_clusters.max()) + 1 if len(unique_clusters) else 1, 1)

    # ------------------------------------------------------------------ Layers

    layers: list[dict] = []

    # 1. EM image layer
    layers.append({
        "type": "image",
        "source": image_source,
        "name": "EM",
        "shader": "void main() { emitGrayscale(toNormalized(getDataValue())); }",
    })

    # 2. Segmentation layer
    layers.append({
        "type": "segmentation",
        "source": seg_source,
        "name": "seg_v117",
        "objectAlpha": 0.5,
        "hideSegmentZero": True,
    })

    # 3. Synapse pairs coloured by predicted cluster
    syn_annotations: list[dict] = []
    for i, (p, q, cl) in enumerate(zip(pre, post, labels)):
        color = _cluster_color(int(cl), n_clusters)
        syn_annotations.append({
            "type": "line",
            "pointA": [round(v, 2) for v in (p / _VOXEL_SIZE_NM)],
            "pointB": [round(v, 2) for v in (q / _VOXEL_SIZE_NM)],
            "id": str(i),
            "props": [color],
        })

    layers.append({
        "type": "annotation",
        "name": "synapses_by_cluster",
        "source": {"url": "local://annotations"},
        "tool": "annotatePoint",
        "annotationColor": "#ffff00",
        "shader": (
            "void main() {\n"
            "  setColor(prop_color());\n"
            "  setLineWidth(2.0);\n"
            "}"
        ),
        "shaderControls": {},
        "annotations": syn_annotations,
    })

    # 4. Uncertainty layer (entropy colouring)
    if entropy_arr is not None:
        max_ent = float(entropy_arr.max()) if len(entropy_arr) else 1.0
        unc_annotations: list[dict] = []
        for i, (p, q, ent) in enumerate(zip(pre, post, entropy_arr)):
            color = _uncertainty_color(float(ent), max_ent)
            unc_annotations.append({
                "type": "line",
                "pointA": [round(v, 2) for v in (p / _VOXEL_SIZE_NM)],
                "pointB": [round(v, 2) for v in (q / _VOXEL_SIZE_NM)],
                "id": str(i),
                "props": [color],
            })
        layers.append({
            "type": "annotation",
            "name": "uncertainty_entropy",
            "source": {"url": "local://annotations"},
            "tool": "annotatePoint",
            "annotations": unc_annotations,
        })

    # 5. Skeleton layer from assembled shapes
    if shapes:
        skel_annotations: list[dict] = []
        n_verts_total = 0
        for cluster_id, fragment in shapes.items():
            if n_verts_total > max_skeleton_verts:
                break
            verts = np.asarray(fragment.vertices_nm, dtype=np.float64)
            edges = np.asarray(fragment.edges, dtype=np.int64)
            color = _cluster_color(int(cluster_id), n_clusters, alpha=0.85)
            verts_vox = verts / _VOXEL_SIZE_NM
            for e_idx, (a, b) in enumerate(edges):
                skel_annotations.append({
                    "type": "line",
                    "pointA": [round(v, 2) for v in verts_vox[a]],
                    "pointB": [round(v, 2) for v in verts_vox[b]],
                    "id": f"sk_{cluster_id}_{e_idx}",
                    "props": [color],
                })
            n_verts_total += len(verts)

        layers.append({
            "type": "annotation",
            "name": "assembled_skeletons",
            "source": {"url": "local://annotations"},
            "tool": "annotatePoint",
            "annotations": skel_annotations,
        })

    # ------------------------------------------------------------------ State
    state = {
        "dimensions": {
            "x": [4e-9, "m"],
            "y": [4e-9, "m"],
            "z": [4e-8, "m"],
        },
        "position": center_vox,
        "crossSectionScale": 1.0,
        "projectionScale": 8192,
        "layers": layers,
        "showAxisLines": True,
        "showSlices": False,
        "layout": "4panel",
    }
    return state


def state_to_url(state: dict, viewer_url: str = _VIEWER_URL) -> str:
    """Encode a Neuroglancer state dict as a shareable URL."""
    state_json = json.dumps(state, separators=(",", ":"))
    return viewer_url + urllib.parse.quote(state_json, safe="")


def state_to_json(state: dict, indent: int = 2) -> str:
    """Pretty-print a Neuroglancer state dict as JSON."""
    return json.dumps(state, indent=indent)


def export_synapse_state(
    region: Any,
    pred_labels: np.ndarray,
    *,
    soft: dict | None = None,
    shapes: dict | None = None,
    side: str = "pre",
    output_path: str | None = None,
) -> str:
    """Convenience wrapper: build state from a Region + partition outputs.

    Extracts pre/post positions from the region object, builds the state,
    and optionally writes it to a file. Always returns the URL string.

    Parameters
    ----------
    region:
        `treestitch.realworld.Region` (or any object with pre_pt_nm, post_pt_nm).
    pred_labels:
        [N] int64 predicted cluster IDs from partition_observations_cc or
        the "pred" key of soft_partition output.
    soft:
        Optional soft_partition dict for entropy colouring.
    shapes:
        Optional assembled neuron shapes for skeleton layer.
    side:
        "pre" uses pre-synaptic positions as anchors; "post" uses post-synaptic.
    output_path:
        If given, write the JSON state to this file path.

    Returns
    -------
    Neuroglancer URL string.
    """
    pre_nm = np.asarray(region.pre_pt_nm, dtype=np.float64)
    post_nm = np.asarray(region.post_pt_nm, dtype=np.float64)

    # pred_labels is per pre-synapse observation; post positions match 1:1
    state = build_neuroglancer_state(
        pre_nm, post_nm, pred_labels,
        soft=soft, shapes=shapes,
    )
    url = state_to_url(state)

    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump(state, f, indent=2)
        print(f"Neuroglancer state written to {output_path}")

    return url
