"""Generate Neuroglancer viewer states from NeuronautS result bundles.

Produces URLs compatible with Neuroglancer and Neuroglass (which accepts
Neuroglancer links via its import dialog).

Coordinate conventions
----------------------
Bundle stores positions in nanometres.  MICrONS voxel size is 8×8×40 nm.
Neuroglancer dimensions are set to 8×8×40 nm so coordinates are in voxels
(x_vox = x_nm / 8, etc.).
"""

from __future__ import annotations

import json
import urllib.parse
from typing import Sequence

import numpy as np

VOXEL_SIZE_NM = (8.0, 8.0, 40.0)

# Palette: distinguishable colours for up to 20 neurons.
_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
]


def _nm_to_vox(pos_nm: Sequence[float]) -> list[float]:
    return [pos_nm[i] / VOXEL_SIZE_NM[i] for i in range(3)]


def _skeleton_lines(vertices_nm: list, edges: list) -> tuple[list, list, list]:
    """Convert skeleton vertices+edges to plotly-style x/y/z lists with None breaks."""
    verts = np.array(vertices_nm, dtype=float)
    xs, ys, zs = [], [], []
    for e in edges:
        a = verts[e[0]]
        b = verts[e[1]]
        xs += [a[0], b[0], None]
        ys += [a[1], b[1], None]
        zs += [a[2], b[2], None]
    return xs, ys, zs


def build_neuroglancer_state(
    bbox_data: dict,
    cluster_ids: list[str] | None = None,
    max_neurons: int = 20,
    include_synapses: bool = True,
    voxel_size_nm: tuple[float, float, float] = VOXEL_SIZE_NM,
) -> dict:
    """Build a Neuroglancer viewer state dict from a bbox bundle entry.

    Parameters
    ----------
    bbox_data:
        A single bbox entry from the bundle ``bboxes`` dict.
    cluster_ids:
        Which clusters to include.  ``None`` = pick up to ``max_neurons`` by
        synapse count (largest first).
    max_neurons:
        Cap on number of skeleton annotation layers to keep URLs manageable.
    include_synapses:
        Add a point-annotation layer for synapse positions coloured by cluster.

    Returns
    -------
    dict
        Neuroglancer viewer state JSON-serialisable dict.
    """
    neurons: dict = bbox_data.get("neurons", {})
    synapses: dict = bbox_data.get("synapses", {})

    # Pick which clusters to show.
    if cluster_ids is None:
        order = sorted(neurons, key=lambda k: -neurons[k].get("n_synapses", 0))
        cluster_ids = order[:max_neurons]

    # Compute bbox centre for the initial view position.
    all_pos = synapses.get("positions_nm", [])
    if all_pos:
        centre_nm = np.array(all_pos, dtype=float).mean(axis=0)
    elif cluster_ids and neurons[cluster_ids[0]].get("vertices_nm"):
        centre_nm = np.array(neurons[cluster_ids[0]]["vertices_nm"], dtype=float).mean(axis=0)
    else:
        centre_nm = np.array([1_250_000.0, 965_000.0, 830_000.0])
    centre_vox = _nm_to_vox(centre_nm.tolist())

    layers = []

    # ── Skeleton layers (one per neuron) ──────────────────────────────────
    for i, cid in enumerate(cluster_ids):
        n = neurons.get(str(cid), neurons.get(cid))
        if n is None:
            continue
        verts = n.get("vertices_nm", [])
        edges = n.get("edges", [])
        if not verts:
            continue
        colour = _PALETTE[i % len(_PALETTE)]
        true_id = n.get("true_root_id", 0)
        label = f"cluster_{cid}" + (f"  [gt:{true_id}]" if true_id else "")

        annotations = []
        for j, e in enumerate(edges):
            annotations.append({
                "type": "line",
                "id": str(j),
                "pointA": _nm_to_vox(verts[e[0]]),
                "pointB": _nm_to_vox(verts[e[1]]),
            })

        if annotations:
            layers.append({
                "type": "annotation",
                "name": label,
                "source": "local://annotations",
                "annotations": annotations,
                "annotationColor": colour,
                "tab": "annotations",
            })

    # ── Synapse point layer ────────────────────────────────────────────────
    if include_synapses and synapses.get("positions_nm"):
        positions = synapses["positions_nm"]
        pred_clusters = synapses.get("pred_cluster", [0] * len(positions))
        selected_set = {str(c) for c in cluster_ids} | {int(c) for c in cluster_ids}

        syn_annotations = []
        for k, (pos, pc) in enumerate(zip(positions, pred_clusters)):
            if str(pc) not in selected_set and pc not in selected_set:
                continue
            syn_annotations.append({
                "type": "point",
                "id": str(k),
                "point": _nm_to_vox(pos),
            })

        if syn_annotations:
            layers.append({
                "type": "annotation",
                "name": "Synapses",
                "source": "local://annotations",
                "annotations": syn_annotations,
                "annotationColor": "#ffff00",
                "tab": "annotations",
            })

    state = {
        "dimensions": {
            "x": [voxel_size_nm[0] * 1e-9, "m"],
            "y": [voxel_size_nm[1] * 1e-9, "m"],
            "z": [voxel_size_nm[2] * 1e-9, "m"],
        },
        "position": centre_vox,
        "crossSectionScale": 1.0,
        "projectionScale": 8192,
        "layers": layers,
        "layout": "4panel",
        "selectedLayer": {"visible": True, "layer": "Synapses"} if include_synapses else {},
    }
    return state


def state_to_neuroglancer_url(
    state: dict,
    base_url: str = "https://neuroglancer-demo.appspot.com",
) -> str:
    """Encode a Neuroglancer state dict to a viewer URL."""
    state_str = json.dumps(state, separators=(",", ":"))
    return f"{base_url}/#!{urllib.parse.quote(state_str)}"


def bundle_to_neuroglancer_url(
    bundle: dict,
    bbox_name: str,
    cluster_ids: list[str] | None = None,
    max_neurons: int = 20,
    include_synapses: bool = True,
    base_url: str = "https://neuroglancer-demo.appspot.com",
) -> str | None:
    """One-shot: build a Neuroglancer URL for a bbox from a result bundle.

    Returns ``None`` if ``bbox_name`` is not in the bundle.
    """
    bbox_data = bundle.get("bboxes", {}).get(bbox_name)
    if bbox_data is None:
        return None
    state = build_neuroglancer_state(
        bbox_data,
        cluster_ids=cluster_ids,
        max_neurons=max_neurons,
        include_synapses=include_synapses,
    )
    return state_to_neuroglancer_url(state, base_url=base_url)


NEUROGLASS_IMPORT_URL = "https://app.neuroglass.com"


def neuroglass_instructions(neuroglancer_url: str) -> str:
    """Return markdown instructions for opening a URL in Neuroglass."""
    return (
        "**To open in Neuroglass:**\n\n"
        "1. Go to [app.neuroglass.com](https://app.neuroglass.com)\n"
        "2. Click **Import** → **Neuroglancer Link**\n"
        "3. Paste the URL below and click **Open**\n\n"
        f"```\n{neuroglancer_url}\n```"
    )
