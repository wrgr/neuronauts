"""Neuroglancer views of two-level stitch products.

Builds on ``treestitch.ngl_export`` (zero-dependency NGL JSON states) to render
what the hierarchical assembly actually did, so every decision can be eyeballed
against the EM + segmentation:

- **observations layer** — synapses coloured by *global* (post-stitch) cluster
- **super_skeletons layer** — per-super merged skeletons coloured by global
  cluster (same colour ⇒ stitched together)
- **stitch_edges layer** — accepted level-1 edges: green = ground-truth
  agrees, red = disagrees, white = unlabelled; forced (exact-channel) merges
  drawn super-centroid → super-centroid in blue
- **rejected_edges layer** — top rejected candidates in gray (why-not view)
- **odd_fragments layer** — skeleton edges of odd-flagged fragments in orange
- **frankenmerges layer** — observations of frankenmerge parents coloured by
  their TRUE object, to inspect whether the halves were kept apart

``export_stitch_viz`` writes one JSON state per view plus ``index.html`` with
clickable viewer links and ``urls.txt``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from treestitch.ngl_export import (
    _MINNIE65_IMAGE_SOURCE,
    _MINNIE65_SEG_SOURCE,
    _VOXEL_SIZE_NM,
    _cluster_color,
    state_to_url,
)


def _vox(p) -> list:
    return [round(float(v), 2) for v in (np.asarray(p, dtype=np.float64) / _VOXEL_SIZE_NM)]


def _line(a, b, ann_id: str, color: str) -> dict:
    return {"type": "line", "pointA": _vox(a), "pointB": _vox(b),
            "id": ann_id, "props": [color]}


def _annotation_layer(name: str, annotations: list[dict]) -> dict:
    return {
        "type": "annotation",
        "name": name,
        "source": {"url": "local://annotations"},
        "tool": "annotatePoint",
        "shader": "void main() {\n  setColor(prop_color());\n  setLineWidth(2.0);\n}",
        "annotations": annotations,
    }


def _base_layers() -> list[dict]:
    return [
        {"type": "image", "source": _MINNIE65_IMAGE_SOURCE, "name": "EM",
         "shader": "void main() { emitGrayscale(toNormalized(getDataValue())); }"},
        {"type": "segmentation", "source": _MINNIE65_SEG_SOURCE,
         "name": "seg", "objectAlpha": 0.5, "hideSegmentZero": True},
    ]


def _state(layers: list[dict], center_nm) -> dict:
    return {
        "dimensions": {"x": [4e-9, "m"], "y": [4e-9, "m"], "z": [4e-8, "m"]},
        "position": _vox(center_nm),
        "crossSectionScale": 1.0,
        "projectionScale": 8192,
        "layers": layers,
        "showAxisLines": True,
        "showSlices": False,
        "layout": "4panel",
    }


def _super_centroid(s) -> np.ndarray:
    return np.asarray(s.skeleton.vertices_nm, dtype=np.float64).mean(axis=0)


def _skeleton_lines(fragment, ann_prefix: str, color: str,
                    max_edges: int = 400) -> list[dict]:
    verts = np.asarray(fragment.vertices_nm, dtype=np.float64)
    edges = np.asarray(fragment.edges, dtype=np.int64).reshape(-1, 2)
    out = []
    step = max(1, len(edges) // max_edges)
    for k in range(0, len(edges), step):
        a, b = edges[k]
        out.append(_line(verts[a], verts[b], f"{ann_prefix}_{k}", color))
    return out


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def stitch_overview_state(
    supers: list,
    super_cluster: np.ndarray,
    obs_pos_nm: np.ndarray,
    obs_global_labels: np.ndarray,
    *,
    max_synapses: int = 1500,
    max_supers: int = 120,
) -> dict:
    """Observations + super skeletons, coloured by global (stitched) cluster.

    Budgets are sized so the encoded URL stays under ~1 MB — beyond that,
    browsers reject the fragment and the state must be pasted into the
    viewer's JSON editor instead."""
    pos = np.asarray(obs_pos_nm, dtype=np.float64)
    glab = np.asarray(obs_global_labels, dtype=np.int64)
    n_clusters = max(int(super_cluster.max()) + 1, 1) if len(super_cluster) else 1

    keep = np.arange(len(pos))
    if len(keep) > max_synapses:
        keep = np.random.default_rng(0).choice(keep, max_synapses, replace=False)
    obs_ann = [
        {"type": "point", "point": _vox(pos[i]), "id": f"o{i}",
         "props": [_cluster_color(int(glab[i]), n_clusters)]}
        for i in keep
    ]

    skel_ann: list[dict] = []
    order = np.argsort([-s.n_obs for s in supers])[:max_supers]
    for si in order:
        color = _cluster_color(int(super_cluster[si]), n_clusters, alpha=0.85)
        skel_ann.extend(_skeleton_lines(
            supers[si].skeleton, f"s{si}", color, max_edges=25))

    layers = _base_layers() + [
        _annotation_layer("observations_by_global_cluster", obs_ann),
        _annotation_layer("super_skeletons", skel_ann),
    ]
    return _state(layers, pos.mean(axis=0))


def stitch_edges_state(
    supers: list,
    result,                      # StitchResult
    *,
    max_rejected: int = 200,
) -> dict:
    """Level-1 decisions: accepted edges (✓ green / ✗ red / ? white), forced
    merges (blue), top rejected candidates (gray)."""
    acc_ann: list[dict] = []
    for k, c in enumerate(result.accepted):
        a, b = supers[c.i], supers[c.j]
        ea = np.asarray(a.skeleton.endpoints_nm)[min(c.ep_i, len(a.skeleton.endpoints_nm) - 1)]
        eb = np.asarray(b.skeleton.endpoints_nm)[min(c.ep_j, len(b.skeleton.endpoints_nm) - 1)]
        if a.majority_label and b.majority_label:
            color = ("rgba(0,220,0,1.0)" if a.majority_label == b.majority_label
                     else "rgba(255,40,40,1.0)")
        else:
            color = "rgba(255,255,255,1.0)"
        acc_ann.append(_line(ea, eb, f"acc{k}", color))

    for k, (i, j) in enumerate(result.forced_pairs):
        acc_ann.append(_line(_super_centroid(supers[i]), _super_centroid(supers[j]),
                             f"forced{k}", "rgba(60,120,255,0.9)"))

    layers = _base_layers() + [_annotation_layer("stitch_decisions", acc_ann)]
    centers = ([_super_centroid(supers[i]) for i, _ in result.forced_pairs[:50]]
               or [_super_centroid(s) for s in supers[:50]])
    return _state(layers, np.mean(centers, axis=0) if centers else np.zeros(3))


def odd_fragments_state(
    fragments: list,
    odd_parents: set,
    *,
    max_fragments: int = 60,
) -> dict:
    """Skeletons of odd-flagged fragments (orange) vs a sample of normal ones
    (teal)."""
    ann: list[dict] = []
    n_odd = n_norm = 0
    for f in fragments:
        if f.base_root_id in odd_parents and n_odd < max_fragments:
            ann.extend(_skeleton_lines(f, f"odd{f.base_root_id}",
                                       "rgba(255,150,0,0.95)", max_edges=40))
            n_odd += 1
        elif f.base_root_id not in odd_parents and n_norm < max_fragments // 3:
            ann.extend(_skeleton_lines(f, f"norm{f.base_root_id}",
                                       "rgba(0,200,200,0.5)", max_edges=25))
            n_norm += 1
    layers = _base_layers() + [_annotation_layer("odd_vs_normal_fragments", ann)]
    all_v = [np.asarray(f.vertices_nm).mean(axis=0) for f in fragments[:200]]
    return _state(layers, np.mean(all_v, axis=0) if all_v else np.zeros(3))


def frankenmerge_state(
    obs_pos_nm: np.ndarray,
    true_labels: np.ndarray,
    parent_ids: np.ndarray,
    pred_labels: np.ndarray,
    *,
    max_parents: int = 20,
) -> dict:
    """Observations of frankenmerge parents: colour = TRUE object; point size
    conveys nothing — check whether same-colour groups landed in different
    predicted clusters via the cluster id in the annotation description."""
    pos = np.asarray(obs_pos_nm, dtype=np.float64)
    true = np.asarray(true_labels, dtype=np.int64)
    par = np.asarray(parent_ids, dtype=np.int64)
    pred = np.asarray(pred_labels, dtype=np.int64)

    ann: list[dict] = []
    n_shown = 0
    for p in np.unique(par):
        m = (par == p) & (true != 0)
        labs = np.unique(true[m])
        if len(labs) < 2:
            continue
        if n_shown >= max_parents:
            break
        n_shown += 1
        lab_index = {int(l): li for li, l in enumerate(labs)}
        for oi in np.where(m)[0]:
            ann.append({
                "type": "point", "point": _vox(pos[oi]), "id": f"fk{p}_{oi}",
                "description": f"parent={p} true={true[oi]} pred={pred[oi]}",
                "props": [_cluster_color(lab_index[int(true[oi])], len(labs))],
            })
    layers = _base_layers() + [_annotation_layer("frankenmerge_parents", ann)]
    pts = [a["point"] for a in ann]
    center = (np.mean(np.asarray(pts) * _VOXEL_SIZE_NM, axis=0)
              if pts else np.zeros(3))
    return _state(layers, center)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_stitch_viz(
    out_dir: str,
    *,
    supers: list,
    result,
    fragments: list,
    odd_parents: set,
    obs_pos_nm: np.ndarray,
    obs_global_labels: np.ndarray,
    true_labels: np.ndarray | None = None,
    parent_ids: np.ndarray | None = None,
    title: str = "two-level stitch",
) -> dict[str, str]:
    """Write all views as JSON states + index.html; return {view: url}."""
    os.makedirs(out_dir, exist_ok=True)
    views: dict[str, dict] = {
        "overview": stitch_overview_state(
            supers, result.super_cluster, obs_pos_nm, obs_global_labels),
        "stitch_edges": stitch_edges_state(supers, result),
        "odd_fragments": odd_fragments_state(fragments, odd_parents),
    }
    if true_labels is not None and parent_ids is not None:
        views["frankenmerges"] = frankenmerge_state(
            obs_pos_nm, true_labels, parent_ids, obs_global_labels)

    urls: dict[str, str] = {}
    for name, state in views.items():
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(state, f, indent=1)
        urls[name] = state_to_url(state)

    with open(os.path.join(out_dir, "urls.txt"), "w") as f:
        for name, url in urls.items():
            f.write(f"{name}\t{url}\n")

    rows = "\n".join(
        f'<li><a href="{url}" target="_blank">{name}</a> '
        f'(<a href="{name}.json">json</a>)</li>'
        for name, url in urls.items()
    )
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(
            "<!doctype html><meta charset='utf-8'>"
            f"<title>{title}</title><h1>{title}</h1>"
            f"<p>Neuroglancer views (open in browser):</p><ul>{rows}</ul>"
        )
    return urls


__all__ = [
    "stitch_overview_state",
    "stitch_edges_state",
    "odd_fragments_state",
    "frankenmerge_state",
    "export_stitch_viz",
]
