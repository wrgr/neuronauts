"""Write a self-contained visualisation bundle and the Neuroglancer state that
opens it.

A bundle is one directory::

    <bundle>/
      mesh/                 precomputed legacy meshes, one segment per result id
        info, <id>:0, <id>.mesh, segment_properties/info
      skeleton/             precomputed skeletons with a radius attribute
        info, <id>, segment_properties/info
      groups/               optional: one merged mesh + skeleton per group
        mesh/ skeleton/       (an assembled neuron as a single segment)
      export/all.obj|.ply   optional, micrometres, coloured by group
      synapses.npz          optional point annotations (pos_nm, pre, post)
      segments.json         manifest: ids, groups, labels, stats, mesh params
      state.json, url.txt, index.html   Neuroglancer state for a served bundle

Grouping is how assembly results are shown. Each result id keeps its own mesh
and the state carries the grouping as Neuroglancer ``equivalences``, so every
member of an assembled neuron is one colour and selects together, while the
meshes never need re-writing to try a different assembly. ``groups/`` is the
same information as physically merged geometry for tools without that notion.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from neuronauts.meshing.formats import (
    write_obj, write_ply, write_precomputed_mesh_dir, write_precomputed_skeleton_dir,
    write_segment_properties, SEGMENT_PROPERTIES_DIR,
)
from neuronauts.meshing.skeleton import SkeletonGeometry, concat_skeletons
from neuronauts.meshing.tube import TriMesh, tube_mesh
from treestitch.ngl_export import _MINNIE65_IMAGE_SOURCE, _MINNIE65_SEG_SOURCE, state_to_url

MANIFEST_NAME = "segments.json"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
VIEWER_URL = "https://neuroglancer-demo.appspot.com/#!"


@dataclass
class MeshParams:
    sides: int = 6
    sphere_level: int = 1
    min_radius_nm: float = 30.0
    max_radius_nm: float | None = None
    radius_scale: float = 1.0
    caps: str = "junctions"

    def mesh(self, skel: SkeletonGeometry) -> TriMesh:
        return tube_mesh(skel, **asdict(self))


# ---------------------------------------------------------------------------
# colours
# ---------------------------------------------------------------------------

def group_color_hex(index: int) -> str:
    """Golden-angle hue per group index; distinct neighbours, stable across runs."""
    h = (index * 137.508) % 360.0 / 360.0
    s, v = 0.65, 0.95
    i = int(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ---------------------------------------------------------------------------
# grouping helpers
# ---------------------------------------------------------------------------

def normalise_groups(groups: Mapping[Any, Any] | None, ids: Sequence[int]) -> dict[int, str]:
    """Accept ``{id: group}`` or ``{group: [ids]}``; return ``{id: str(group)}``
    restricted to ``ids``. Ids without a group are left out (they stay single)."""
    if not groups:
        return {}
    id_set = {int(i) for i in ids}
    out: dict[int, str] = {}
    sample = next(iter(groups.values()))
    if isinstance(sample, (list, tuple, set, np.ndarray)):
        for g, members in groups.items():
            for m in members:
                if int(m) in id_set:
                    out[int(m)] = str(g)
    else:
        for i, g in groups.items():
            if int(i) in id_set:
                out[int(i)] = str(g)
    return out


def group_index(id_to_group: Mapping[int, str]) -> dict[str, int]:
    """Stable 1-based integer per group name (sorted), for the groups/ layer."""
    return {g: k + 1 for k, g in enumerate(sorted(set(id_to_group.values())))}


def equivalence_classes(id_to_group: Mapping[int, str]) -> list[list[int]]:
    by: dict[str, list[int]] = {}
    for i, g in id_to_group.items():
        by.setdefault(g, []).append(int(i))
    return [sorted(m) for g, m in sorted(by.items()) if len(m) > 1]


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def export_bundle(
    out_dir: str | Path,
    skeletons: Mapping[int, SkeletonGeometry],
    *,
    groups: Mapping[Any, Any] | None = None,
    labels: Mapping[int, str] | None = None,
    numbers: Mapping[str, Mapping[int, float]] | None = None,
    params: MeshParams | None = None,
    formats: Sequence[str] = ("precomputed",),
    write_group_meshes: bool = False,
    synapses: Mapping[str, np.ndarray] | None = None,
    obj_scale: float = 1e-3,
    title: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    served_root: str | Path | None = None,
    source: Mapping[str, Any] | None = None,
    clean: bool = False,
    verbose: bool = False,
) -> dict:
    """Mesh every skeleton and write the bundle. Returns the manifest dict.

    Parameters
    ----------
    skeletons
        ``{segment_id: SkeletonGeometry}``; ids must fit in uint64 (v117 root
        ids, fragment ids, or any stable integer you assign).
    groups
        Assembly: ``{segment_id: group}`` or ``{group: [segment_ids]}``.
    labels, numbers
        Shown in the viewer's segment list (searchable label, sortable columns).
    formats
        Any of ``precomputed`` (Neuroglancer), ``obj``, ``ply``.
    write_group_meshes
        Also write ``groups/`` with one merged mesh + skeleton per group.
    synapses
        ``{"pos_nm": [S,3], "pre": [S], "post": [S]}`` -> point annotations.
    base_url, served_root
        Where the bundle will be served from; ``served_root`` defaults to the
        bundle's parent so ``serve <parent>`` works out of the box.
    clean
        Remove an existing bundle directory first (otherwise files are
        overwritten in place and stale segments could linger).
    """
    params = params or MeshParams()
    formats = tuple(f.lower() for f in formats)
    unknown = set(formats) - {"precomputed", "obj", "ply"}
    if unknown:
        raise ValueError(f"unknown formats {sorted(unknown)}")
    out = Path(out_dir)
    if clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    title = title or out.name

    ids = sorted(int(i) for i in skeletons.keys())
    id_to_group = normalise_groups(groups, ids)
    gidx = group_index(id_to_group)
    colors = {i: group_color_hex(gidx[id_to_group[i]] if i in id_to_group else 1000 + k)
              for k, i in enumerate(ids)}

    t0 = time.time()
    meshes: dict[int, TriMesh] = {}
    per_segment: dict[str, dict] = {}
    lo_all = np.full(3, np.inf)
    hi_all = np.full(3, -np.inf)
    n_faces_total = 0
    for k, sid in enumerate(ids):
        skel = skeletons[sid].drop_invalid()
        mesh = params.mesh(skel)
        meshes[sid] = mesh
        lo, hi = mesh.bounds_nm() if not mesh.is_empty else skel.bounds_nm()
        if mesh.n_vertices or skel.n_vertices:
            lo_all = np.minimum(lo_all, lo)
            hi_all = np.maximum(hi_all, hi)
        n_faces_total += mesh.n_faces
        per_segment[str(sid)] = {
            "n_skeleton_vertices": int(skel.n_vertices),
            "n_skeleton_edges": int(skel.n_edges),
            "cable_nm": float(skel.cable_length_nm()),
            "n_mesh_vertices": int(mesh.n_vertices),
            "n_mesh_faces": int(mesh.n_faces),
            "group": id_to_group.get(sid),
            "label": (labels or {}).get(sid),
            "color": colors[sid],
        }
        if verbose and (k + 1) % 50 == 0:
            print(f"  meshed {k + 1}/{len(ids)} ({n_faces_total:,} faces)", flush=True)
    if not np.isfinite(lo_all).all():
        lo_all = hi_all = np.zeros(3)

    props_numbers: dict[str, dict[int, float]] = {
        "cable_um": {s: per_segment[str(s)]["cable_nm"] / 1000.0 for s in ids},
        "n_faces": {s: float(per_segment[str(s)]["n_mesh_faces"]) for s in ids},
    }
    for name, col in (numbers or {}).items():
        props_numbers[name] = {int(i): float(v) for i, v in col.items()}
    seg_labels = {s: (labels or {}).get(s) or (
        f"{id_to_group[s]}" if s in id_to_group else f"seg {s}") for s in ids}

    if "precomputed" in formats:
        write_precomputed_mesh_dir(out / "mesh", meshes, segment_properties=True)
        write_precomputed_skeleton_dir(out / "skeleton", skeletons, segment_properties=True)
        for sub in ("mesh", "skeleton"):
            write_segment_properties(out / sub / SEGMENT_PROPERTIES_DIR, ids,
                                     labels=seg_labels, numbers=props_numbers)
        if write_group_meshes and gidx:
            gm: dict[int, TriMesh] = {}
            gs: dict[int, SkeletonGeometry] = {}
            for g, gi in gidx.items():
                members = [s for s in ids if id_to_group.get(s) == g]
                gm[gi] = TriMesh.concat([meshes[s] for s in members])
                gs[gi] = concat_skeletons([skeletons[s] for s in members])
            write_precomputed_mesh_dir(out / "groups" / "mesh", gm, segment_properties=True)
            write_precomputed_skeleton_dir(out / "groups" / "skeleton", gs,
                                           segment_properties=True)
            g_ids = sorted(gidx.values())
            inv = {gi: g for g, gi in gidx.items()}
            g_nums = {"n_members": {gi: float(sum(1 for s in ids if id_to_group.get(s) == inv[gi]))
                                    for gi in g_ids},
                      "cable_um": {gi: gs[gi].cable_length_nm() / 1000.0 for gi in g_ids}}
            for sub in ("mesh", "skeleton"):
                write_segment_properties(out / "groups" / sub / SEGMENT_PROPERTIES_DIR, g_ids,
                                         labels={gi: inv[gi] for gi in g_ids}, numbers=g_nums)

    export_dir = out / "export"
    if "obj" in formats:
        write_obj(export_dir / "all.obj", meshes, scale=obj_scale,
                  names={s: f"seg_{s}" + (f"__{id_to_group[s]}" if s in id_to_group else "")
                         for s in ids})
    if "ply" in formats:
        write_ply(export_dir / "all.ply", meshes, scale=obj_scale,
                  colors={s: hex_to_rgb(colors[s]) for s in ids})

    if synapses is not None and len(synapses.get("pos_nm", [])):
        np.savez_compressed(
            out / "synapses.npz",
            pos_nm=np.asarray(synapses["pos_nm"], np.float32).reshape(-1, 3),
            pre=np.asarray(synapses.get("pre", np.zeros(len(synapses["pos_nm"]))), np.uint64),
            post=np.asarray(synapses.get("post", np.zeros(len(synapses["pos_nm"]))), np.uint64),
        )

    manifest = {
        "title": title,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 2),
        "n_segments": len(ids),
        "n_groups": len(gidx),
        "n_faces_total": int(n_faces_total),
        "ids": [str(s) for s in ids],
        "groups": {str(s): g for s, g in id_to_group.items()},
        "group_index": gidx,
        "labels": {str(s): seg_labels[s] for s in ids},
        "colors": {str(s): colors[s] for s in ids},
        "bounds_lo_nm": lo_all.tolist(),
        "bounds_hi_nm": hi_all.tolist(),
        "center_nm": ((lo_all + hi_all) / 2.0).tolist(),
        "mesh_params": asdict(params),
        "formats": list(formats),
        "has_group_meshes": bool(write_group_meshes and gidx and "precomputed" in formats),
        "has_synapses": (out / "synapses.npz").exists(),
        "source": dict(source or {}),
        "segments": per_segment,
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1))

    if "precomputed" in formats:
        state = build_state(out, base_url=base_url, served_root=served_root)
        write_state(out, state)
    return manifest


def load_manifest(bundle_dir: str | Path) -> dict:
    return json.loads((Path(bundle_dir) / MANIFEST_NAME).read_text())


# ---------------------------------------------------------------------------
# neuroglancer state
# ---------------------------------------------------------------------------

def bundle_url(bundle_dir: str | Path, base_url: str = DEFAULT_BASE_URL,
               served_root: str | Path | None = None) -> str:
    """URL prefix of the bundle when ``served_root`` is served at ``base_url``."""
    bundle = Path(bundle_dir).resolve()
    root = Path(served_root).resolve() if served_root else bundle.parent
    rel = os.path.relpath(bundle, root).replace(os.sep, "/")
    base = base_url.rstrip("/")
    return base if rel == "." else f"{base}/{rel}"


def build_state(
    bundle_dir: str | Path,
    *,
    base_url: str = DEFAULT_BASE_URL,
    served_root: str | Path | None = None,
    include_em: bool = True,
    include_graphene_seg: bool = False,
    layout: str = "4panel",
    max_synapses: int = 5000,
    select_all: bool = True,
) -> dict:
    """Neuroglancer JSON state for a served bundle.

    Coordinates are declared in nanometres, so the mesh and skeleton sources
    (which are in nm) and the EM volume (which carries its own voxel scale)
    line up without any conversion in the state.
    """
    m = load_manifest(bundle_dir)
    prefix = bundle_url(bundle_dir, base_url, served_root)
    ids = list(m["ids"])
    groups = {str(k): v for k, v in m.get("groups", {}).items()}
    colors = m.get("colors", {})
    lo = np.asarray(m["bounds_lo_nm"], np.float64)
    hi = np.asarray(m["bounds_hi_nm"], np.float64)
    extent = float(max((hi - lo).max(), 1000.0))

    layers: list[dict] = []
    if include_em:
        layers.append({
            "type": "image", "name": "EM", "source": _MINNIE65_IMAGE_SOURCE,
            "shader": "void main() { emitGrayscale(toNormalized(getDataValue())); }",
        })
    if include_graphene_seg:
        layers.append({
            "type": "segmentation", "name": "minnie65_public", "source": _MINNIE65_SEG_SOURCE,
            "visible": False, "objectAlpha": 0.4, "hideSegmentZero": True,
        })

    by_group: dict[str, list[str]] = {}
    for s, g in groups.items():
        by_group.setdefault(g, []).append(s)
    equivalences = [sorted(v, key=int) for g, v in sorted(by_group.items()) if len(v) > 1]

    main = {
        "type": "segmentation",
        "name": m.get("title", "results"),
        "source": [f"precomputed://{prefix}/mesh", f"precomputed://{prefix}/skeleton"],
        "segments": ids if select_all else [],
        "segmentColors": {s: colors[s] for s in ids if s in colors},
        "objectAlpha": 1.0,
        "hideSegmentZero": True,
        "skeletonRendering": {"mode2d": "lines_and_points", "mode3d": "lines"},
    }
    if equivalences:
        main["equivalences"] = equivalences
    layers.append(main)

    if m.get("has_group_meshes"):
        g_ids = [str(v) for v in sorted(m.get("group_index", {}).values())]
        layers.append({
            "type": "segmentation",
            "name": f"{m.get('title', 'results')} (merged groups)",
            "source": [f"precomputed://{prefix}/groups/mesh",
                       f"precomputed://{prefix}/groups/skeleton"],
            "segments": g_ids,
            "segmentColors": {str(gi): group_color_hex(gi) for gi in map(int, g_ids)},
            "visible": False,
            "objectAlpha": 1.0,
            "hideSegmentZero": True,
        })

    syn_path = Path(bundle_dir) / "synapses.npz"
    if m.get("has_synapses") and syn_path.exists():
        with np.load(syn_path, allow_pickle=False) as z:
            pos, pre, post = z["pos_nm"], z["pre"], z["post"]
        if len(pos) > max_synapses:
            keep = np.random.default_rng(0).choice(len(pos), max_synapses, replace=False)
            pos, pre, post = pos[keep], pre[keep], post[keep]
        ann = []
        for k in range(len(pos)):
            owner = str(int(pre[k])) if str(int(pre[k])) in colors else str(int(post[k]))
            ann.append({
                "type": "point", "id": f"s{k}",
                "point": [round(float(v), 1) for v in pos[k]],
                "props": [colors.get(owner, "#ffff00")],
                "description": f"pre {int(pre[k])} -> post {int(post[k])}",
            })
        layers.append({
            "type": "annotation", "name": "synapses",
            "source": {"url": "local://annotations",
                       "transform": {"outputDimensions": {
                           "x": [1e-9, "m"], "y": [1e-9, "m"], "z": [1e-9, "m"]}}},
            "annotationProperties": [{"id": "color", "type": "rgb", "default": "#ffff00"}],
            "shader": "void main() {\n  setColor(prop_color());\n  setPointMarkerSize(5.0);\n}",
            "annotations": ann,
        })

    center = np.asarray(m["center_nm"], np.float64)
    return {
        "dimensions": {"x": [1e-9, "m"], "y": [1e-9, "m"], "z": [1e-9, "m"]},
        "position": [float(v) for v in center],
        "crossSectionScale": max(extent / 1000.0, 1.0),
        "projectionScale": extent * 1.2,
        "layers": layers,
        "selectedLayer": {"layer": main["name"], "visible": True},
        "layout": layout,
        "showSlices": layout != "3d",
    }


def write_state(bundle_dir: str | Path, state: dict, *, viewer_url: str = VIEWER_URL) -> str:
    """Write ``state.json``, ``url.txt`` and ``index.html``; return the URL."""
    out = Path(bundle_dir)
    (out / "state.json").write_text(json.dumps(state, indent=1))
    url = state_to_url(state, viewer_url=viewer_url)
    (out / "url.txt").write_text(url + "\n")
    m = load_manifest(out)
    src = [layer["source"] for layer in state["layers"] if layer.get("type") == "segmentation"
           and isinstance(layer.get("source"), list)]
    served = src[0][0].split("://", 1)[1].rsplit("/mesh", 1)[0] if src else ""
    (out / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'>"
        f"<title>{m['title']}</title><h1>{m['title']}</h1>"
        f"<p>{m['n_segments']} segments, {m['n_groups']} groups, "
        f"{m['n_faces_total']:,} triangles.</p>"
        f"<p><a href='{url}' target='_blank'>Open in Neuroglancer</a> "
        "(<a href='state.json'>state.json</a>)</p>"
        f"<p>This state expects the bundle at <code>{served}</code>. Serve it with:</p>"
        "<pre>python scripts/mesh_results.py serve --dir &lt;served root&gt;</pre>"
        "<p>Or paste <code>state.json</code> into the viewer's JSON editor "
        "({} key) after changing the source URLs.</p>"
    )
    return url


__all__ = [
    "DEFAULT_BASE_URL", "MANIFEST_NAME", "VIEWER_URL", "MeshParams",
    "build_state", "bundle_url", "equivalence_classes", "export_bundle",
    "group_color_hex", "group_index", "hex_to_rgb", "load_manifest",
    "normalise_groups", "write_state",
]
