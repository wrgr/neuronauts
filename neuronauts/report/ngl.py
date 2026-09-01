"""Base Neuroglancer viewer: plain-JSON states in nanometre coordinates.

Two state builders already exist in this repo and disagree about the voxel
grid (``dashboard/neuroglancer_export.py`` divides by 8 nm, ``treestitch/
ngl_export.py`` by 4 nm). Both are avoidable: Neuroglancer's global coordinate
space is independent of any layer's native grid, so this module declares the
global dimensions as **1 nm** and passes every position straight through in
nanometres. Image and segmentation layers carry their own scale metadata and
are placed correctly by the viewer; annotation layers get an explicit
``outputDimensions`` in nm so they never inherit a stale grid.

The output is a dict that serialises to a viewer state. It is built without
the ``neuroglancer`` package so it can run anywhere; when the package is
installed, ``validate`` round-trips the dict through its ``ViewerState`` and
``serve`` opens a local viewer for states too large for a URL.

Segmentation at v117: the graphene layer accepts a ``timestamp`` (Unix
seconds) so the segmentation can be shown as it was when the atoms were
defined. This follows the nglui convention; it has not been verified in a
browser from this codebase, so treat a wrong-looking segmentation as a
question about that field first.

Views:

  region_view      a cube of interest as a bounding box
  experiment_view  the box, anchor soma and anchor root a benchmark recorded
  atom_view        one v117 atom: real L2 skeleton, endpoints, the atom's
                   synapses, and the segment selected in the graphene layer
"""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

NM = [1e-9, "m"]
DIMENSIONS_NM = {"x": NM, "y": NM, "z": NM}

EM_SOURCE = "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em"
SEG_GRAPHENE_SOURCE = "graphene://https://minnie.microns-daf.com/segmentation/table/minnie65_public"
SEG_FLAT_SOURCE = "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/seg"
V117_TIMESTAMP = 1623399000

VIEWERS = {
    "demo": "https://neuroglancer-demo.appspot.com/#!",
    "spelunker": "https://spelunker.cave-explorer.org/#!",
}
DEFAULT_VIEWER = "spelunker"

#: A URL longer than this is not something a browser or a Markdown file will
#: reliably carry; write the JSON and serve it instead.
MAX_URL_BYTES = 200_000

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def _pts(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).reshape(-1, 3)


def _round(p: Iterable[float]) -> list[float]:
    return [round(float(v), 1) for v in p]


def _subsample(n: int, cap: Optional[int], seed: int = 0) -> np.ndarray:
    if cap is None or n <= cap:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))


class NglState:
    """Incrementally build a viewer state; every coordinate is in nm."""

    def __init__(self, *, show_em: bool = True, show_seg: bool = True,
                 seg_source: str = SEG_GRAPHENE_SOURCE,
                 seg_timestamp: Optional[int] = None,
                 layout: str = "4panel", max_annotations: int = 20_000):
        self.layers: list[dict] = []
        self.layout = layout
        self.max_annotations = max_annotations
        self._position: Optional[np.ndarray] = None
        self._extent_nm: float = 30_000.0
        self.truncated: dict[str, tuple[int, int]] = {}
        if show_em:
            self.layers.append({"type": "image", "source": EM_SOURCE, "name": "em",
                                "shader": "void main() { emitGrayscale(toNormalized(getDataValue())); }"})
        if show_seg:
            layer: dict = {"type": "segmentation", "source": seg_source,
                           "name": "segmentation", "segments": [],
                           "objectAlpha": 0.6, "hideSegmentZero": True}
            if seg_timestamp is not None:
                layer["timestamp"] = int(seg_timestamp)
                layer["name"] = f"segmentation@{seg_timestamp}"
            self.layers.append(layer)

    # -- view placement ----------------------------------------------------
    def look_at(self, position_nm: Sequence[float], extent_nm: Optional[float] = None) -> "NglState":
        self._position = _pts(position_nm)[0]
        if extent_nm:
            self._extent_nm = float(extent_nm)
        return self

    def fit(self, points_nm: Any) -> "NglState":
        p = _pts(points_nm)
        if len(p):
            lo, hi = p.min(axis=0), p.max(axis=0)
            self.look_at((lo + hi) / 2, float(np.max(hi - lo)) or self._extent_nm)
        return self

    # -- segments ----------------------------------------------------------
    def select_segments(self, ids: Iterable[int]) -> "NglState":
        for layer in self.layers:
            if layer["type"] == "segmentation":
                have = set(layer["segments"])
                layer["segments"] += [str(int(i)) for i in ids if str(int(i)) not in have]
        return self

    # -- annotation layers -------------------------------------------------
    def _annotation_layer(self, name: str, annotations: list[dict], color: str,
                          per_annotation_color: bool = False, **extra) -> dict:
        layer: dict = {
            "type": "annotation", "name": name,
            "source": {"url": "local://annotations",
                       "transform": {"outputDimensions": DIMENSIONS_NM}},
            "annotationColor": color,
            "annotations": annotations,
            **extra,
        }
        if per_annotation_color:
            layer["annotationProperties"] = [
                {"id": "color", "type": "rgb", "default": color}]
            layer["shader"] = "void main() { setColor(prop_color()); }"
        self.layers.append(layer)
        return layer

    def _cap(self, name: str, n: int) -> np.ndarray:
        idx = _subsample(n, self.max_annotations)
        if len(idx) < n:
            self.truncated[name] = (len(idx), n)
        return idx

    def add_points(self, name: str, points_nm: Any, *, color: str = "#ffff00",
                   colors: Optional[Sequence[str]] = None,
                   descriptions: Optional[Sequence[str]] = None) -> "NglState":
        p = _pts(points_nm)
        idx = self._cap(name, len(p))
        ann = []
        for k in idx:
            a = {"type": "point", "id": f"{name}-{k}", "point": _round(p[k])}
            if descriptions is not None:
                a["description"] = str(descriptions[k])
            if colors is not None:
                a["props"] = [colors[k]]
            ann.append(a)
        self._annotation_layer(self._named(name), ann, color,
                               per_annotation_color=colors is not None)
        return self

    def add_lines(self, name: str, a_nm: Any, b_nm: Any, *, color: str = "#ffffff",
                  colors: Optional[Sequence[str]] = None) -> "NglState":
        a, b = _pts(a_nm), _pts(b_nm)
        if len(a) != len(b):
            raise ValueError(f"{name}: {len(a)} line starts vs {len(b)} ends")
        idx = self._cap(name, len(a))
        ann = []
        for k in idx:
            item = {"type": "line", "id": f"{name}-{k}",
                    "pointA": _round(a[k]), "pointB": _round(b[k])}
            if colors is not None:
                item["props"] = [colors[k]]
            ann.append(item)
        self._annotation_layer(self._named(name), ann, color,
                               per_annotation_color=colors is not None)
        return self

    def add_skeleton(self, name: str, vertices_nm: Any, edges: Any, *,
                     color: str = "#ffffff") -> "NglState":
        v = _pts(vertices_nm)
        e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        return self.add_lines(name, v[e[:, 0]], v[e[:, 1]], color=color)

    def add_box(self, name: str, lo_nm: Sequence[float], hi_nm: Sequence[float], *,
                color: str = "#2a78d6", description: str = "") -> "NglState":
        ann = [{"type": "axis_aligned_bounding_box", "id": f"{name}-box",
                "pointA": _round(lo_nm), "pointB": _round(hi_nm)}]
        if description:
            ann[0]["description"] = description
        self._annotation_layer(name, ann, color)
        return self

    def _named(self, name: str) -> str:
        if name in self.truncated:
            kept, total = self.truncated[name]
            return f"{name} ({kept:,} of {total:,})"
        return name

    # -- output ------------------------------------------------------------
    def to_dict(self) -> dict:
        pos = self._position
        if pos is None:
            pos = np.array([663_000.0, 591_000.0, 860_000.0])
        extent = max(self._extent_nm, 1_000.0)
        return {
            "dimensions": DIMENSIONS_NM,
            "position": _round(pos),
            "crossSectionScale": round(max(extent / 900.0, 4.0), 2),
            "projectionScale": round(extent * 1.6, 1),
            "layers": self.layers,
            "showSlices": False,
            "layout": self.layout,
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        return dumps_state(self.to_dict(), indent)

    def save(self, path: str | Path) -> Path:
        return save_state(self.to_dict(), path)

    def to_url(self, viewer: str = DEFAULT_VIEWER) -> Optional[str]:
        """Shareable URL, or None when the state is too large to carry in one."""
        return state_to_url(self.to_dict(), viewer)


def _n_annotations(state: dict) -> int:
    return sum(len(l.get("annotations", [])) for l in state.get("layers", []))


def dumps_state(state: dict, indent: Optional[int] = 2) -> str:
    """Pretty JSON for small states, compact for large ones.

    Indented JSON puts every coordinate on its own line, which turns a
    20k-annotation state into ~10 MB; compact is a third of that.
    """
    if indent is None or _n_annotations(state) > 500:
        return json.dumps(state, separators=(",", ":"))
    return json.dumps(state, indent=indent)


def save_state(state: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dumps_state(state))
    return p


def state_to_url(state: dict, viewer: str = DEFAULT_VIEWER) -> Optional[str]:
    base = VIEWERS.get(viewer, viewer)
    encoded = urllib.parse.quote(json.dumps(state, separators=(",", ":")), safe="")
    if len(encoded) > MAX_URL_BYTES:
        return None
    return base + encoded


def url_to_state(url: str) -> dict:
    """Inverse of :func:`state_to_url`, for tests and for reading shared links."""
    frag = url.split("#!", 1)[1]
    return json.loads(urllib.parse.unquote(frag))


def validate(state: dict) -> dict:
    """Round-trip through the ``neuroglancer`` package's ``ViewerState``.

    Raises ``ImportError`` without the package and whatever the package
    raises on a malformed state; returns the re-serialised dict otherwise.
    """
    from neuroglancer import viewer_state
    return viewer_state.ViewerState(json_data=state).to_json()


def serve(state: dict, bind_address: str = "127.0.0.1", port: int = 0) -> str:
    """Open a local Neuroglancer viewer on ``state`` and return its URL.

    Needs the ``neuroglancer`` package. The server lives as long as the
    calling process; a script should sleep or wait for input after this.
    """
    import neuroglancer
    from neuroglancer import viewer_state
    neuroglancer.set_server_bind_address(bind_address, bind_port=port)
    viewer = neuroglancer.Viewer()
    viewer.set_state(viewer_state.ViewerState(json_data=state))
    return viewer.get_viewer_url()


# ---------------------------------------------------------------------------
# views over project data
# ---------------------------------------------------------------------------

def region_view(centre_um: Sequence[float], side_um: float | Sequence[float], *,
                seg_timestamp: Optional[int] = None) -> NglState:
    centre = np.asarray(centre_um, float) * 1000.0
    half = np.broadcast_to(np.asarray(side_um, float) * 500.0, (3,))
    st = NglState(seg_timestamp=seg_timestamp)
    st.add_box("region", centre - half, centre + half,
               description=f"{side_um} um cube at {list(centre_um)} um")
    st.look_at(centre, float(np.max(half) * 2))
    return st


def experiment_view(record: Any, *, seg_timestamp: Optional[int] = None) -> Optional[NglState]:
    """The spatial context a benchmark JSON recorded, or None if it has none."""
    prov = getattr(record, "provenance", record) or {}
    bbox = prov.get("bbox_nm")
    anchor = prov.get("anchor_soma_nm")
    root = prov.get("anchor_target_root")
    if bbox is None and anchor is None:
        return None
    st = NglState(seg_timestamp=seg_timestamp)
    if bbox is not None:
        lo, hi = np.asarray(bbox[0], float), np.asarray(bbox[1], float)
        st.add_box("experiment bbox", lo, hi,
                   description=f"{getattr(record, 'id', 'experiment')} box")
        st.look_at((lo + hi) / 2, float(np.max(hi - lo)))
    if anchor is not None:
        st.add_points("anchor soma", [anchor], color=PALETTE[1],
                      descriptions=[f"anchor target root {root}"])
        if bbox is None:
            st.look_at(anchor, 30_000)
    if root:
        st.select_segments([int(root)])
    return st


class L2Positions:
    """Sorted lookup from L2 id to ``rep_coord_nm`` over the attribute cache.

    The cache holds ~21M rows; loading it costs a few seconds and ~0.5 GB,
    so keep one instance around when viewing many atoms.
    """

    def __init__(self, attrs_npz: str | Path):
        with np.load(Path(attrs_npz), allow_pickle=False) as z:
            ids, pos = z["l2_id"], z["pos_nm"]
        order = np.argsort(ids)
        self.ids = ids[order]
        self.pos = pos[order]

    def lookup(self, l2_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Positions for ``l2_ids`` and a mask of which were found."""
        q = np.asarray(l2_ids, dtype=np.uint64)
        i = np.searchsorted(self.ids, q)
        i = np.clip(i, 0, len(self.ids) - 1)
        found = self.ids[i] == q
        out = np.full((len(q), 3), np.nan, dtype=np.float32)
        out[found] = self.pos[i[found]]
        return out, found


def load_atom_geometry(atom_id: int, geom_dir: str | Path) -> Optional[dict]:
    """L2 ids and real adjacency of one atom from the shard store."""
    shard_dir = Path(geom_dir) / "shards"
    for f in sorted(shard_dir.glob("*.npz")):
        with np.load(f, allow_pickle=False) as z:
            hits = np.nonzero(z["atom_id"] == np.uint64(atom_id))[0]
            if len(hits) == 0:
                continue
            i = int(hits[0])
            npt, ept = z["node_ptr"], z["edge_ptr"]
            return {"atom": int(atom_id), "shard": f.name,
                    "l2_ids": z["l2_ids"][npt[i]:npt[i + 1]],
                    "edges": z["edges"][ept[i]:ept[i + 1]].reshape(-1, 2)}
    return None


def atom_view(atom_id: int, *, geom_dir: str | Path, positions: L2Positions,
              topology_npz: Optional[str | Path] = None,
              population_npz: Optional[str | Path] = None,
              seg_timestamp: Optional[int] = V117_TIMESTAMP,
              max_annotations: int = 20_000) -> tuple[NglState, dict]:
    """One atom: L2 skeleton, endpoints, synapses, selected segment.

    Returns the state and a summary dict (node/edge counts, how many nodes
    had coordinates) so the caller can print what the view contains.
    """
    geom = load_atom_geometry(atom_id, geom_dir)
    if geom is None:
        raise KeyError(f"atom {atom_id} is not in {geom_dir}/shards")
    l2 = geom["l2_ids"]
    pos, found = positions.lookup(l2)
    order = np.argsort(l2)
    srt = l2[order]
    e = geom["edges"]
    loc = order[np.searchsorted(srt, e.reshape(-1))].reshape(-1, 2)
    ok_edge = (srt[np.searchsorted(srt, e.reshape(-1))] == e.reshape(-1)).reshape(-1, 2).all(axis=1)
    ok_edge &= found[loc[:, 0]] & found[loc[:, 1]]
    loc = loc[ok_edge]

    st = NglState(seg_timestamp=seg_timestamp, max_annotations=max_annotations)
    st.select_segments([atom_id])
    if len(loc):
        st.add_lines("L2 skeleton", pos[loc[:, 0]], pos[loc[:, 1]], color=PALETTE[0])
    summary = {"atom": int(atom_id), "n_l2": int(len(l2)), "n_edges": int(len(e)),
               "n_l2_with_coords": int(found.sum()), "n_edges_drawn": int(len(loc)),
               "shard": geom["shard"]}

    if topology_npz is not None:
        with np.load(Path(topology_npz), allow_pickle=False) as z:
            m = z["ep_atom"] == np.uint64(atom_id)
            ep = z["ep_pos_nm"][m]
            leaf = z["ep_seg_len_nm"][m]
            cal = z["ep_caliber_nm"][m]
        if len(ep):
            st.add_points("endpoints", ep, color=PALETTE[1],
                          descriptions=[f"leaf {l:.0f} nm, caliber {c:.0f} nm"
                                        for l, c in zip(leaf, cal)])
        summary["n_endpoints"] = int(len(ep))

    if population_npz is not None:
        with np.load(Path(population_npz), allow_pickle=False) as z:
            pre = z["syn_atom_pre"] == np.uint64(atom_id)
            post = z["syn_atom_post"] == np.uint64(atom_id)
            ctr = z["syn_ctr_nm"]
            syn_pre, syn_post = ctr[pre], ctr[post]
        if len(syn_pre):
            st.add_points("synapses (atom is pre)", syn_pre, color=PALETTE[2])
        if len(syn_post):
            st.add_points("synapses (atom is post)", syn_post, color=PALETTE[4])
        summary["n_pre"], summary["n_post"] = int(len(syn_pre)), int(len(syn_post))

    if found.any():
        st.fit(pos[found])
    return st, summary
