"""Data fetching from MICrONS via CloudVolume and CAVEclient."""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

MICRONS_EM_PATH = "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em"
MICRONS_DATASTACK = "minnie65_public"
CAVE_SERVER = "https://global.daf-apis.com"
DEFAULT_BOX_SIDE_UM = 6.0
DEFAULT_BOX_SIDE_NM = int(DEFAULT_BOX_SIDE_UM * 1000)
SYNAPSE_VOXEL_SIZE_NM = (4, 4, 40)

MIP_VOXEL_SIZES = {
    0: (8, 8, 40),
    1: (16, 16, 40),
    2: (32, 32, 40),
    3: (64, 64, 40),
}


def _install_system_trust_store() -> None:
    """Use the platform trust store when available.

    Python 3.14 environments created from Homebrew can fail certificate
    validation against the DAF APIs even when curl succeeds. truststore keeps
    TLS verification enabled while aligning requests with the system trust
    roots.
    """
    try:
        import truststore
    except ImportError:
        return

    truststore.inject_into_ssl()


@dataclass
class VolumeChunk:
    data: np.ndarray
    voxel_size_nm: Tuple
    bbox_voxels: Tuple
    mip: int


@dataclass
class SynapseTable:
    """Per-synapse data for one bounding box.

    Core fields (always present)
    ----------------------------
    pre_pt, post_pt : float32 [N, 3]
        Synapse pre- and post-synaptic positions in box-relative voxel coords.
    pre_root_id, post_root_id : int64 [N]
        CAVE root IDs — used as ground-truth grouping labels for the line-graph
        F1 metric.
    synapse_id : int64 [N]
        Unique synapse identifiers from the materialization table.

    Scaffold fields (optional, populated when CAVE segment data is available)
    -------------------------------------------------------------------------
    pre_seg_id, post_seg_id : int64 [N] or None
        Per-synapse segment IDs at the time of materialization.  These are the
        "noisy scaffold" labels: multiple root IDs may share a segment ID
        (under-merge) or a single root ID may span several segment IDs
        (fragmentation).  When present they are used by ``_merge_role_groups``
        to pre-initialize scaffold groups before local agent evidence is
        applied.  ``None`` when not available (e.g. synthetic benchmarks or
        older fetch paths).
    """

    pre_pt: np.ndarray
    post_pt: np.ndarray
    pre_root_id: np.ndarray
    post_root_id: np.ndarray
    synapse_id: np.ndarray
    pre_seg_id: np.ndarray | None = None
    post_seg_id: np.ndarray | None = None


@dataclass(frozen=True)
class SkeletonData:
    """Skeleton geometry for a single root at one materialization version."""

    root_id: int
    materialization_version: int
    vertices: np.ndarray
    edges: np.ndarray
    radius: np.ndarray | None = None


@dataclass(frozen=True)
class SyntheticBenchmarkConfig:
    shape: Tuple[int, int, int] = (96, 96, 96)
    n_synapses: int = 30
    membrane_planes: int = 10
    min_neuron_groups: int = 6
    max_neuron_groups: int = 15
    anchor_margin: int = 12
    pre_cluster_std: float = 4.0
    post_cluster_std: float = 4.0


@dataclass(frozen=True)
class RealBoxSpec:
    center_nm: Tuple[int, int, int]
    side_um: float = DEFAULT_BOX_SIDE_UM
    mip: int = 2

    @property
    def bbox_nm(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        return make_cube_bbox_nm(self.center_nm, self.side_um)

    @property
    def cache_key(self) -> str:
        payload = json.dumps(
            {"center_nm": self.center_nm, "side_um": self.side_um, "mip": self.mip},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def make_cube_bbox_nm(
    center_nm: Tuple[int, int, int],
    side_um: float = DEFAULT_BOX_SIDE_UM,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Build a cubic nanometer bounding box around a center point.

    The default is approximately a 6 x 6 x 6 micron cube, i.e. 6000 nm per side.
    """
    side_nm = int(side_um * 1000)
    half = side_nm // 2
    x, y, z = center_nm
    return (
        (x - half, y - half, z - half),
        (x + half, y + half, z + half),
    )


def fetch_volume(
    bbox_nm: Tuple[Tuple, Tuple],
    mip: int = 2,
    em_path: str = MICRONS_EM_PATH,
) -> VolumeChunk:
    _install_system_trust_store()
    try:
        from cloudvolume import CloudVolume
    except ImportError as exc:
        raise ImportError("pip install cloud-volume") from exc

    cv = CloudVolume(em_path, mip=mip, use_https=True, progress=False, fill_missing=True)
    vox = MIP_VOXEL_SIZES[mip]
    x0 = int(bbox_nm[0][0] / vox[0])
    y0 = int(bbox_nm[0][1] / vox[1])
    z0 = int(bbox_nm[0][2] / vox[2])
    x1 = int(bbox_nm[1][0] / vox[0])
    y1 = int(bbox_nm[1][1] / vox[1])
    z1 = int(bbox_nm[1][2] / vox[2])

    data = np.squeeze(cv[x0:x1, y0:y1, z0:z1])
    return VolumeChunk(
        data=data.astype(np.uint8),
        voxel_size_nm=vox,
        bbox_voxels=((x0, y0, z0), (x1, y1, z1)),
        mip=mip,
    )


def fetch_synapses(
    bbox_nm: Tuple[Tuple, Tuple],
    mip: int = 2,
    version: int | None = None,
    datastack: str = MICRONS_DATASTACK,
    cave_server: str = CAVE_SERVER,
    token: Optional[str] = None,
) -> SynapseTable:
    _install_system_trust_store()
    try:
        from caveclient import CAVEclient
    except ImportError as exc:
        raise ImportError("pip install caveclient") from exc

    client = CAVEclient(datastack, server_address=cave_server, auth_token=token)
    if version is not None:
        client.version = version
    (x0, y0, z0), (x1, y1, z1) = bbox_nm
    syn_vox = np.array(SYNAPSE_VOXEL_SIZE_NM, dtype=np.float32)
    bbox_synapse_units = [
        (np.array([x0, y0, z0], dtype=np.float32) / syn_vox).astype(np.int64).tolist(),
        (np.array([x1, y1, z1], dtype=np.float32) / syn_vox).astype(np.int64).tolist(),
    ]
    df = client.materialize.synapse_query(
        bounding_box=bbox_synapse_units,
        bounding_box_column="ctr_pt_position",
    )

    if len(df) == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return SynapseTable(
            pre_pt=empty,
            post_pt=empty,
            pre_root_id=np.array([], dtype=np.int64),
            post_root_id=np.array([], dtype=np.int64),
            synapse_id=np.array([], dtype=np.int64),
        )

    vox = MIP_VOXEL_SIZES[mip]
    bbox_origin_vox = np.array(
        [
            bbox_nm[0][0] / vox[0],
            bbox_nm[0][1] / vox[1],
            bbox_nm[0][2] / vox[2],
        ],
        dtype=np.float32,
    )

    def pts_to_voxels(col: str) -> np.ndarray:
        pts = np.stack(df[col].values)
        pts_nm = pts * syn_vox
        pts_vox = pts_nm / np.array(vox, dtype=np.float32)
        return (pts_vox - bbox_origin_vox).astype(np.float32)

    # Pull segment IDs if present in the materialization table.
    # Column names vary by datastack version; try common variants gracefully.
    def _try_seg_id(col: str) -> np.ndarray | None:
        if col in df.columns:
            return df[col].values.astype(np.int64)
        return None

    _pre = _try_seg_id("pre_pt_supervoxel_id")
    pre_seg_id = _pre if _pre is not None else _try_seg_id("pre_pt_seg_id")
    _post = _try_seg_id("post_pt_supervoxel_id")
    post_seg_id = _post if _post is not None else _try_seg_id("post_pt_seg_id")

    return SynapseTable(
        pre_pt=pts_to_voxels("pre_pt_position"),
        post_pt=pts_to_voxels("post_pt_position"),
        pre_root_id=df["pre_pt_root_id"].values.astype(np.int64),
        post_root_id=df["post_pt_root_id"].values.astype(np.int64),
        synapse_id=df.index.values.astype(np.int64),
        pre_seg_id=pre_seg_id,
        post_seg_id=post_seg_id,
    )


def fetch_root_skeleton(
    root_id: int,
    *,
    version: int,
    datastack: str = MICRONS_DATASTACK,
    cave_server: str = CAVE_SERVER,
    token: Optional[str] = None,
    skeleton_service_version: int = 4,
    cache_dir: str | Path | None = None,
    max_retries: int = 4,
    initial_backoff_s: float = 1.0,
    client=None,
) -> SkeletonData:
    """Fetch one root skeleton at a specific materialization version."""
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"v{int(version)}_rid{int(root_id)}_skv{int(skeleton_service_version)}.npz"
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            radius = cached["radius"] if "radius" in cached else None
            return SkeletonData(
                root_id=int(root_id),
                materialization_version=int(version),
                vertices=cached["vertices"].astype(np.float32, copy=False),
                edges=cached["edges"].astype(np.int64, copy=False),
                radius=None if radius is None else radius.astype(np.float32, copy=False),
            )

    _install_system_trust_store()
    if client is None:
        try:
            from caveclient import CAVEclient
        except ImportError as exc:
            raise ImportError("pip install caveclient") from exc
        client = CAVEclient(datastack, server_address=cave_server, auth_token=token)
        client.version = int(version)

    last_exc: Exception | None = None
    raw = None
    for attempt in range(max(1, int(max_retries))):
        try:
            raw = client.skeleton.get_skeleton(
                int(root_id),
                datastack_name=datastack,
                skeleton_version=int(skeleton_service_version),
                output_format="dict",
            )
            break
        except Exception as exc:  # pragma: no cover - exercised via mocked fallback tests
            last_exc = exc
            if attempt + 1 >= max(1, int(max_retries)):
                raise
            time.sleep(float(initial_backoff_s) * (2 ** attempt))
    if raw is None:
        assert last_exc is not None
        raise last_exc

    vertices = np.asarray(raw.get("vertices", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    edges = np.asarray(raw.get("edges", np.zeros((0, 2), dtype=np.int64)), dtype=np.int64)
    radius_raw = raw.get("radius", None)
    radius = None if radius_raw is None else np.asarray(radius_raw, dtype=np.float32)
    sk = SkeletonData(
        root_id=int(root_id),
        materialization_version=int(version),
        vertices=vertices,
        edges=edges,
        radius=radius,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"vertices": sk.vertices, "edges": sk.edges}
        if sk.radius is not None:
            arrays["radius"] = sk.radius
        np.savez_compressed(cache_path, **arrays)
    return sk


def fetch_root_skeletons(
    root_ids: np.ndarray | list[int],
    *,
    version: int,
    datastack: str = MICRONS_DATASTACK,
    cave_server: str = CAVE_SERVER,
    token: Optional[str] = None,
    skeleton_service_version: int = 4,
    cache_dir: str | Path | None = None,
    allow_missing: bool = True,
    max_retries: int = 4,
) -> dict[int, SkeletonData]:
    """Fetch skeletons keyed by root ID for one materialization version."""
    _install_system_trust_store()
    try:
        from caveclient import CAVEclient
    except ImportError as exc:
        raise ImportError("pip install caveclient") from exc

    client = CAVEclient(datastack, server_address=cave_server, auth_token=token)
    client.version = int(version)
    unique_roots = sorted({int(root_id) for root_id in np.asarray(root_ids, dtype=np.int64).tolist() if int(root_id) > 0})
    out: dict[int, SkeletonData] = {}
    for root_id in unique_roots:
        try:
            out[root_id] = fetch_root_skeleton(
                root_id,
                version=version,
                datastack=datastack,
                cave_server=cave_server,
                token=token,
                skeleton_service_version=skeleton_service_version,
                cache_dir=cache_dir,
                max_retries=max_retries,
                client=client,
            )
        except Exception:
            if not allow_missing:
                raise
            out[root_id] = SkeletonData(
                root_id=int(root_id),
                materialization_version=int(version),
                vertices=np.zeros((0, 3), dtype=np.float32),
                edges=np.zeros((0, 2), dtype=np.int64),
                radius=None,
            )
    return out


class CAVEDataFetcher:
    """Small convenience wrapper around the module-level CAVE fetch helpers.

    This keeps the public API close to the design described in the whitepaper
    while preserving the original functional helpers used elsewhere in the repo.
    """

    def __init__(
        self,
        datastack: str = MICRONS_DATASTACK,
        *,
        cave_server: str = CAVE_SERVER,
        token: str | None = None,
    ) -> None:
        self.datastack = datastack
        self.cave_server = cave_server
        self.token = token

    def fetch_volume(self, bbox_nm: Tuple[Tuple, Tuple], *, mip: int = 2) -> VolumeChunk:
        return fetch_volume(bbox_nm=bbox_nm, mip=mip)

    def fetch_synapses(self, bbox_nm: Tuple[Tuple, Tuple], *, mip: int = 2) -> SynapseTable:
        return fetch_synapses(
            bbox_nm=bbox_nm,
            mip=mip,
            datastack=self.datastack,
            cave_server=self.cave_server,
            token=self.token,
        )

    def fetch_root_skeleton(
        self,
        root_id: int,
        *,
        version: int,
        skeleton_service_version: int = 4,
    ) -> SkeletonData:
        return fetch_root_skeleton(
            root_id,
            version=version,
            datastack=self.datastack,
            cave_server=self.cave_server,
            token=self.token,
            skeleton_service_version=skeleton_service_version,
        )


def membrane_cache_paths(
    box: RealBoxSpec,
    cache_dir: str | Path,
) -> tuple[Path, Path]:
    cache_dir = Path(cache_dir)
    stem = f"{box.cache_key}_mip{box.mip}"
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.json"


def load_cached_membrane(
    box: RealBoxSpec,
    cache_dir: str | Path,
) -> np.ndarray | None:
    membrane_path, _ = membrane_cache_paths(box, cache_dir)
    if not membrane_path.exists():
        return None
    membrane = np.load(membrane_path)
    return membrane.astype(np.float32, copy=False)


def save_cached_membrane(
    box: RealBoxSpec,
    cache_dir: str | Path,
    membrane: np.ndarray,
    *,
    source: str,
    extra_metadata: dict[str, object] | None = None,
) -> Path:
    membrane_path, metadata_path = membrane_cache_paths(box, cache_dir)
    membrane_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(membrane_path, membrane.astype(np.float32))
    metadata = {
        "center_nm": box.center_nm,
        "side_um": box.side_um,
        "mip": box.mip,
        "shape": list(membrane.shape),
        "source": source,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return membrane_path


def skeleton_tortuosity(
    points: np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    """Return the tortuosity of a polyline skeleton.

    Tortuosity is defined as the ratio of the arc length (sum of step
    distances) to the straight-line end-to-end distance.  A perfectly
    straight path has tortuosity 1.0; a wound path has tortuosity > 1.

    Parameters
    ----------
    points:
        Array of shape ``[N, 3]`` (or ``[N, 2]``) giving the ordered skeleton
        vertices in voxel or physical units.
    eps:
        Small constant added to the denominator to avoid division by zero for
        degenerate or length-1 paths.

    Returns
    -------
    float
        Tortuosity >= 1.0, or 1.0 for degenerate inputs.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 1.0
    step_dists = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
    arc_length = float(step_dists.sum())
    end_to_end = float(np.linalg.norm(pts[-1] - pts[0]))
    return arc_length / max(end_to_end, eps)


def skeleton_stepwise_features(
    points: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute per-step skeleton features for use with ``TorchPathEncoder``.

    Returns a float32 array of shape ``[T, 3]`` where ``T = len(points) - 1``
    (one row per edge).  Columns are:

    0. step distance (nm or voxel units, as supplied)
    1. cumulative arc-length normalised to [0, 1] along the path
    2. local turning angle in radians (0 for straight, pi for U-turn)

    Parameters
    ----------
    points:
        Ordered skeleton vertices, shape ``[N, 3]``.
    eps:
        Stability constant.
    """
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return np.zeros((0, 3), dtype=np.float32)

    diffs = np.diff(pts, axis=0).astype(np.float32)
    step_dists = np.linalg.norm(diffs, axis=-1).astype(np.float32)

    cumulative = np.concatenate([[0.0], np.cumsum(step_dists)]).astype(np.float32)
    total = cumulative[-1]
    norm_cum = (cumulative[:-1] / max(total, eps)).astype(np.float32)

    units = diffs / np.clip(step_dists[:, None], eps, None)
    if len(units) >= 2:
        cos_angle = np.clip((units[:-1] * units[1:]).sum(axis=-1), -1.0, 1.0)
        turning = np.arccos(cos_angle).astype(np.float32)
        # First step has no predecessor; pad with 0.
        turning = np.concatenate([[0.0], turning]).astype(np.float32)
    else:
        turning = np.zeros(len(step_dists), dtype=np.float32)

    return np.stack([step_dists, norm_cum, turning], axis=-1).astype(np.float32)


def mesh_volume_surface_ratio(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    """Estimate the volume-to-surface-area ratio of a triangular mesh.

    Uses the divergence theorem to approximate the signed volume enclosed by
    the mesh.  The absolute value is taken so the sign of face winding does
    not matter for a near-closed surface.

    Parameters
    ----------
    vertices:
        Float array of shape ``[V, 3]``.
    faces:
        Integer array of shape ``[F, 3]`` giving triangle vertex indices.
    eps:
        Stability constant for surface area denominator.

    Returns
    -------
    float
        Non-negative volume/area ratio.  Returns 0.0 for degenerate inputs.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64)

    if verts.ndim != 2 or verts.shape[1] != 3 or tris.ndim != 2 or tris.shape[1] != 3:
        return 0.0
    if len(tris) == 0:
        return 0.0

    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    # Signed volume via divergence theorem (1/6 * sum of v0 · (v1 × v2)).
    signed_vol = float(np.abs((v0 * cross).sum(axis=-1).sum()) / 6.0)
    # Surface area = half sum of |cross products|.
    surface = float(np.linalg.norm(cross, axis=-1).sum() / 2.0)

    return signed_vol / max(surface, eps)


def mesh_stepwise_features(
    vertices: np.ndarray,
    faces: np.ndarray,
    waypoints: np.ndarray,
    *,
    k_nearest: int = 8,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute per-step mesh features aligned to a set of skeleton waypoints.

    For each waypoint the ``k_nearest`` mesh vertices are gathered and a
    small local surface patch is characterised by:

    0. mean distance to the ``k`` nearest mesh vertices (local scale proxy)
    1. local volume/area ratio of the patch's sub-faces
    2. mean face area of the patch

    Returns a float32 array of shape ``[len(waypoints), 3]``.

    Parameters
    ----------
    vertices:
        Mesh vertex array, shape ``[V, 3]``.
    faces:
        Mesh face array, shape ``[F, 3]``.
    waypoints:
        Ordered skeleton step positions, shape ``[T, 3]``.
    k_nearest:
        Number of mesh vertices to gather per waypoint.
    eps:
        Stability constant.
    """
    verts = np.asarray(vertices, dtype=np.float32)
    tris = np.asarray(faces, dtype=np.int64)
    wps = np.asarray(waypoints, dtype=np.float32)
    T = len(wps)

    if len(verts) == 0 or len(tris) == 0 or T == 0:
        return np.zeros((T, 3), dtype=np.float32)

    try:
        from scipy.spatial import cKDTree as _cKDTree
    except ImportError:
        # Fallback: return zeros if scipy is unavailable.
        return np.zeros((T, 3), dtype=np.float32)

    tree = _cKDTree(verts)
    k = min(k_nearest, len(verts))
    dists, neighbor_idx = tree.query(wps, k=k)

    out = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        nidx = neighbor_idx[t] if k > 1 else np.array([neighbor_idx[t]])
        ndist = dists[t] if k > 1 else np.array([dists[t]])
        out[t, 0] = float(ndist.mean())

        # Find faces that touch at least one neighbour.
        neighbor_set = set(nidx.tolist())
        patch_faces = tris[
            np.any(np.isin(tris, list(neighbor_set)), axis=1)
        ]
        if len(patch_faces) == 0:
            continue
        v0 = verts[patch_faces[:, 0]]
        v1 = verts[patch_faces[:, 1]]
        v2 = verts[patch_faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0).astype(np.float64)
        face_areas = np.linalg.norm(cross, axis=-1) / 2.0
        surface = float(face_areas.sum())
        signed_vol = float(np.abs((v0.astype(np.float64) * cross).sum(axis=-1).sum()) / 6.0)
        out[t, 1] = float(signed_vol / max(surface, eps))
        out[t, 2] = float(face_areas.mean()) if len(face_areas) else 0.0

    return out


def make_test_volume(
    config: SyntheticBenchmarkConfig | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
):
    config = config or SyntheticBenchmarkConfig()
    if rng is None:
        rng = np.random.default_rng(seed)

    shape = tuple(config.shape)
    n_synapses = int(config.n_synapses)
    data = rng.integers(80, 180, size=shape, dtype=np.uint8)

    for _ in range(config.membrane_planes):
        axis = rng.integers(0, 3)
        idx = rng.integers(10, shape[axis] - 10)
        slc = [slice(None)] * 3
        slc[axis] = slice(idx - 1, idx + 2)
        data[tuple(slc)] = rng.integers(0, 40, size=data[tuple(slc)].shape)

    chunk = VolumeChunk(
        data=data,
        voxel_size_nm=(32, 32, 30),
        bbox_voxels=((0, 0, 0), shape),
        mip=2,
    )

    # Build a recoverable synthetic graph: synapses sharing a root ID are also
    # spatially near one another, so line-graph edges are not random noise.
    max_groups = max(config.min_neuron_groups, config.max_neuron_groups)
    n_pre_neurons = int(rng.integers(config.min_neuron_groups, max_groups + 1))
    n_post_neurons = int(rng.integers(config.min_neuron_groups, max_groups + 1))
    margin = config.anchor_margin
    pre_anchors = rng.uniform(margin, np.array(shape) - margin, size=(n_pre_neurons, 3)).astype(np.float32)
    post_anchors = rng.uniform(margin, np.array(shape) - margin, size=(n_post_neurons, 3)).astype(np.float32)

    pre_assign = rng.integers(0, n_pre_neurons, size=n_synapses)
    post_assign = rng.integers(0, n_post_neurons, size=n_synapses)

    pts_pre = pre_anchors[pre_assign] + rng.normal(0, config.pre_cluster_std, size=(n_synapses, 3)).astype(np.float32)
    pts_post = post_anchors[post_assign] + rng.normal(0, config.post_cluster_std, size=(n_synapses, 3)).astype(np.float32)
    pts_pre = np.clip(pts_pre, 5, np.array(shape) - 5).astype(np.float32)
    pts_post = np.clip(pts_post, 5, np.array(shape) - 5).astype(np.float32)

    pre_root_id = (pre_assign + 1).astype(np.int64)
    post_root_id = (post_assign + 10_001).astype(np.int64)

    # Synthetic scaffold: seg_ids match root_ids exactly (perfect scaffold).
    # This lets downstream code exercise the scaffold init path without
    # requiring a live CAVE connection.
    synapses = SynapseTable(
        pre_pt=pts_pre,
        post_pt=pts_post,
        pre_root_id=pre_root_id,
        post_root_id=post_root_id,
        synapse_id=np.arange(n_synapses, dtype=np.int64),
        pre_seg_id=pre_root_id.copy(),
        post_seg_id=post_root_id.copy(),
    )
    return chunk, synapses
