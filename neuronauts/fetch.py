"""Data fetching from MICrONS via CloudVolume and CAVEclient."""

from dataclasses import dataclass
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
    pre_pt: np.ndarray
    post_pt: np.ndarray
    pre_root_id: np.ndarray
    post_root_id: np.ndarray
    synapse_id: np.ndarray


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

    return SynapseTable(
        pre_pt=pts_to_voxels("pre_pt_position"),
        post_pt=pts_to_voxels("post_pt_position"),
        pre_root_id=df["pre_pt_root_id"].values.astype(np.int64),
        post_root_id=df["post_pt_root_id"].values.astype(np.int64),
        synapse_id=df.index.values.astype(np.int64),
    )


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

    synapses = SynapseTable(
        pre_pt=pts_pre,
        post_pt=pts_post,
        pre_root_id=(pre_assign + 1).astype(np.int64),
        post_root_id=(post_assign + 10_001).astype(np.int64),
        synapse_id=np.arange(n_synapses, dtype=np.int64),
    )
    return chunk, synapses
