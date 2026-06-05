"""
Field computation for agent sensing.

The runtime can consume either:
- a cached learned membrane probability volume, or
- a deterministic Sobel fallback computed from raw EM intensity.

These helpers implement the field transforms once a membrane scalar field is
available. `compute_membrane_field()` remains the fallback path for runs that
do not have a cached learned membrane volume.
"""

import numpy as np

from .._scipy_compat import gaussian_filter, sobel


def compute_membrane_field(
    volume: np.ndarray,
    sigma: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    vol = volume.astype(np.float32)
    if sigma > 0:
        vol = gaussian_filter(vol, sigma=sigma)

    gx = sobel(vol, axis=0)
    gy = sobel(vol, axis=1)
    gz = sobel(vol, axis=2)
    mag = np.sqrt(gx**2 + gy**2 + gz**2)

    if normalize:
        vmax = mag.max()
        if vmax > 0:
            mag /= vmax

    return mag.astype(np.float32)


def compute_membrane_vectors(
    membrane_field: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    smooth = gaussian_filter(membrane_field, sigma=sigma)

    gx = sobel(smooth, axis=0)
    gy = sobel(smooth, axis=1)
    gz = sobel(smooth, axis=2)

    from ..helpers import safe_normalize

    vectors = np.stack([-gx, -gy, -gz], axis=-1)
    return safe_normalize(vectors, axis=-1).astype(np.float32)


def compute_synapse_attraction_field(
    shape: tuple,
    synapse_pts: np.ndarray,
    radius: float = 10.0,
    falloff: str = "gaussian",
) -> np.ndarray:
    field = np.zeros(shape, dtype=np.float32)
    if len(synapse_pts) == 0:
        return field

    xi, yi, zi = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
    )
    coords = np.stack([xi, yi, zi], axis=-1).astype(np.float32)

    for pt in synapse_pts:
        dist = np.linalg.norm(coords - pt, axis=-1)
        if falloff == "gaussian":
            contribution = np.exp(-0.5 * (dist / (radius / 2)) ** 2)
        else:
            contribution = np.clip(1.0 - dist / radius, 0, 1)
        field = np.maximum(field, contribution)

    return field


def compute_exploration_field(shape: tuple) -> np.ndarray:
    return np.ones(shape, dtype=np.float32)


def sample_field_trilinear(field: np.ndarray, pt: np.ndarray) -> float:
    shape = np.array(field.shape)
    pt = np.clip(pt, 0, shape - 1 - 1e-6)

    x0, y0, z0 = pt.astype(int)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    x1 = min(x1, shape[0] - 1)
    y1 = min(y1, shape[1] - 1)
    z1 = min(z1, shape[2] - 1)

    xd, yd, zd = pt - np.array([x0, y0, z0], dtype=float)

    c000 = field[x0, y0, z0]
    c100 = field[x1, y0, z0]
    c010 = field[x0, y1, z0]
    c110 = field[x1, y1, z0]
    c001 = field[x0, y0, z1]
    c101 = field[x1, y0, z1]
    c011 = field[x0, y1, z1]
    c111 = field[x1, y1, z1]

    return float(
        c000 * (1 - xd) * (1 - yd) * (1 - zd)
        + c100 * xd * (1 - yd) * (1 - zd)
        + c010 * (1 - xd) * yd * (1 - zd)
        + c110 * xd * yd * (1 - zd)
        + c001 * (1 - xd) * (1 - yd) * zd
        + c101 * xd * (1 - yd) * zd
        + c011 * (1 - xd) * yd * zd
        + c111 * xd * yd * zd
    )


def sample_vector_field_trilinear(field: np.ndarray, pt: np.ndarray) -> np.ndarray:
    return np.array([sample_field_trilinear(field[..., i], pt) for i in range(3)], dtype=np.float32)
