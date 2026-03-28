"""Centralized scipy fallbacks.

Every module that needs ``cKDTree``, ``cdist``, ``gaussian_filter``, or
``sobel`` should import from here.  When scipy is available the real
implementations are used; otherwise lightweight numpy-only substitutes
are provided so the core package remains usable without scipy installed.
"""

from __future__ import annotations

import numpy as np

# ── scipy.spatial.cKDTree ────────────────────────────────────────────

try:
    from scipy.spatial import cKDTree
except ImportError:

    class cKDTree:  # type: ignore[no-redef]
        """Minimal fallback with the subset of scipy.spatial.cKDTree used here."""

        def __init__(self, data: np.ndarray) -> None:
            self.data = np.asarray(data, dtype=np.float32)

        def query_ball_point(self, points: np.ndarray, r: float):
            pts = np.asarray(points, dtype=np.float32)
            scalar_input = pts.ndim == 1
            if scalar_input:
                pts = pts[None, :]
            result = []
            for point in pts:
                dist = np.linalg.norm(self.data - point, axis=1)
                result.append(np.flatnonzero(dist <= r).tolist())
            return result[0] if scalar_input else result

        def query_pairs(self, r: float, output_type: str = "set"):
            pairs = []
            for i in range(len(self.data)):
                dist = np.linalg.norm(self.data[i + 1 :] - self.data[i], axis=1)
                neighbors = np.flatnonzero(dist <= r)
                for offset in neighbors.tolist():
                    pairs.append((i, i + 1 + offset))
            if output_type == "ndarray":
                return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
            return {tuple(pair) for pair in pairs}

        def query(self, point: np.ndarray, k: int = 1):
            point_arr = np.asarray(point, dtype=np.float32)
            if point_arr.ndim == 1:
                dist = np.linalg.norm(self.data - point_arr, axis=1)
                if k == 1:
                    idx = int(np.argmin(dist))
                    return float(dist[idx]), idx
                top_k = min(k, len(self.data))
                order = np.argsort(dist)[:top_k]
                return dist[order], order
            # Batch query
            results_d, results_i = [], []
            for p in point_arr:
                d, i = self.query(p, k=k)
                results_d.append(d)
                results_i.append(i)
            return np.array(results_d), np.array(results_i)


# ── scipy.spatial.distance.cdist ─────────────────────────────────────

try:
    from scipy.spatial.distance import cdist
except ImportError:

    def cdist(left: np.ndarray, right: np.ndarray) -> np.ndarray:  # type: ignore[misc]
        left_arr = np.asarray(left, dtype=np.float32)
        right_arr = np.asarray(right, dtype=np.float32)
        if len(left_arr) == 0 or len(right_arr) == 0:
            return np.zeros((len(left_arr), len(right_arr)), dtype=np.float32)
        diff = left_arr[:, None, :] - right_arr[None, :, :]
        return np.linalg.norm(diff, axis=-1).astype(np.float32, copy=False)


# ── scipy.ndimage filters ───────────────────────────────────────────

try:
    from scipy.ndimage import gaussian_filter, sobel
except ImportError:

    def gaussian_filter(volume: np.ndarray, sigma: float = 1.0) -> np.ndarray:  # type: ignore[misc]
        del sigma
        return volume.astype(np.float32, copy=False)

    def sobel(volume: np.ndarray, axis: int) -> np.ndarray:  # type: ignore[misc]
        return np.gradient(volume.astype(np.float32, copy=False), axis=axis)


__all__ = ["cKDTree", "cdist", "gaussian_filter", "sobel"]
