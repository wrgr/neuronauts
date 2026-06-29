"""Public data loader API for the MICrONS Minnie65 dataset (mouse V1, ~1 mm³).

Data sources
------------
- **Nucleus table** (v1412): static CSV.gz on public Google Cloud Storage.
  No auth required.  ~4.7 MB download; column 3 (0-indexed) is root_id int64.
  URL: https://storage.googleapis.com/mat_dbs/public/minnie65_phase3_v1/v1412/
       nucleus_detection_v0_merged.csv.gz

- **Cell type annotations**: coarse excitatory/inhibitory labels from AIBS.
  Tried from public GCS; returns None when unavailable.
  URL: https://storage.googleapis.com/iarpa_microns/minnie/minnie65/tables/
       aibs_soma_coarse_type.csv

- **Skeleton cache**: neuroglancer precomputed binary format from the CAVE
  skeleton cache.  Requires a bearer token (see DEFAULT_TOKEN).
  URL: https://minnie.microns-daf.com/skeletoncache/api/v1/
       minnie65_public/precomputed/skeleton/{root_id}

Binary layout of each skeleton response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  4 bytes  uint32  n_verts
  4 bytes  uint32  n_edges
  n_verts*3 × float32  xyz coordinates (nm)
  n_edges*2 × uint32   edge pairs (vertex indices)
  n_verts   × float32  radii (nm)  — may be absent; 300 nm used as fallback
"""

from __future__ import annotations

import gzip
import io
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

DEFAULT_TOKEN = "a08cdcba8581846f48d5742a75c53311"

_NUCLEUS_URL = (
    "https://storage.googleapis.com/mat_dbs/public/minnie65_phase3_v1/"
    "v1412/nucleus_detection_v0_merged.csv.gz"
)
_SKELETON_BASE = (
    "https://minnie.microns-daf.com/skeletoncache/api/v1/"
    "minnie65_public/precomputed/skeleton"
)
_CELL_TYPE_URLS = [
    "https://storage.googleapis.com/iarpa_microns/minnie/minnie65/tables/aibs_soma_coarse_type.csv",
    "https://storage.googleapis.com/mat_dbs/public/minnie65_phase3_v1/v1412/aibs_soma_coarse_type.csv",
]

_INTER_REQUEST_DELAY = 0.05  # seconds between skeleton requests per worker


# ---------------------------------------------------------------------------
# Nucleus table
# ---------------------------------------------------------------------------

def load_nucleus_table(cache_path: Optional[str] = None) -> pd.DataFrame:
    """Download (or load from cache) the v1412 proofread nucleus table.

    Returns a DataFrame with at least column ``root_id`` (int64).
    Caches to *cache_path* if provided so subsequent calls skip the ~4.7 MB
    download.

    Parameters
    ----------
    cache_path:
        Optional path to a gzip'd CSV cache file.  Written on first fetch,
        read on subsequent calls when the file already exists.
    """
    if cache_path is not None:
        p = Path(cache_path)
        if p.exists():
            return pd.read_csv(p, compression="gzip")

    resp = requests.get(_NUCLEUS_URL, timeout=60)
    resp.raise_for_status()

    rows: list[dict] = []
    with gzip.open(io.BytesIO(resp.content)) as fh:
        for line in fh:
            parts = line.decode().strip().split(",")
            if len(parts) < 4:
                continue
            try:
                root_id = int(parts[3])
            except ValueError:
                continue
            if root_id != 0:
                rows.append({"root_id": root_id})

    df = pd.DataFrame(rows)
    df["root_id"] = df["root_id"].astype(np.int64)

    if cache_path is not None:
        df.to_csv(cache_path, index=False, compression="gzip")

    return df


# ---------------------------------------------------------------------------
# Cell type annotations
# ---------------------------------------------------------------------------

def load_cell_types(cache_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Download (or load from cache) cell type annotations for Minnie65.

    Returns a DataFrame with ``root_id`` and ``cell_type`` columns, or
    ``None`` when no publicly accessible table can be reached.  Tries several
    known GCS paths in order.

    Parameters
    ----------
    cache_path:
        Optional path to a gzip'd CSV cache file.
    """
    if cache_path is not None:
        p = Path(cache_path)
        if p.exists():
            return pd.read_csv(p, compression="gzip")

    for url in _CELL_TYPE_URLS:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            df = pd.read_csv(io.StringIO(resp.text))
            # Normalise column names: find root_id and cell_type equivalents.
            col_map: dict[str, str] = {}
            for col in df.columns:
                lc = col.lower()
                if "root" in lc and "id" in lc and "root_id" not in col_map:
                    col_map[col] = "root_id"
                elif any(k in lc for k in ("cell_type", "coarse", "type", "label")) and "cell_type" not in col_map:
                    col_map[col] = "cell_type"
            if "root_id" not in col_map.values() or "cell_type" not in col_map.values():
                continue
            df = df.rename(columns=col_map)[["root_id", "cell_type"]]
            df["root_id"] = df["root_id"].astype(np.int64)
            if cache_path is not None:
                df.to_csv(cache_path, index=False, compression="gzip")
            return df
        except Exception:
            continue

    return None


# ---------------------------------------------------------------------------
# Single skeleton
# ---------------------------------------------------------------------------

def load_skeleton(
    root_id: int,
    token: str = DEFAULT_TOKEN,
) -> Optional[dict]:
    """Fetch one skeleton from the CAVE skeleton cache.

    Parameters
    ----------
    root_id:
        Proofread neuron root ID at v1412.
    token:
        CAVE bearer auth token.

    Returns
    -------
    dict with keys:

    - ``vertices_nm``: ``np.ndarray`` [V, 3] float32 — xyz in nanometres
    - ``edges``: ``np.ndarray`` [E, 2] int64 — vertex-index pairs
    - ``radii_nm``: ``np.ndarray`` [V] float32 — inscribed-sphere radii

    or ``None`` on any failure (not found, rate-limited, parse error, etc.).
    """
    url = f"{_SKELETON_BASE}/{root_id}"
    try:
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if resp.status_code != 200:
            return None
        data = resp.content
    except Exception:
        return None

    try:
        offset = 0
        n_verts = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        n_edges = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        if n_verts < 3:
            return None
        vertices = (
            np.frombuffer(data, dtype="<f4", count=n_verts * 3, offset=offset)
            .reshape(n_verts, 3)
            .copy()
        )
        offset += n_verts * 3 * 4
        edges = (
            np.frombuffer(data, dtype="<u4", count=n_edges * 2, offset=offset)
            .reshape(n_edges, 2)
            .astype(np.int64)
            .copy()
        )
        offset += n_edges * 2 * 4
        if len(data) - offset >= n_verts * 4:
            radii = np.frombuffer(data, dtype="<f4", count=n_verts, offset=offset).copy()
        else:
            radii = np.full(n_verts, 300.0, dtype=np.float32)
    except Exception:
        return None

    return {"vertices_nm": vertices, "edges": edges, "radii_nm": radii}


# ---------------------------------------------------------------------------
# Concurrent skeleton loader
# ---------------------------------------------------------------------------

def load_skeletons(
    root_ids: list[int],
    token: str = DEFAULT_TOKEN,
    *,
    max_workers: int = 4,
    progress: bool = True,
) -> dict[int, dict]:
    """Fetch multiple skeletons concurrently.

    Failed / missing root IDs are silently omitted from the returned mapping.
    Each worker sleeps 50 ms between requests to stay gentle on the cache.

    Parameters
    ----------
    root_ids:
        List of v1412 root IDs to fetch.
    token:
        CAVE bearer auth token.
    max_workers:
        Thread-pool size.
    progress:
        Show a tqdm progress bar when the package is installed; ignored
        gracefully when tqdm is absent.

    Returns
    -------
    dict mapping ``root_id`` → skeleton dict (same schema as :func:`load_skeleton`).
    """
    try:
        from tqdm import tqdm
        _wrap = tqdm if progress else (lambda x, **kw: x)
    except ImportError:
        _wrap = lambda x, **kw: x  # noqa: E731

    def _fetch(rid: int) -> tuple[int, Optional[dict]]:
        result = load_skeleton(rid, token)
        time.sleep(_INTER_REQUEST_DELAY)
        return rid, result

    out: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch, rid): rid for rid in root_ids}
        for fut in _wrap(as_completed(futures), total=len(futures), desc="skeletons"):
            rid, skel = fut.result()
            if skel is not None:
                out[rid] = skel

    return out


# ---------------------------------------------------------------------------
# Sample neurons
# ---------------------------------------------------------------------------

def sample_neurons(
    n: int,
    *,
    cell_type: Optional[str] = None,
    seed: int = 0,
    cache_path: Optional[str] = None,
) -> list[int]:
    """Sample *n* random proofread root IDs, optionally filtered by cell type.

    Parameters
    ----------
    n:
        Number of root IDs to return.
    cell_type:
        Coarse type string to filter on, e.g. ``"E"`` for excitatory or
        ``"I"`` for inhibitory.  Pass ``None`` (default) to sample from
        all types.  Raises ``ValueError`` if a type is requested but the
        cell type table is unavailable.
    seed:
        Random seed for reproducibility.
    cache_path:
        Passed through to :func:`load_nucleus_table` to cache the download.

    Returns
    -------
    list of int  (length exactly *n*)

    Raises
    ------
    ValueError
        When *cell_type* is specified but the annotation table cannot be
        fetched, or when fewer than *n* neurons match the filter.
    """
    nucleus_df = load_nucleus_table(cache_path=cache_path)
    root_ids = nucleus_df["root_id"].tolist()

    if cell_type is not None:
        ct_df = load_cell_types()
        if ct_df is None:
            raise ValueError(
                "cell_type filter requested but cell type table is not publicly available"
            )
        allowed = set(ct_df.loc[ct_df["cell_type"] == cell_type, "root_id"].tolist())
        root_ids = [r for r in root_ids if r in allowed]
        if len(root_ids) < n:
            raise ValueError(
                f"Only {len(root_ids)} neurons match cell_type={cell_type!r}, "
                f"but {n} were requested"
            )

    rng = np.random.default_rng(seed)
    chosen = rng.choice(root_ids, size=n, replace=False)
    return [int(r) for r in chosen]


# ---------------------------------------------------------------------------
# CLI summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== MICrONS Minnie65 v1412 — data summary ===\n")

    print("Loading nucleus table …")
    nucleus = load_nucleus_table()
    print(f"  Proofread neurons at v1412: {len(nucleus):,}\n")

    print("Loading cell type annotations …")
    ct = load_cell_types()
    if ct is not None:
        breakdown = ct["cell_type"].value_counts().to_dict()
        print("  Cell type breakdown:")
        for k, v in sorted(breakdown.items()):
            print(f"    {k}: {v:,}")
    else:
        print("  Cell type table not publicly available.")
    print()

    print("Sampling 3 random neurons and fetching skeletons …")
    sample = sample_neurons(3, seed=42)
    skeletons = load_skeletons(sample, max_workers=3, progress=False)
    for rid in sample:
        skel = skeletons.get(rid)
        if skel is None:
            print(f"  root_id={rid}: fetch failed")
        else:
            print(
                f"  root_id={rid}: "
                f"{skel['vertices_nm'].shape[0]} vertices, "
                f"{skel['edges'].shape[0]} edges"
            )
