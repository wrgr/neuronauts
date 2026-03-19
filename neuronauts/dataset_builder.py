"""Real MICrONS box selection, fetching, and disk caching for training.

Usage
-----
Quick start — CAVE-only, bigger boxes (recommended)::

    python scripts/train.py build-dataset \\
        --cache-dir data/boxes30 \\
        --n-boxes 80 \\
        --box-side-um 30 \\
        --min-positive-pairs 5 \\
        --no-em

This fetches only synapse tables from CAVE (no EM volume download) and
filters for boxes that contain at least 5 same-root-id synapse pairs —
guaranteeing the grammar has real positive examples to learn from.

Or with the Python API::

    from neuronauts.dataset_builder import BoxCache, build_dataset, select_synapse_seeded_boxes
    specs   = select_synapse_seeded_boxes(n=80, box_side_um=30)
    cache   = BoxCache("data/boxes30")
    records = build_dataset(specs, cache, no_em=True, min_positive_pairs=5, verbose=True)

Data format
-----------
Each box is persisted as two files under ``cache_dir/``::

    <hash>.npz   — [volume,] pre_pt, post_pt, pre_root_id, post_root_id,
                   synapse_id, [pre_seg_id, post_seg_id]
                   (volume is absent for CAVE-only / --no-em boxes)
    <hash>.json  — metadata (center_nm, side_um, mip, n_synapses,
                   n_positive_pairs, has_volume, …)

An ``index.json`` in the root of the cache dir lists all records so the
cache can be loaded without scanning the filesystem.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .fetch import (
    CAVE_SERVER,
    MICRONS_DATASTACK,
    SYNAPSE_VOXEL_SIZE_NM,
    RealBoxSpec,
    SynapseTable,
    VolumeChunk,
    fetch_synapses,
    fetch_volume,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MICrONS Minnie65 approximate interior extent (nm, mip-independent)
# These are conservative inward margins to avoid edge tiles with missing data.
#
# NOTE: The full dataset footprint is large but CAVE synapse annotations only
# cover the proofread neuropil core (~1 mm³).  Random sampling over the full
# extents yields mostly empty boxes.  Use select_synapse_seeded_boxes() instead
# of select_random_boxes() unless you have the nucleus table available.
# ---------------------------------------------------------------------------
MINNIE65_X_NM = (300_000, 3_800_000)
MINNIE65_Y_NM = (300_000, 2_700_000)
MINNIE65_Z_NM = (50_000,  780_000)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BoxRecord:
    """Metadata for a single cached training box.

    The ``box_hash`` is derived from the ``RealBoxSpec.cache_key`` and is
    used to name the on-disk files.  All numeric fields are stored in the
    companion ``<hash>.json`` file so the cache can be reconstructed from
    the index without loading the heavy ``.npz`` arrays.

    Attributes
    ----------
    n_positive_pairs:
        Number of same-root-id synapse pairs (pre-side + post-side combined).
        0 means unknown (boxes cached before this field was added).
    has_volume:
        False for CAVE-only boxes built with ``no_em=True`` / ``--no-em``.
        Grammar training works fine without the volume; only GAT/agent
        simulation requires it.
    """

    box_hash: str
    center_nm: tuple
    side_um: float
    mip: int
    n_synapses: int
    n_positive_pairs: int = 0
    has_volume: bool = True
    root_id_version: int | None = None

    def to_spec(self) -> RealBoxSpec:
        return RealBoxSpec(
            center_nm=tuple(self.center_nm),   # type: ignore[arg-type]
            side_um=self.side_um,
            mip=self.mip,
        )


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

class BoxCache:
    """Disk-backed cache of EM volumes + synapse tables.

    The cache is append-only: saving a box that already exists is a no-op.
    Thread-safety is *not* guaranteed — build datasets sequentially.

    Parameters
    ----------
    cache_dir:
        Root directory for the cache.  Created automatically if absent.
    """

    _INDEX_FILE = "index.json"

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index: list[dict] = self._load_index()

    # ── Persistence helpers ───────────────────────────────────────────────

    def _index_path(self) -> Path:
        return self.cache_dir / self._INDEX_FILE

    def _load_index(self) -> list[dict]:
        p = self._index_path()
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return []

    def _save_index(self) -> None:
        self._index_path().write_text(
            json.dumps(self._index, indent=2), encoding="utf-8"
        )

    def _npz_path(self, box_hash: str) -> Path:
        return self.cache_dir / f"{box_hash}.npz"

    def _meta_path(self, box_hash: str) -> Path:
        return self.cache_dir / f"{box_hash}.json"

    # ── Public API ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def all_records(self) -> list[BoxRecord]:
        """Return all cached box records.

        Supplies default values for fields added after the initial cache was
        built so old caches remain readable.
        """
        _defaults = {
            "n_positive_pairs": 0,
            "has_volume": True,
            "root_id_version": None,
        }
        return [BoxRecord(**{**_defaults, **entry}) for entry in self._index]

    def contains(self, spec: RealBoxSpec) -> bool:
        return any(entry["box_hash"] == spec.cache_key for entry in self._index)

    def save(
        self,
        spec: RealBoxSpec,
        volume: VolumeChunk,
        synapses: SynapseTable,
        n_positive_pairs: int = 0,
        root_id_version: int | None = None,
    ) -> BoxRecord:
        """Persist a (volume, synapses) pair.  Returns the new record.

        If the box is already cached this is a no-op and the existing record
        is returned.
        """
        box_hash = spec.cache_key
        if self.contains(spec):
            _defaults = {"n_positive_pairs": 0, "has_volume": True}
            existing = next(
                BoxRecord(**{**_defaults, **e})
                for e in self._index if e["box_hash"] == box_hash
            )
            return existing

        arrays: dict[str, np.ndarray] = {
            "volume": volume.data.astype(np.uint8),
            "pre_pt": synapses.pre_pt,
            "post_pt": synapses.post_pt,
            "pre_root_id": synapses.pre_root_id,
            "post_root_id": synapses.post_root_id,
            "synapse_id": synapses.synapse_id,
        }
        if synapses.pre_seg_id is not None:
            arrays["pre_seg_id"] = synapses.pre_seg_id
        if synapses.post_seg_id is not None:
            arrays["post_seg_id"] = synapses.post_seg_id

        np.savez_compressed(self._npz_path(box_hash), **arrays)

        meta = {
            "center_nm": list(spec.center_nm),
            "side_um": spec.side_um,
            "mip": spec.mip,
            "voxel_size_nm": list(volume.voxel_size_nm),
            "volume_shape": list(volume.data.shape),
            "n_synapses": int(len(synapses.pre_pt)),
            "n_positive_pairs": n_positive_pairs,
            "has_volume": True,
            "root_id_version": root_id_version,
        }
        self._meta_path(box_hash).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        record = BoxRecord(
            box_hash=box_hash,
            center_nm=tuple(spec.center_nm),
            side_um=spec.side_um,
            mip=spec.mip,
            n_synapses=int(len(synapses.pre_pt)),
            n_positive_pairs=n_positive_pairs,
            has_volume=True,
            root_id_version=root_id_version,
        )
        self._index.append(asdict(record))
        self._save_index()
        return record

    def save_synapse_only(
        self,
        spec: RealBoxSpec,
        synapses: SynapseTable,
        n_positive_pairs: int = 0,
        root_id_version: int | None = None,
    ) -> BoxRecord:
        """Persist synapse table only — no EM volume (CAVE-only mode).

        Grammar training works entirely from synapse geometry and root IDs;
        the EM volume is only needed for agent simulation (GAT training).
        Skipping the volume fetch makes data collection ~10× faster and
        allows much larger box sizes without memory pressure.
        """
        box_hash = spec.cache_key
        if self.contains(spec):
            _defaults = {"n_positive_pairs": 0, "has_volume": False}
            existing = next(
                BoxRecord(**{**_defaults, **e})
                for e in self._index if e["box_hash"] == box_hash
            )
            return existing

        arrays: dict[str, np.ndarray] = {
            "pre_pt":      synapses.pre_pt,
            "post_pt":     synapses.post_pt,
            "pre_root_id": synapses.pre_root_id,
            "post_root_id":synapses.post_root_id,
            "synapse_id":  synapses.synapse_id,
        }
        if synapses.pre_seg_id is not None:
            arrays["pre_seg_id"] = synapses.pre_seg_id
        if synapses.post_seg_id is not None:
            arrays["post_seg_id"] = synapses.post_seg_id

        np.savez_compressed(self._npz_path(box_hash), **arrays)

        meta = {
            "center_nm": list(spec.center_nm),
            "side_um": spec.side_um,
            "mip": spec.mip,
            "n_synapses": int(len(synapses.pre_pt)),
            "n_positive_pairs": n_positive_pairs,
            "has_volume": False,
            "root_id_version": root_id_version,
        }
        self._meta_path(box_hash).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        record = BoxRecord(
            box_hash=box_hash,
            center_nm=tuple(spec.center_nm),
            side_um=spec.side_um,
            mip=spec.mip,
            n_synapses=int(len(synapses.pre_pt)),
            n_positive_pairs=n_positive_pairs,
            has_volume=False,
            root_id_version=root_id_version,
        )
        self._index.append(asdict(record))
        self._save_index()
        return record

    def load(self, record: BoxRecord) -> tuple[VolumeChunk, SynapseTable]:
        """Load a (VolumeChunk, SynapseTable) pair from disk.

        For CAVE-only boxes (``record.has_volume is False``) the returned
        ``VolumeChunk`` has an empty ``data`` array.  Grammar training ignores
        the volume; only GAT/agent-simulation steps need a real volume.
        """
        npz = np.load(self._npz_path(record.box_hash), allow_pickle=False)
        meta = json.loads(self._meta_path(record.box_hash).read_text(encoding="utf-8"))

        if "volume" in npz and meta.get("has_volume", True):
            volume = VolumeChunk(
                data=npz["volume"],
                voxel_size_nm=tuple(meta["voxel_size_nm"]),
                bbox_voxels=((0, 0, 0), tuple(meta["volume_shape"])),
                mip=record.mip,
            )
        else:
            # Synapse-only box: return a stub volume so callers don't need
            # to special-case the return type.
            volume = VolumeChunk(
                data=np.zeros((0, 0, 0), dtype=np.uint8),
                voxel_size_nm=(32.0, 32.0, 40.0),
                bbox_voxels=((0, 0, 0), (0, 0, 0)),
                mip=record.mip,
            )

        synapses = SynapseTable(
            pre_pt=npz["pre_pt"],
            post_pt=npz["post_pt"],
            pre_root_id=npz["pre_root_id"],
            post_root_id=npz["post_root_id"],
            synapse_id=npz["synapse_id"],
            pre_seg_id=npz.get("pre_seg_id"),
            post_seg_id=npz.get("post_seg_id"),
        )
        return volume, synapses

    def iter_records(
        self,
        *,
        shuffle: bool = False,
        rng: np.random.Generator | None = None,
    ) -> Iterator[BoxRecord]:
        """Iterate over all cached records, optionally shuffled."""
        records = self.all_records()
        if shuffle:
            if rng is None:
                rng = np.random.default_rng()
            order = rng.permutation(len(records))
            records = [records[i] for i in order]
        yield from records


# ---------------------------------------------------------------------------
# Box selection strategies
# ---------------------------------------------------------------------------

def select_random_boxes(
    n: int,
    *,
    box_side_um: float = 6.0,
    mip: int = 2,
    seed: int = 42,
    x_range_nm: tuple[int, int] = MINNIE65_X_NM,
    y_range_nm: tuple[int, int] = MINNIE65_Y_NM,
    z_range_nm: tuple[int, int] = MINNIE65_Z_NM,
) -> list[RealBoxSpec]:
    """Uniformly sample ``n`` box centres within the Minnie65 interior.

    The half-width of each box (``box_side_um / 2 * 1000`` nm) is subtracted
    from each boundary so the sampled boxes never extend outside the declared
    range.

    Parameters
    ----------
    n:
        Number of box specs to generate.
    box_side_um:
        Side length in microns (default 6 µm ≈ the typical agent run box).
    mip:
        MIP level for fetching the EM volume.
    seed:
        RNG seed for reproducibility.
    x/y/z_range_nm:
        Override the default Minnie65 interior bounds.
    """
    rng = np.random.default_rng(seed)
    half_nm = int(box_side_um * 1000 / 2)
    specs = []
    for _ in range(n):
        cx = int(rng.integers(x_range_nm[0] + half_nm, x_range_nm[1] - half_nm))
        cy = int(rng.integers(y_range_nm[0] + half_nm, y_range_nm[1] - half_nm))
        cz = int(rng.integers(z_range_nm[0] + half_nm, z_range_nm[1] - half_nm))
        specs.append(RealBoxSpec(center_nm=(cx, cy, cz), side_um=box_side_um, mip=mip))
    return specs


def select_synapse_seeded_boxes(
    n: int,
    *,
    box_side_um: float = 6.0,
    mip: int = 2,
    seed: int = 42,
    token: str | None = None,
    sample_limit: int = 2000,
    datastack: str = MICRONS_DATASTACK,
    cave_server: str = CAVE_SERVER,
    cave_version: int | None = None,
) -> list[RealBoxSpec]:
    """Pick box centres from real CAVE synapse positions.

    Queries up to ``sample_limit`` synapses from CAVE and randomly draws ``n``
    of them to use as box centres.  Because the centres come from actual synapse
    positions every resulting box is guaranteed to be inside the annotated
    neuropil — unlike ``select_random_boxes`` which samples the full (mostly
    empty) dataset extent.

    No bounding box filter is applied; CAVE returns a representative spatial
    sample across the annotated region.

    Parameters
    ----------
    n:
        Number of box specs to return.
    box_side_um:
        Side length in microns.
    mip:
        MIP level for EM volume fetching.
    seed:
        RNG seed for reproducibility.
    token:
        Optional CAVE auth token (not required for public minnie65 access).
    sample_limit:
        Maximum number of synapses to pull from CAVE; must be ≥ n.
    """
    from .fetch import _install_system_trust_store

    _install_system_trust_store()
    try:
        from caveclient import CAVEclient
    except ImportError as exc:
        raise ImportError("pip install caveclient") from exc

    rng = np.random.default_rng(seed)
    client = CAVEclient(datastack, server_address=cave_server, auth_token=token)
    if cave_version is not None:
        client.version = cave_version

    # Pull a spatial sample from the synapse table.  Using synapse_query with
    # no bounding box may be refused for very large tables; fall back to
    # query_table with a limit if needed.
    syn_vox = np.array(SYNAPSE_VOXEL_SIZE_NM, dtype=np.float64)
    positions_nm: list[tuple[int, int, int]] = []

    try:
        df = client.materialize.synapse_query(limit=sample_limit)
        if len(df) > 0 and "ctr_pt_position" in df.columns:
            for pos in df["ctr_pt_position"].values:
                arr = np.asarray(pos, dtype=np.float64)
                nm = (arr * syn_vox).astype(int)
                positions_nm.append((int(nm[0]), int(nm[1]), int(nm[2])))
    except Exception:
        pass

    # Fallback: query the raw table with select_columns to reduce bandwidth.
    if not positions_nm:
        try:
            tbl = client.materialize.query_table(
                "synapses_pni_2",
                select_columns=["ctr_pt_position"],
                limit=sample_limit,
            )
            if len(tbl) > 0 and "ctr_pt_position" in tbl.columns:
                for pos in tbl["ctr_pt_position"].values:
                    arr = np.asarray(pos, dtype=np.float64)
                    nm = (arr * syn_vox).astype(int)
                    positions_nm.append((int(nm[0]), int(nm[1]), int(nm[2])))
        except Exception as exc:
            raise RuntimeError(
                f"Could not fetch synapse positions from CAVE: {exc}\n"
                "Check network access and try again, or use --strategy random "
                "with --n-boxes set much higher (e.g. 500) to compensate for "
                "empty-region sampling."
            ) from exc

    if not positions_nm:
        raise RuntimeError(
            "CAVE returned no synapse positions.  "
            "Check that the datastack is accessible and try again."
        )

    k = min(n, len(positions_nm))
    chosen = rng.choice(len(positions_nm), size=k, replace=False)
    return [
        RealBoxSpec(center_nm=positions_nm[i], side_um=box_side_um, mip=mip)
        for i in chosen
    ]


def select_boxes_from_nucleus_table(
    counts_tsv: str,
    nucleus_csv: str,
    n: int,
    *,
    min_syn: int = 30,
    max_syn: int = 500,
    box_side_um: float = 6.0,
    mip: int = 2,
    supervoxel_vox_nm: tuple[int, int, int] = (8, 8, 40),
    seed: int = 42,
) -> list[RealBoxSpec]:
    """Pick box centres from soma positions of high-synapse-count neurons.

    Requires the two static files produced by ``synapse_root_counts_static.py``:

    - ``counts_tsv``  — ``run_logs/synapse_root_counts_static.tsv``
    - ``nucleus_csv`` — ``data/microns_static/v<ver>/nucleus_detection_v0.csv``

    Neurons with ``min_syn ≤ total_synapse_count ≤ max_syn`` are preferred.
    The nucleus table must have position columns (``pt_position_x``,
    ``pt_position_y``, ``pt_position_z`` in supervoxel voxels at
    ``supervoxel_vox_nm`` resolution).  If position columns are absent the
    function raises a clear ``ValueError``.

    Parameters
    ----------
    counts_tsv:
        TSV path from ``synapse_root_counts_static.py``.
    nucleus_csv:
        Nucleus detection CSV (downloaded by ``ensure_static_files``).
    n:
        Number of boxes to return (may be fewer if fewer matching neurons exist).
    min_syn / max_syn:
        Synapse count range for root ID selection.
    supervoxel_vox_nm:
        Voxel size (nm) of the supervoxel resolution used in the nucleus table.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pip install pandas") from exc

    rng = np.random.default_rng(seed)

    # ── Load counts table ──────────────────────────────────────────────────
    counts_df = pd.read_csv(counts_tsv, sep="\t")
    mask = (
        (counts_df["total_synapse_count"] >= min_syn)
        & (counts_df["total_synapse_count"] <= max_syn)
    )
    good_roots = set(counts_df.loc[mask, "root_id"].astype("int64").tolist())
    if not good_roots:
        raise ValueError(
            f"No root IDs with {min_syn} ≤ total_synapse_count ≤ {max_syn} "
            f"found in {counts_tsv}"
        )

    # ── Load nucleus positions ─────────────────────────────────────────────
    nuc_df = pd.read_csv(nucleus_csv)

    # Locate position columns — handle several naming conventions.
    pos_col_variants = [
        ("pt_position_x", "pt_position_y", "pt_position_z"),
        ("position_x", "position_y", "position_z"),
        ("x", "y", "z"),
    ]
    pos_cols: tuple[str, str, str] | None = None
    for variant in pos_col_variants:
        if all(c in nuc_df.columns for c in variant):
            pos_cols = variant
            break

    if pos_cols is None:
        raise ValueError(
            "nucleus_csv has no recognisable position columns.  "
            f"Available columns: {list(nuc_df.columns)}"
        )

    root_col = "pt_root_id" if "pt_root_id" in nuc_df.columns else "root_id"
    nuc_df = nuc_df[nuc_df[root_col].isin(good_roots)].copy()
    if nuc_df.empty:
        raise ValueError(
            "No rows in nucleus_csv match the selected root IDs. "
            "Check that the materialization versions align."
        )

    # Shuffle and take up to n.
    order = rng.permutation(len(nuc_df))
    nuc_df = nuc_df.iloc[order[:n]].reset_index(drop=True)

    vox = np.array(supervoxel_vox_nm, dtype=np.float64)
    specs = []
    for _, row in nuc_df.iterrows():
        cx_nm = int(float(row[pos_cols[0]]) * vox[0])
        cy_nm = int(float(row[pos_cols[1]]) * vox[1])
        cz_nm = int(float(row[pos_cols[2]]) * vox[2])
        specs.append(
            RealBoxSpec(center_nm=(cx_nm, cy_nm, cz_nm), side_um=box_side_um, mip=mip)
        )
    return specs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_positive_pairs(synapses: SynapseTable) -> int:
    """Count same-root-id synapse pairs (pre-side + post-side combined).

    A positive pair (y_ij = 1) is any two synapses that share the same
    pre_root_id or the same post_root_id — i.e. they are on the same neuron
    on at least one side.  This is the quantity the merge head is trained on.

    Boxes with few positive pairs give the grammar almost no positive
    examples to learn from; use ``min_positive_pairs`` in ``build_dataset``
    to filter them out.
    """
    from collections import Counter

    total = 0
    for root_ids in (synapses.pre_root_id, synapses.post_root_id):
        for n in Counter(root_ids).values():
            if n >= 2:
                total += n * (n - 1) // 2
    return total


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    specs: list[RealBoxSpec],
    cache: BoxCache,
    *,
    min_synapses: int = 10,
    max_synapses: int = 300,
    min_positive_pairs: int = 0,
    no_em: bool = False,
    token: str | None = None,
    cave_version: int | None = None,
    verbose: bool = True,
) -> list[BoxRecord]:
    """Fetch each box spec from MICrONS and persist to *cache*.

    Already-cached boxes are skipped.  Boxes failing the synapse or
    positive-pair filters are skipped and not cached.

    Parameters
    ----------
    specs:
        Box specs to fetch (from any ``select_*`` helper).
    cache:
        Destination ``BoxCache``.
    min_synapses / max_synapses:
        Total synapse-count filter applied after fetching.
    min_positive_pairs:
        Minimum number of same-root-id synapse pairs required.  Boxes below
        this threshold contain almost no positive training examples and should
        be discarded.  Recommended: 5 for 30 µm boxes, 2 for 15 µm boxes.
        Default 0 keeps all boxes (backward-compatible).
    no_em:
        If True, skip the EM volume fetch and store only the synapse table.
        Grammar training requires only synapse geometry and root IDs, so this
        is safe for all grammar-only workflows.  Makes data collection ~10×
        faster and allows much larger boxes without memory pressure.
    token:
        Optional CAVE auth token (not required for public minnie65 access).
    verbose:
        Print progress.

    Returns
    -------
    List of ``BoxRecord`` for every usable box now present in the cache
    (including previously cached boxes that pass the filters).
    """
    records: list[BoxRecord] = []
    n_skip_cached = 0
    n_skip_synapse = 0
    n_skip_pairs = 0
    n_fetched = 0

    for i, spec in enumerate(specs):
        if cache.contains(spec):
            existing = next(r for r in cache.all_records() if r.box_hash == spec.cache_key)
            passes = (
                min_synapses <= existing.n_synapses <= max_synapses
                and existing.n_positive_pairs >= min_positive_pairs
            )
            if passes:
                records.append(existing)
            n_skip_cached += 1
            continue

        if verbose:
            em_tag = "" if no_em else " + EM"
            print(
                f"  [{i+1}/{len(specs)}] fetching center={spec.center_nm} "
                f"side={spec.side_um}µm{em_tag} …",
                end=" ",
                flush=True,
            )

        try:
            # fetch_synapses will set CAVEclient.version when passed.
            synapses = fetch_synapses(
                spec.bbox_nm,
                mip=spec.mip,
                token=token,
                version=cave_version,
            )
            n_syn = int(len(synapses.pre_pt))

            if n_syn < min_synapses or n_syn > max_synapses:
                if verbose:
                    print(f"skip (n_synapses={n_syn})")
                n_skip_synapse += 1
                continue

            n_pos = count_positive_pairs(synapses)
            if n_pos < min_positive_pairs:
                if verbose:
                    print(f"skip (n_synapses={n_syn}, positive_pairs={n_pos} < {min_positive_pairs})")
                n_skip_pairs += 1
                continue

            if no_em:
                record = cache.save_synapse_only(
                    spec,
                    synapses,
                    n_positive_pairs=n_pos,
                    root_id_version=cave_version,
                )
                if verbose:
                    print(f"ok (n_synapses={n_syn}, positive_pairs={n_pos})")
            else:
                volume = fetch_volume(spec.bbox_nm, mip=spec.mip)
                record = cache.save(
                    spec,
                    volume,
                    synapses,
                    n_positive_pairs=n_pos,
                    root_id_version=cave_version,
                )
                if verbose:
                    print(f"ok (n_synapses={n_syn}, positive_pairs={n_pos}, shape={volume.data.shape})")

            records.append(record)
            n_fetched += 1

        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", spec.center_nm, exc)
            if verbose:
                print(f"FAILED: {exc}")

    if verbose:
        print(
            f"\nDataset build complete: {n_fetched} new, "
            f"{n_skip_cached} already cached, "
            f"{n_skip_synapse} skipped (synapse count), "
            f"{n_skip_pairs} skipped (too few positive pairs). "
            f"Total usable records: {len(records)}"
        )
    return records


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_dataset(
    cache_dir: str | Path,
    *,
    min_synapses: int = 0,
    max_synapses: int = 999_999,
    min_positive_pairs: int = 0,
) -> tuple[BoxCache, list[BoxRecord]]:
    """Load an existing cache and return (cache, records) filtered by synapse count
    and positive-pair count."""
    cache = BoxCache(cache_dir)
    records = [
        r for r in cache.all_records()
        if min_synapses <= r.n_synapses <= max_synapses
        and r.n_positive_pairs >= min_positive_pairs
    ]
    return cache, records
