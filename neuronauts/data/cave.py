"""Fetch v117 synapse + skeleton data from CAVE for synapse co-assignment.

The pipeline produces four arrays that feed directly into ``build_synapse_graph``:

- ``positions_nm [N, 3]`` — global nm coordinates of pre-synaptic terminals
- ``seg_ids [N]`` — v117 root IDs (the noisy scaffold labels)
- ``gt_labels [N]`` — v1412 root IDs (expert-proofread ground truth, 0 = unknown)
- ``skeletons`` — one kimimaro skeleton per unique v117 segment ID

The split between fetching (this module) and DNA encoding (represent/) is
intentional: fetching requires a network connection; encoding requires a GPU.
Cache the skeletons to disk and re-encode as needed.

Example
-------
::

    from neuronauts.data.cave import fetch_v117_region, CAVE_TOKEN
    from neuronauts.represent.skeleton_gnn import SkeletonGNN, encode_fragments_gnn
    from neuronauts.coassign import build_synapse_graph, SynapseCoassigner, train

    region = fetch_v117_region(bbox_nm, token=CAVE_TOKEN)
    # Encode DNA for each skeleton
    seg_dna = encode_seg_dna(region.skeletons, region.seg_ids)
    graph = build_synapse_graph(region.positions_nm, region.seg_ids,
                                region.gt_labels, seg_dna)
    model = SynapseCoassigner(node_dim=graph.node_dim)
    train(model, [graph])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..fetch import (
    CAVE_SERVER,
    MICRONS_DATASTACK,
    SkeletonData,
    fetch_root_skeleton,
    fetch_synapses,
)
from ..cave_root_mapping import map_roots_between_versions

log = logging.getLogger(__name__)

# Default auth token for the public MICrONS / minnie65_public datastack.
# Override by passing token= to fetch_v117_region.
CAVE_TOKEN: str | None = None

# Canonical version numbers for the MICrONS minnie65 dataset.
V117 = 117
V1412 = 1412

# Synapse position columns are in CAVE's native 4×4×40 nm synapse voxels.
# To convert to global nm: pts_nm = pts_voxels * (4, 4, 40) + bbox_origin_nm
# However, fetch_synapses already returns box-relative MIP-2 voxels (32×32×40 nm).
# We convert back to global nm using the bbox origin here.
MIP2_VOX = np.array([32, 32, 40], dtype=np.float32)


@dataclass
class V117Region:
    """Fetched v117 data for one spatial region, ready for co-assignment.

    All arrays are parallel (same length N = number of synapses).

    Attributes
    ----------
    positions_nm:
        Global nm coordinates of each synapse's pre-synaptic terminal. ``[N, 3]``
    seg_ids:
        v117 root ID of the pre-synaptic segment for each synapse. ``[N]``
        These are the *noisy* scaffold labels — the evidence the model works with.
    gt_labels:
        v1412 root ID of each synapse's pre-synaptic neuron. ``[N]``
        These are the *ground-truth* labels used for training and evaluation.
        0 = the v117 segment did not map to any v1412 root (segment went extinct
        between versions, or was a false-merge that was split and both pieces
        are ambiguous).
    skeletons:
        One ``SkeletonData`` per unique seg_id in ``seg_ids``.  Keyed by v117
        root ID.  Segments with zero vertices were unreachable and are included
        as empty skeletons so callers can safely index by seg_id.
    bbox_nm:
        The bounding box used to fetch synapses, ``((x0,y0,z0), (x1,y1,z1))``.
    n_synapses:
        Number of synapses (= len(positions_nm)).
    n_segments:
        Number of unique v117 segments with at least one synapse.
    """

    positions_nm: np.ndarray
    seg_ids: np.ndarray
    gt_labels: np.ndarray
    skeletons: dict
    bbox_nm: tuple

    @property
    def n_synapses(self) -> int:
        return len(self.positions_nm)

    @property
    def n_segments(self) -> int:
        return len(self.skeletons)


def _robust_map_roots(
    seg_ids: list[int],
    old_version: int,
    new_version: int,
    *,
    token: str | None,
    batch_size: int = 50,
    per_batch_timeout_s: float = 60.0,
    max_attempts: int = 3,
) -> dict[int, int]:
    """Map root IDs version-to-version with per-batch timeout and retry.

    The chunkedgraph ``get_latest_roots`` endpoint occasionally hangs with no
    timeout of its own — a single stalled call would block the whole pipeline
    forever. We split the IDs into small batches and run each in a worker
    thread bounded by ``per_batch_timeout_s``; a stalled batch is retried up to
    ``max_attempts`` times. IDs that never map (all attempts time out) are
    simply omitted from the result, so callers treat them as label 0.
    """
    import threading

    def _run_batch(batch: list[int], box: dict) -> None:
        # Runs in a daemon thread; a stalled call is abandoned (the thread is
        # left to die with the process) rather than blocking the pipeline.
        try:
            box["result"] = map_roots_between_versions(
                batch, old_version, new_version, token=token,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller via box
            box["error"] = exc

    mapping: dict[int, int] = {}
    batches = [seg_ids[i:i + batch_size] for i in range(0, len(seg_ids), batch_size)]
    for bi, batch in enumerate(batches):
        for attempt in range(1, max_attempts + 1):
            box: dict = {}
            t = threading.Thread(target=_run_batch, args=(batch, box), daemon=True)
            t.start()
            t.join(per_batch_timeout_s)
            if "result" in box:
                mapping.update(box["result"])
                break
            if t.is_alive():
                log.warning(
                    "Mapping batch %d/%d stalled past %.0fs (attempt %d/%d); abandoning",
                    bi + 1, len(batches), per_batch_timeout_s, attempt, max_attempts,
                )
            elif "error" in box:
                log.warning(
                    "Mapping batch %d/%d errored (attempt %d/%d): %s",
                    bi + 1, len(batches), attempt, max_attempts, repr(box["error"])[:200],
                )
        log.info("Mapped %d/%d batches (%d ids resolved so far)",
                 bi + 1, len(batches), len(mapping))
    return mapping


def _fetch_skeletons_parallel(
    root_ids: list[int],
    *,
    version: int,
    token: str | None,
    datastack: str,
    cave_server: str,
    cache_dir: str | None,
    workers: int = 8,
) -> dict[int, "SkeletonData"]:
    """Fetch skeletons concurrently, keyed by root ID.

    v117 roots are skeletonised on demand by the service (~1/s serially), so a
    sequential fetch of a few hundred roots takes many minutes. Each skeleton
    is independent network I/O, so a small thread pool overlaps the waits and
    cuts wall time roughly ``workers``-fold. Every worker writes its own
    ``.npz`` to ``cache_dir``, so an interrupted run resumes from cache.
    Unreachable roots (extinct / no skeleton) are returned as empty skeletons.
    """
    import concurrent.futures
    import threading

    progress = {"done": 0}
    lock = threading.Lock()
    total = len(root_ids)

    def _one(rid: int) -> tuple[int, "SkeletonData"]:
        try:
            sk = fetch_root_skeleton(
                int(rid), version=version, token=token, datastack=datastack,
                cave_server=cave_server, cache_dir=cache_dir, max_retries=2,
            )
        except Exception:  # noqa: BLE001 - missing/extinct roots are expected
            sk = SkeletonData(
                root_id=int(rid), materialization_version=int(version),
                vertices=np.zeros((0, 3), dtype=np.float32),
                edges=np.zeros((0, 2), dtype=np.int64), radius=None,
            )
        with lock:
            progress["done"] += 1
            if progress["done"] % 25 == 0 or progress["done"] == total:
                log.info("  skeletons fetched: %d/%d", progress["done"], total)
        return int(rid), sk

    out: dict[int, "SkeletonData"] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for rid, sk in pool.map(_one, [int(r) for r in root_ids]):
            out[rid] = sk
    return out


def fetch_v117_region(
    bbox_nm: tuple,
    *,
    token: str | None = CAVE_TOKEN,
    min_seg_synapses: int = 2,
    max_segs: int = 500,
    skeleton_cache_dir: str | None = None,
    v117_version: int = V117,
    v1412_version: int = V1412,
    datastack: str = MICRONS_DATASTACK,
    cave_server: str = CAVE_SERVER,
    skeleton_workers: int = 8,
) -> V117Region:
    """Fetch synapse co-assignment data for a spatial region at version 117.

    Parameters
    ----------
    bbox_nm:
        Bounding box ``((x0,y0,z0), (x1,y1,z1))`` in global nanometers.
        Use ``neuronauts.fetch.make_cube_bbox_nm`` to build one from a center.
    token:
        CAVE auth token.  Required for private datastacks; the public
        minnie65_public datastack works without one but rate-limits unauthenticated
        requests heavily.
    min_seg_synapses:
        Minimum number of synapses a v117 segment must have to be included.
        Segments with fewer synapses are discarded — they contribute almost no
        training signal and inflate skeleton-fetch cost.
    max_segs:
        Hard cap on the number of unique v117 segments to fetch skeletons for.
        The segments with the most synapses are kept.  Set to ``None`` to fetch all.
    skeleton_cache_dir:
        If given, skeletons are cached as ``.npz`` files in this directory.
        Subsequent calls to the same root IDs skip the network fetch.
    v117_version:
        CAVE materialization version to use for synapse and skeleton fetching.
    v1412_version:
        CAVE materialization version to use for ground-truth label mapping.

    Returns
    -------
    V117Region
        Synapse positions, v117 seg_ids, v1412 gt_labels, and skeletons.

    Notes
    -----
    The co-assignment pipeline uses *pre*-synaptic positions and seg_ids.
    Pre-synaptic terminals lie on axons; one axon rarely has two synapses on
    the same target, so pre-side clustering gives cleaner neuron separation
    than post-side clustering (which would mix excitatory + inhibitory inputs).
    """
    # ------------------------------------------------------------------ #
    # 1. Fetch synapse table at v117
    # ------------------------------------------------------------------ #
    log.info("Fetching synapses in bbox=%s at version %d", bbox_nm, v117_version)
    syn = fetch_synapses(
        bbox_nm,
        version=v117_version,
        token=token,
        datastack=datastack,
        cave_server=cave_server,
    )
    if syn.n_synapses == 0:
        log.warning("No synapses found in bbox %s at v%d", bbox_nm, v117_version)
        return V117Region(
            positions_nm=np.zeros((0, 3), dtype=np.float32),
            seg_ids=np.zeros(0, dtype=np.int64),
            gt_labels=np.zeros(0, dtype=np.int64),
            skeletons={},
            bbox_nm=bbox_nm,
        )

    # Convert box-relative MIP-2 voxels → global nm
    bbox_origin = np.array(bbox_nm[0], dtype=np.float32)
    positions_nm = (syn.pre_pt * MIP2_VOX + bbox_origin).astype(np.float32)
    seg_ids_raw = syn.pre_root_id.astype(np.int64)

    # ------------------------------------------------------------------ #
    # 2. Filter: keep only segments with >= min_seg_synapses synapses
    # ------------------------------------------------------------------ #
    unique_segs, counts = np.unique(seg_ids_raw, return_counts=True)
    keep_segs = set(unique_segs[counts >= min_seg_synapses].tolist())
    mask = np.array([int(s) in keep_segs for s in seg_ids_raw.tolist()], dtype=bool)
    positions_nm = positions_nm[mask]
    seg_ids = seg_ids_raw[mask]

    if len(seg_ids) == 0:
        log.warning("All segments have fewer than %d synapses; region is empty", min_seg_synapses)
        return V117Region(
            positions_nm=np.zeros((0, 3), dtype=np.float32),
            seg_ids=np.zeros(0, dtype=np.int64),
            gt_labels=np.zeros(0, dtype=np.int64),
            skeletons={},
            bbox_nm=bbox_nm,
        )

    # ------------------------------------------------------------------ #
    # 3. Cap to max_segs by synapse count (largest segments first)
    # ------------------------------------------------------------------ #
    if max_segs is not None:
        unique_segs, counts = np.unique(seg_ids, return_counts=True)
        if len(unique_segs) > max_segs:
            top_idx = np.argsort(counts)[-max_segs:]
            keep_segs = set(unique_segs[top_idx].tolist())
            mask = np.array([int(s) in keep_segs for s in seg_ids.tolist()], dtype=bool)
            positions_nm = positions_nm[mask]
            seg_ids = seg_ids[mask]
            log.info("Capped to %d segments (dropped %d small ones)",
                     max_segs, len(unique_segs) - max_segs)

    unique_seg_ids = np.unique(seg_ids).tolist()
    log.info("%d synapses across %d unique v%d segments",
             len(seg_ids), len(unique_seg_ids), v117_version)

    # ------------------------------------------------------------------ #
    # 4. Map v117 seg_ids → v1412 ground-truth labels
    # ------------------------------------------------------------------ #
    log.info("Mapping %d v%d seg_ids → v%d", len(unique_seg_ids), v117_version, v1412_version)
    v117_to_v1412 = _robust_map_roots(
        unique_seg_ids, v117_version, v1412_version, token=token,
    )
    gt_labels = np.array(
        [v117_to_v1412.get(int(s), 0) for s in seg_ids.tolist()],
        dtype=np.int64,
    )
    n_mapped = int((gt_labels > 0).sum())
    log.info("%d / %d synapses have a valid v%d label (%.0f%%)",
             n_mapped, len(gt_labels), v1412_version,
             100 * n_mapped / max(len(gt_labels), 1))

    # ------------------------------------------------------------------ #
    # 5. Fetch v117 skeletons for each unique segment (in parallel)
    # ------------------------------------------------------------------ #
    log.info("Fetching skeletons for %d v%d segments (%d workers)",
             len(unique_seg_ids), v117_version, skeleton_workers)
    skeletons = _fetch_skeletons_parallel(
        unique_seg_ids,
        version=v117_version,
        token=token,
        datastack=datastack,
        cave_server=cave_server,
        cache_dir=skeleton_cache_dir,
        workers=skeleton_workers,
    )
    n_empty = sum(1 for sk in skeletons.values() if len(sk.vertices) == 0)
    if n_empty:
        log.warning("%d / %d skeletons were empty (root unreachable at v%d)",
                    n_empty, len(skeletons), v117_version)

    return V117Region(
        positions_nm=positions_nm,
        seg_ids=seg_ids,
        gt_labels=gt_labels,
        skeletons=skeletons,
        bbox_nm=bbox_nm,
    )


def encode_seg_dna(
    skeletons: dict,
    seg_ids: np.ndarray | Sequence[int],
    *,
    dna_dim: int = 64,
    device: str = "cpu",
) -> dict[int, np.ndarray]:
    """Encode DNA embeddings for each unique segment using SkeletonGNN.

    Parameters
    ----------
    skeletons:
        Dict mapping seg_id → ``SkeletonData`` (from ``fetch_v117_region``).
    seg_ids:
        Array of seg_ids present in the synapse data (used to pick which
        skeletons to encode; a seg may be in skeletons but have zero synapses).
    dna_dim:
        Dimension of the output DNA embedding.
    device:
        Torch device string ("cpu" or "cuda").

    Returns
    -------
    dict[int, np.ndarray]
        Maps each seg_id to a float32 ``[dna_dim]`` DNA embedding.
        Segments with empty skeletons get a zero embedding.
    """
    from ..represent.skeleton_gnn import SkeletonGNN, encode_fragments_gnn
    from ..schemas import Fragment

    unique_segs = list({int(s) for s in np.asarray(seg_ids, dtype=np.int64).tolist()})
    fragments = []
    for seg_id in unique_segs:
        sk = skeletons.get(seg_id)
        if sk is None or len(sk.vertices) == 0:
            continue
        radii = sk.radius if sk.radius is not None else np.zeros(len(sk.vertices), np.float32)
        # Build a minimal Fragment for the SkeletonGNN encoder
        frag = Fragment(
            fragment_id=seg_id,
            region_id="v117",
            base_root_id=seg_id,
            vertices_nm=sk.vertices,
            edges=sk.edges,
            radius_nm=radii,
            endpoints_nm=_leaf_vertices(sk.vertices, sk.edges),
            synapse_indices=np.empty(0, dtype=np.int64),
        )
        fragments.append(frag)

    if not fragments:
        return {int(s): np.zeros(dna_dim, dtype=np.float32) for s in unique_segs}

    encoder = SkeletonGNN(output_dim=dna_dim)
    encoded = encode_fragments_gnn(encoder, fragments, device=device)

    seg_dna: dict[int, np.ndarray] = {}
    for frag in encoded:
        dna = frag.dna if frag.dna is not None else np.zeros(dna_dim, dtype=np.float32)
        seg_dna[int(frag.fragment_id)] = dna.astype(np.float32)

    # Fill in zeros for segments with empty skeletons
    for seg_id in unique_segs:
        if seg_id not in seg_dna:
            seg_dna[seg_id] = np.zeros(dna_dim, dtype=np.float32)

    return seg_dna


def _leaf_vertices(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return positions of leaf vertices (degree <= 1) in the skeleton."""
    if len(vertices) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if len(edges) == 0:
        return vertices[:1]
    degree = np.zeros(len(vertices), dtype=np.int32)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    leaf_idx = np.where(degree <= 1)[0]
    return vertices[leaf_idx]
