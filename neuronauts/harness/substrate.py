"""Cached region substrate: real L2 geometry, v117 atoms, proofread labels.

The substrate is built proofread-cell-first, which is what makes it tractable.
Going supervoxel-first from a dense region is not: a 400x350x350 um slab holds
43.7M synapses over 86.8M unique supervoxels, and mapping those to roots at
~2k/s would take about twelve hours. Walking outward from the ~250 proofread
cells instead costs about ten minutes and yields only labelled geometry.

Construction, per proofread cell at the target version:

  1. ``level2_chunk_graph(root, bounds)`` -- REAL L2 adjacency inside the
     region. This is the substrate earlier rounds lacked; EXP-053B and EXP-056
     had to approximate topology with a kNN/MST over synapse endpoints and
     both flagged that as a limitation.
  2. ``rep_coord_nm`` for every L2 node, pooled across all cells into large
     batched requests. Pooling matters: the attributes endpoint is keyed on
     L2 ids, not roots, so per-root batching wastes the request budget.
  3. ``roots_at(l2_ids, v117_timestamp)`` -- the v117 root each L2 node
     belonged to. These v117 roots are the atoms the grammar must reassemble,
     and the proofread cell id is the ground-truth label.

Measured on a 5-cell pilot: 100% coordinate coverage, 96.1% v117 resolution,
and 45-99 v117 fragments per proofread cell.

Every network stage caches to disk, so a rebuild is free and experiments run
fully offline.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import requests

from neuronauts.data import lineage as L

PT_VOXEL_NM = np.asarray([4.0, 4.0, 40.0])
SEG_VOXEL_NM = np.asarray([8.0, 8.0, 40.0])
GT_DIR = Path("data/gt_manifest")


@dataclass
class Substrate:
    """A region of real, labelled connectome geometry.

    Nodes are L2 chunks. ``node_frag`` is the v117 root that owned the node --
    the atom to be assembled. ``node_cell`` is the proofread root at the target
    version -- the ground-truth neuron. ``edge_index`` is real L2 adjacency.
    """

    node_l2_id: np.ndarray      # [N] uint64
    node_pos_nm: np.ndarray     # [N, 3] float32
    node_frag: np.ndarray       # [N] uint64, v117 atom (0 = unresolved)
    node_cell: np.ndarray       # [N] uint64, proofread ground truth
    edge_index: np.ndarray      # [E, 2] int32, into node arrays
    cell_ids: np.ndarray        # [C] uint64
    cell_soma_nm: np.ndarray    # [C, 3] float32
    meta: dict

    @property
    def n_nodes(self) -> int:
        return len(self.node_l2_id)

    @property
    def n_cells(self) -> int:
        return len(self.cell_ids)

    def fragment_table(self, min_nodes: int = 1):
        """Per-atom summary: node count, owning cells, centroid.

        A fragment owned by more than one cell is a real v117 false merge --
        the atomization signal. One owned by exactly one cell is a false split
        to be repaired.
        """
        keep = self.node_frag > 0
        frags, inv = np.unique(self.node_frag[keep], return_inverse=True)
        pos = self.node_pos_nm[keep]
        cells = self.node_cell[keep]
        counts = np.bincount(inv, minlength=len(frags))
        centroid = np.zeros((len(frags), 3), np.float32)
        for d in range(3):
            centroid[:, d] = np.bincount(inv, weights=pos[:, d],
                                         minlength=len(frags)) / counts
        n_owners = np.zeros(len(frags), np.int32)
        owner = np.zeros(len(frags), np.uint64)
        for i in range(len(frags)):
            u = np.unique(cells[inv == i])
            n_owners[i] = len(u)
            owner[i] = u[0]
        m = counts >= min_nodes
        return {"frag_id": frags[m], "n_nodes": counts[m],
                "centroid_nm": centroid[m], "n_owner_cells": n_owners[m],
                "owner_cell": owner[m]}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, node_l2_id=self.node_l2_id, node_pos_nm=self.node_pos_nm,
            node_frag=self.node_frag, node_cell=self.node_cell,
            edge_index=self.edge_index, cell_ids=self.cell_ids,
            cell_soma_nm=self.cell_soma_nm,
            meta=np.frombuffer(json.dumps(self.meta).encode(), np.uint8))


def load_substrate(path: str | Path) -> Substrate:
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(bytes(z["meta"]).decode()) if "meta" in z else {}
        return Substrate(
            node_l2_id=z["node_l2_id"], node_pos_nm=z["node_pos_nm"],
            node_frag=z["node_frag"], node_cell=z["node_cell"],
            edge_index=z["edge_index"], cell_ids=z["cell_ids"],
            cell_soma_nm=z["cell_soma_nm"], meta=meta)


# ---------------------------------------------------------------------------
# Network stages (each cached)
# ---------------------------------------------------------------------------

def load_proofread_table(version: int):
    import pandas as pd
    df = pd.read_csv(GT_DIR / f"proofreading_status_v{version}.csv.gz")
    df["x_nm"] = df["pt_position_x"] * PT_VOXEL_NM[0]
    df["y_nm"] = df["pt_position_y"] * PT_VOXEL_NM[1]
    df["z_nm"] = df["pt_position_z"] * PT_VOXEL_NM[2]
    return df


def select_cells(df, centre_um, side_um, tier: str = "gold"):
    """Proofread cells whose soma sits inside the region, at a quality tier."""
    centre = np.asarray(centre_um, float) * 1000.0
    half = np.asarray(side_um, float) * 1000.0 / 2.0
    pos = df[["x_nm", "y_nm", "z_nm"]].to_numpy()
    m = np.all(np.abs(pos - centre) <= half, axis=1)
    m &= df["status_dendrite"].astype(bool).to_numpy()
    m &= df["status_axon"].astype(bool).to_numpy()
    m &= (df["strategy_dendrite"].astype(str) == "dendrite_extended").to_numpy()
    if tier == "gold":
        m &= (df["strategy_axon"].astype(str) == "axon_fully_extended").to_numpy()
    return df.loc[m]


def region_bounds(centre_um, side_um):
    centre = np.asarray(centre_um, float) * 1000.0
    half = np.asarray(side_um, float) * 1000.0 / 2.0
    lo, hi = centre - half, centre + half
    # The chunkedgraph parses bounds with int(); floats produce a 500.
    seg = np.array([[lo[i] / SEG_VOXEL_NM[i], hi[i] / SEG_VOXEL_NM[i]]
                    for i in range(3)], dtype=int)
    return lo, hi, seg


def fetch_l2_graphs(roots, seg_bounds, cache_dir: Path, workers: int = 6,
                    verbose: bool = True) -> dict[int, np.ndarray]:
    """Real L2 adjacency per root, cached as one NPZ each."""
    from caveclient import CAVEclient

    cache_dir.mkdir(parents=True, exist_ok=True)
    client = CAVEclient("minnie65_public")

    def go(root: int):
        f = cache_dir / f"{root}.npz"
        if f.exists():
            try:
                with np.load(f, allow_pickle=False) as z:
                    return root, z["edges"], None
            except Exception:
                pass
        try:
            e = np.asarray(client.chunkedgraph.level2_chunk_graph(
                int(root), bounds=seg_bounds), dtype=np.uint64)
        except Exception as exc:  # noqa: BLE001
            return root, None, f"{type(exc).__name__}: {exc}"
        if e.ndim != 2 or e.shape[0] == 0:
            e = np.zeros((0, 2), np.uint64)
        np.savez_compressed(f, edges=e)
        return root, e, None

    out, errs = {}, []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (root, e, err) in enumerate(ex.map(go, roots), 1):
            if err:
                errs.append((root, err))
            elif e is not None and len(e):
                out[root] = e
            if verbose and i % 25 == 0:
                print(f"    L2 graphs {i}/{len(roots)}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
    if errs and verbose:
        print(f"    {len(errs)} roots failed; first: {errs[0][1][:120]}")
    return out


def fetch_l2_coords(l2_ids: np.ndarray, token: str, cache_path: Path,
                    batch: int = 2000, workers: int = 8,
                    verbose: bool = True) -> dict[int, np.ndarray]:
    """Pooled rep_coord_nm for every L2 id, cached as a single NPZ."""
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            known = dict(zip(z["l2_id"].tolist(), z["pos_nm"]))
        missing = np.array([i for i in l2_ids.tolist() if i not in known],
                           dtype=np.uint64)
        if len(missing) == 0:
            return known
    else:
        known, missing = {}, l2_ids

    url = f"{L.L2_CACHE_SERVER}/l2cache/api/v1/table/{L.L2_TABLE}/attributes"
    hdr = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    chunks = [missing[i:i + batch].tolist() for i in range(0, len(missing), batch)]

    def go(chunk):
        for attempt in range(3):
            try:
                r = requests.post(url, headers=hdr, timeout=180,
                                  json={"l2_ids": chunk,
                                        "attribute_names": ["rep_coord_nm"]})
                if r.status_code == 200:
                    return {int(k): np.asarray(v["rep_coord_nm"], np.float32)
                            for k, v in r.json().items() if v.get("rep_coord_nm")}
            except Exception:
                pass
            time.sleep(1.0 + attempt)
        return {}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, d in enumerate(ex.map(go, chunks), 1):
            known.update(d)
            if verbose and i % 20 == 0:
                print(f"    coords {i}/{len(chunks)} batches "
                      f"({time.time()-t0:.0f}s)", flush=True)

    ids = np.array(list(known.keys()), np.uint64)
    pos = np.stack([known[int(i)] for i in ids]) if len(ids) else np.zeros((0, 3), np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, l2_id=ids, pos_nm=pos.astype(np.float32))
    return known


def fetch_v117_map(l2_ids: np.ndarray, token: str, cache_path: Path,
                   workers: int = 8, verbose: bool = True) -> dict[int, int]:
    """Map each L2 node to the v117 root that owned it, cached.

    This is the step that turns current geometry into historical atoms. The
    pilot resolved 96.1% of nodes; a sharp drop from that is a signal something
    is wrong, not a data property.
    """
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            known = dict(zip(z["l2_id"].tolist(), z["v117"].tolist()))
        missing = np.array([i for i in l2_ids.tolist() if i not in known],
                           dtype=np.uint64)
        if len(missing) == 0:
            return known
    else:
        known, missing = {}, l2_ids

    batch = L._ROOTS_BATCH
    chunks = [missing[i:i + batch] for i in range(0, len(missing), batch)]

    def go(chunk):
        for attempt in range(3):
            try:
                r = L.roots_at(chunk, L.V117_TIMESTAMP, token=token)
                if r is not None:
                    return chunk, r
            except Exception:
                pass
            time.sleep(1.0 + attempt)
        return chunk, np.zeros(len(chunk), np.uint64)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (chunk, r) in enumerate(ex.map(go, chunks), 1):
            known.update(zip(chunk.tolist(), r.tolist()))
            if verbose and i % 20 == 0:
                print(f"    v117 {i}/{len(chunks)} batches "
                      f"({time.time()-t0:.0f}s)", flush=True)

    ids = np.array(list(known.keys()), np.uint64)
    vals = np.array([known[int(i)] for i in ids], np.uint64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, l2_id=ids, v117=vals)
    return known


def build_substrate(centre_um, side_um, *, tier: str = "gold",
                    version: int = 1822, cache_dir: str | Path = "data/substrate",
                    name: str = "region", limit_cells: int = 0,
                    workers: int = 6, token: Optional[str] = None,
                    verbose: bool = True) -> Substrate:
    token = token or L.DEFAULT_TOKEN
    cache = Path(cache_dir) / name
    cache.mkdir(parents=True, exist_ok=True)

    df = load_proofread_table(version)
    sel = select_cells(df, centre_um, side_um, tier)
    if limit_cells:
        sel = sel.head(limit_cells)
    roots = [int(r) for r in sel["pt_root_id"].to_numpy()]
    soma = sel[["x_nm", "y_nm", "z_nm"]].to_numpy(np.float32)
    if verbose:
        print(f"[1/5] {len(roots)} proofread cells [{tier}] "
              f"in {side_um} um @ {centre_um}", flush=True)
    if not roots:
        raise RuntimeError("no proofread cells selected for this region/tier")

    lo, hi, seg = region_bounds(centre_um, side_um)

    if verbose:
        print("[2/5] real L2 adjacency graphs ...", flush=True)
    graphs = fetch_l2_graphs(roots, seg, cache / "l2_graphs", workers, verbose)
    if verbose:
        print(f"      {len(graphs)}/{len(roots)} cells returned geometry")
    if not graphs:
        raise RuntimeError("no L2 graphs retrieved")

    pool = np.unique(np.concatenate([np.unique(e) for e in graphs.values()]))
    if verbose:
        print(f"[3/5] pooled L2 nodes: {len(pool):,} -> rep_coord_nm", flush=True)
    coords = fetch_l2_coords(pool, token, cache / "l2_coords.npz",
                             workers=max(workers, 8), verbose=verbose)

    if verbose:
        print(f"[4/5] v117 atom labels for {len(pool):,} nodes", flush=True)
    v117 = fetch_v117_map(pool, token, cache / "l2_v117.npz",
                          workers=max(workers, 8), verbose=verbose)

    if verbose:
        print("[5/5] assembling substrate", flush=True)
    node_ids, node_pos, node_frag, node_cell = [], [], [], []
    index: dict[int, int] = {}
    for root, edges in graphs.items():
        for l2 in np.unique(edges).tolist():
            if l2 in index or l2 not in coords:
                continue
            p = coords[l2]
            if not np.all((p >= lo) & (p < hi)):
                continue  # exact clip; bounds only restrict to whole chunks
            index[l2] = len(node_ids)
            node_ids.append(l2)
            node_pos.append(p)
            node_frag.append(v117.get(l2, 0))
            node_cell.append(root)

    ei = []
    for edges in graphs.values():
        for a, b in edges.tolist():
            ia, ib = index.get(a), index.get(b)
            if ia is not None and ib is not None and ia != ib:
                ei.append((ia, ib))
    ei = np.unique(np.sort(np.asarray(ei, np.int32), axis=1), axis=0) if ei \
        else np.zeros((0, 2), np.int32)

    sub = Substrate(
        node_l2_id=np.asarray(node_ids, np.uint64),
        node_pos_nm=np.asarray(node_pos, np.float32),
        node_frag=np.asarray(node_frag, np.uint64),
        node_cell=np.asarray(node_cell, np.uint64),
        edge_index=ei,
        cell_ids=np.asarray(list(graphs.keys()), np.uint64),
        cell_soma_nm=np.asarray(
            [soma[roots.index(r)] for r in graphs.keys()], np.float32),
        meta={"region_centre_um": list(map(float, centre_um)),
              "region_side_um": (list(map(float, side_um))
                                 if np.ndim(side_um) else float(side_um)),
              "tier": tier, "target_version": version,
              "base_version": 117, "v117_timestamp": int(L.V117_TIMESTAMP),
              "lower_nm": lo.tolist(), "upper_nm": hi.tolist(),
              "n_cells_requested": len(roots),
              "l2_topology": "real level2_chunk_graph adjacency"},
    )
    sub.save(cache / "substrate.npz")
    return sub
