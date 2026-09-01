"""Label-blind atom population for a region.

The atoms are **v117 segmentation objects**, not chunks. v117 is a full
flood-fill + agglomeration segmentation with light proofreading, so a v117 root
is a real object -- a neurite, a glial process, a passing axon -- carrying real
errors. Those errors are the task: an object spanning two neurons is a false
merge, a neuron shattered across many objects is a false split.

The population must be enumerated **without ground truth**, otherwise the task
is rigged. Selecting atoms by "belongs to a proofread cell" would mean every
atom belongs to some real neuron and the assembler never meets a distractor.
The GT-free filter used here is the intended one: every v117 object carrying at
least ``min_synapses`` synapses inside the region. Under a full segmentation
that admits all the glia and passing axons too, which is the confuser
population we want.

Ground truth is attached later, and only where it exists -- most atoms will have
none, which is correct.

Cost is dominated by mapping synapse supervoxels to v117 roots: measured at
5,832 supervoxels/s with 8 workers, 100% resolved. A 100 um cube maps in ~5 min.
Cached, so it is paid once.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from neuronauts.data import lineage as L


@dataclass
class AtomPopulation:
    """v117 objects with synapses in a region, enumerated label-blind."""

    atom_id: np.ndarray          # [A] uint64, v117 root
    n_synapses: np.ndarray       # [A] int32, synapses in region
    centroid_nm: np.ndarray      # [A, 3] float32, from its synapses
    syn_atom_pre: np.ndarray     # [S] uint64, v117 atom of each synapse's pre side
    syn_atom_post: np.ndarray    # [S] uint64
    syn_ctr_nm: np.ndarray       # [S, 3] float32
    syn_id: np.ndarray           # [S] int64
    meta: dict

    def filter_min_synapses(self, k: int) -> "AtomPopulation":
        m = self.n_synapses >= k
        return AtomPopulation(
            atom_id=self.atom_id[m], n_synapses=self.n_synapses[m],
            centroid_nm=self.centroid_nm[m],
            syn_atom_pre=self.syn_atom_pre, syn_atom_post=self.syn_atom_post,
            syn_ctr_nm=self.syn_ctr_nm, syn_id=self.syn_id,
            meta={**self.meta, "min_synapses": k})

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, atom_id=self.atom_id, n_synapses=self.n_synapses,
            centroid_nm=self.centroid_nm, syn_atom_pre=self.syn_atom_pre,
            syn_atom_post=self.syn_atom_post, syn_ctr_nm=self.syn_ctr_nm,
            syn_id=self.syn_id,
            meta=np.frombuffer(json.dumps(self.meta).encode(), np.uint8))


def load_population(path: str | Path) -> AtomPopulation:
    with np.load(Path(path), allow_pickle=False) as z:
        return AtomPopulation(
            atom_id=z["atom_id"], n_synapses=z["n_synapses"],
            centroid_nm=z["centroid_nm"], syn_atom_pre=z["syn_atom_pre"],
            syn_atom_post=z["syn_atom_post"], syn_ctr_nm=z["syn_ctr_nm"],
            syn_id=z["syn_id"],
            meta=json.loads(bytes(z["meta"]).decode()) if "meta" in z else {})


def map_supervoxels(svids: np.ndarray, timestamp: int, *,
                    token: Optional[str] = None, workers: int = 8,
                    cache_path: Optional[Path] = None,
                    verbose: bool = True) -> dict[int, int]:
    """Batched, threaded supervoxel -> root at a timestamp, cached to disk.

    The serial per-call pattern elsewhere in the codebase runs at ~600/s; this
    reaches ~5.8k/s, which is the difference between minutes and hours.
    """
    token = token or L.DEFAULT_TOKEN
    known: dict[int, int] = {}
    if cache_path is not None and Path(cache_path).exists():
        with np.load(cache_path, allow_pickle=False) as z:
            known = dict(zip(z["sv"].tolist(), z["root"].tolist()))
        missing = np.asarray([s for s in svids.tolist() if s not in known],
                             dtype=np.uint64)
    else:
        missing = svids
    if len(missing) == 0:
        return known

    batch = L._ROOTS_BATCH
    chunks = [missing[i:i + batch] for i in range(0, len(missing), batch)]

    def go(chunk):
        for attempt in range(3):
            try:
                r = L.roots_at(chunk, timestamp, token=token)
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
            if verbose and i % 50 == 0:
                done = i * batch
                print(f"    {done:,}/{len(missing):,} supervoxels "
                      f"({done/(time.time()-t0):,.0f}/s)", flush=True)

    if cache_path is not None:
        sv = np.asarray(list(known.keys()), np.uint64)
        rt = np.asarray([known[int(s)] for s in sv], np.uint64)
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, sv=sv, root=rt)
    return known


def build_population(region_npz: str | Path, centre_um, side_um, *,
                     cache_dir: str | Path, token: Optional[str] = None,
                     workers: int = 8, verbose: bool = True) -> AtomPopulation:
    """Enumerate every v117 object with a synapse in the region."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / "population.npz"
    if out.exists():
        if verbose:
            print(f"  population cached at {out}")
        return load_population(out)

    centre = np.asarray(centre_um, float) * 1000.0
    half = np.asarray(side_um, float) * 1000.0 / 2.0

    if verbose:
        print(f"[pop 1/3] loading {region_npz}", flush=True)
    with np.load(Path(region_npz), allow_pickle=False) as z:
        m = np.all(np.abs(z["ctr_nm"] - centre) <= half, axis=1)
        ctr = z["ctr_nm"][m]
        pre_sv, post_sv = z["pre_sv"][m], z["post_sv"][m]
        syn_id = z["synapse_id"][m]
    if verbose:
        print(f"          {len(ctr):,} synapses in region", flush=True)

    sv = np.unique(np.concatenate([pre_sv, post_sv]))
    sv = sv[sv > 0]
    if verbose:
        print(f"[pop 2/3] mapping {len(sv):,} supervoxels -> v117 roots",
              flush=True)
    sv2root = map_supervoxels(sv, L.V117_TIMESTAMP, token=token,
                              workers=workers, cache_path=cache / "sv_v117.npz",
                              verbose=verbose)

    lut = np.zeros(len(sv), np.uint64)
    for i, s in enumerate(sv.tolist()):
        lut[i] = sv2root.get(s, 0)
    order = np.argsort(sv)
    a_pre = lut[order][np.searchsorted(sv[order], pre_sv)]
    a_post = lut[order][np.searchsorted(sv[order], post_sv)]

    if verbose:
        print("[pop 3/3] aggregating atoms", flush=True)
    stacked = np.concatenate([a_pre, a_post])
    pts = np.concatenate([ctr, ctr], axis=0)
    keep = stacked > 0
    atoms, inv = np.unique(stacked[keep], return_inverse=True)
    counts = np.bincount(inv, minlength=len(atoms)).astype(np.int32)
    cen = np.zeros((len(atoms), 3), np.float32)
    for d in range(3):
        cen[:, d] = np.bincount(inv, weights=pts[keep][:, d],
                                minlength=len(atoms)) / counts

    pop = AtomPopulation(
        atom_id=atoms, n_synapses=counts, centroid_nm=cen,
        syn_atom_pre=a_pre, syn_atom_post=a_post,
        syn_ctr_nm=ctr.astype(np.float32), syn_id=syn_id,
        meta={"centre_um": list(map(float, np.atleast_1d(centre_um))),
              "side_um": float(np.atleast_1d(side_um)[0])
              if np.ndim(side_um) == 0 else list(map(float, side_um)),
              "base_version": 117, "v117_timestamp": int(L.V117_TIMESTAMP),
              "selection": "label-blind: every v117 object with a synapse "
                           "whose centre lies in the region",
              "n_synapses": int(len(ctr)), "n_supervoxels": int(len(sv))})
    pop.save(out)
    return pop
