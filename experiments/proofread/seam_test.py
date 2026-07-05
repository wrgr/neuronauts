"""Task 26 core test: does the two-cue signal separate a real MERGE SEAM from a
real CONTINUATION when both cues are sampled *on the neurite at the edit site*?

This directly tests the FINDINGS diagnosis — the complementarity run failed because
cues were read at synapse-cleft positions, not on the neurite at the seam.  Here we
place them correctly:

* **SEAM (should CUT, label 1):** a real m343 false-merge = two *current* roots that
  were wrongly joined then split apart.  Their closest-approach skeleton points
  ``(pa, pb)`` are the historical seam — two different cells' cross-sections.  Cues:
  ``local_evidence(pa, pb)`` (cut-face sim low / barrier high => different process)
  and grammar ``join_delta_energy(A, B)`` (negative => joining is ungrammatical).
* **CONTINUATION (should KEEP, label 0):** two skeleton vertices of ONE clean,
  proofread neuron ~``gap_nm`` apart along the cable — a true continuous process
  (cut-face sim high / barrier low).

If the local cue separates seams from continuations *here* (it did not on cleft
positions), the diagnosis holds and the deployable signal is at the seam.  Grammar
is expected to fire only on the multi-soma / cross-compartment seams — the
complementarity boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from experiments.proofread.local_evidence import local_evidence
from experiments.proofread.grammar_energy import join_delta_energy


@dataclass
class SeamRow:
    kind: str            # "seam" or "continuation"
    label: int           # 1 = should cut (different process), 0 = continuation
    cutface_sim: float
    barrier: float
    ok: int
    gap_nm: float
    grammar_join_de: float   # NaN for continuations
    ids: tuple


def _closest_approach(vA, vB):
    from scipy.spatial import cKDTree
    d, i = cKDTree(vA).query(vB, k=1)
    j = int(np.argmin(d))
    return int(i[j]), int(j), float(d[j])


def _geodesic_pair(verts, edges, gap_nm, rng):
    """Pick two vertices ~gap_nm apart along the cable (BFS from a random start)."""
    from collections import deque
    n = len(verts)
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
    for _ in range(20):
        s = int(rng.integers(n))
        # BFS accumulating path length until we exceed gap_nm
        seen = {s: 0.0}; q = deque([s])
        best = None
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w in seen:
                    continue
                seen[w] = seen[u] + float(np.linalg.norm(verts[w] - verts[u]))
                if seen[w] >= gap_nm:
                    best = w; break
                q.append(w)
            if best is not None:
                break
        if best is not None:
            return s, best, seen[best]
    return None


def build_seam_rows(merges, clean_roots, embed_fn, *, version=1822, token=None,
                    client=None, cache_dir="cache/skel_current", min_verts=150,
                    gap_nm=2000.0, cont_per_neuron=6, mip=1, seed=0, verbose=True):
    from neuronauts.fetch import fetch_root_skeleton
    rng = np.random.default_rng(seed)
    rows: list[SeamRow] = []

    # ---- SEAM positives: real m343 false-merges (two current roots) ----
    for m in merges:
        try:
            lat = [int(x) for x in np.atleast_1d(client.chunkedgraph.get_latest_roots(m))]
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  [seam {m}] latest-roots failed: {e}")
            continue
        sk = []
        for rid in lat:
            try:
                s = fetch_root_skeleton(rid, version=version, token=token,
                                        cache_dir=cache_dir, client=client)
            except Exception:
                continue
            if len(s.vertices) >= min_verts:
                sk.append((rid, s))
        if len(sk) != 2:
            continue
        (ida, A), (idb, B) = sk
        vA = A.vertices.astype(float); vB = B.vertices.astype(float)
        ia, jb, gap = _closest_approach(vA, vB)
        pa, pb = vA[ia], vB[jb]
        ev = local_evidence(pa, pb, embed_fn, mip=mip)
        gde = join_delta_energy(vA, A.edges, A.radius, vB, B.edges, B.radius)
        rows.append(SeamRow("seam", 1, ev.cutface_sim, ev.barrier, int(ev.ok),
                            gap, float(gde), (ida, idb)))
        if verbose:
            print(f"  SEAM {ida}/{idb}: sim={ev.cutface_sim:+.3f} barrier={ev.barrier:.3f} "
                  f"gap={gap:.0f}nm joinDE={gde:+.2f} ok={ev.ok}")

    # ---- CONTINUATION negatives: within-neuron cable pairs ----
    for r in clean_roots:
        try:
            s = fetch_root_skeleton(r, version=version, token=token,
                                    cache_dir=cache_dir, client=client)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  [cont {r}] skeleton failed: {e}")
            continue
        v = s.vertices.astype(float); e = np.asarray(s.edges)
        for _ in range(cont_per_neuron):
            pick = _geodesic_pair(v, e, gap_nm, rng)
            if pick is None:
                continue
            i, j, glen = pick
            ev = local_evidence(v[i], v[j], embed_fn, mip=mip)
            rows.append(SeamRow("continuation", 0, ev.cutface_sim, ev.barrier,
                                int(ev.ok), glen, float("nan"), (int(r), int(r))))
            if verbose:
                print(f"  CONT {r}: sim={ev.cutface_sim:+.3f} barrier={ev.barrier:.3f} "
                      f"geo={glen:.0f}nm ok={ev.ok}")
    return rows


def summarize(rows: list[SeamRow]) -> dict:
    ok = [r for r in rows if r.ok]
    seam = [r for r in ok if r.label == 1]
    cont = [r for r in ok if r.label == 0]
    out = {"n_total": len(rows), "n_ok": len(ok),
           "n_seam_ok": len(seam), "n_cont_ok": len(cont)}
    if seam and cont:
        s_sim = np.array([r.cutface_sim for r in seam])
        c_sim = np.array([r.cutface_sim for r in cont])
        s_bar = np.array([r.barrier for r in seam])
        c_bar = np.array([r.barrier for r in cont])
        out["cutface_sim_seam_mean"] = float(s_sim.mean())
        out["cutface_sim_cont_mean"] = float(c_sim.mean())
        out["barrier_seam_mean"] = float(s_bar.mean())
        out["barrier_cont_mean"] = float(c_bar.mean())
        # AUC of "is a seam?" from (1 - cutface_sim) and from barrier
        y = np.array([1] * len(seam) + [0] * len(cont))
        try:
            from sklearn.metrics import roc_auc_score
            out["auc_cutface"] = float(roc_auc_score(y, np.concatenate([-s_sim, -c_sim])))
            out["auc_barrier"] = float(roc_auc_score(y, np.concatenate([s_bar, c_bar])))
        except Exception:  # noqa: BLE001
            pass
    grj = [r.grammar_join_de for r in seam if np.isfinite(r.grammar_join_de)]
    if grj:
        out["grammar_seam_join_de_mean"] = float(np.mean(grj))
        out["grammar_seam_reject_frac"] = float(np.mean([g < 0 for g in grj]))
    return out
