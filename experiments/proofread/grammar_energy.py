"""Pillar 1 — the global shape *grammar* as an energy over neuron morphology.

A valid neuron is a "grammatical sentence": exactly one soma; axon and dendrite
both emanate from the soma (no axon↔dendrite fusion away from it); caliber varies
smoothly (no jumps); a single connected tree.  ``grammar_energy`` scores how
*ungrammatical* a reconstructed object is (higher = worse).  A proofreading **edit**
(a cut or a join) changes the parse; ``edit_delta_energy`` returns how much the edit
*reduces* the ungrammaticality — the global cue for whether the edit is correct.

This is the AutoProof-style global/shape cue, but as an interpretable generative
energy (it both proposes edits — where the energy is high — and scores each edit's
global effect), reusing the verified grammar pieces:
``neuronauts.soma_clusters`` (multi-soma) and, when synapses are available,
``experiments.pcfg.compartment_grammar.object_signals`` (A↔D crossing).

Terms are skeleton-only by default (no synapses / EM needed), so it is cheap enough
to score every candidate edit densely.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuronauts.soma_clusters import soma_clusters


@dataclass
class EnergyTerms:
    soma: float          # (n_soma - 1)+  : multi-soma is a merge of two cells
    caliber: float       # abrupt radius jumps along edges (foreign process grafted)
    disconnect: float    # >1 connected component (should be one tree)
    ad: float            # axon↔dendrite crossing not via soma (needs compartments)
    total: float

    def as_dict(self) -> dict:
        return {"soma": self.soma, "caliber": self.caliber,
                "disconnect": self.disconnect, "ad": self.ad, "total": self.total}


def _n_components(nv, edges) -> int:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    if nv == 0:
        return 0
    if len(edges) == 0:
        return nv
    g = csr_matrix((np.ones(len(edges) * 2),
                    (np.concatenate([edges[:, 0], edges[:, 1]]),
                     np.concatenate([edges[:, 1], edges[:, 0]]))), shape=(nv, nv))
    return int(connected_components(g, directed=False)[0])


def _caliber_energy(vertices, edges, radius, *, rel_jump=1.5, min_nm=300.0) -> float:
    """Count of *extreme* relative radius jumps across edges (a foreign-caliber graft).

    A smooth process tapers gently (ratio well under ``1+rel_jump``); a merge grafts
    cable of a different diameter, producing a step.  Threshold is set high
    (``rel_jump=1.5`` => >2.5x diameter change) so normal tapering scores ~0 and only
    genuine steps count; tiny calibers are ignored."""
    if radius is None or len(edges) == 0:
        return 0.0
    r = np.asarray(radius, float)
    ra, rb = r[edges[:, 0]], r[edges[:, 1]]
    ok = (ra > min_nm) & (rb > min_nm) & np.isfinite(ra) & np.isfinite(rb)
    if not ok.any():
        return 0.0
    ratio = np.maximum(ra[ok], rb[ok]) / (np.minimum(ra[ok], rb[ok]) + 1e-9)
    return float((ratio > (1.0 + rel_jump)).sum())


def grammar_energy(vertices_nm, edges, radius=None, *, compartment_labels=None,
                   w_soma=1.0, w_caliber=0.2, w_disconnect=1.0, w_ad=1.0) -> EnergyTerms:
    """Ungrammaticality energy of a reconstructed object (higher = worse)."""
    V = np.asarray(vertices_nm, float)
    edges = np.asarray(edges, np.int64).reshape(-1, 2)
    nv = len(V)

    n_soma = len(soma_clusters(V, radius))
    soma_e = float(max(0, n_soma - 1))                       # 2 somas => a merge
    caliber_e = _caliber_energy(V, edges, radius)
    disc_e = float(max(0, _n_components(nv, edges) - 1))

    ad_e = 0.0
    if compartment_labels is not None:
        from experiments.pcfg.compartment_grammar import object_signals
        ad_e = float(object_signals(compartment_labels).ad_score)

    total = (w_soma * soma_e + w_caliber * caliber_e +
             w_disconnect * disc_e + w_ad * ad_e)
    return EnergyTerms(soma_e, caliber_e, disc_e, ad_e, total)


def cut_delta_energy(vertices_nm, edges, radius, cut_edge, **kw) -> float:
    """ΔEnergy of CUTTING ``cut_edge`` = energy(whole) − Σ energy(resulting pieces).

    The two sides become *separate objects*, so each is scored on its own (a
    2-soma object splits into two 1-soma cells, each grammatical).  Positive =>
    the cut removes ungrammaticality (a good merge-correction).
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    V = np.asarray(vertices_nm, float)
    edges = np.asarray(edges, np.int64).reshape(-1, 2)
    before = grammar_energy(V, edges, radius, **kw).total
    keep = ~((edges[:, 0] == cut_edge[0]) & (edges[:, 1] == cut_edge[1]) |
             (edges[:, 0] == cut_edge[1]) & (edges[:, 1] == cut_edge[0]))
    e2 = edges[keep]
    nv = len(V)
    if len(e2) == 0:
        return before
    g = csr_matrix((np.ones(len(e2) * 2),
                    (np.concatenate([e2[:, 0], e2[:, 1]]),
                     np.concatenate([e2[:, 1], e2[:, 0]]))), shape=(nv, nv))
    ncc, comp = connected_components(g, directed=False)
    after = 0.0
    for cid in range(ncc):
        sel = np.where(comp == cid)[0]
        if len(sel) < 2:
            continue
        remap = -np.ones(nv, int); remap[sel] = np.arange(len(sel))
        m = np.isin(e2[:, 0], sel) & np.isin(e2[:, 1], sel)
        se = np.stack([remap[e2[m, 0]], remap[e2[m, 1]]], axis=1)
        sr = None if radius is None else np.asarray(radius, float)[sel]
        after += grammar_energy(V[sel], se, sr, **kw).total
    return float(before - after)


def join_delta_energy(vA, eA, rA, vB, eB, rB, *, bridge=None, **kw) -> float:
    """ΔEnergy of JOINING two objects = energy(A)+energy(B) − energy(joined).

    Positive => joining makes the result *more* grammatical (a good split-fix, e.g.
    reuniting two dendrite arcs of one cell — no new soma, smooth caliber).  A join
    that fuses two somata or grafts a foreign caliber goes negative (rejected)."""
    vA = np.asarray(vA, float); vB = np.asarray(vB, float)
    eA = np.asarray(eA, np.int64).reshape(-1, 2); eB = np.asarray(eB, np.int64).reshape(-1, 2)
    rA = None if rA is None else np.asarray(rA, float)
    rB = None if rB is None else np.asarray(rB, float)
    eB2 = eB + len(vA)
    V = np.vstack([vA, vB])
    if bridge is None:
        from scipy.spatial import cKDTree
        d, i = cKDTree(vA).query(vB, k=1); j = int(np.argmin(d)); bridge = (int(i[j]), int(j) + len(vA))
    E = np.vstack([eA, eB2, [list(bridge)]])
    R = None if (rA is None or rB is None) else np.concatenate([rA, rB])
    e_join = grammar_energy(V, E, R, **kw).total
    e_a = grammar_energy(vA, eA, rA, **kw).total
    e_b = grammar_energy(vB, eB, rB, **kw).total
    return float(e_a + e_b - e_join)
