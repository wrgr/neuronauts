"""Whole-cell shape descriptors and the graft corruption, for EXP-083.

A cell's skeleton is re-rooted at its soma and stored breadth-first so every
parent index is smaller than its child's.  That single invariant makes every
tree quantity below computable by depth level with numpy, with no per-vertex
Python loop, which is what lets a few thousand assemblies be scored in a
minute.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

R = Path(__file__).resolve().parents[1]
SKEL = R / "data/external/cell_skeletons"

UM = 1000.0          # nm per micron


# --------------------------------------------------------------------------
# skeleton -> breadth-first rooted tree
# --------------------------------------------------------------------------
def load_tree(path):
    d = np.load(path)
    V = d["vertices"].astype(np.float64)
    E = d["edges"].astype(np.int64)
    rad = d["radius"].astype(np.float64)
    comp = d["compartment"].astype(np.int64)
    n = len(V)
    adj = [[] for _ in range(n)]
    for a, b in E.tolist():
        adj[a].append(b)
        adj[b].append(a)
    soma = int(np.flatnonzero(comp == 1)[0]) if (comp == 1).any() else int(np.argmax(rad))

    parent = np.full(n, -1, np.int64)
    order = [soma]
    seen = np.zeros(n, bool)
    seen[soma] = True
    q = deque([soma])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                parent[v] = u
                order.append(v)
                q.append(v)
    order = np.asarray(order, np.int64)          # breadth-first, soma first
    newid = np.full(n, -1, np.int64)
    newid[order] = np.arange(len(order))
    par = np.where(parent[order] >= 0, newid[parent[order]], -1)
    par[0] = -1
    t = dict(V=V[order], rad=rad[order], comp=comp[order], par=par,
             n=len(order), n_dropped=int(n - len(order)))
    _annotate(t)
    return t


def _annotate(t):
    """Depth levels, edge lengths, geodesic distance, subtree cable."""
    par, V = t["par"], t["V"]
    n = t["n"]
    depth = np.zeros(n, np.int64)
    for v in range(1, n):                        # par[v] < v, one cheap pass
        depth[v] = depth[par[v]] + 1
    levels = [np.flatnonzero(depth == d) for d in range(int(depth.max()) + 1)]
    elen = np.zeros(n)
    elen[1:] = np.linalg.norm(V[1:] - V[par[1:]], axis=1)
    geo = np.zeros(n)
    for lv in levels[1:]:
        geo[lv] = geo[par[lv]] + elen[lv]
    t.update(depth=depth, levels=levels, elen=elen, geo=geo)
    t["sub_cable"] = subtree_sum(t, elen)
    t["is_tip"] = np.bincount(par[1:], minlength=n) == 0
    t["is_tip"][0] = False
    return t


def subtree_sum(t, val):
    """Sum of ``val`` over each vertex's subtree (itself included)."""
    acc = np.array(val, np.float64, copy=True)
    par = t["par"]
    for lv in t["levels"][:0:-1]:
        np.add.at(acc, par[lv], acc[lv])
    return acc


def subtree_mask(t, u):
    """Boolean mask of the subtree rooted at ``u``."""
    par, n = t["par"], t["n"]
    m = np.zeros(n, bool)
    m[u] = True
    for lv in t["levels"][int(t["depth"][u]) + 1:]:
        m[lv] = m[par[lv]]
    return m


# --------------------------------------------------------------------------
# assembly: base tree with a subtree replaced by a (possibly foreign) one
# --------------------------------------------------------------------------
def assemble(A, drop_masks, graft=None):
    """A's arbor minus ``drop_masks``, optionally with ``graft`` attached.

    ``graft`` is ``(B, w, at)``: the subtree of cell ``B`` rooted at ``w``,
    rigidly translated so ``w`` lands exactly where A's vertex ``at`` sat, and
    hung from A's parent of ``at``.  The first edge of the graft is therefore
    identical in length and direction to the edge it replaces -- the join looks
    the same locally whether the piece is A's own or another cell's.
    """
    keep = np.ones(A["n"], bool)
    for m in drop_masks:
        keep &= ~m
    keep[0] = True
    idx = np.flatnonzero(keep)
    newid = np.full(A["n"], -1, np.int64)
    newid[idx] = np.arange(len(idx))
    par = np.where(A["par"][idx] >= 0, newid[A["par"][idx]], -1)
    par[0] = -1
    if (par[1:] < 0).any():                       # a kept vertex lost its parent
        return None
    V, rad, comp = A["V"][idx], A["rad"][idx], A["comp"][idx]

    n_graft = 0
    if graft is not None:
        B, w, at = graft
        gm = subtree_mask(B, w)
        gidx = np.flatnonzero(gm)
        gnew = np.full(B["n"], -1, np.int64)
        gnew[gidx] = np.arange(len(gidx)) + len(idx)
        gpar = gnew[B["par"][gidx]]
        anchor = newid[A["par"][at]]
        if anchor < 0:
            return None
        gpar[0] = anchor                          # gidx[0] == w (breadth-first)
        shift = A["V"][at] - B["V"][w]
        V = np.concatenate([V, B["V"][gidx] + shift])
        rad = np.concatenate([rad, B["rad"][gidx]])
        comp = np.concatenate([comp, B["comp"][gidx]])
        par = np.concatenate([par, gpar])
        n_graft = len(gidx)
    t = dict(V=V, rad=rad, comp=comp, par=par, n=len(V), n_graft=n_graft)
    _annotate(t)
    return t


# --------------------------------------------------------------------------
# whole-cell shape descriptors
# --------------------------------------------------------------------------
SIZE_COLS = ["log_cable", "log_nvert"]
TOPO_COLS = ["n_tips", "n_branch", "tips_per_mm", "mean_seg_um", "strahler_max",
             "mean_tip_order", "asym_mean", "max_children"]
EXTENT_COLS = ["max_geo_um", "med_tip_geo_um", "max_radial_um", "radial_p50_um",
               "radial_p90_um", "geo_over_rad", "tort_med", "pca_e1", "pca_e3",
               "sholl_peak", "sholl_peak_frac", "frac_inward", "self_overlap",
               "frac_beyond_100um"]
RADIUS_COLS = ["rad_mean_um", "rad_cv", "rad_slope", "rad_distal_over_prox"]
COMP_COLS = ["frac_axon", "switch_per_mm", "dend_beyond_100um", "axon_prox_frac"]
ALL_COLS = SIZE_COLS + TOPO_COLS + EXTENT_COLS + RADIUS_COLS + COMP_COLS


def descriptors(t):
    par, V, geo, elen = t["par"], t["V"], t["geo"], t["elen"]
    n = t["n"]
    cable = float(elen.sum())
    cable_um = cable / UM
    nkid = np.bincount(par[1:], minlength=n)
    is_tip = (nkid == 0)
    is_tip[0] = False
    is_branch = nkid >= 2
    tips = np.flatnonzero(is_tip)
    n_tips, n_branch = int(is_tip.sum()), int(is_branch.sum())

    # --- topology ---------------------------------------------------------
    order = np.ones(n, np.int64)                 # Strahler
    br_order = np.zeros(n, np.int64)             # branch points to the soma
    for lv in t["levels"][:0:-1]:
        p = par[lv]
        mx = np.zeros(n, np.int64)
        np.maximum.at(mx, p, order[lv])
        eq = np.bincount(p[order[lv] == mx[p]], minlength=n)
        internal = np.flatnonzero(nkid > 0)
        order[internal] = np.where(eq[internal] >= 2, mx[internal] + 1,
                                   np.maximum(mx[internal], 1))
    for lv in t["levels"][1:]:
        br_order[lv] = br_order[par[lv]] + is_branch[par[lv]]
    # partition asymmetry: cable split at each branch point
    sub = subtree_sum(t, elen)
    asym = []
    if n_branch:
        bp = np.flatnonzero(is_branch)
        kids = {int(b): [] for b in bp}
        ch = np.flatnonzero(is_branch[par[1:]]) + 1
        for c in ch.tolist():
            kids[int(par[c])].append(sub[c])
        for v in kids.values():
            v = sorted(v)
            if len(v) >= 2 and v[-1] > 0:
                asym.append((v[-1] - v[0]) / (v[-1] + v[0] + 1e-9))
    asym_mean = float(np.mean(asym)) if asym else 0.0

    # --- extent -----------------------------------------------------------
    rad_d = np.linalg.norm(V - V[0], axis=1)
    w = elen.copy()
    w[0] = 0.0
    wsum = max(w.sum(), 1e-9)
    ordr = np.argsort(rad_d)
    cw = np.cumsum(w[ordr]) / wsum
    radial_p50 = float(rad_d[ordr][np.searchsorted(cw, 0.5)])
    radial_p90 = float(rad_d[ordr][np.searchsorted(cw, 0.9)])
    tort = geo[tips] / np.maximum(rad_d[tips], 1e-9)
    C = V - np.average(V, axis=0, weights=np.maximum(w, 1e-12))
    ev = np.linalg.eigvalsh(np.cov((C * np.sqrt(np.maximum(w, 0))[:, None]).T
                                   / max(np.sqrt(wsum), 1e-9)))
    ev = np.sort(np.maximum(ev, 0))[::-1]
    evs = max(ev.sum(), 1e-12)

    # Sholl on 40 shells
    rmax = float(rad_d.max())
    shells = np.linspace(rmax / 40.0, rmax, 40)
    r0, r1 = rad_d[par[1:]], rad_d[1:]
    lo, hi = np.minimum(r0, r1), np.maximum(r0, r1)
    cross = ((lo[:, None] < shells[None, :]) & (hi[:, None] >= shells[None, :])).sum(0)
    k = int(np.argmax(cross))

    frac_inward = float(w[1:][r1 < r0].sum() / wsum)

    # cable that doubles back through the arbor: two vertices within 1.5 um of
    # each other but 20 um apart along the cable
    tree = cKDTree(V)
    pairs = tree.query_pairs(1500.0, output_type="ndarray")
    if len(pairs):
        far = np.abs(geo[pairs[:, 0]] - geo[pairs[:, 1]]) > 20_000.0
        self_overlap = float(2.0 * far.sum() / n)
    else:
        self_overlap = 0.0

    # --- radius -----------------------------------------------------------
    rr = t["rad"] / UM
    rad_mean = float(np.average(rr, weights=np.maximum(w, 1e-12)))
    rad_cv = float(np.std(rr) / max(rad_mean, 1e-9))
    g_um = geo / UM
    if g_um.std() > 0:
        slope = float(np.polyfit(g_um, rr, 1)[0] * 100.0)
    else:
        slope = 0.0
    prox = rr[g_um < np.percentile(g_um, 25)]
    dist = rr[g_um > np.percentile(g_um, 75)]
    rad_ratio = float(np.mean(dist) / max(np.mean(prox), 1e-9))

    # --- compartment ------------------------------------------------------
    comp = t["comp"]
    ax = comp == 2
    frac_axon = float(w[ax].sum() / wsum)
    sw = float(w[1:][comp[1:] != comp[par[1:]]].sum())
    switch_per_mm = sw / max(cable / 1e6, 1e-9) / 1e3
    dend = comp == 3
    dend_far = float(w[dend & (rad_d > 100_000)].sum() / max(w[dend].sum(), 1e-9))
    axon_prox = float(w[ax & (rad_d < 30_000)].sum() / max(w[ax].sum(), 1e-9))

    f = dict(
        log_cable=float(np.log10(max(cable_um, 1e-3))),
        log_nvert=float(np.log10(n)),
        n_tips=float(n_tips), n_branch=float(n_branch),
        tips_per_mm=n_tips / max(cable_um / 1000.0, 1e-9),
        mean_seg_um=cable_um / max(n_tips + n_branch, 1),
        strahler_max=float(order[0] if n > 1 else 1),
        mean_tip_order=float(br_order[tips].mean()) if len(tips) else 0.0,
        asym_mean=asym_mean, max_children=float(nkid.max()),
        max_geo_um=float(geo.max() / UM),
        med_tip_geo_um=float(np.median(geo[tips]) / UM) if len(tips) else 0.0,
        max_radial_um=rmax / UM,
        radial_p50_um=radial_p50 / UM, radial_p90_um=radial_p90 / UM,
        geo_over_rad=float(geo.max() / max(rmax, 1e-9)),
        tort_med=float(np.median(tort)) if len(tips) else 1.0,
        pca_e1=float(ev[0] / evs), pca_e3=float(ev[2] / evs),
        sholl_peak=float(cross.max()),
        sholl_peak_frac=float(shells[k] / max(rmax, 1e-9)),
        frac_inward=frac_inward, self_overlap=self_overlap,
        frac_beyond_100um=float(w[rad_d > 100_000].sum() / wsum),
        rad_mean_um=rad_mean, rad_cv=rad_cv, rad_slope=slope,
        rad_distal_over_prox=rad_ratio,
        frac_axon=frac_axon, switch_per_mm=switch_per_mm,
        dend_beyond_100um=dend_far, axon_prox_frac=axon_prox,
    )
    return f
