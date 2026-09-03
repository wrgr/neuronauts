"""EXP-079 -- a morphology grammar over real proofread arbors, applied to the
corrected contact panels.

Everything the repository's grammar work scored before was a skeleton cut in
software and re-joined; both halves of such a cut share caliber and tangent by
construction. This scores the real task instead: 99 corrected contact panels
(66 needing a join, 33 genuine arbor terminals), against the pairwise geometric
baseline that EXP-076 measured at median rank 5 of 2,440 and top-1 on 22 of 66.

The grammar's productions are estimated from the 103 proofread skeletons in
``data/external/cell_skeletons`` -- real morphology, never a synthetic cut --
and every panel is scored by a grammar fitted with that panel's own cell held
out.
"""
from __future__ import annotations

import glob
import json
from collections import deque
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
SKEL = R / "data/external/cell_skeletons"
PANELS = R / "data/external/panels"
CARDS = R / "data/external/cell_cards"

WIN = 1500.0        # the panel's local window, nm
TAPER_TIP = 300.0   # panel end_ratio: caliber over the last 300 nm
TAPER_BACK = (1000.0, 1300.0)


# --------------------------------------------------------------------------
# skeleton -> rooted tree with geodesic distance
# --------------------------------------------------------------------------
def load_skeleton(path):
    d = np.load(path)
    V = d["vertices"].astype(float)
    E = d["edges"].astype(int)
    rad = d["radius"].astype(float)
    comp = d["compartment"].astype(int)
    n = len(V)
    adj = [[] for _ in range(n)]
    for a, b in E.tolist():
        adj[a].append(b)
        adj[b].append(a)
    soma = int(np.flatnonzero(comp == 1)[0]) if (comp == 1).any() else int(np.argmax(rad))
    # BFS from soma over the (tree) graph
    parent = np.full(n, -1, np.int64)
    dist = np.full(n, np.inf)
    dist[soma] = 0.0
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
                dist[v] = dist[u] + float(np.linalg.norm(V[v] - V[u]))
                order.append(v)
                q.append(v)
    deg = np.array([len(a) for a in adj])
    return dict(V=V, rad=rad, comp=comp, adj=adj, soma=soma, parent=parent,
                dist=dist, order=np.array(order), deg=deg, seen=seen)


def walk_proximal(sk, v, max_nm, step=50.0):
    """Radius sampled every ``step`` nm along the cable from ``v`` toward the soma.

    Skeleton vertices sit a median 1,786 nm apart, so the panel's 300 nm taper
    window falls between them; radius is read by linear interpolation along the
    edge, which is how a per-vertex radius is defined between vertices. The
    consequence is stated in the evaluation: the grammar's taper is a smoothed
    version of the panel's, not the same measurement.
    """
    V, rad, parent = sk["V"], sk["rad"], sk["parent"]
    ss, rr = [0.0], [float(rad[v])]
    cur, acc = v, 0.0
    while acc < max_nm:
        p = int(parent[cur])
        if p < 0:
            break
        seg = float(np.linalg.norm(V[cur] - V[p]))
        if seg <= 0:
            cur = p
            continue
        k = max(int(seg // step), 1)
        for j in range(1, k + 1):
            f = min(j * step, seg) / seg
            ss.append(acc + f * seg)
            rr.append(float(rad[cur] * (1 - f) + rad[p] * f))
        acc += seg
        cur = p
    return np.array(ss), np.array(rr), cur


def walk_proximal_verts(sk, v, max_nm):
    """Vertices on the path from ``v`` toward the soma, with path length from v."""
    out = [(v, 0.0)]
    cur, acc = v, 0.0
    while acc < max_nm:
        p = int(sk["parent"][cur])
        if p < 0:
            break
        acc += float(np.linalg.norm(sk["V"][cur] - sk["V"][p]))
        out.append((p, acc))
        cur = p
    return out


def taper_ratio(sk, v, tip_nm=TAPER_TIP, back=TAPER_BACK):
    """The panel's ``end_ratio`` computed on a skeleton: caliber over the last
    ``tip_nm`` of cable before ``v`` divided by caliber ``back`` nm back."""
    s, r, _ = walk_proximal(sk, v, back[1] + 200.0)
    near = r[s <= tip_nm]
    far = r[(s >= back[0]) & (s <= back[1])]
    if not len(near) or not len(far) or np.mean(far) <= 0:
        return np.nan
    return float(np.mean(near) / np.mean(far))


def local_axis(P):
    if len(P) < 3:
        return None
    return np.linalg.svd(P - P.mean(0), full_matrices=False)[2][0]


def collect_ends(path):
    """Every cable end of one arbor, labelled TIP (a real terminal) or CUT (an
    interior point where a cut would leave a face), with the features the panel
    measures at the seed's end."""
    sk = load_skeleton(path)
    V, rad, deg = sk["V"], sk["rad"], sk["deg"]
    n = len(V)
    tips = [i for i in range(n) if deg[i] == 1 and i != sk["soma"] and sk["seen"][i]]
    interior = [i for i in range(n) if deg[i] == 2 and sk["seen"][i]]
    # a cut null must not sit on the last stretch of cable before a real tip
    if tips:
        gd = np.full(n, np.inf)
        q = deque()
        for t in tips:
            gd[t] = 0.0
            q.append(t)
        while q:                                   # geodesic distance to nearest tip
            u = q.popleft()
            for v in sk["adj"][u]:
                d = gd[u] + float(np.linalg.norm(V[v] - V[u]))
                if d < gd[v] - 1e-6:
                    gd[v] = d
                    q.append(v)
    else:
        gd = np.full(n, np.inf)
    rng = np.random.default_rng(0)
    interior = [i for i in interior if gd[i] > 3000.0]
    if len(interior) > 400:
        interior = list(rng.choice(interior, 400, replace=False))

    rows = []
    for kind, ids in (("tip", tips), ("cut", interior)):
        for v in ids:
            tr = taper_ratio(sk, v)
            if not np.isfinite(tr):
                continue
            se, re_, _ = walk_proximal(sk, v, 4000.0)
            rows.append(dict(kind=kind, taper=tr,
                             taper_wide=taper_ratio(sk, v, 1000.0, (2000.0, 4000.0)),
                             rad_end=float(np.mean(re_[se <= WIN])),
                             rad_box=float(np.mean(re_)),
                             comp=int(sk["comp"][v]),
                             soma_nm=float(np.linalg.norm(V[v] - V[sk["soma"]])),
                             path_nm=float(sk["dist"][v])))
    return rows


def collect_continuations(path):
    """What a cable looks like a fragment's worth further along itself.

    At an interior point x this measures exactly the quantities a panel holds
    for a candidate: the caliber ratio between the cable beyond x and the cable
    at x, the |cos| between the two local axes (``collin``), and the |cos|
    between x's axis and the direction to the far cable's centroid (``along``).
    """
    sk = load_skeleton(path)
    V, rad = sk["V"], sk["rad"]
    n = len(V)
    rng = np.random.default_rng(1)
    cand = [i for i in range(n) if sk["deg"][i] == 2 and sk["seen"][i]]
    if len(cand) > 300:
        cand = list(rng.choice(cand, 300, replace=False))
    rows = []
    # children lookup for walking distally
    children = [[] for _ in range(n)]
    for v in range(n):
        p = int(sk["parent"][v])
        if p >= 0:
            children[p].append(v)
    for x in cand:
        # proximal side = the "seed"
        w = walk_proximal_verts(sk, x, WIN)
        prox = np.array([a for a, _ in w])
        if len(prox) < 3:
            continue
        aS = local_axis(V[prox])
        if aS is None:
            continue
        # distal side = the "candidate", a run of cable of a fragment's length
        for L in (2000.0, 5000.0, 10000.0):
            seq, acc, cur = [], 0.0, x
            while acc < L:
                ch = children[cur]
                if not ch:
                    break
                nxt = ch[0] if len(ch) == 1 else ch[int(rng.integers(len(ch)))]
                acc += float(np.linalg.norm(V[nxt] - V[cur]))
                seq.append(nxt)
                cur = nxt
            if len(seq) < 3 or acc < 0.6 * L:
                continue
            seq = np.array(seq)
            far = seq[np.array([float(np.linalg.norm(V[s] - V[x])) for s in seq]) <= WIN]
            aC = local_axis(V[far]) if len(far) >= 3 else local_axis(V[seq])
            if aC is None:
                continue
            u = V[seq].mean(0) - V[prox].mean(0)
            nu = float(np.linalg.norm(u))
            rows.append(dict(L=L,
                             ratio=float(np.mean(rad[seq]) / max(np.mean(rad[prox]), 1e-9)),
                             collin=abs(float(aS @ aC)),
                             along=abs(float(aS @ (u / nu))) if nu > 0 else 0.0,
                             rad_par=float(np.mean(rad[prox])),
                             comp=int(sk["comp"][x])))
    return rows


if __name__ == "__main__":
    import sys
    fs = sorted(glob.glob(str(SKEL / "*_skv4.npz")))
    which = sys.argv[1] if len(sys.argv) > 1 else "ends"
    out = []
    for i, f in enumerate(fs):
        cell = int(Path(f).name.split("_")[0])
        rows = collect_ends(f) if which == "ends" else collect_continuations(f)
        for r in rows:
            r["cell"] = cell
        out.extend(rows)
        if i % 20 == 0:
            print(f"  {i}/{len(fs)} {len(out)} rows", flush=True)
    dest = R / f"data/external/exp079_{which}.json"
    json.dump(out, open(dest, "w"))
    print(f"wrote {len(out)} rows -> {dest}")
