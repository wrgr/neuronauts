"""Tree-level features: what a candidate looks like from the whole arbor.

A contact panel says how two objects meet. None of these four say anything
without the seed's entire cloud and its soma, which is the difference the
grammar is being asked to exploit:

  oproj     signed projection of the candidate's centroid on the direction the
            seed's cable was heading when it ran out (a panel's ``along`` is
            unsigned, so a candidate lying back along the parent scores as high
            as one lying ahead of it)
  occupy    fraction of the candidate that runs within 1.5 um of the seed's
            arbor away from the contact -- a process running alongside the
            parent is not its continuation
  extent    the candidate's own span, nm
  dsoma     how much further from the soma the candidate sits than the contact

The same four are measured on the 103 proofread skeletons at interior points,
where the cable beyond the point is a continuation by construction. That is the
grammar's CONTINUE production; the background is the panel candidate field.
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

R = Path(__file__).resolve().parents[1]
WIN = 1500.0
NEAR = 1500.0          # "runs alongside the parent" threshold
AWAY = 5000.0          # ignore the contact neighbourhood, where every
                       # continuation is legitimately close to its parent


def axis_of(P):
    if len(P) < 3:
        return None
    return np.linalg.svd(P - P.mean(0), full_matrices=False)[2][0]


def outward_axis(S, ctr):
    loc = S[np.linalg.norm(S - ctr, axis=1) < WIN]
    a = axis_of(loc)
    if a is None:
        return None
    v = ctr - loc.mean(0)
    return a if float(a @ v) >= 0 else -a


def features(C, ctr, h, S_tree, soma, S_far_mask_pts=None):
    """The four tree features for one candidate point set."""
    if len(C) == 0:
        return None
    e = C.mean(0) - ctr
    ne = float(np.linalg.norm(e))
    oproj = float(e @ h / ne) if ne > 0 else 0.0
    far = C[np.linalg.norm(C - ctr, axis=1) > AWAY]
    if len(far):
        d, _ = S_tree.query(far, k=1)
        occupy = float(np.mean(d < NEAR))
    else:
        occupy = np.nan
    if len(C) >= 3:
        a = axis_of(C)
        t = C @ a
        extent = float(t.max() - t.min())
    else:
        extent = float(np.linalg.norm(C.max(0) - C.min(0))) if len(C) > 1 else 0.0
    dsoma = float(np.median(np.linalg.norm(C - soma, axis=1)) -
                  np.linalg.norm(ctr - soma))
    return dict(oproj=oproj, occupy=occupy, extent=extent, dsoma=dsoma)


# --------------------------------------------------------------------------
def panel_rows():
    import exp079_contacts as EC
    con = json.load(open(R / "data/external/exp079_contacts.json"))
    out = []
    for f in sorted(glob.glob(str(R / "data/external/panels/*.npz"))):
        key = Path(f).stem.split("_")[1]
        if key not in con:
            continue
        c = con[key]
        d = np.load(f)
        ctr = np.asarray(c["ctr"], float)
        soma = np.asarray(c["soma"], float)
        S = EC.cloud(c["seed"])
        h = outward_axis(S, ctr)
        if h is None:
            print("  no seed axis", key); continue
        S_tree = cKDTree(S)
        objs = d["obj"].astype(np.uint64).tolist()
        rec = []
        for i, a in enumerate(objs):
            C = EC.cloud(a)
            fe = features(C, ctr, h, S_tree, soma) if len(C) else None
            rec.append(dict(obj=int(a), covered=bool(len(C)),
                            in_target=bool(d["in_target"][i]),
                            gap=float(d["gap_nm"][i]), along=float(d["along"][i]),
                            collin=float(d["collin"][i]), n_vox=int(d["n_vox"][i]),
                            cal_cand=float(d["cal_cand"][i]), **(fe or {})))
        out.append(dict(key=key, cell=c["cell"], rule=c["rule"],
                        soma_nm=c["soma_nm"], cal_seed=float(d["cal_seed"]),
                        end_ratio=float(d["end_ratio"]),
                        already_whole=bool(d["already_whole"]), cands=rec))
        print(f"  {key}: {len(rec)} candidates, {sum(r['covered'] for r in rec)} with clouds", flush=True)
    return out


def skeleton_rows():
    """The CONTINUE production: the same four features where the continuation
    is known, on real arbors."""
    import sys
    sys.path.insert(0, str(R / "scripts"))
    from exp079_morphology_grammar import load_skeleton, walk_proximal_verts

    def densify(V, idx, step=250.0):
        """Skeleton vertices sit a median 1,786 nm apart -- too sparse for the
        panel's 1,500 nm local window, and too sparse to stand in for a mip-5
        cloud. Positions are linearly interpolated along each edge."""
        pts = [V[idx[0]]]
        for a, b in zip(idx[:-1], idx[1:]):
            seg = float(np.linalg.norm(V[b] - V[a]))
            k = max(int(seg // step), 1)
            for j in range(1, k + 1):
                f = min(j * step, seg) / max(seg, 1e-9)
                pts.append(V[a] * (1 - f) + V[b] * f)
        return np.asarray(pts)

    rows = []
    for f in sorted(glob.glob(str(R / "data/external/cell_skeletons/*_skv4.npz"))):
        cell = int(Path(f).name.split("_")[0])
        sk = load_skeleton(f)
        V, n = sk["V"], len(sk["V"])
        soma = V[sk["soma"]]
        children = [[] for _ in range(n)]
        for v in range(n):
            p = int(sk["parent"][v])
            if p >= 0:
                children[p].append(v)
        rng = np.random.default_rng(2)
        cand = [i for i in range(n) if sk["deg"][i] == 2 and sk["seen"][i]]
        if len(cand) > 120:
            cand = list(rng.choice(cand, 120, replace=False))
        for x in cand:
            w = walk_proximal_verts(sk, x, 2 * WIN)
            prox = np.array([a for a, _ in w])
            if len(prox) < 2:
                continue
            ctr = V[x]
            Pprox = densify(V, prox)
            h = outward_axis(Pprox, ctr)
            if h is None:
                continue
            # everything on the soma side of x is the arbor already occupied
            sub, stack = [], [x]
            while stack:                       # the distal subtree of x
                u = stack.pop()
                sub.append(u)
                stack.extend(children[u])
            distal = np.array(sub)
            occ_mask = np.ones(n, bool)
            occ_mask[distal] = False
            if occ_mask.sum() < 5:
                continue
            occ_idx = np.flatnonzero(occ_mask)
            S_tree = cKDTree(V[occ_idx])
            for L in (5000.0, 15000.0, 40000.0):    # a fragment's worth of cable
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
                fe = features(densify(V, np.r_[x, np.array(seq)]), ctr, h, S_tree, soma)
                if fe is None:
                    continue
                fe.update(cell=cell, L=L, comp=int(sk["comp"][x]))
                rows.append(fe)
        print(f"  {cell}: {len(rows)} rows", flush=True)
    return rows


if __name__ == "__main__":
    import sys
    which = sys.argv[1]
    if which == "panels":
        out = panel_rows()
        json.dump(out, open(R / "data/external/exp079_panel_tree.json", "w"))
        print(f"wrote {len(out)} panels")
    else:
        out = skeleton_rows()
        json.dump(out, open(R / "data/external/exp079_skel_tree.json", "w"))
        print(f"wrote {len(out)} rows")
