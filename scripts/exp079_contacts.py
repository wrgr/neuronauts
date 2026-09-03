"""Recover each panel's contact point and the tree geometry around it.

``build_contact_panels.py`` stores no coordinates, so the contact is recomputed
here by the same rule the builder used (closest approach of the seed cloud to
the target cloud, or the builder's interior-terminal choice). The result is
checked against the cell card's own link point, which was computed by different
code.
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

R = Path(__file__).resolve().parents[1]
CUBE_C = np.array([663.0, 591.0, 860.0]) * 1000.0
LO_NM, HI_NM = CUBE_C - 50_000.0, CUBE_C + 50_000.0
HALF_NM = 4000.0

_z = np.load(R / "data/substrate/c100um/object_clouds_mip5.npz", allow_pickle=False)
_obj, _ptr, _pos = _z["object_id"], _z["node_ptr"], _z["pos_nm"]
_rowi = {int(a): k for k, a in enumerate(_obj.tolist())}


def cloud(a):
    k = _rowi.get(int(a))
    return _pos[int(_ptr[k]):int(_ptr[k + 1])] if k is not None else np.empty((0, 3))


def contact_of(card):
    """The builder's box centre for this cell, plus which rule produced it."""
    seed = int(card["seed"]["v117_fragment"])
    tgt = set(card["structure"]["seeded_target"]) - {seed}
    Ps = cloud(seed)
    if not len(Ps):
        return None
    Pt = [cloud(t) for t in tgt if len(cloud(t))]
    if Pt:
        Pt = np.vstack(Pt)
        d, j = cKDTree(Pt).query(Ps, k=1)
        m = int(np.argmin(d))
        return dict(ctr=(Ps[m] + Pt[int(j[m])]) / 2.0, rule="cut", seed=seed)
    soma = np.asarray(card["seed"]["pos_nm"], float)
    inside = np.all((Ps - LO_NM > 2 * HALF_NM) & (HI_NM - Ps > 2 * HALF_NM), axis=1)
    if not inside.any():
        return None
    cand_pts = Ps[inside]
    outward = cand_pts - soma
    dirs = outward / np.maximum(np.linalg.norm(outward, axis=1, keepdims=True), 1.0)
    tree_seed = cKDTree(Ps)
    tip_ok = np.zeros(len(cand_pts), bool)
    for i in range(len(cand_pts)):
        nb = Ps[tree_seed.query_ball_point(cand_pts[i], r=3000.0)]
        if len(nb) < 3:
            continue
        tip_ok[i] = not np.any((nb - cand_pts[i]) @ dirs[i] > 500.0)
    if not tip_ok.any():
        return None
    ends = cand_pts[tip_ok]
    ctr = ends[int(np.argmax(np.linalg.norm(ends - soma, axis=1)))]
    return dict(ctr=ctr, rule="terminal", seed=seed)


if __name__ == "__main__":
    cards = {}
    for f in sorted(glob.glob(str(R / "data/external/cell_cards/*.json"))):
        if Path(f).name.startswith("_"):
            continue
        c = json.load(open(f))
        cards[str(c["cell"])[-8:]] = c
    out, checks = {}, []
    for f in sorted(glob.glob(str(R / "data/external/panels/*.npz"))):
        key = Path(f).stem.split("_")[1]
        c = cards.get(key)
        if c is None:
            print("no card", key); continue
        r = contact_of(c)
        if r is None:
            print("no contact", key); continue
        soma = np.asarray(c["seed"]["pos_nm"], float)
        out[key] = dict(ctr=r["ctr"].tolist(), rule=r["rule"], seed=r["seed"],
                        cell=int(c["cell"]), soma=soma.tolist(),
                        soma_nm=float(np.linalg.norm(r["ctr"] - soma)))
        # independent check: the card's own link point, computed by other code
        sc = c.get("split_challenges") or []
        if r["rule"] == "cut" and sc:
            pts = [np.asarray(s["at_a_nm"], float) for s in sc] + \
                  [np.asarray(s["at_b_nm"], float) for s in sc]
            checks.append(float(min(np.linalg.norm(p - r["ctr"]) for p in pts)))
    json.dump(out, open(R / "data/external/exp079_contacts.json", "w"))
    checks = np.array(checks)
    print(f"{len(out)} contacts; cut-panel centres vs the card's own link points: "
          f"median {np.median(checks):.0f} nm, p90 {np.percentile(checks,90):.0f} nm, "
          f"n={len(checks)}, within 2um {int((checks<2000).sum())}/{len(checks)}")
