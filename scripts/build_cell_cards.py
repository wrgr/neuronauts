"""One environment card per soma-seeded cell: what it looks like in the box,
where its split and merge challenges are, and what proofreading did to it.

Reads only cached data (cell graphs, skeletons, edit history, object clouds,
labels, population, CB2). Reports coverage per input so a card built from
partial caches says so instead of looking complete. Writes
data/external/cell_cards/<root>.json and <root>.png.
"""
import glob, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "/Users/wgray13/projects/neuronauts")
from neuronauts.harness.box_truth import box_components, seeded_target
from neuronauts.harness.labels import TIER_NONE, load_labels
from neuronauts.harness.population import load_population
R = Path("/Users/wgray13/projects/neuronauts"); X = R / "data/external"; OUT = X / "cell_cards"; OUT.mkdir(exist_ok=True)
LO = np.array([613000., 541000., 810000.]); HI = np.array([713000., 641000., 910000.])
COMP = {1: "soma", 2: "axon", 3: "dendrite", 4: "apical"}

# --- shared tables -------------------------------------------------------------
seeds = {int(s["root_v1822"]): s for s in json.load(open(X / "soma_viz/seed_census.json"))["seeds"] if s["evaluable"]}
g = np.load(R / "data/substrate/geom/objgeom_kall.npz", allow_pickle=False)
ol2, opos, oa, optr = g["l2_id"], g["pos_nm"], g["atom_id"], g["node_ptr"]
ores = g["resolved"]        # a few L2 nodes carry no coordinate; objgeom.points()
                            # drops them and so must this -- feeding NaN to a
                            # cKDTree raises, which is how the first run died at card 23
o = np.argsort(ol2); ol2s, oposs = ol2[o], opos[o]; l2atom = np.repeat(oa, np.diff(optr))[o]
row_of = {int(a): k for k, a in enumerate(oa.tolist())}
def atom_pts(a):
    k = row_of.get(int(a))
    if k is None: return np.empty((0, 3))
    sl = slice(int(optr[k]), int(optr[k + 1]))
    return opos[sl][ores[sl]]
att = np.load(X / "soma_viz/connective_l2_attrs.npz", allow_pickle=False); c_ = np.argsort(att["l2_id"]); cl2s, cposs = att["l2_id"][c_], att["pos_nm"][c_]
lab = load_labels(R / "data/substrate/c100um/labels_v1822.npz")
t = np.load(R / "data/substrate/topology/kall.npz", allow_pickle=False); atoms = t["atom_id"]
i = lab.index_of(atoms); has = i >= 0
own = np.zeros(len(atoms), np.int64); pure = np.zeros(len(atoms), bool); tier = np.full(len(atoms), TIER_NONE, np.int8); nroots = np.zeros(len(atoms), np.int32)
own[has] = lab.owner[i[has]].astype(np.int64); pure[has] = lab.pure[i[has]]; tier[has] = lab.owner_tier[i[has]]; nroots[has] = lab.n_roots[i[has]]
keep = pure & (tier > TIER_NONE) & (own > 0); ids, ow = atoms[keep], own[keep]
# per-atom v1822 root sets from synapse sides, for merge challenges (which OTHER cell shares an atom)
pop = load_population(R / "data/substrate/c100um/population.npz")
sv = np.load(R / "data/substrate/c100um/sv_v1822.npz", allow_pickle=False); svo = np.argsort(sv["sv"]); svs, svr = sv["sv"][svo], sv["root"][svo]
def roots_of_sv(x):
    j = np.clip(np.searchsorted(svs, x), 0, len(svs) - 1); return np.where(svs[j] == x, svr[j], np.uint64(0))
# per-side supervoxels live in the region synapse file, aligned to the population by synapse id
reg = np.load(R / "data/regions/dense_v1_synapses.npz", allow_pickle=False)
reg_id = reg["synapse_id"]          # 100% of population syn_id values are found here (checked)
ro = np.argsort(reg_id); rid_s = reg_id[ro]; jj = np.clip(np.searchsorted(rid_s, pop.syn_id), 0, len(rid_s) - 1)
hit = rid_s[jj] == pop.syn_id
assert hit.mean() > 0.99, f"population synapse ids not found in the region file: {hit.mean():.1%}"
pre_sv = np.where(hit, reg["pre_sv"][ro][jj], np.uint64(0)); post_sv = np.where(hit, reg["post_sv"][ro][jj], np.uint64(0))
side_atom = np.concatenate([pop.syn_atom_pre, pop.syn_atom_post]); side_root = roots_of_sv(np.concatenate([pre_sv, post_sv]))
# CB2 located decisions
cb2 = json.load(open(X / "cb2/incube_edits.json")); res = {r["root_id"]: int(r["v117_root"]) for r in json.load(open(X / "cb2/final_resolution.json"))}

def positions_for(nodes):
    P = np.full((len(nodes), 3), np.nan); j = np.clip(np.searchsorted(ol2s, nodes), 0, len(ol2s) - 1); h = ol2s[j] == nodes; P[h] = oposs[j[h]]
    m = ~h
    if m.any():
        k = np.clip(np.searchsorted(cl2s, nodes[m]), 0, len(cl2s) - 1); h2 = cl2s[k] == nodes[m]; P[np.flatnonzero(m)[h2]] = cposs[k[h2]]
    return P, h, np.where(h, l2atom[j], np.uint64(0))

def build(cell):
    s = seeds[cell]; card = {"cell": cell, "seed": s, "coverage": {}}
    # cell type is a prediction unless a human typed it; carry which, so a card
    # never launders a model call into a fact (92.5% agreement, all disagreements
    # 23P vs 4P -- see scripts/add_human_cell_types.py)
    card["cell_type"] = {"final": s.get("cell_type_final", s.get("cell_type", "unknown")),
                         "source": s.get("cell_type_source", "model"),
                         "human": s.get("cell_type_human"), "model": s.get("cell_type_fine")}
    gf = X / f"cell_l2_graphs/{cell}.npz"
    if not gf.exists(): card["coverage"]["graph"] = False; return card
    E = np.load(gf, allow_pickle=False)["edges"]; nodes = np.unique(E); pos = {int(v): k for k, v in enumerate(nodes.tolist())}
    P, known, natom = positions_for(nodes); positioned = float(np.isfinite(P).all(axis=1).mean())
    card["coverage"]["graph"] = True; card["coverage"]["nodes_positioned"] = round(positioned, 4)
    # set here, not inside the links block: a cell with nothing to join still HAS a
    # skeleton, and reporting 67/103 coverage when all 103 were fetched is misleading
    card["coverage"]["skeleton"] = (X / f"cell_skeletons/{cell}_skv4.npz").exists()
    members = set(ids[ow == cell].tolist())
    frag = np.array([int(x) if int(x) in members else 0 for x in natom.tolist()], np.int64)
    ei = np.array([[pos[int(a)], pos[int(b)]] for a, b in E.tolist()])
    bt = box_components(ei, P, frag, LO, HI); seed_frag = int(s["v117_fragment"]); tgt = seeded_target(bt, seed_frag)
    card["structure"] = {"labelled_fragments": bt.n_fragments, "components": [len(c) for c in bt.components],
                         "seeded_target": tgt, "already_whole": bool(seed_frag in {f for c in bt.components for f in c} and not tgt),
                         "soma_fragment_l2_nodes": int(np.diff(optr)[row_of[seed_frag]]) if seed_frag in row_of else None,
                         "l2_nodes_in_cube": int(np.all((P >= LO) & (P <= HI), axis=1).sum()), "l2_nodes_total": int(len(nodes))}
    # split challenges: spanning links inside the seeded target, with gap and compartment at each end
    links = []
    if len(tgt) > 1:
        pts = {f: atom_pts(f) for f in tgt}; trees = {f: cKDTree(p) for f, p in pts.items() if len(p)}
        # every p is already finite (atom_pts applies `resolved`); assert rather
        # than trust, because a silent NaN here would corrupt every gap measured
        assert all(np.isfinite(p).all() for p in pts.values() if len(p)), "non-finite atom points"
        n = len(tgt); D = np.full((n, n), np.inf); closest = {}
        for a_ in range(n):
            for b_ in range(a_ + 1, n):
                fa, fb = tgt[a_], tgt[b_]
                if fa in trees and fb in trees and len(pts[fa]):
                    d, jj = trees[fb].query(pts[fa], k=1); k = int(np.argmin(d)); D[a_, b_] = D[b_, a_] = float(d[k]); closest[(a_, b_)] = (pts[fa][k], pts[fb][jj[k]])
        intree, rest = [0], list(range(1, n))
        skf = X / f"cell_skeletons/{cell}_skv4.npz"; sk = np.load(skf, allow_pickle=False) if skf.exists() else None
        skv = sk["vertices"] if sk is not None else None
        skok = np.isfinite(skv).all(axis=1) if skv is not None else None
        sktree = cKDTree(skv[skok]) if skv is not None and skok.any() else None
        skcomp = sk["compartment"][skok] if sk is not None and "compartment" in sk.files else None
        card["coverage"]["skeleton"] = sk is not None
        def comp_at(p):
            if sktree is None or skcomp is None or not np.isfinite(p).all(): return None
            return COMP.get(int(skcomp[sktree.query(p)[1]]), "?")
        while rest:
            _, a_, b_ = min((D[p, q], p, q) for p in intree for q in rest)
            pa, pb = closest[(min(a_, b_), max(a_, b_))]
            links.append({"a": tgt[a_], "b": tgt[b_], "gap_nm": round(float(D[a_, b_])), "at_a_nm": pa.round().astype(int).tolist(),
                          "at_b_nm": pb.round().astype(int).tolist(), "compartment_a": comp_at(pa), "compartment_b": comp_at(pb)})
            intree.append(b_); rest.remove(b_)
    card["split_challenges"] = links
    # merge challenges: mixed atoms among this cell's fragments (owner==cell but n_roots>=2), and which other root shares them
    mixed_here = [int(a) for a, o_, nr in zip(atoms, own, nroots) if o_ == cell and nr >= 2]
    merges = []
    for a in mixed_here:
        other = []
        if side_root is not None:
            rs = side_root[side_atom == a]; u, cnt = np.unique(rs[(rs > 0) & (rs != cell)], return_counts=True)
            other = [{"root": int(r_), "sides": int(c)} for r_, c in sorted(zip(u.tolist(), cnt.tolist()), key=lambda kv: -kv[1])[:3]]
        pa = atom_pts(a); merges.append({"atom": a, "l2_nodes": int(len(pa)), "centroid_nm": pa.mean(0).round().astype(int).tolist() if len(pa) else None, "other_roots": other})
    card["merge_challenges"] = merges
    # edit history
    ef = X / f"edit_history/{cell}.json"
    if ef.exists():
        eh = json.load(open(ef)); ops = eh["ops"]
        pts_in = [p for o_ in ops for p in o_.get("edit_points_nm", []) if np.all((np.array(p) >= LO) & (np.array(p) <= HI))]
        card["edit_history"] = {"n_ops": eh["n_ops"], "n_merges": eh["n_merges"], "n_splits": eh["n_splits"], "located": bool(eh.get("located")),
                                "n_edit_points_in_cube": len(pts_in), "first_ts_ms": min(o_["timestamp_ms"] for o_ in ops) if ops else None,
                                "last_ts_ms": max(o_["timestamp_ms"] for o_ in ops) if ops else None,
                                "n_users": len({o_["user_id"] for o_ in ops}), "in_cube_points": pts_in[:2000]}
        card["coverage"]["edit_history"] = True
    else: card["coverage"]["edit_history"] = False
    cb = [d for d in cb2 if any(res.get(x) in members or res.get(x) == seed_frag for x in d["before_root_ids"] + d["after_root_ids"])]
    card["cb2_decisions"] = {"n": len(cb), "split_edit": sum(d["sample_type"] == "split_edit" for d in cb), "merge_edit": sum(d["sample_type"] == "merge_edit" for d in cb)}
    return card

def thumbnail(card, P_nodes, frag_nodes, tgt):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=90)
    for k, (i_, j_, lab_) in enumerate(((0, 1, "x–y"), (0, 2, "x–z"))):
        a = ax[k]; a.set_title(f"…{str(card['cell'])[-6:]} {lab_}", fontsize=9); a.set_aspect("equal")
        m = np.isfinite(P_nodes).all(axis=1)
        a.scatter(P_nodes[m, i_] / 1e3, P_nodes[m, j_] / 1e3, s=1, c="#c8d0da", lw=0)
        for f in np.unique(frag_nodes[frag_nodes > 0]):
            mm = m & (frag_nodes == f); col = "#d9534f" if f in tgt else "#3b6fb6"
            a.scatter(P_nodes[mm, i_] / 1e3, P_nodes[mm, j_] / 1e3, s=2, c=col, lw=0)
        s = card["seed"]["pos_nm"]; a.plot(s[i_] / 1e3, s[j_] / 1e3, "o", mfc="none", mec="#f0a830", ms=9, mew=2)
        for p in card.get("edit_history", {}).get("in_cube_points", [])[:400]: a.plot(p[i_] / 1e3, p[j_] / 1e3, "+", c="#222", ms=4, mew=0.7)
        a.set_xlim(LO[i_] / 1e3, HI[i_] / 1e3); a.set_ylim(LO[j_] / 1e3, HI[j_] / 1e3); a.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(OUT / f"{card['cell']}.png"); plt.close(fig)

if __name__ == "__main__":
    only = [int(x) for x in sys.argv[1:]] or sorted(seeds)
    n_ok = 0
    for cell in only:
        card = build(cell)
        if card["coverage"].get("graph"):
            E = np.load(X / f"cell_l2_graphs/{cell}.npz", allow_pickle=False)["edges"]; nodes = np.unique(E); P, known, natom = positions_for(nodes)
            members = set(ids[ow == cell].tolist()); frag = np.array([int(x) if int(x) in members else 0 for x in natom.tolist()])
            try: thumbnail(card, P, frag, set(card["structure"]["seeded_target"]))
            except Exception as ex: card["coverage"]["thumbnail_error"] = f"{type(ex).__name__}: {ex}"
            n_ok += 1
        json.dump(card, open(OUT / f"{cell}.json", "w"), default=int)
    print(f"cards written: {len(only)} ({n_ok} with graphs) -> {OUT}", flush=True)
