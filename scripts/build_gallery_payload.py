"""Payload for the environment gallery: every cell's stats, a downscaled
thumbnail, and interactive skeletons for the representatives.

Skeletons rather than meshes for the 3D: they are already cached for all 103
cells, and a whole-cell skeleton is a few thousand vertices where a mesh is
1.37M faces. Each vertex is assigned to the v117 fragment whose object cloud is
nearest, so the shattering is visible without fetching anything.
"""
import base64, io, json, math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
R = Path("/Users/wgray13/projects/neuronauts"); X = R / "data/external"
OUT = "/private/tmp/claude-501/-Users-wgray13-projects-neuronauts/8c2bcfd4-b48d-453f-ae78-fb9ed1b00ae7/scratchpad/gallery.json"
REPS = [864691135788476, 0]  # replaced below from the aggregate
LO = np.array([613000., 541000., 810000.]); HI = np.array([713000., 641000., 910000.])
A = json.load(open(X / "cell_cards/_aggregate.json"))["rows"]
cards = {r["cell"]: json.load(open(X / f"cell_cards/{r['cell']}.json")) for r in A}
g = np.load(R / "data/substrate/geom/objgeom_kall.npz", allow_pickle=False)
opos, oa, optr, ores = g["pos_nm"], g["atom_id"], g["node_ptr"], g["resolved"]
row_of = {int(a): k for k, a in enumerate(oa.tolist())}
def pts(a):
    k = row_of.get(int(a))
    if k is None: return np.empty((0, 3))
    sl = slice(int(optr[k]), int(optr[k + 1])); return opos[sl][ores[sl]]

def thumb(cell, w=340):
    p = X / f"cell_cards/{cell}.png"
    if not p.exists(): return None
    im = Image.open(p).convert("L")
    im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, format="PNG", optimize=True)
    return base64.b64encode(b.getvalue()).decode()

# representatives, chosen in the Phase B read
reps = {}
def pick(pred):
    g_ = sorted([r for r in A if pred(r)], key=lambda r: -(r["n_links"] or 0))
    return g_[0]["cell"] if g_ else None
for label, pred in (("already whole", lambda r: r["already_whole"] and r["l2_in_cube"] > 2000),
                    ("dendrite-only", lambda r: r["kind"] == "dendrite-only splits"),
                    ("axon splits + merges", lambda r: r["links_aa"] >= 3 and r["n_merges_mixed_atoms"] >= 3),
                    ("compartment-crossing", lambda r: r["n_dropped_crossing"] >= 2),
                    ("dense plexus", lambda r: r["type"] == "NGC")):
    c = pick(pred)
    if c: reps[label] = c
print("representatives:", reps, flush=True)

skels = {}
for label, cell in reps.items():
    f = X / f"cell_skeletons/{cell}_skv4.npz"
    if not f.exists(): continue
    z = np.load(f, allow_pickle=False)
    V, E = z["vertices"].astype(float), z["edges"].astype(int)
    ok = np.isfinite(V).all(axis=1)
    card = cards[cell]; tgt = set(card["structure"]["seeded_target"])
    # always include the soma's own fragment: an "already whole" cell has an empty
    # target and no shared merged objects, so without it nothing would be coloured
    soma_frag = int(card["seed"]["v117_fragment"])
    frags = sorted(tgt | {int(m["atom"]) for m in card["merge_challenges"]} | {soma_frag})
    clouds = {f_: pts(f_) for f_ in frags}
    clouds = {f_: p for f_, p in clouds.items() if len(p)}
    lab = np.full(len(V), -1, np.int64)
    if clouds:
        keys = list(clouds); tree = cKDTree(np.vstack([clouds[k] for k in keys]))
        owner = np.concatenate([[i] * len(clouds[k]) for i, k in enumerate(keys)])
        d, j = tree.query(V[ok], k=1)
        near = np.where(d <= 1500, owner[j], -1)      # only claim a vertex within 1.5 um
        lab[np.flatnonzero(ok)] = near
        keys_out = [str(k) for k in keys]
    else: keys_out = []
    inside = np.all((V >= LO) & (V <= HI), axis=1)
    skels[str(cell)] = {"v": np.round(V).astype(int).tolist(), "e": E.tolist(),
                        "label": lab.tolist(), "frag_ids": keys_out,
                        "in_target": [int(int(k) in tgt) for k in keys_out],
                        "soma_frag": str(soma_frag),
                        "inside": inside.astype(int).tolist(),
                        "soma_nm": cards[cell]["seed"]["pos_nm"]}
    print(f"  {label}: {cell} -> {len(V)} vertices, {len(keys_out)} fragments", flush=True)

rows = []
for r in A:
    c = cards[r["cell"]]
    rows.append({**{k: r[k] for k in ("cell","type","type_source","kind","fragments","components","seeded",
                                      "n_links","n_dropped_crossing","gap_med_nm","links_dd","links_aa",
                                      "n_merges_mixed_atoms","edits","edits_in_cube","l2_in_cube","already_whole")},
                 # carry the compartment WITH each gap: the histogram splits by it,
                 # and attributing gaps to compartments from per-cell counts would
                 # be a fabricated chart that looks measured
                 "links": [{"gap": l["gap_nm"],
                            "comp": "-".join(sorted([l.get("compartment_a") or "?",
                                                     l.get("compartment_b") or "?"]))}
                           for l in c["split_challenges"] if l.get("scored", True)],
                 "gaps": [l["gap_nm"] for l in c["split_challenges"] if l.get("scored", True)],
                 "thumb": thumb(r["cell"])})
def finite(x):
    """JSON.parse rejects bare NaN/Infinity; json.dump emits them happily.

    A cell with no scored links has an empty gap list, so its median is nan.
    One such token invalidates the whole payload in the browser and the page
    renders nothing, with no error until you open a console. allow_nan=False
    below turns that silent breakage into a build failure.
    """
    if isinstance(x, float) and not math.isfinite(x): return None
    if isinstance(x, dict):  return {k: finite(v) for k, v in x.items()}
    if isinstance(x, list):  return [finite(v) for v in x]
    return x

payload = finite({"rows": rows, "reps": {k: str(v) for k, v in reps.items()}, "skeletons": skels,
                  "cube": {"lo_nm": LO.tolist(), "hi_nm": HI.tolist()}})
json.dump(payload, open(OUT, "w"), separators=(",", ":"), allow_nan=False)
import os; print(f"\nwrote {OUT}  {os.path.getsize(OUT)/1e6:.1f} MB  ({len(rows)} cells, {len(skels)} skeletons)")
