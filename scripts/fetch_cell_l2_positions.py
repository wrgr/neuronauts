"""Coordinates for every level-2 node of every cached cell graph that the
population's objgeom does not already position, so box connectivity can be
decided for ALL cells -- not only the 40 whose connective nodes were fetched
earlier. Extends the same attribute cache the re-cut reads."""
import glob, sys
import numpy as np
sys.path.insert(0, "/Users/wgray13/projects/neuronauts")
from neuronauts.harness.geometry import fetch_l2_attributes
R = "/Users/wgray13/projects/neuronauts/"
S = "/private/tmp/claude-501/-Users-wgray13-projects-neuronauts/8c2bcfd4-b48d-453f-ae78-fb9ed1b00ae7/scratchpad/"
ol2 = np.sort(np.load(R + "data/substrate/geom/objgeom_kall.npz", allow_pickle=False)["l2_id"])
unk = []
for f in sorted(glob.glob(R + "data/external/cell_l2_graphs/*.npz")):
    E = np.load(f, allow_pickle=False)["edges"]
    if not len(E): continue
    nodes = np.unique(E); j = np.clip(np.searchsorted(ol2, nodes), 0, len(ol2) - 1)
    unk.append(nodes[ol2[j] != nodes])
unk = np.unique(np.concatenate(unk))
print(f"{len(unk):,} level-2 nodes across all cached cell graphs are not population nodes", flush=True)
attrs = fetch_l2_attributes(unk, S + "connective_l2_attrs.npz", verbose=True)
have = np.isin(unk, attrs["l2_id"]); print(f"positioned after fetch: {int(have.sum()):,}/{len(unk):,} ({have.mean():.1%})", flush=True)
import shutil; shutil.copy(S + "connective_l2_attrs.npz", R + "data/external/soma_viz/connective_l2_attrs.npz")
print("DONE_L2POS", flush=True)
