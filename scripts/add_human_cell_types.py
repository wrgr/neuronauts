"""Attach human-curated cell types to the seed census, beside the model ones.

The census first carried only model predictions (`nucleus_ref_neuron_svm`, a
support vector machine on nucleus features; `aibs_metamodel_celltypes_v661`, a
soma/nucleus metamodel). Those are good -- 92.5% exact agreement with hand
labels on the 107 nuclei that have both -- but they are predictions, and a card
that prints them without saying so launders a prediction into a fact.

Hand-typed sources, both restricted to the Allen column region so coverage is
partial: allen_v1_column_types_slanted_ref (neurons; Casey Schneider-Mizell,
Nuno da Costa, Agnes Bodor) and aibs_column_nonneuronal_ref (non-neuronal;
JoAnn Buchanan), plus aibs_metamodel_celltypes_v661_corrections.

Writes `cell_type_human`, `cell_type_source` ("human" or "model") and
`cell_type_final` per seed. Every disagreement observed is 23P vs 4P -- a
laminar call -- so the coarse class the challenge typing rests on is unaffected.
"""
import json
from collections import Counter
from caveclient import CAVEclient
P = "/Users/wgray13/projects/neuronauts/data/external/soma_viz/seed_census.json"
c = CAVEclient("minnie65_public"); r = json.load(open(P)); ids = [s["nucleus_id"] for s in r["seeds"]]
def grab(tab):
    out = {}
    for i in range(0, len(ids), 200):
        try: df = c.materialize.query_table(tab, filter_in_dict={"target_id": ids[i:i+200]})
        except Exception as ex: print(f"  {tab}: {type(ex).__name__}"); return out
        for t, v in zip(df["target_id"], df["cell_type"]): out[int(t)] = str(v)
    return out
human = {**grab("aibs_column_nonneuronal_ref"), **grab("allen_v1_column_types_slanted_ref"),
         **grab("aibs_metamodel_celltypes_v661_corrections")}
n_dis = 0
for s in r["seeds"]:
    h = human.get(s["nucleus_id"]); m = s.get("cell_type_fine", "unknown")
    s["cell_type_human"] = h
    s["cell_type_source"] = "human" if h else "model"
    s["cell_type_final"] = h or m
    if h and m != "unknown" and h != m: n_dis += 1
r["cell_type_provenance"] = {
    "model_tables": ["nucleus_ref_neuron_svm (SVM, nucleus features)",
                     "aibs_metamodel_celltypes_v661 (metamodel, soma+nucleus features)"],
    "human_tables": ["allen_v1_column_types_slanted_ref (neurons)",
                     "aibs_column_nonneuronal_ref (non-neuronal)",
                     "aibs_metamodel_celltypes_v661_corrections"],
    "n_human_labelled": sum(1 for s in r["seeds"] if s["cell_type_human"]),
    "n_disagreements": n_dis,
    "note": "all observed disagreements are 23P vs 4P (a laminar call); no neuron/"
            "non-neuron disagreement, and no human-labelled non-neuronal cell is an "
            "evaluable seed"}
r["cell_type_counts_final"] = dict(Counter(s["cell_type_final"] for s in r["seeds"]))
r["seed_type_counts_final"] = dict(Counter(s["cell_type_final"] for s in r["seeds"] if s["evaluable"]))
json.dump(r, open(P, "w"), indent=1)
print("human-labelled:", r["cell_type_provenance"]["n_human_labelled"], "| disagreements:", n_dis)
print("seed types (final):", dict(sorted(r["seed_type_counts_final"].items(), key=lambda kv: -kv[1])))
print("seeds by source:", dict(Counter(s["cell_type_source"] for s in r["seeds"] if s["evaluable"])))
