"""Turn the per-cell environment cards into the distributions Phase B reads,
plus a markdown table. Runs on whatever cards exist and says how many."""
import glob, json
from collections import Counter
from pathlib import Path
import numpy as np
X = Path("/Users/wgray13/projects/neuronauts/data/external/cell_cards")
cards = [json.load(open(f)) for f in sorted(X.glob("*.json"))]
ok = [c for c in cards if c["coverage"].get("graph")]
print(f"cards: {len(cards)}, with graphs: {len(ok)}, with edit history: {sum(c['coverage'].get('edit_history', False) for c in ok)}, "
      f"with skeleton: {sum(c['coverage'].get('skeleton', False) for c in ok)}")
def q(a, p): a = np.asarray(a, float); return float(np.percentile(a, p)) if len(a) else float("nan")
rows = []
for c in ok:
    s, st = c["seed"], c["structure"]; links = c["split_challenges"]; eh = c.get("edit_history", {})
    gaps = [l["gap_nm"] for l in links]
    comp = Counter(f"{l['compartment_a']}-{l['compartment_b']}" for l in links)
    rows.append({"cell": c["cell"], "type": c.get("cell_type", {}).get("final", s.get("cell_type", "?")),
                 "type_source": c.get("cell_type", {}).get("source", "model"), "system": s.get("cell_type_system", "?"),
                 "fragments": st["labelled_fragments"], "components": len(st["components"]), "largest": max(st["components"]) if st["components"] else 0,
                 "seeded": len(st["seeded_target"]), "already_whole": st["already_whole"],
                 "soma_frag_l2": st["soma_fragment_l2_nodes"], "l2_in_cube": st["l2_nodes_in_cube"], "l2_total": st["l2_nodes_total"],
                 "n_links": len(links), "gap_med_nm": q(gaps, 50), "gap_max_nm": max(gaps) if gaps else None,
                 "links_dd": comp.get("dendrite-dendrite", 0), "links_aa": comp.get("axon-axon", 0),
                 "links_mixed_comp": sum(v for k, v in comp.items() if k not in ("dendrite-dendrite", "axon-axon")),
                 "n_merges_mixed_atoms": len(c["merge_challenges"]), "other_roots_max_sides": max([o["sides"] for m in c["merge_challenges"] for o in m["other_roots"]] or [0]),
                 "edits": eh.get("n_ops"), "edit_merges": eh.get("n_merges"), "edit_splits": eh.get("n_splits"), "edits_in_cube": eh.get("n_edit_points_in_cube"),
                 "cb2": c["cb2_decisions"]["n"]})
# --- a first cut at challenge types, stated as rules so they can be argued with ---
def kind(r):
    if r["already_whole"]: return "already whole"
    if r["seeded"] == 0: return "no in-box target"
    if r["n_merges_mixed_atoms"] >= 3: return "frankenmerge-heavy"
    if r["links_aa"] + r["links_mixed_comp"] == 0 and r["links_dd"] > 0: return "dendrite-only splits"
    if r["links_aa"] > 0: return "axon splits present"
    return "other"
for r in rows: r["kind"] = kind(r)
print("\nchallenge types:", dict(Counter(r["kind"] for r in rows)))
print("cell types     :", dict(Counter(r["type"] for r in rows)))
print("type provenance:", dict(Counter(r["type_source"] for r in rows)))
print(f"\nper cell: fragments med {q([r['fragments'] for r in rows],50):.0f} (p90 {q([r['fragments'] for r in rows],90):.0f}); "
      f"components med {q([r['components'] for r in rows],50):.0f}; seeded target med {q([r['seeded'] for r in rows],50):.0f}; "
      f"already whole {sum(r['already_whole'] for r in rows)}")
gaps_all = [l["gap_nm"] for c in ok for l in c["split_challenges"]]
print(f"split links: {len(gaps_all)} total; gap med {q(gaps_all,50):.0f} nm, p90 {q(gaps_all,90):.0f}, max {max(gaps_all) if gaps_all else 0:.0f}")
cc = Counter(f"{l['compartment_a']}-{l['compartment_b']}" for c in ok for l in c["split_challenges"]); print("link compartments:", dict(cc))
ed = [r["edits"] for r in rows if r["edits"] is not None]
if ed: print(f"edits per cell (n={len(ed)}): med {q(ed,50):.0f}, p90 {q(ed,90):.0f}, max {max(ed)}; in-cube edit points med {q([r['edits_in_cube'] for r in rows if r['edits_in_cube'] is not None],50):.0f}")
json.dump({"n_cards": len(cards), "rows": rows}, open(X / "_aggregate.json", "w"), indent=1)
cols = ["cell", "type", "kind", "fragments", "components", "seeded", "n_links", "gap_med_nm", "links_dd", "links_aa", "n_merges_mixed_atoms", "edits", "edits_in_cube"]
with open(X / "_table.md", "w") as f:
    f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
    for r in sorted(rows, key=lambda r: (-r["n_merges_mixed_atoms"], -r["n_links"])):
        f.write("| " + " | ".join("" if r[c] is None else (f"{r[c]:.0f}" if isinstance(r[c], float) else str(r[c])[-8:] if c == "cell" else str(r[c])) for c in cols) + " |\n")
print("wrote _aggregate.json and _table.md")
