#!/usr/bin/env python3
"""EXP-050 synthetic companion for blind interneuron-stratified validation.

This table is intentionally separate from the real EXP-049 results.  Ground
truth subtype is retained in a sidecar mapping and never embedded in tokens.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neuronauts.assemble import HungarianBipartiteAssembler
from neuronauts.global_merge.eval.benchmark import compute_pairwise_partition_metrics
from neuronauts.morpho_grammar.santiago_v2_grammar import (
    induce_cell_type_from_observables,
    type_segment_v2,
)
from benchmark_exp047_hungarian_bipartite import links_to_pred_map

SUBTYPE_COUNTS = {"Pyramidal": 100, "Basket": 25, "Martinotti": 15, "VIP": 10, "Glia": 15}
INTERNEURON_SUBTYPES = {"Basket", "Martinotti", "VIP"}


def build_population(seed: int = 48) -> tuple[list[dict], dict[str, str], dict[str, str], dict[str, str]]:
    """Return blind tokens plus GT maps held strictly outside inference data."""
    rng = np.random.default_rng(seed)
    tokens, gt_map, subtype_map, compartment_map = [], {}, {}, {}
    centers = rng.uniform(0, 250_000, (sum(SUBTYPE_COUNTS.values()), 3))
    index = 0
    for subtype, count in SUBTYPE_COUNTS.items():
        for local in range(count):
            cell_id = f"{subtype.lower()}-{local}"
            center = centers[index]
            index += 1
            if subtype == "Glia":
                specs = [("process", 230.0, 0, 0)]
            else:
                soma_r = {"Pyramidal": 2_500, "Basket": 800, "Martinotti": 900, "VIP": 600}[subtype]
                pre_ratio = {"Pyramidal": .10, "Basket": .85, "Martinotti": .65, "VIP": .70}[subtype]
                dend_r = {"Pyramidal": 280, "Basket": 130, "Martinotti": 150, "VIP": 90}[subtype]
                specs = [("soma", soma_r, 1, 8), ("dendrite", dend_r, 1, 9),
                         ("axon", 75.0, round(10 * pre_ratio), round(10 * (1 - pre_ratio)))]
            for piece, (compartment, radius, n_pre, n_post) in enumerate(specs):
                fid = f"{cell_id}-p{piece}"
                point = center + [piece * 650, 0, 0] + rng.normal(0, 20, 3)
                token = {"fragment_id": fid, "start_nm": point, "end_nm": point + [400, 0, 0],
                         "mean_radius_nm": radius, "max_radius_nm": radius,
                         "n_pre": n_pre, "n_post": n_post,
                         "bouton_density": n_pre / 50.0}
                if subtype == "Glia":
                    token["is_glia"] = True
                tokens.append(token)
                gt_map[fid], subtype_map[fid], compartment_map[fid] = cell_id, subtype, compartment
    forbidden = {"gt_cell_type", "cell_type", "subtype", "gt_label"}
    assert all(not forbidden.intersection(token) for token in tokens)
    return tokens, gt_map, subtype_map, compartment_map


class BlindHandshake:
    def evaluate_bidirectional_handshake(self, left: dict, right: dict) -> float:
        distance = np.linalg.norm(np.asarray(left["end_nm"]) - np.asarray(right["start_nm"]))
        return float(np.exp(-distance / 1_200) * (1.0 if type_segment_v2(left) == type_segment_v2(right) else .2))


def compute_stratified_metrics(pred: dict[str, str], gt: dict[str, str], subtype: dict[str, str],
                               compartments: dict[str, str], tokens: list[dict]) -> dict:
    output = {"overall": compute_pairwise_partition_metrics(pred, gt)}
    strata = {"Pyramidal": {"Pyramidal"}, "Interneuron": INTERNEURON_SUBTYPES, "Glia": {"Glia"}}
    for name, allowed in strata.items():
        subset = {fid: label for fid, label in gt.items() if subtype[fid] in allowed}
        output[name] = compute_pairwise_partition_metrics(pred, subset)
    expected = {"soma": "Soma", "dendrite": "Dendrite", "axon": "Axon", "process": "Glia"}
    correct, total = Counter(), Counter()
    induced_correct = induced_total = 0
    token_by_id = {token["fragment_id"]: token for token in tokens}
    for fid, actual in compartments.items():
        stratum = "Interneuron" if subtype[fid] in INTERNEURON_SUBTYPES else subtype[fid]
        correct[stratum] += type_segment_v2(token_by_id[fid]) == expected[actual]
        total[stratum] += 1
        if actual == "soma" and stratum in {"Pyramidal", "Interneuron"}:
            induced_correct += induce_cell_type_from_observables(token_by_id[fid]) == stratum
            induced_total += 1
    output["segment_typing_accuracy_by_stratum"] = {k: correct[k] / total[k] for k in total}
    output["cell_type_induce_accuracy"] = induced_correct / max(induced_total, 1)
    clusters = defaultdict(set)
    for fid, group in pred.items():
        clusters[group].add("I" if subtype[fid] in INTERNEURON_SUBTYPES else "E" if subtype[fid] == "Pyramidal" else "G")
    mixed = sum(types.issuperset({"E", "I"}) for types in clusters.values())
    output["chimera_rate_EI"] = mixed / max(len(clusters), 1)
    glia = [fid for fid in gt if subtype[fid] == "Glia"]
    output["glia_isolation_rate"] = sum(all(subtype[x] == "Glia" for x, p in pred.items() if p == pred[fid]) for fid in glia) / max(len(glia), 1)
    return output


def main() -> None:
    import json
    tokens, gt, subtype, compartments = build_population()
    assembler = HungarianBipartiteAssembler(BlindHandshake(), max_search_dist_nm=4_000)
    links, _ = assembler.assemble_volume_bipartite(tokens, candidate_pool=tokens)
    pred = links_to_pred_map(list(gt), links)
    print("EXP-050 SYNTHETIC COMPANION — do not combine with EXP-049 real-data table")
    print(f"Population: {SUBTYPE_COUNTS}; candidate pool: {len(tokens)}")
    print(json.dumps(compute_stratified_metrics(pred, gt, subtype, compartments, tokens), indent=2))


if __name__ == "__main__":
    main()
