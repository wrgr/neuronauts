#!/usr/bin/env python3
"""EXP-049: spatially disjoint, dense v117 subvolume evaluation."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neuronauts.assemble import HungarianBipartiteAssembler
from neuronauts.data.cave import V117Region, fetch_v117_region
from neuronauts.fetch import make_cube_bbox_nm
from neuronauts.global_merge.eval.benchmark import compute_pairwise_partition_metrics
from neuronauts.morpho_grammar.santiago_v2_grammar import type_segment_v2

DEFAULT_TEST_CENTER_NM = (700_000, 500_000, 20_000)
DEFAULT_TRAIN_CENTER_NM = (140_000, 60_000, 20_000)


def boxes_overlap(a: tuple, b: tuple) -> bool:
    """Return whether two half-open 3-D boxes have positive-volume overlap."""
    return all(a[0][axis] < b[1][axis] and b[0][axis] < a[1][axis] for axis in range(3))


def _majority_labels(region: V117Region) -> dict[int, int]:
    votes: dict[int, Counter] = defaultdict(Counter)
    for seg, gt in zip(region.seg_ids.tolist(), region.gt_labels.tolist()):
        if int(gt) > 0:
            votes[int(seg)][int(gt)] += 1
    return {seg: counts.most_common(1)[0][0] for seg, counts in votes.items()}


def build_tokens_from_region(region: V117Region) -> tuple[list[dict], dict[str, str], dict[str, dict]]:
    """Build blind tokens for *all* fetched segments and labeled evaluation maps."""
    labels = _majority_labels(region)
    syn_counts = Counter(int(x) for x in region.seg_ids.tolist())
    tokens, gt_map, pieces = [], {}, {}
    for seg_id in sorted(int(x) for x in region.skeletons):
        skel = region.skeletons[seg_id]
        vertices = np.asarray(skel.vertices, dtype=float)
        radii = np.asarray(skel.radius, dtype=float) if skel.radius is not None else np.zeros(len(vertices))
        centroid = vertices.mean(axis=0) if len(vertices) else np.zeros(3)
        start, end = (vertices[0], vertices[-1]) if len(vertices) else (centroid, centroid)
        token = {
            "fragment_id": str(seg_id), "start_nm": start, "end_nm": end,
            "centroid_nm": centroid, "mean_radius_nm": float(radii.mean()) if len(radii) else 0.0,
            "max_radius_nm": float(radii.max()) if len(radii) else 0.0,
            "n_pre": int(syn_counts[seg_id]), "n_post": 0,
        }
        # Blindness contract: no gt label or cell type is stored in a token.
        tokens.append(token)
        pieces[str(seg_id)] = {"vertices_nm": vertices, "edges": np.asarray(skel.edges)}
        if seg_id in labels:
            gt_map[str(seg_id)] = str(labels[seg_id])
    return tokens, gt_map, pieces


class GeometricHandshake:
    def evaluate_bidirectional_handshake(self, left: Mapping, right: Mapping) -> float:
        distance = np.linalg.norm(np.asarray(left["end_nm"]) - np.asarray(right["start_nm"]))
        same_polarity = type_segment_v2(left) == type_segment_v2(right)
        return float(np.exp(-distance / 2_000.0) * (1.0 if same_polarity else 0.05))


def _partition(ids: list[str], links: list[tuple[str, str]]) -> dict[str, str]:
    from benchmark_exp047_hungarian_bipartite import links_to_pred_map
    return links_to_pred_map(ids, links)


def stratified_metrics(pred: dict[str, str], gt: dict[str, str], cell_types: Mapping[str, str]) -> dict:
    """Apply optional cell-type strata after inference; labels never touch tokens."""
    output = {"overall": compute_pairwise_partition_metrics(pred, gt)}
    for stratum in ("E", "I", "Glia"):
        subset = {fid: label for fid, label in gt.items() if cell_types.get(label) == stratum}
        output[stratum] = compute_pairwise_partition_metrics(pred, subset) if subset else None
    ei_links = total_links = 0
    cluster_gt = defaultdict(set)
    for fid, cluster in pred.items():
        if fid in gt:
            cluster_gt[cluster].add(gt[fid])
    for labels in cluster_gt.values():
        types = {cell_types.get(label) for label in labels}
        total_links += len(labels) > 1
        ei_links += "E" in types and "I" in types
    output["chimera_rate_EI"] = ei_links / total_links if total_links else 0.0
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--center-nm", nargs=3, type=int, default=DEFAULT_TEST_CENTER_NM)
    parser.add_argument("--side-um", type=float, default=30.0)
    parser.add_argument("--train-center-nm", nargs=3, type=int, default=DEFAULT_TRAIN_CENTER_NM)
    parser.add_argument("--train-side-um", type=float, default=100.0)
    parser.add_argument("--token", default=os.environ.get("CAVE_TOKEN"))
    parser.add_argument("--cache-dir", default="cache/v117_exp049")
    parser.add_argument("--cell-types-json", help="Optional v1412 root -> E/I/Glia post-hoc labels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_bbox = make_cube_bbox_nm(tuple(args.center_nm), args.side_um)
    train_bbox = make_cube_bbox_nm(tuple(args.train_center_nm), args.train_side_um)
    if boxes_overlap(train_bbox, test_bbox):
        raise SystemExit("Invalid design: train and test bounding boxes overlap")
    print(f"Spatial overlap: false\nTrain bbox: {train_bbox}\nTest bbox:  {test_bbox}")
    region = fetch_v117_region(test_bbox, token=args.token, min_seg_synapses=3,
                               max_segs=None, skeleton_cache_dir=args.cache_dir)
    tokens, gt_map, pieces = build_tokens_from_region(region)
    if not tokens:
        raise SystemExit("No v117 segments found in the requested box")
    assembler = HungarianBipartiteAssembler(GeometricHandshake(), max_search_dist_nm=10_000)
    links, meta = assembler.assemble_volume_bipartite(tokens, pieces, candidate_pool=tokens)
    pred = _partition(list(gt_map), links)
    cell_types = json.loads(Path(args.cell_types_json).read_text()) if args.cell_types_json else {}
    metrics = stratified_metrics(pred, gt_map, cell_types)
    volume_um3 = args.side_um ** 3
    coverage = len(gt_map) / len(tokens)
    soma_clusters = defaultdict(list)
    for token in tokens:
        soma_clusters[pred.get(token["fragment_id"], token["fragment_id"])].append(type_segment_v2(token) == "Soma")
    single_soma = sum(sum(values) == 1 for values in soma_clusters.values()) / max(len(soma_clusters), 1)
    print(f"Candidate pool size (all v117 segments): {len(tokens)}")
    print(f"Fragment density: {len(tokens) / volume_um3:.6f} fragments/um^3")
    print(f"v1412 coverage: {coverage:.1%}; single-soma compliance: {single_soma:.1%}")
    print(json.dumps({"metrics": metrics, "diagnostics": meta}, indent=2))


if __name__ == "__main__":
    main()
