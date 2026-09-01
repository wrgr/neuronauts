#!/usr/bin/env python3
"""Step 4 (learned) — EdgePartitionGNN + correlation clustering on bench_v1.

Runs the clean-track model (the Phase 2.11 configuration) against the locked
splits and scores it on exactly the metrics the baselines used, so the numbers
are directly comparable to `results/bench_v1/RESULTS.md`.

Protocol, enforced by construction:

- The model is trained **only** on the train split.
- `cc_bias` is swept on **val only**; the winner is applied **once** to test.
- Ground-truth labels reach the trainer as supervision and the scorer as
  evaluation. They are never node or edge features — the graph builder puts
  position, fragment id, and embedding similarity in the features, and
  `ObservationGraph.labels` is a separate array.

Fragment morphology here is the real synapse point cloud, the Phase 2.3
substrate. That is a genuine limitation and not a small one: STATUS.md measured
union-find ARI 0.305 with synapse-cloud fragments vs 0.838 once real L2
skeletons supplied endpoint adjacency. Expect this to underperform what the
same model can do with L2 geometry.

Usage
-----
    python scripts/model_bench_v1.py --epochs 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from neuronauts.data.versions import BASE_VERSION, LABEL_VERSION  # noqa: E402
from neuronauts.results_schema import ResultsRecord, write_results  # noqa: E402
from scripts.baseline_bench_v1 import (  # noqa: E402
    DATASET_DIR,
    evaluate,
    fmt,
    load_split,
)


def build_region_and_fragments(data: dict, region_idx: int, region_name: str):
    """Wrap one bench_v1 region as a Region + cloud Fragments."""
    from neuronauts.schemas import Region
    from treestitch.realworld import _cloud_fragment

    sel = data["region_of"] == region_idx
    pos = data["positions_nm"][sel].astype(np.float32)
    base = data["base_roots"][sel]
    label = data["label_roots"][sel]
    n = len(pos)

    region = Region(
        region_id=region_name,
        bbox_nm=((float(pos[:, 0].min()), float(pos[:, 1].min()), float(pos[:, 2].min())),
                 (float(pos[:, 0].max()), float(pos[:, 1].max()), float(pos[:, 2].max()))),
        voxel_size_nm=(4.0, 4.0, 40.0),
        seg_version=BASE_VERSION,
        label_version=LABEL_VERSION,
        pre_pt_nm=pos,
        post_pt_nm=pos,
        pre_root_id=label.astype(np.int64),
        post_root_id=label.astype(np.int64),
        synapse_id=np.arange(n, dtype=np.int64),
        pre_seg_id=base.astype(np.int64),
        post_seg_id=base.astype(np.int64),
    ).validate()

    fragments = []
    for fid, root in enumerate(np.unique(base)):
        idx = np.flatnonzero(base == root)
        fragments.append(
            _cloud_fragment(int(root), region_name, pos[idx], idx.astype(np.int64))
        )
    return region, fragments


def build_graph_for_region(data, region_idx, region_name, *, k_spatial, endpoint_radius_nm):
    from treestitch.embed import encode_fragments_morphological
    from treestitch.graph import build_observation_graph

    region, fragments = build_region_and_fragments(data, region_idx, region_name)
    fragments = encode_fragments_morphological(fragments)
    graph = build_observation_graph(
        region, fragments, side="pre",
        k_spatial=k_spatial,
        endpoint_radius_nm=endpoint_radius_nm,
    )
    return graph


def build_split_graphs(data: dict, *, k_spatial: int, endpoint_radius_nm: float):
    graphs = []
    for i, name in enumerate(data["regions"]):
        g = build_graph_for_region(
            data, i, name,
            k_spatial=k_spatial, endpoint_radius_nm=endpoint_radius_nm)
        n_by_type = {int(t): int((g.edge_type == t).sum())
                     for t in np.unique(g.edge_type)}
        print(f"    [{name}] nodes={len(g.node_pos):,} edges={len(g.edge_src):,} "
              f"by_type={n_by_type}", flush=True)
        graphs.append(g)
    return graphs


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--k-spatial", type=int, default=8)
    ap.add_argument(
        "--endpoint-radius-nm", type=float, default=2_000.0,
        help="Endpoint-adjacency radius. The 10,000 nm used for SKELETON "
             "fragments explodes on this substrate: a synapse-cloud "
             "'endpoint' is just an extreme of the point cloud, and in dense "
             "neuropil thousands fall within 10 um. Measured on the val "
             "region (1,311 fragments): 0 endpoint edges at None, 1,324 at "
             "1 um, 13,396 at 2 um, 167,324 at 5 um, 1,176,878 at 10 um. "
             "2 um keeps the signal without the blow-up.")
    ap.add_argument("--franken-hard-frac", type=float, default=0.30)
    ap.add_argument("--cc-biases", type=float, nargs="*",
                    default=[-1.0, 0.0, 1.0, 2.0, 3.0],
                    help="Swept on val only. Positive values merge more "
                         "aggressively; the negative end is conservative and "
                         "collapses to making no joins at all on this "
                         "substrate.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/bench_v1")
    args = ap.parse_args()

    from treestitch.graph import concat_observation_graphs
    from treestitch.partition import partition_observations_cc, train_edge_partition

    dataset = json.loads((DATASET_DIR / "manifests" / "dataset.json").read_text())
    manifest_sha = __import__("hashlib").sha256(
        json.dumps(dataset["manifest_sha256"], sort_keys=True).encode()
    ).hexdigest()

    splits = {s: load_split(s) for s in ("train", "val", "test")}

    print("building observation graphs (real synapse-cloud fragments)…")
    graphs = {}
    for s in ("train", "val", "test"):
        print(f"  {s}:")
        graphs[s] = build_split_graphs(
            splits[s], k_spatial=args.k_spatial,
            endpoint_radius_nm=args.endpoint_radius_nm)

    train_graph = concat_observation_graphs(graphs["train"])
    print(f"\ntrain graph: {len(train_graph.node_pos):,} nodes, "
          f"{len(train_graph.edge_src):,} edges")

    print(f"\ntraining EdgePartitionGNN ({args.epochs} epochs, train split only)…")
    model, hist = train_edge_partition(
        train_graph,
        n_epochs=args.epochs,
        franken_hard_frac=args.franken_hard_frac,
        seed=args.seed,
    )
    final = {k: (round(float(v[-1]), 4) if isinstance(v, list) and v else v)
             for k, v in hist.items() if isinstance(v, list) and v}
    print(f"  final training stats: {final}")

    # --- calibrate cc_bias on VAL ONLY -------------------------------------
    print("\nsweeping cc_bias on VAL (test untouched)…")
    val_graph = concat_observation_graphs(graphs["val"])
    sweep = []
    for bias in args.cc_biases:
        pred = partition_observations_cc(model, val_graph, bias=bias)
        m = evaluate(pred, splits["val"])
        sweep.append((bias, m))
        print(f"  bias={bias:>5.1f}  ARI={m['ari']:>8.4f}  "
              f"cross_merge P={fmt(m['cross_merge_precision'])} "
              f"R={fmt(m['cross_merge_recall'])} "
              f"F1={fmt(m['cross_merge_f1'])}  "
              f"joins={m['n_cross_pairs_predicted']:>10,}  "
              f"clusters={m['n_pred_clusters']:,}", flush=True)

    # Configurations that make no cross-fragment join at all are just the
    # untouched-v117 baseline wearing a model; selecting one would report the
    # do-nothing floor as a model result. Require at least one join.
    usable = [(b, m) for b, m in sweep if m["n_cross_pairs_predicted"] > 0]
    if not usable:
        print("\n  !! every cc_bias made zero cross-fragment joins: this model "
              "reduces to the untouched-v117 baseline. Nothing to apply to "
              "test that is not already reported as the floor.")
        best_bias, best_m = None, None
    else:
        best_bias, best_m = max(usable, key=lambda t: t[1]["cross_merge_f1"] or 0.0)
        print(f"\n  -> selected cc_bias={best_bias} on val "
              f"(cross_merge_F1={fmt(best_m['cross_merge_f1'])}, "
              f"{best_m['n_cross_pairs_correct']:,}/"
              f"{best_m['n_cross_pairs_predicted']:,} joins correct)")

    if best_bias is None:
        write_results(ResultsRecord(
            experiment="bench_v1_edge_partition_gnn_val_sweep",
            split="val",
            metrics={"sweep": [{"cc_bias": b, **mm} for b, mm in sweep],
                     "outcome": "no cc_bias produced any cross-fragment join",
                     **sweep[-1][1]},
            base_version=BASE_VERSION, label_version=LABEL_VERSION,
            synthetic=False, data_manifest_sha=manifest_sha,
            notes=("Model reduced to the untouched-v117 baseline at every "
                   "swept cc_bias; test deliberately not touched."),
        ), REPO / args.out_dir / "edge_partition_gnn_val_sweep__val.json")
        print("test left untouched.")
        return 0

    # --- one run on TEST ----------------------------------------------------
    print("\napplying once to TEST…")
    test_graph = concat_observation_graphs(graphs["test"])
    pred_test = partition_observations_cc(model, test_graph, bias=best_bias)
    m_test = evaluate(pred_test, splits["test"])
    print(f"  test  ARI={m_test['ari']:.4f}  pair_F1={m_test['pair_f1']:.4f}  "
          f"cross_merge P={fmt(m_test['cross_merge_precision'])} "
          f"R={fmt(m_test['cross_merge_recall'])} "
          f"F1={fmt(m_test['cross_merge_f1'])}")
    print(f"        predicted {m_test['n_cross_pairs_predicted']:,} cross joins, "
          f"{m_test['n_cross_pairs_correct']:,} correct, "
          f"{m_test['n_cross_pairs_true']:,} true ones exist")

    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    notes = ("EdgePartitionGNN + GAEC correlation clustering. Trained on train "
             "only; cc_bias calibrated on val only; test evaluated once. "
             "Fragment morphology is the real synapse point cloud, NOT L2 "
             "skeletons - STATUS.md measured union-find ARI 0.305 vs 0.838 for "
             "that substrate difference, so this understates what the same "
             "model can do with L2 geometry.")

    for name, split, metrics in [
        ("edge_partition_gnn_val_sweep", "val",
         {"selected_cc_bias": best_bias,
          "sweep": [{"cc_bias": b, **mm} for b, mm in sweep], **best_m}),
        ("edge_partition_gnn", "test",
         {**m_test, "selected_cc_bias": best_bias, "calibrated_on": "val",
          "epochs": args.epochs, "fragment_substrate": "synapse_cloud"}),
    ]:
        write_results(ResultsRecord(
            experiment=f"bench_v1_{name}",
            split=split,
            metrics=metrics,
            base_version=BASE_VERSION,
            label_version=LABEL_VERSION,
            synthetic=False,
            data_manifest_sha=manifest_sha,
            n_observations=metrics.get("n_observations"),
            notes=notes,
        ), out / f"{name}__{split}.json")
    print(f"\nwrote stamped records to {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
