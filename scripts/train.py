#!/usr/bin/env python3
"""End-to-end training: fetch/cache real boxes → grammar + GAT → evaluate.

Quick-start
-----------
1.  Build a box cache (requires network access to MICrONS/CAVE)::

        python scripts/train.py build-dataset \\
            --cache-dir data/boxes \\
            --n-boxes 50 \\
            --min-synapses 15

    Or from a local nucleus table (produced by synapse_root_counts_static.py)::

        python scripts/train.py build-dataset \\
            --cache-dir data/boxes \\
            --n-boxes 50 \\
            --counts-tsv run_logs/synapse_root_counts_static.tsv \\
            --nucleus-csv data/microns_static/v1078/nucleus_detection_v0.csv

2.  Train grammar and (optionally) GAT::

        python scripts/train.py train \\
            --cache-dir data/boxes \\
            --grammar-output models/shared_grammar.pt \\
            --gat-output models/gat.pt \\
            --epochs 30 \\
            --train-gat

3.  Everything in one shot::

        python scripts/train.py run \\
            --cache-dir data/boxes \\
            --n-boxes 50

Training strategy
-----------------
Grammar (SharedGrammarModel)
    Built from cached synapse tables alone — no agent simulation required.
    Each box contributes merge examples (positive = same root_id cluster,
    negative = nearby but distinct clusters) and topology/atomicity examples
    (positive = single root cluster, negative = two merged clusters).
    This is fast: ~0.2–0.5 s per box per epoch on CPU.

GlobalAssemblyGAT  (only when --train-gat is set)
    Requires agent path simulation on each box.  With 700 agents × 450 steps
    a single box takes ~20–60 s on CPU.  The GAT training step then runs on
    the resulting ConnectivityGraph.  Enable only if you have adequate compute
    or want to use a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add repo root to path so the script works from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise SystemExit(
            "torch is required.  Install with:\n"
            "  pip install torch\n"
            "or (full extras):\n"
            "  pip install -e '.[topology]'"
        ) from exc


def _grammar_batch_from_synapses(synapses, device, *, max_merge=256, max_topo=128):
    """Build merge + topology batches from a SynapseTable (no simulation).

    Returns ``(merge_batch, topo_batch)`` or ``(None, None)`` if the synapse
    table is too sparse to produce any examples.
    """
    import torch
    from neuronauts.merge_dataset import build_merge_examples, examples_to_arrays
    from neuronauts.topology_dataset import (
        build_cluster_examples,
        examples_to_branch_sequence_arrays,
    )

    merge_examples = build_merge_examples(synapses)
    topo_examples = build_cluster_examples(synapses, membrane_field=None)  # membrane ignored

    if not merge_examples or not topo_examples:
        return None, None

    # Cap to avoid memory spikes on large boxes.
    merge_examples = merge_examples[:max_merge]
    topo_examples = topo_examples[:max_topo]

    lx, lm, rx, rm, y_merge = examples_to_arrays(merge_examples)
    merge_batch = {
        "left_x":   torch.from_numpy(lx).to(device),
        "left_mask":  torch.from_numpy(lm).to(device),
        "right_x":  torch.from_numpy(rx).to(device),
        "right_mask": torch.from_numpy(rm).to(device),
        "y": torch.from_numpy(y_merge.astype(np.float32)).to(device),
    }

    bx, bsm, bm = examples_to_branch_sequence_arrays(topo_examples)
    y_topo = np.array([ex.label for ex in topo_examples], dtype=np.float32)
    topo_batch = {
        "branch_x":             torch.from_numpy(bx).to(device),
        "branch_sequence_mask": torch.from_numpy(bsm).to(device),
        "branch_mask":          torch.from_numpy(bm).to(device),
        "y":                    torch.from_numpy(y_topo).to(device),
    }
    return merge_batch, topo_batch


def _accuracy_from_logits(logits_np, y_np) -> float:
    preds = (logits_np >= 0.0).astype(np.int64)
    return float((preds == y_np.astype(np.int64)).mean())


def _tsv_row(fields: dict) -> str:
    return "\t".join(str(v) for v in fields.values())


def _tsv_header(fields: dict) -> str:
    return "\t".join(fields.keys())


# ---------------------------------------------------------------------------
# Subcommand: build-dataset
# ---------------------------------------------------------------------------

def cmd_build_dataset(args: argparse.Namespace) -> int:
    from neuronauts.dataset_builder import (
        BoxCache,
        build_dataset,
        select_boxes_from_nucleus_table,
        select_random_boxes,
        select_synapse_seeded_boxes,
    )

    cache = BoxCache(args.cache_dir)
    print(f"Cache dir: {args.cache_dir}  (existing: {len(cache)} boxes)")

    strategy = getattr(args, "strategy", "synapse-seeded")

    if args.counts_tsv and args.nucleus_csv:
        print(
            f"Selecting up to {args.n_boxes} boxes from nucleus table "
            f"(syn range [{args.min_synapses}, {args.max_synapses}]) …"
        )
        specs = select_boxes_from_nucleus_table(
            counts_tsv=args.counts_tsv,
            nucleus_csv=args.nucleus_csv,
            n=args.n_boxes,
            min_syn=args.min_synapses,
            max_syn=args.max_synapses,
            box_side_um=args.box_side_um,
            seed=args.seed,
        )
    elif strategy == "random":
        print(
            f"Randomly sampling {args.n_boxes} box positions from Minnie65 …\n"
            "  (tip: most random boxes are empty; expect many 'skip' messages)"
        )
        specs = select_random_boxes(
            n=args.n_boxes,
            box_side_um=args.box_side_um,
            seed=args.seed,
        )
    else:
        print(
            f"Synapse-seeded sampling: pulling positions from CAVE then selecting "
            f"{args.n_boxes} box centres …"
        )
        specs = select_synapse_seeded_boxes(
            n=args.n_boxes,
            box_side_um=args.box_side_um,
            seed=args.seed,
            token=args.cave_token,
        )

    records = build_dataset(
        specs,
        cache,
        min_synapses=args.min_synapses,
        max_synapses=args.max_synapses,
        token=args.cave_token,
        verbose=True,
    )
    print(f"\nDone.  {len(records)} usable boxes in cache.")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> int:  # noqa: C901
    torch = _require_torch()

    from neuronauts.dataset_builder import load_dataset
    from neuronauts.shared_grammar_model import (
        GATTrainingConfig,
        GlobalAssemblyGAT,
        SharedGrammarModel,
        gat_train_step,
        load_global_assembly_gat,
        load_shared_grammar_model,
        multitask_train_step,
        save_global_assembly_gat,
        save_shared_grammar_model,
    )

    # ── Load / create models ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    grammar_path = Path(args.grammar_output)
    if grammar_path.exists() and not args.reset:
        print(f"Resuming grammar from {grammar_path}")
        grammar_model = load_shared_grammar_model(str(grammar_path)).to(device)
    else:
        grammar_model = SharedGrammarModel().to(device)
        print("Initialised fresh grammar model.")

    grammar_optimizer = torch.optim.Adam(grammar_model.parameters(), lr=args.lr)

    gat_model = None
    gat_optimizer = None
    if args.train_gat:
        gat_path = Path(args.gat_output)
        if gat_path.exists() and not args.reset:
            print(f"Resuming GAT from {gat_path}")
            gat_model = load_global_assembly_gat(str(gat_path)).to(device)
        else:
            emb_dim = grammar_model.path_encoder.output_dim
            gat_model = GlobalAssemblyGAT(node_dim=emb_dim).to(device)
            print("Initialised fresh GAT model.")
        gat_optimizer = torch.optim.Adam(gat_model.parameters(), lr=args.lr)

    # ── Load dataset ──────────────────────────────────────────────────────
    cache, all_records = load_dataset(
        args.cache_dir,
        min_synapses=args.min_synapses,
        max_synapses=args.max_synapses,
    )
    if not all_records:
        print(
            f"No cached boxes in {args.cache_dir} with "
            f"{args.min_synapses}–{args.max_synapses} synapses.\n"
            "Run:  python scripts/train.py build-dataset --cache-dir <dir>"
        )
        return 1

    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_records)  # type: ignore[arg-type]
    n_val = max(1, int(len(all_records) * args.val_fraction))
    val_records = all_records[:n_val]
    train_records = all_records[n_val:]
    print(
        f"Dataset: {len(train_records)} train + {len(val_records)} val boxes "
        f"({sum(r.n_synapses for r in train_records)} train synapses)"
    )

    # ── Logging ───────────────────────────────────────────────────────────
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_log.tsv"

    best_val_f1 = -1.0
    history: dict[str, list] = {
        k: [] for k in (
            "epoch", "train_merge_acc", "train_topo_acc",
            "train_gat_f1", "val_f1", "val_precision", "val_recall",
        )
    }

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        grammar_model.train()
        epoch_merge_accs: list[float] = []
        epoch_topo_accs: list[float] = []
        epoch_gat_f1s: list[float] = []

        order = rng.permutation(len(train_records))
        for idx in order:
            record = train_records[idx]
            try:
                volume_chunk, synapses = cache.load(record)
            except Exception as exc:
                print(f"  [W] failed to load {record.box_hash}: {exc}")
                continue

            # ── Grammar step (fast: no simulation) ────────────────────────
            merge_batch, topo_batch = _grammar_batch_from_synapses(
                synapses, device,
                max_merge=args.max_merge_per_box,
                max_topo=args.max_topo_per_box,
            )
            if merge_batch is not None:
                grammar_metrics = multitask_train_step(
                    grammar_model, grammar_optimizer,
                    merge_batch=merge_batch,
                    topology_batch=topo_batch,
                )
                epoch_merge_accs.append(grammar_metrics.get("merge_accuracy", 0.0))
                epoch_topo_accs.append(grammar_metrics.get("atomicity_accuracy", 0.0))

            # ── GAT step (slow: needs path simulation) ─────────────────────
            if args.train_gat and epoch % args.gat_every_n_epochs == 0:
                _run_gat_training_step(
                    volume_chunk, synapses, gat_model, grammar_model,
                    gat_optimizer, device, args, epoch_gat_f1s,
                )

        # ── Validation ────────────────────────────────────────────────────
        grammar_model.eval()
        val_f1s: list[float] = []
        val_precisions: list[float] = []
        val_recalls: list[float] = []

        for record in val_records:
            m = _validate_box(record, cache, grammar_model, gat_model, args, device)
            if m is not None:
                val_f1s.append(m.f1)
                val_precisions.append(m.precision)
                val_recalls.append(m.recall)

        val_f1  = float(np.mean(val_f1s))     if val_f1s     else 0.0
        val_pre = float(np.mean(val_precisions)) if val_precisions else 0.0
        val_rec = float(np.mean(val_recalls))  if val_recalls  else 0.0

        # ── Checkpoint on best val F1 ──────────────────────────────────────
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            grammar_path.parent.mkdir(parents=True, exist_ok=True)
            save_shared_grammar_model(str(grammar_path), grammar_model)
            if gat_model is not None:
                Path(args.gat_output).parent.mkdir(parents=True, exist_ok=True)
                save_global_assembly_gat(str(args.gat_output), gat_model)

        # ── Logging ───────────────────────────────────────────────────────
        row = {
            "epoch":           epoch,
            "train_merge_acc": f"{np.mean(epoch_merge_accs):.4f}" if epoch_merge_accs else "n/a",
            "train_topo_acc":  f"{np.mean(epoch_topo_accs):.4f}"  if epoch_topo_accs  else "n/a",
            "train_gat_f1":    f"{np.mean(epoch_gat_f1s):.4f}"    if epoch_gat_f1s    else "n/a",
            "val_f1":          f"{val_f1:.4f}",
            "val_precision":   f"{val_pre:.4f}",
            "val_recall":      f"{val_rec:.4f}",
            "best_val_f1":     f"{best_val_f1:.4f}",
            "elapsed_s":       f"{time.time() - t_epoch:.1f}",
        }
        if epoch == 1 and not log_path.exists():
            log_path.write_text(_tsv_header(row) + "\n", encoding="utf-8")
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(_tsv_row(row) + "\n")

        print(
            f"epoch {epoch:3d}/{args.epochs} | "
            f"merge_acc={row['train_merge_acc']} "
            f"topo_acc={row['train_topo_acc']} "
            f"gat_f1={row['train_gat_f1']} | "
            f"val_f1={val_f1:.4f} (best={best_val_f1:.4f}) | "
            f"{time.time() - t_epoch:.1f}s"
        )

    print(
        f"\nTraining complete.  Best val F1 = {best_val_f1:.4f}\n"
        f"  Grammar → {grammar_path}\n"
        + (f"  GAT     → {args.gat_output}\n" if args.train_gat else "")
        + f"  Log     → {log_path}"
    )
    return 0


def _run_gat_training_step(
    volume_chunk, synapses, gat_model, grammar_model,
    gat_optimizer, device, args, gat_f1_acc: list,
):
    """Run agent simulation → build graph → gat_train_step on one box."""
    from neuronauts.fields import compute_membrane_field
    from neuronauts.run import HeuristicConfig, _build_graph, simulate_paths_and_hits
    from neuronauts.shared_grammar_model import gat_train_step

    try:
        mf = compute_membrane_field(volume_chunk.data)
        path_arr, synapse_hits, path_lengths, _ = simulate_paths_and_hits(
            volume_chunk.data,
            synapses.pre_pt,
            synapses.post_pt,
            verbose=False,
            membrane_field_override=mf,
        )
        graph = _build_graph(
            path_arr=path_arr,
            path_lengths=path_lengths,
            synapse_hits=synapse_hits,
            pre_pts=synapses.pre_pt,
            post_pts=synapses.post_pt,
            pre_seg_ids=synapses.pre_seg_id,
            post_seg_ids=synapses.post_seg_id,
            heuristic_config=HeuristicConfig.learned(),
        )
        if not graph.edges:
            return

        m = gat_train_step(
            gat_model, grammar_model.path_encoder, gat_optimizer,
            graph=graph,
            pre_root_ids=synapses.pre_root_id,
            post_root_ids=synapses.post_root_id,
            soft_f1_weight=args.gat_soft_f1_weight,
        )
        if m["n_edges"] > 0:
            gat_f1_acc.append(m["pred_f1"])
    except Exception as exc:
        print(f"  [W] GAT step failed: {exc}")


def _make_live_merge_score_fn(grammar_model):
    """Build a merge-score closure that calls the live in-memory model.

    Unlike ``_load_shared_merge_score_fn`` (which is lru_cache'd and reads from
    disk), this always uses the current model weights so validation tracks the
    model as it trains.
    """
    import torch

    grammar_model.eval()

    def score_fn(left_sequence: np.ndarray, right_sequence: np.ndarray) -> float:
        left = torch.from_numpy(left_sequence[None, ...]).float()
        right = torch.from_numpy(right_sequence[None, ...]).float()
        left_mask = torch.zeros((1, left.shape[1]), dtype=torch.bool)
        right_mask = torch.zeros((1, right.shape[1]), dtype=torch.bool)
        with torch.no_grad():
            logits = grammar_model.score_merge(left, left_mask, right, right_mask)
        return float(logits.squeeze().cpu())

    return score_fn


def _validate_box(record, cache, grammar_model, gat_model, args, device):
    """Run full inference on one validation box and return LineGraphMetrics."""
    import torch
    from neuronauts.fields import compute_membrane_field
    from neuronauts.run import HeuristicConfig, _build_graph, simulate_paths_and_hits
    from neuronauts.line_graph import evaluate

    try:
        volume_chunk, synapses = cache.load(record)
        if len(synapses.pre_pt) < 5:
            return None

        mf = compute_membrane_field(volume_chunk.data)
        path_arr, synapse_hits, path_lengths, _ = simulate_paths_and_hits(
            volume_chunk.data,
            synapses.pre_pt,
            synapses.post_pt,
            verbose=False,
            membrane_field_override=mf,
        )

        # Always use the live in-memory model so validation tracks training progress.
        # Avoid _load_shared_merge_score_fn which is lru_cache'd and would return
        # stale epoch-1 weights for the entire training run.
        _score_fn = _make_live_merge_score_fn(grammar_model)
        hcfg = HeuristicConfig.learned()

        graph = _build_graph(
            path_arr=path_arr,
            path_lengths=path_lengths,
            synapse_hits=synapse_hits,
            pre_pts=synapses.pre_pt,
            post_pts=synapses.post_pt,
            pre_seg_ids=synapses.pre_seg_id,
            post_seg_ids=synapses.post_seg_id,
            learned_merge_score_fn=_score_fn,
            heuristic_config=hcfg,
        )

        # Optional GAT refinement on the validation graph.
        if gat_model is not None and graph.edges:
            from neuronauts.assembly import gat_refine_connectivity
            grammar_model.eval()
            with torch.no_grad():
                graph = gat_refine_connectivity(
                    graph,
                    path_encoder=grammar_model.path_encoder,
                    gat_model=gat_model,
                    edge_threshold=args.gat_edge_threshold,
                )

        return evaluate(graph, synapses.pre_root_id, synapses.post_root_id)

    except Exception as exc:
        print(f"  [W] validation failed for {record.box_hash}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Subcommand: run (build + train in one shot)
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    """Build dataset then train."""
    # Forward to build-dataset then train.
    rc = cmd_build_dataset(args)
    if rc != 0:
        return rc
    return cmd_train(args)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-dir", default="data/boxes",
        help="Directory for cached box data.",
    )
    parser.add_argument(
        "--min-synapses", type=int, default=15,
        help="Minimum synapse pairs per box.",
    )
    parser.add_argument(
        "--max-synapses", type=int, default=200,
        help="Maximum synapse pairs per box.",
    )
    parser.add_argument("--seed", type=int, default=42)


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    _add_common_args(parser)
    parser.add_argument(
        "--n-boxes", type=int, default=50,
        help="Number of box specs to request.",
    )
    parser.add_argument(
        "--box-side-um", type=float, default=6.0,
        help="Box side length in microns.",
    )
    parser.add_argument(
        "--strategy", default="synapse-seeded",
        choices=["synapse-seeded", "random"],
        help=(
            "Box selection strategy.  'synapse-seeded' (default) queries CAVE "
            "for real synapse positions and uses those as box centres — every "
            "box is guaranteed to be inside annotated neuropil.  'random' "
            "uniformly samples the full dataset extent and skips empty boxes "
            "(expect many 'skip' messages)."
        ),
    )
    parser.add_argument(
        "--counts-tsv", default=None,
        help="Optional: path to synapse_root_counts_static.tsv.",
    )
    parser.add_argument(
        "--nucleus-csv", default=None,
        help="Optional: path to nucleus_detection_v0.csv.",
    )
    parser.add_argument(
        "--cave-token", default=None,
        help="CAVE auth token (not required for public access).",
    )


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    _add_common_args(parser)
    parser.add_argument(
        "--grammar-output", default="models/shared_grammar_real.pt",
        help="Output path for the grammar model checkpoint.",
    )
    parser.add_argument(
        "--gat-output", default="models/gat_real.pt",
        help="Output path for the GAT model checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--val-fraction", type=float, default=0.15,
        help="Fraction of boxes reserved for validation.",
    )
    parser.add_argument(
        "--max-merge-per-box", type=int, default=256,
        help="Cap on merge examples per box per step.",
    )
    parser.add_argument(
        "--max-topo-per-box", type=int, default=128,
        help="Cap on topology examples per box per step.",
    )
    parser.add_argument(
        "--train-gat", action="store_true",
        help="Also train the GlobalAssemblyGAT (requires agent simulation; slow).",
    )
    parser.add_argument(
        "--gat-every-n-epochs", type=int, default=5,
        help="Only run GAT training steps every N grammar epochs.",
    )
    parser.add_argument(
        "--gat-soft-f1-weight", type=float, default=0.5,
        help="Soft-F1 loss weight in gat_train_step.",
    )
    parser.add_argument(
        "--gat-edge-threshold", type=float, default=0.5,
        help="Sigmoid threshold for keeping edges in GAT refinement.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Ignore existing checkpoints and start from scratch.",
    )
    parser.add_argument(
        "--log-dir", default="run_logs",
        help="Directory for training TSV log.",
    )


def parse_args(argv=None) -> argparse.Namespace:
    root = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = root.add_subparsers(dest="command", required=True)

    # build-dataset
    p_build = sub.add_parser("build-dataset", help="Fetch and cache real MICrONS boxes.")
    _add_dataset_args(p_build)
    p_build.set_defaults(func=cmd_build_dataset)

    # train
    p_train = sub.add_parser("train", help="Train grammar (+optional GAT) on cached boxes.")
    _add_train_args(p_train)
    p_train.set_defaults(func=cmd_train)

    # run  (shortcut: build + train) — adds all args without duplication
    p_run = sub.add_parser("run", help="build-dataset then train in one shot.")
    # common
    p_run.add_argument("--cache-dir",    default="data/boxes")
    p_run.add_argument("--min-synapses", type=int, default=15)
    p_run.add_argument("--max-synapses", type=int, default=200)
    p_run.add_argument("--seed",         type=int, default=42)
    # dataset-specific
    p_run.add_argument("--n-boxes",     type=int,   default=50)
    p_run.add_argument("--box-side-um", type=float, default=6.0)
    p_run.add_argument("--strategy",    default="synapse-seeded", choices=["synapse-seeded", "random"])
    p_run.add_argument("--counts-tsv",  default=None)
    p_run.add_argument("--nucleus-csv", default=None)
    p_run.add_argument("--cave-token",  default=None)
    # train-specific
    p_run.add_argument("--grammar-output",      default="models/shared_grammar_real.pt")
    p_run.add_argument("--gat-output",          default="models/gat_real.pt")
    p_run.add_argument("--epochs",              type=int,   default=30)
    p_run.add_argument("--lr",                  type=float, default=3e-4)
    p_run.add_argument("--val-fraction",        type=float, default=0.15)
    p_run.add_argument("--max-merge-per-box",   type=int,   default=256)
    p_run.add_argument("--max-topo-per-box",    type=int,   default=128)
    p_run.add_argument("--train-gat",           action="store_true")
    p_run.add_argument("--gat-every-n-epochs",  type=int,   default=5)
    p_run.add_argument("--gat-soft-f1-weight",  type=float, default=0.5)
    p_run.add_argument("--gat-edge-threshold",  type=float, default=0.5)
    p_run.add_argument("--reset",               action="store_true")
    p_run.add_argument("--log-dir",             default="run_logs")
    p_run.set_defaults(func=cmd_run)

    return root.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
