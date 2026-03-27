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


def _grammar_batch_from_synapses(
    synapses,
    device,
    *,
    path_feature_mode=None,
    max_merge=256,
    max_topo=128,
    max_negative_pairs_per_role=64,
    topo_balanced=True,
):
    """Build merge + topology batches from a SynapseTable (no simulation).

    Returns ``(merge_batch, topo_batch)`` or ``(None, None)`` if the synapse
    table is too sparse to produce any examples.

    When ``topo_balanced=True`` (default), topology examples are stratified so
    we take roughly equal numbers of atomic (label=1) and non-atomic (label=0)
    examples. This avoids the "predict-1-always" shortcut when pos_frac is high.
    """
    import torch
    from neuronauts.grammar import DEFAULT_PATH_FEATURE_MODE
    from neuronauts.merge_dataset import build_merge_examples, examples_to_arrays
    from neuronauts.topology_dataset import (
        build_cluster_examples,
        examples_to_branch_sequence_arrays,
    )
    if path_feature_mode is None:
        path_feature_mode = DEFAULT_PATH_FEATURE_MODE

    # Balanced merge examples (negatives scale with positives), then shuffle so
    # the per-box cap doesn't accidentally select all-positives.
    merge_examples = build_merge_examples(synapses, path_feature_mode=path_feature_mode)
    topo_examples = build_cluster_examples(
        synapses,
        membrane_field=None,  # membrane ignored
        path_feature_mode=path_feature_mode,
        max_negative_pairs_per_role=max_negative_pairs_per_role,
    )

    if not merge_examples or not topo_examples:
        return None, None

    # Cap to avoid memory spikes on large boxes.
    # Deterministic shuffle based on synapse IDs so batches are stable across runs.
    seed = int(np.uint32(np.sum(synapses.synapse_id) % (2**32 - 1)))
    rng = np.random.default_rng(seed)
    if len(merge_examples) > max_merge:
        idx = rng.permutation(len(merge_examples))[:max_merge]
        merge_examples = [merge_examples[i] for i in idx]
    else:
        merge_examples = [merge_examples[i] for i in rng.permutation(len(merge_examples))]

    # Stratified topology sampling: take roughly equal pos/neg to avoid trivial
    # majority prediction (e.g. pos_frac ~0.9 → predict-1 yields 90% acc).
    if topo_balanced:
        pos = [i for i, ex in enumerate(topo_examples) if ex.label == 1]
        neg = [i for i, ex in enumerate(topo_examples) if ex.label == 0]
        rng.shuffle(pos)
        rng.shuffle(neg)
        n_each = min(len(pos), len(neg), max_topo // 2)
        if n_each > 0:
            idx = list(pos[:n_each]) + list(neg[:n_each])
            if len(topo_examples) > max_topo:
                remainder = [i for i in range(len(topo_examples)) if i not in idx]
                rng.shuffle(remainder)
                idx.extend(remainder[: max_topo - len(idx)])
            else:
                # Oversample minority to balance when we have few examples.
                need = max_topo - len(idx)
                if need > 0:
                    majority = pos if len(pos) >= len(neg) else neg
                    minority = neg if len(pos) >= len(neg) else pos
                    pool_maj = majority[n_each:]
                    need_maj = min(need // 2, len(pool_maj))
                    need_min = need - need_maj
                    idx.extend(pool_maj[:need_maj])
                    if need_min > 0 and minority:
                        idx.extend(
                            list(rng.choice(minority, size=min(need_min, max_topo - len(idx)), replace=True))
                        )
            idx = idx[:max_topo]
            rng.shuffle(idx)
            topo_examples = [topo_examples[i] for i in idx]
        elif len(topo_examples) > max_topo:
            idx = rng.permutation(len(topo_examples))[:max_topo]
            topo_examples = [topo_examples[i] for i in idx]
        else:
            topo_examples = [topo_examples[i] for i in rng.permutation(len(topo_examples))]
    elif len(topo_examples) > max_topo:
        idx = rng.permutation(len(topo_examples))[:max_topo]
        topo_examples = [topo_examples[i] for i in idx]
    else:
        topo_examples = [topo_examples[i] for i in rng.permutation(len(topo_examples))]

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


def _maybe_map_synapse_roots(synapses, base_version: int, target_version: int):
    """Optionally remap root IDs between materialization versions.

    If ``base_version == target_version`` this is a no-op and the original
    ``synapses`` object is returned.  Otherwise, root IDs are treated as being
    defined at ``base_version`` and are mapped forward into ``target_version``
    using the CAVE chunkedgraph.  Any synapse whose pre- or post-synaptic root
    maps to 0 (no corresponding body at the target version) is dropped.
    """
    from neuronauts.fetch import SynapseTable

    if base_version == target_version:
        return synapses

    pre_roots = np.asarray(synapses.pre_root_id, dtype=np.int64)
    post_roots = np.asarray(synapses.post_root_id, dtype=np.int64)

    # Expect a precomputed root remap table when versions differ.
    mapping = getattr(
        _maybe_map_synapse_roots,
        "_root_mapping",
        None,
    )
    if mapping is None:
        # No mapping loaded: fall back to identity and warn once.
        if not getattr(_maybe_map_synapse_roots, "_warned_no_mapping", False):
            print(
                "[root-mapping] base_version != target_version "
                "but no precomputed mapping was loaded. "
                "Proceeding without remapping."
            )
            _maybe_map_synapse_roots._warned_no_mapping = True  # type: ignore[attr-defined]
        return synapses

    pre_mapped = np.array(
        [mapping.get(int(r), 0) for r in pre_roots],
        dtype=np.int64,
    )
    post_mapped = np.array(
        [mapping.get(int(r), 0) for r in post_roots],
        dtype=np.int64,
    )

    keep_mask = (pre_mapped != 0) & (post_mapped != 0)
    if not np.any(keep_mask):
        # All synapses vanished under the mapping; return the original and let
        # downstream sampling logic skip boxes with no usable pairs.
        return synapses

    def _mask_or_none(arr):
        if arr is None:
            return None
        return np.asarray(arr)[keep_mask]

    return SynapseTable(
        pre_pt=np.asarray(synapses.pre_pt)[keep_mask],
        post_pt=np.asarray(synapses.post_pt)[keep_mask],
        pre_root_id=pre_mapped[keep_mask],
        post_root_id=post_mapped[keep_mask],
        synapse_id=np.asarray(synapses.synapse_id)[keep_mask],
        pre_seg_id=_mask_or_none(synapses.pre_seg_id),
        post_seg_id=_mask_or_none(synapses.post_seg_id),
    )


def _accuracy_from_logits(logits_np, y_np) -> float:
    preds = (logits_np >= 0.0).astype(np.int64)
    return float((preds == y_np.astype(np.int64)).mean())


def _tsv_row(fields: dict) -> str:
    return "\t".join(str(v) for v in fields.values())


def _tsv_header(fields: dict) -> str:
    return "\t".join(fields.keys())


def _normalize_graph_source_args(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve graph-source defaults and enforce leakage guards."""
    from neuronauts.skeleton_graph import validate_skeleton_graph_config

    if getattr(args, "skeleton_version", None) is None:
        args.skeleton_version = int(args.base_version)
    if getattr(args, "graph_source", "agents") == "skeleton":
        try:
            validate_skeleton_graph_config(
                base_version=int(args.base_version),
                target_version=int(args.target_version),
                skeleton_version=int(args.skeleton_version),
                graph_source=str(args.graph_source),
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    return args


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
    elif strategy == "proofread-core":
        try:
            from experiments.root_neighborhood_dataset import (
                build_root_neighborhood_cache,
                sample_proofread_roots,
            )
        except ImportError as exc:
            raise SystemExit(
                "proofread-core dataset building requires optional dependencies "
                "(notably caveclient and pandas). Install the project extras "
                "used for MICrONS data access, then retry."
            ) from exc

        if not getattr(args, "no_em", False):
            print(
                "[proofread-core] using synapse-only cache entries; "
                "--no-em is implied for this strategy."
            )
        if args.box_side_um != 6.0:
            print(
                "[proofread-core] ignoring --box-side-um; "
                "use --proofread-radius-um to control neighborhood size."
            )

        root_ids: list[int]
        if getattr(args, "proofread_roots_tsv", None):
            import csv

            with open(args.proofread_roots_tsv, "r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                if reader.fieldnames is None or "root_id" not in reader.fieldnames:
                    raise SystemExit(
                        "--proofread-roots-tsv must include a 'root_id' column. "
                        f"Found: {reader.fieldnames}"
                    )
                root_ids = [int(row["root_id"]) for row in reader if row.get("root_id")]
            print(
                f"[proofread-core] loaded {len(root_ids)} proofread roots from "
                f"{args.proofread_roots_tsv}"
            )
        else:
            print(
                f"[proofread-core] sampling {args.proofread_n_roots} proofread roots "
                f"at materialization v{args.cave_version} …"
            )
            root_ids = sample_proofread_roots(
                datastack=args.proofread_datastack,
                version=args.cave_version,
                n_roots=args.proofread_n_roots,
                seed=args.seed,
                token=args.cave_token,
                require_dendrite=args.proofread_require_dendrite,
                require_axon=args.proofread_require_axon,
            )

        build_root_neighborhood_cache(
            cache_dir=args.cache_dir,
            datastack=args.proofread_datastack,
            version=args.cave_version,
            root_ids=root_ids,
            radius_um=args.proofread_radius_um,
            mip=2,
            token=args.cave_token,
            anchor_side=args.proofread_anchor_side,
            min_anchor_synapses=args.proofread_min_anchor_synapses,
            max_synapses=args.max_synapses,
            seed=args.seed,
            verbose=True,
            per_root_timeout_s=args.proofread_per_root_timeout_s,
        )
        refreshed = BoxCache(args.cache_dir)
        print(f"\nDone.  {len(refreshed)} usable proofread-core boxes in cache.")
        return 0
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
            cave_version=getattr(args, "cave_version", None),
        )

    records = build_dataset(
        specs,
        cache,
        min_synapses=args.min_synapses,
        max_synapses=args.max_synapses,
        min_positive_pairs=getattr(args, "min_positive_pairs", 0),
        no_em=getattr(args, "no_em", False),
        token=args.cave_token,
        cave_version=getattr(args, "cave_version", None),
        verbose=True,
    )
    print(f"\nDone.  {len(records)} usable boxes in cache.")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> int:  # noqa: C901
    args = _normalize_graph_source_args(args)
    torch = _require_torch()

    from neuronauts.dataset_builder import load_dataset
    from neuronauts.grammar import DEFAULT_PATH_FEATURE_MODE, path_feature_dim
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

    # ── Optional root-ID remap table ─────────────────────────────────────
    if args.base_version != args.target_version and args.root_remap_tsv:
        print(f"Loading precomputed root-ID mapping from {args.root_remap_tsv} …")
        import csv

        mapping: dict[int, int] = {}
        with open(args.root_remap_tsv, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            if reader.fieldnames is None or not {"root_base", "root_target"}.issubset(reader.fieldnames):
                raise SystemExit(
                    "--root-remap-tsv must have columns 'root_base' and 'root_target'. "
                    f"Found: {reader.fieldnames}"
                )
            for row in reader:
                b = int(row["root_base"])
                t = int(row["root_target"])
                mapping[b] = t
        # Attach to the mapper function so it can be used cheaply per box.
        _maybe_map_synapse_roots._root_mapping = mapping  # type: ignore[attr-defined]
        print(
            f"[root-mapping] loaded {len(mapping):,} base→target root IDs "
            f"(v{args.base_version} → v{args.target_version})"
        )
    elif args.base_version != args.target_version:
        print(
            f"[root-mapping] base_version (v{args.base_version}) != target_version "
            f"(v{args.target_version}) but no --root-remap-tsv was provided. "
            "Proceeding without remapping."
        )
    else:
        print(f"[root-mapping] disabled (base_version == target_version == v{args.base_version})")

    # ── Load / create models ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    grammar_path = Path(args.grammar_output)
    if grammar_path.exists() and not args.reset:
        print(f"Resuming grammar from {grammar_path}")
        grammar_model = load_shared_grammar_model(str(grammar_path)).to(device)
        loaded_mode = getattr(grammar_model, "path_feature_mode", DEFAULT_PATH_FEATURE_MODE)
        if getattr(args, "path_feature_mode", loaded_mode) != loaded_mode:
            print(
                f"[W] Ignoring --path-feature-mode={args.path_feature_mode!r}; "
                f"checkpoint expects {loaded_mode!r}."
            )
        args.path_feature_mode = loaded_mode
    else:
        grammar_model = SharedGrammarModel(
            input_dim=path_feature_dim(args.path_feature_mode),
            path_feature_mode=args.path_feature_mode,
        ).to(device)
        print(f"Initialised fresh grammar model ({args.path_feature_mode}).")

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
        min_positive_pairs=getattr(args, "min_positive_pairs", 0),
    )
    if not all_records:
        print(
            f"No cached boxes in {args.cache_dir} with "
            f"{args.min_synapses}–{args.max_synapses} synapses.\n"
            "Run:  python scripts/train.py build-dataset --cache-dir <dir>"
        )
        return 1

    cached_versions = {
        r.root_id_version for r in all_records if getattr(r, "root_id_version", None) is not None
    }
    if cached_versions:
        if len(cached_versions) > 1:
            print(
                f"[W] Cache contains multiple root_id_version values: {sorted(cached_versions)}. "
                "Using --base-version for remapping may mix label spaces."
            )
        if args.base_version not in cached_versions:
            min_ver = min(cached_versions)
            print(
                f"[W] Cached roots are at version(s) {sorted(cached_versions)}, "
                f"but --base-version={args.base_version}. "
                "Set --base-version to match the cache, or rebuild the cache with "
                f"--cave-version={min_ver} (or your desired value)."
            )
    else:
        print(
            "[W] Cache entries do not record root_id_version. "
            "Assuming cached roots correspond to --base-version "
            f"(currently {args.base_version})."
        )

    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_records)  # type: ignore[arg-type]
    n_val = max(1, int(len(all_records) * args.val_fraction))
    val_records = all_records[:n_val]
    train_records = all_records[n_val:]
    print(
        f"Dataset: {len(train_records)} train + {len(val_records)} val boxes "
        f"({sum(r.n_synapses for r in train_records)} train synapses)"
    )
    n_train_with_volume = sum(1 for r in train_records if getattr(r, "has_volume", True))
    n_val_with_volume = sum(1 for r in val_records if getattr(r, "has_volume", True))
    if args.train_gat and getattr(args, "graph_source", "agents") == "agents" and n_train_with_volume == 0:
        print(
            "[W] --train-gat requested but the training cache has no EM volumes. "
            "Disabling GAT training for this run."
        )
        args.train_gat = False
    if getattr(args, "val_sim_every_n", 0) > 0 and getattr(args, "graph_source", "agents") == "agents" and n_val_with_volume == 0:
        print(
            "[W] Slow simulation validation requested but validation cache has no EM volumes. "
            "Disabling --val-sim-every-n for this run."
        )
        args.val_sim_every_n = 0

    # ── Logging ───────────────────────────────────────────────────────────
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_log.tsv"

    best_val_bce = float("inf")   # lower is better; used for checkpointing
    best_val_f1  = -1.0           # tracked when --val-sim-every-n is set
    history: dict[str, list] = {
        k: [] for k in (
            "epoch", "train_merge_acc", "train_topo_acc",
            "train_gat_f1",
            "val_merge_acc", "val_merge_bce", "val_topo_acc", "val_topo_bce",
            "val_f1", "val_precision", "val_recall",
            "val_sampled_f1", "val_sampled_precision", "val_sampled_recall",
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
                volume_chunk, base_synapses = cache.load(record)
                synapses = _maybe_map_synapse_roots(
                    base_synapses,
                    base_version=getattr(args, "base_version", args.target_version),
                    target_version=args.target_version,
                )
            except Exception as exc:
                print(f"  [W] failed to load {record.box_hash}: {exc}")
                continue

            # ── Grammar step (fast: no simulation) ────────────────────────
            merge_batch, topo_batch = _grammar_batch_from_synapses(
                synapses, device,
                path_feature_mode=args.path_feature_mode,
                max_merge=args.max_merge_per_box,
                max_topo=args.max_topo_per_box,
                max_negative_pairs_per_role=getattr(
                    args, "max_negative_pairs_per_role", 64
                ),
                topo_balanced=not getattr(args, "no_topo_balanced", False),
            )
            if merge_batch is not None:
                # Debug: print label balance for merge and topology (first train box each epoch).
                if idx == int(order[0]):
                    y = merge_batch["y"].detach().cpu().numpy()
                    frac_pos = float(y.mean()) if y.size else float("nan")
                    y_topo = topo_batch["y"].detach().cpu().numpy()
                    topo_frac = float(y_topo.mean()) if y_topo.size else float("nan")
                    print(
                        f"  [debug] epoch {epoch}: merge_batch n={y.size} pos_frac={frac_pos:.3f} | "
                        f"topo_batch n={y_topo.size} pos_frac={topo_frac:.3f}"
                    )
                grammar_metrics = multitask_train_step(
                    grammar_model, grammar_optimizer,
                    merge_batch=merge_batch,
                    topology_batch=topo_batch,
                    atomicity_loss_weight=getattr(
                        args, "atomicity_loss_weight", 1.0
                    ),
                )
                epoch_merge_accs.append(grammar_metrics.get("merge_accuracy", 0.0))
                epoch_topo_accs.append(grammar_metrics.get("atomicity_accuracy", 0.0))

            # ── GAT step (slow: needs path simulation) ─────────────────────
            if args.train_gat and epoch % args.gat_every_n_epochs == 0:
                _run_gat_training_step(
                    record, volume_chunk, base_synapses, synapses, gat_model, grammar_model,
                    gat_optimizer, device, args, epoch_gat_f1s,
                )

        # ── Fast validation (every epoch, no simulation) ──────────────────
        grammar_model.eval()
        fast_accs: list[float] = []
        fast_bces: list[float] = []
        fast_topo_accs: list[float] = []
        fast_topo_bces: list[float] = []

        for record in val_records:
            result = _validate_box_fast(record, cache, grammar_model, device, args.path_feature_mode)
            if result is not None:
                fast_accs.append(result["merge_acc"])
                fast_bces.append(result["merge_bce"])
                fast_topo_accs.append(result["topo_acc"])
                fast_topo_bces.append(result["topo_bce"])

        val_merge_acc = float(np.mean(fast_accs)) if fast_accs else 0.0
        val_merge_bce = float(np.mean(fast_bces)) if fast_bces else float("inf")
        val_topo_acc = float(np.mean(fast_topo_accs)) if fast_topo_accs else 0.0
        val_topo_bce = float(np.mean(fast_topo_bces)) if fast_topo_bces else float("inf")

        # ── Slow simulation validation (optional, every N epochs) ─────────
        val_f1 = float("nan")
        val_pre = float("nan")
        val_rec = float("nan")
        val_sampled_f1 = float("nan")
        val_sampled_pre = float("nan")
        val_sampled_rec = float("nan")
        sim_every = getattr(args, "val_sim_every_n", 0)
        if sim_every > 0 and epoch % sim_every == 0:
            val_f1s: list[float] = []
            val_precisions: list[float] = []
            val_recalls: list[float] = []
            val_sampled_f1s: list[float] = []
            val_sampled_precisions: list[float] = []
            val_sampled_recalls: list[float] = []
            for record in val_records:
                m, diag = _validate_box(
                    record, cache, grammar_model, gat_model, args, device
                )
                if m is not None:
                    val_f1s.append(m.f1)
                    val_precisions.append(m.precision)
                    val_recalls.append(m.recall)
                    if "sampled_f1" in diag:
                        val_sampled_f1s.append(float(diag["sampled_f1"]))
                        val_sampled_precisions.append(float(diag["sampled_precision"]))
                        val_sampled_recalls.append(float(diag["sampled_recall"]))
                    if epoch == sim_every:  # first simulation pass
                        print(
                            f"  val box {record.box_hash[:8]}: "
                            f"F1={m.f1:.3f} P={m.precision:.3f} R={m.recall:.3f} "
                            f"sampled_F1={diag.get('sampled_f1', float('nan')):.3f} "
                            f"true_e={m.n_true_edges} est_e={m.n_estimated_edges} "
                            f"syn={diag.get('n_synapses',0)} "
                            f"cands={diag.get('n_merge_candidates',0)} "
                            f"accepted={diag.get('n_merge_accepted',0)} "
                            f"mean_score={diag.get('mean_score', float('nan')):.3f}"
                        )
            val_f1  = float(np.mean(val_f1s))        if val_f1s        else 0.0
            val_pre = float(np.mean(val_precisions))  if val_precisions else 0.0
            val_rec = float(np.mean(val_recalls))     if val_recalls    else 0.0
            val_sampled_f1 = float(np.mean(val_sampled_f1s)) if val_sampled_f1s else 0.0
            val_sampled_pre = float(np.mean(val_sampled_precisions)) if val_sampled_precisions else 0.0
            val_sampled_rec = float(np.mean(val_sampled_recalls)) if val_sampled_recalls else 0.0
            if val_f1 >= best_val_f1:
                best_val_f1 = val_f1

        # ── Checkpoint on best val BCE (lower = better) ───────────────────
        if val_merge_bce <= best_val_bce:
            best_val_bce = val_merge_bce
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
            "val_merge_acc":   f"{val_merge_acc:.4f}",
            "val_merge_bce":   f"{val_merge_bce:.4f}",
            "val_topo_acc":    f"{val_topo_acc:.4f}",
            "val_topo_bce":    f"{val_topo_bce:.4f}",
            "val_f1":          f"{val_f1:.4f}" if not np.isnan(val_f1) else "n/a",
            "val_precision":   f"{val_pre:.4f}" if not np.isnan(val_pre) else "n/a",
            "val_recall":      f"{val_rec:.4f}" if not np.isnan(val_rec) else "n/a",
            "val_sampled_f1":  f"{val_sampled_f1:.4f}" if not np.isnan(val_sampled_f1) else "n/a",
            "val_sampled_precision": f"{val_sampled_pre:.4f}" if not np.isnan(val_sampled_pre) else "n/a",
            "val_sampled_recall":    f"{val_sampled_rec:.4f}" if not np.isnan(val_sampled_rec) else "n/a",
            "best_val_bce":    f"{best_val_bce:.4f}",
            "elapsed_s":       f"{time.time() - t_epoch:.1f}",
        }
        if epoch == 1 and not log_path.exists():
            log_path.write_text(_tsv_header(row) + "\n", encoding="utf-8")
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(_tsv_row(row) + "\n")

        sim_suffix = (
            f" val_f1={val_f1:.4f} sampled_f1={val_sampled_f1:.4f} (best={best_val_f1:.4f})"
            if not np.isnan(val_f1) else ""
        )
        print(
            f"epoch {epoch:3d}/{args.epochs} | "
            f"merge_acc={row['train_merge_acc']} "
            f"topo_acc={row['train_topo_acc']} "
            f"gat_f1={row['train_gat_f1']} | "
            f"val_merge_acc={val_merge_acc:.4f} "
            f"val_bce={val_merge_bce:.4f} "
            f"val_topo_acc={val_topo_acc:.4f} "
            f"val_topo_bce={val_topo_bce:.4f} "
            f"(best={best_val_bce:.4f})"
            f"{sim_suffix} | "
            f"{time.time() - t_epoch:.1f}s"
        )

    print(
        f"\nTraining complete.  Best val BCE = {best_val_bce:.4f}"
        + (f"  Best val F1 = {best_val_f1:.4f}" if best_val_f1 > -1.0 else "")
        + f"\n  Grammar → {grammar_path}\n"
        + (f"  GAT     → {args.gat_output}\n" if args.train_gat else "")
        + f"  Log     → {log_path}"
    )
    return 0


def _run_gat_training_step(
    record, volume_chunk, base_synapses, label_synapses, gat_model, grammar_model,
    gat_optimizer, device, args, gat_f1_acc: list,
):
    """Run graph construction → gat_train_step on one box."""
    from neuronauts.fields import compute_membrane_field
    from neuronauts.run import HeuristicConfig, _build_graph, simulate_paths_and_hits
    from neuronauts.skeleton_graph import build_skeleton_connectivity_graph
    from neuronauts.shared_grammar_model import gat_train_step

    try:
        if getattr(args, "graph_source", "agents") == "skeleton":
            graph = build_skeleton_connectivity_graph(
                record.to_spec(),
                base_synapses,
                version=args.skeleton_version,
                datastack=getattr(args, "proofread_datastack", "minnie65_public"),
                cave_server=getattr(args, "cave_server", "https://global.daf-apis.com"),
                token=getattr(args, "cave_token", None),
                skeleton_cache_dir=getattr(args, "skeleton_cache_dir", "cache/skeletons"),
            )
        else:
            mf = compute_membrane_field(volume_chunk.data)
            path_arr, synapse_hits, path_lengths, _ = simulate_paths_and_hits(
                volume_chunk.data,
                base_synapses.pre_pt,
                base_synapses.post_pt,
                verbose=False,
                membrane_field_override=mf,
            )
            graph = _build_graph(
                path_arr=path_arr,
                path_lengths=path_lengths,
                synapse_hits=synapse_hits,
                pre_pts=base_synapses.pre_pt,
                post_pts=base_synapses.post_pt,
                pre_seg_ids=base_synapses.pre_seg_id,
                post_seg_ids=base_synapses.post_seg_id,
                heuristic_config=HeuristicConfig.learned(),
            )
        if not graph.edges:
            return

        m = gat_train_step(
            gat_model, grammar_model.path_encoder, gat_optimizer,
            graph=graph,
            pre_root_ids=label_synapses.pre_root_id,
            post_root_ids=label_synapses.post_root_id,
            soft_f1_weight=args.gat_soft_f1_weight,
        )
        if m["n_edges"] > 0:
            gat_f1_acc.append(m["pred_f1"])
    except Exception as exc:
        print(f"  [W] GAT step failed: {exc}")


def _validate_box_fast(record, cache, grammar_model, device, path_feature_mode):
    """Fast no-simulation validation: build grammar batches from cached synapses
    and compute grammar-task validation metrics.

    Returns a dict with keys ``merge_acc``, ``merge_bce``, ``topo_acc``,
    ``topo_bce``, ``n_pairs``, ``n_topo``, or ``None`` if the box has too few
    synapses to produce examples.
    """
    import torch
    import torch.nn.functional as F

    try:
        _volume_chunk, synapses = cache.load(record)
        merge_batch, topo_batch = _grammar_batch_from_synapses(
            synapses,
            device,
            path_feature_mode=path_feature_mode,
        )
        if merge_batch is None:
            return None

        grammar_model.eval()
        with torch.no_grad():
            logits = grammar_model.score_merge(
                merge_batch["left_x"],
                merge_batch["left_mask"],
                merge_batch["right_x"],
                merge_batch["right_mask"],
            )
            y = merge_batch["y"]
            bce = float(F.binary_cross_entropy_with_logits(logits, y).item())
            acc = _accuracy_from_logits(
                logits.cpu().numpy(), y.cpu().numpy()
            )
            topo_logits = grammar_model.score_atomicity(
                topo_batch["branch_x"],
                topo_batch["branch_sequence_mask"],
                topo_batch["branch_mask"],
            )
            y_topo = topo_batch["y"]
            topo_bce = float(F.binary_cross_entropy_with_logits(topo_logits, y_topo).item())
            topo_acc = _accuracy_from_logits(
                topo_logits.cpu().numpy(), y_topo.cpu().numpy()
            )
        return {
            "merge_acc": acc,
            "merge_bce": bce,
            "topo_acc": topo_acc,
            "topo_bce": topo_bce,
            "n_pairs": int(y.shape[0]),
            "n_topo": int(y_topo.shape[0]),
        }
    except Exception as exc:
        print(f"  [W] fast val failed for {record.box_hash}: {exc}")
        return None


def _make_live_merge_score_fn(grammar_model):
    """Build a merge-score closure that calls the live in-memory model.

    Unlike ``_load_shared_merge_score_fn`` (which is lru_cache'd and reads from
    disk), this always uses the current model weights so validation tracks the
    model as it trains.
    """
    import torch

    grammar_model.eval()
    path_feature_mode = getattr(grammar_model, "path_feature_mode", "raw_delta3+skeleton")

    def score_fn(left_sequence: np.ndarray, right_sequence: np.ndarray) -> float:
        left = torch.from_numpy(left_sequence[None, ...]).float()
        right = torch.from_numpy(right_sequence[None, ...]).float()
        left_mask = torch.zeros((1, left.shape[1]), dtype=torch.bool)
        right_mask = torch.zeros((1, right.shape[1]), dtype=torch.bool)
        with torch.no_grad():
            logits = grammar_model.score_merge(left, left_mask, right, right_mask)
        return float(logits.squeeze().cpu())

    score_fn.path_feature_mode = path_feature_mode
    return score_fn


def _validate_box(record, cache, grammar_model, gat_model, args, device):
    """Run full inference on one validation box.

    Returns ``(LineGraphMetrics | None, diag_dict)`` where ``diag_dict``
    contains diagnostic counters useful for debugging F1 stagnation.
    """
    import torch
    from neuronauts.fields import compute_membrane_field
    from neuronauts.run import HeuristicConfig, _build_graph, simulate_paths_and_hits
    from neuronauts.line_graph import evaluate, evaluate_sampled

    diag: dict = {}
    try:
        volume_chunk, base_synapses = cache.load(record)
        if len(base_synapses.pre_pt) < 5:
            return None, diag
        label_synapses = _maybe_map_synapse_roots(
            base_synapses,
            base_version=getattr(args, "base_version", args.target_version),
            target_version=args.target_version,
        )
        if (
            getattr(args, "graph_source", "agents") == "agents"
            and (getattr(volume_chunk, "data", None) is None or volume_chunk.data.size == 0)
        ):
            diag["skipped_no_volume"] = True
            return None, diag

        if getattr(args, "graph_source", "agents") == "skeleton":
            from neuronauts.skeleton_graph import build_skeleton_connectivity_graph

            graph = build_skeleton_connectivity_graph(
                record.to_spec(),
                base_synapses,
                version=args.skeleton_version,
                datastack=getattr(args, "proofread_datastack", "minnie65_public"),
                cave_server=getattr(args, "cave_server", "https://global.daf-apis.com"),
                token=getattr(args, "cave_token", None),
                skeleton_cache_dir=getattr(args, "skeleton_cache_dir", "cache/skeletons"),
            )
            diag["n_merge_candidates"] = len(graph.edges)
            diag["n_merge_accepted"] = len(graph.edges)
            diag["mean_score"] = float("nan")
            diag["n_synapses"] = len(base_synapses.pre_pt)
        else:
            mf = compute_membrane_field(volume_chunk.data)
            path_arr, synapse_hits, path_lengths, _ = simulate_paths_and_hits(
                volume_chunk.data,
                base_synapses.pre_pt,
                base_synapses.post_pt,
                verbose=False,
                membrane_field_override=mf,
            )

            _base_score_fn = _make_live_merge_score_fn(grammar_model)
            hcfg = HeuristicConfig.learned()
            _n_candidates = [0]
            _n_accepted = [0]
            _scores: list[float] = []

            def _counting_score_fn(left_seq, right_seq):
                s = _base_score_fn(left_seq, right_seq)
                _n_candidates[0] += 1
                _scores.append(s)
                if s >= 0.0:
                    _n_accepted[0] += 1
                return s

            graph = _build_graph(
                path_arr=path_arr,
                path_lengths=path_lengths,
                synapse_hits=synapse_hits,
                pre_pts=base_synapses.pre_pt,
                post_pts=base_synapses.post_pt,
                pre_seg_ids=base_synapses.pre_seg_id,
                post_seg_ids=base_synapses.post_seg_id,
                learned_merge_score_fn=_counting_score_fn,
                heuristic_config=hcfg,
            )

            diag["n_merge_candidates"] = _n_candidates[0]
            diag["n_merge_accepted"]   = _n_accepted[0]
            diag["mean_score"] = float(np.mean(_scores)) if _scores else float("nan")
            diag["n_synapses"] = len(base_synapses.pre_pt)

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
        sampled = evaluate_sampled(
            graph,
            label_synapses.pre_root_id,
            label_synapses.post_root_id,
            max_pairs=args.val_sampled_max_pairs,
            seed=args.seed,
        )
        diag["sampled_f1"] = sampled.f1
        diag["sampled_precision"] = sampled.precision
        diag["sampled_recall"] = sampled.recall

        return evaluate(graph, label_synapses.pre_root_id, label_synapses.post_root_id), diag

    except Exception as exc:
        print(f"  [W] validation failed for {record.box_hash}: {exc}")
        return None, diag


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
        "--max-synapses", type=int, default=20000,
        help="Maximum synapse pairs per box (default 20000; was 200 for 6µm boxes).",
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
        choices=["synapse-seeded", "random", "proofread-core"],
        help=(
            "Box selection strategy.  'synapse-seeded' (default) queries CAVE "
            "for real synapse positions and uses those as box centres — every "
            "box is guaranteed to be inside annotated neuropil.  'random' "
            "uniformly samples the full dataset extent and skips empty boxes "
            "(expect many 'skip' messages).  'proofread-core' samples "
            "proofread anchor roots and caches their local synapse "
            "neighborhoods for grammar training."
        ),
    )
    parser.add_argument(
        "--no-em", action="store_true",
        help=(
            "Skip EM volume fetch — store only the CAVE synapse table.  "
            "Grammar training requires only synapse geometry and root IDs, "
            "so this is safe for all grammar-only workflows and makes data "
            "collection ~10× faster.  Recommended when using larger boxes "
            "(--box-side-um 30+)."
        ),
    )
    parser.add_argument(
        "--min-positive-pairs", type=int, default=0,
        help=(
            "Minimum number of same-root-id synapse pairs required per box.  "
            "Boxes below this threshold have almost no positive training "
            "examples; filtering them out improves training quality.  "
            "Recommended: 5 for 30 µm boxes, 2 for 15 µm boxes."
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
    parser.add_argument(
        "--cave-version",
        type=int,
        default=1412,
        help="Materialization version used when caching synapse/root IDs.",
    )
    parser.add_argument(
        "--proofread-datastack",
        default="minnie65_public",
        help="Datastack used when strategy=proofread-core.",
    )
    parser.add_argument(
        "--proofread-n-roots",
        type=int,
        default=25,
        help="Number of proofread anchor roots to sample when strategy=proofread-core.",
    )
    parser.add_argument(
        "--proofread-roots-tsv",
        default=None,
        help="Optional TSV with a root_id column to seed strategy=proofread-core.",
    )
    parser.add_argument(
        "--proofread-radius-um",
        type=float,
        default=40.0,
        help="Neighborhood radius in microns when strategy=proofread-core.",
    )
    parser.add_argument(
        "--proofread-anchor-side",
        choices=["both", "pre", "post"],
        default="both",
        help="Keep anchor-root synapses on both sides, or only pre/post, for strategy=proofread-core.",
    )
    parser.add_argument(
        "--proofread-min-anchor-synapses",
        type=int,
        default=50,
        help="Minimum anchor-root synapses required per neighborhood for strategy=proofread-core.",
    )
    parser.add_argument(
        "--proofread-per-root-timeout-s",
        type=int,
        default=180,
        help="Per-root synapse-fetch timeout in seconds for strategy=proofread-core.",
    )
    parser.add_argument(
        "--proofread-require-dendrite",
        action="store_true",
        default=True,
        help="Require dendrite proofreading when sampling roots for strategy=proofread-core.",
    )
    parser.add_argument(
        "--no-proofread-require-dendrite",
        dest="proofread_require_dendrite",
        action="store_false",
        help="Disable dendrite-proofread filtering for strategy=proofread-core.",
    )
    parser.add_argument(
        "--proofread-require-axon",
        action="store_true",
        default=False,
        help="Also require axon proofreading when sampling roots for strategy=proofread-core.",
    )


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    _add_common_args(parser)
    parser.add_argument(
        "--graph-source",
        default="agents",
        choices=["agents", "skeleton"],
        help="Source used to build connectivity graphs for GAT/slow validation.",
    )
    parser.add_argument(
        "--skeleton-version",
        type=int,
        default=None,
        help=(
            "Materialization version used when fetching skeletons for "
            "--graph-source skeleton. Defaults to --base-version and must match it "
            "to avoid target-label leakage."
        ),
    )
    parser.add_argument(
        "--skeleton-cache-dir",
        default="cache/skeletons",
        help="On-disk cache for fetched base-materialization skeletons.",
    )
    parser.add_argument(
        "--path-feature-mode",
        default="raw_delta3+skeleton",
        choices=["legacy_geom3", "raw_delta3", "raw_delta3+skeleton"],
        help="Per-step path representation used by grammar training and GAT node encoding.",
    )
    parser.add_argument(
        "--min-positive-pairs", type=int, default=0,
        help="Only train on boxes with at least this many same-root-id pairs.",
    )
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
        "--max-negative-pairs-per-role", type=int, default=64,
        help=(
            "Max negative (non-atomic) pairs per pre/post role when building "
            "topology examples. Higher values improve class balance (default 64)."
        ),
    )
    parser.add_argument(
        "--no-topo-balanced", action="store_true",
        help="Disable stratified topology sampling (use raw pos/neg ratio).",
    )
    parser.add_argument(
        "--atomicity-loss-weight", type=float, default=1.0,
        help="Weight for atomicity/topology loss relative to merge loss.",
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
        "--val-sim-every-n", type=int, default=0,
        help=(
            "Run the slow agent-simulation validation every N epochs and log "
            "val_f1.  0 (default) disables simulation validation entirely — "
            "the primary metric is then val_merge_bce (pairwise BCE on held-out "
            "synapse pairs, no simulation needed)."
        ),
    )
    parser.add_argument(
        "--val-sampled-max-pairs", type=int, default=10000,
        help="Number of sampled synapse pairs for the sampled-pair validation F1 diagnostic.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Ignore existing checkpoints and start from scratch.",
    )
    parser.add_argument(
        "--log-dir", default="run_logs",
        help="Directory for training TSV log.",
    )
    parser.add_argument(
        "--base-version",
        type=int,
        default=1412,
        help=(
            "Materialization version where cached box root IDs are defined. "
            "If this matches --target-version (default), no root-ID remapping "
            "is performed."
        ),
    )
    parser.add_argument(
        "--target-version",
        type=int,
        default=1412,
        help=(
            "Materialization version to map root IDs into for supervision. "
            "When different from --base-version, root IDs are mapped forward "
            "via chunkedgraph.get_latest_roots and synapses whose mapped pre "
            "or post root is 0 (vanished bodies) are dropped."
        ),
    )
    parser.add_argument(
        "--root-remap-tsv",
        type=str,
        default=None,
        help=(
            "Optional TSV with two columns 'root_base' and 'root_target' giving a "
            "precomputed root-ID mapping from --base-version to --target-version. "
            "When provided and base_version != target_version, this mapping is "
            "used during training instead of live chunkedgraph calls."
        ),
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
    p_run.add_argument("--strategy",            default="synapse-seeded", choices=["synapse-seeded", "random", "proofread-core"])
    p_run.add_argument("--no-em",               action="store_true")
    p_run.add_argument("--min-positive-pairs",  type=int, default=0)
    p_run.add_argument("--counts-tsv",          default=None)
    p_run.add_argument("--nucleus-csv",         default=None)
    p_run.add_argument("--cave-token",          default=None)
    p_run.add_argument("--cave-version",       type=int, default=1412)
    p_run.add_argument("--proofread-datastack", default="minnie65_public")
    p_run.add_argument("--proofread-n-roots",   type=int, default=25)
    p_run.add_argument("--proofread-roots-tsv", default=None)
    p_run.add_argument("--proofread-radius-um", type=float, default=40.0)
    p_run.add_argument("--proofread-anchor-side", choices=["both", "pre", "post"], default="both")
    p_run.add_argument("--proofread-min-anchor-synapses", type=int, default=50)
    p_run.add_argument("--proofread-per-root-timeout-s", type=int, default=180)
    p_run.add_argument("--proofread-require-dendrite", action="store_true", default=True)
    p_run.add_argument("--no-proofread-require-dendrite", dest="proofread_require_dendrite", action="store_false")
    p_run.add_argument("--proofread-require-axon", action="store_true", default=False)
    # train-specific
    p_run.add_argument("--grammar-output",      default="models/shared_grammar_real.pt")
    p_run.add_argument("--graph-source",        default="agents", choices=["agents", "skeleton"])
    p_run.add_argument("--skeleton-version",    type=int, default=None)
    p_run.add_argument("--skeleton-cache-dir",  default="cache/skeletons")
    p_run.add_argument("--path-feature-mode",   default="raw_delta3+skeleton", choices=["legacy_geom3", "raw_delta3", "raw_delta3+skeleton"])
    p_run.add_argument("--gat-output",          default="models/gat_real.pt")
    p_run.add_argument("--epochs",              type=int,   default=30)
    p_run.add_argument("--lr",                  type=float, default=3e-4)
    p_run.add_argument("--val-fraction",        type=float, default=0.15)
    p_run.add_argument("--max-merge-per-box",   type=int,   default=256)
    p_run.add_argument("--max-topo-per-box",    type=int,   default=128)
    p_run.add_argument("--max-negative-pairs-per-role", type=int, default=64)
    p_run.add_argument("--no-topo-balanced",    action="store_true")
    p_run.add_argument("--atomicity-loss-weight", type=float, default=1.0)
    p_run.add_argument("--train-gat",           action="store_true")
    p_run.add_argument("--gat-every-n-epochs",  type=int,   default=5)
    p_run.add_argument("--gat-soft-f1-weight",  type=float, default=0.5)
    p_run.add_argument("--gat-edge-threshold",  type=float, default=0.5)
    p_run.add_argument("--val-sim-every-n",     type=int,   default=0)
    p_run.add_argument("--val-sampled-max-pairs", type=int, default=10000)
    p_run.add_argument("--reset",               action="store_true")
    p_run.add_argument("--log-dir",             default="run_logs")
    p_run.set_defaults(func=cmd_run)

    # remap-roots: build a precomputed root-ID mapping TSV for a cache
    p_remap = sub.add_parser(
        "remap-roots",
        help=(
            "Scan a cached dataset, collect all unique root IDs, map them from "
            "--base-version to --target-version via chunkedgraph, and write a "
            "TSV mapping table (root_base, root_target)."
        ),
    )
    _add_common_args(p_remap)
    p_remap.add_argument(
        "--base-version",
        type=int,
        required=True,
        help="Materialization version where cached root IDs are valid.",
    )
    p_remap.add_argument(
        "--target-version",
        type=int,
        required=True,
        help="Target materialization version to map roots into.",
    )
    p_remap.add_argument(
        "--output",
        type=str,
        default="root_remap.tsv",
        help="Output TSV path for the base→target root-ID mapping.",
    )
    p_remap.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress after each mapping batch (roots/s, elapsed).",
    )
    p_remap.set_defaults(func=cmd_remap_roots)

    # remap-cache-roots: write a new cache whose synapse root IDs match target_version
    p_remap_cache = sub.add_parser(
        "remap-cache-roots",
        help=(
            "Remap root IDs inside a cached dataset using a precomputed "
            "--root-remap-tsv table, drop synapses that vanish at the target, "
            "and recompute per-box n_positive_pairs. Writes a new cache."
        ),
    )
    _add_common_args(p_remap_cache)
    p_remap_cache.add_argument(
        "--out-cache-dir",
        type=str,
        required=True,
        help="Destination cache directory for the remapped dataset.",
    )
    p_remap_cache.add_argument(
        "--base-version",
        type=int,
        required=True,
        help="Materialization version where cached root IDs live.",
    )
    p_remap_cache.add_argument(
        "--target-version",
        type=int,
        required=True,
        help="Materialization version to map roots into.",
    )
    p_remap_cache.add_argument(
        "--root-remap-tsv",
        type=str,
        required=True,
        help="TSV mapping table (columns: root_base, root_target).",
    )
    p_remap_cache.set_defaults(func=cmd_remap_cache_roots)

    # analyze-root-remap: compute stats from a root remap TSV
    p_analyze = sub.add_parser(
        "analyze-root-remap",
        help="Compute survival/churn and in-degree (merge size) stats from a root remap TSV.",
    )
    p_analyze.add_argument(
        "--root-remap-tsv",
        type=str,
        required=True,
        help="TSV mapping table (columns: root_base, root_target).",
    )
    p_analyze.add_argument(
        "--top-k-targets",
        type=int,
        default=20,
        help="Number of largest in-degree target roots to print.",
    )
    p_analyze.add_argument(
        "--max-hist-k",
        type=int,
        default=50,
        help="Max in_degree value to include in the printed histogram.",
    )
    p_analyze.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write JSON summary stats.",
    )
    p_analyze.set_defaults(func=cmd_analyze_root_remap)

    return root.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return args.func(args)


def cmd_remap_roots(args: argparse.Namespace) -> int:
    """Build a precomputed root-ID mapping TSV for a cached dataset.

    This scans all cached boxes in ``--cache-dir``, collects unique root IDs
    from their synapse tables, maps those roots from ``--base-version`` into
    ``--target-version`` via the CAVE chunkedgraph, and writes a TSV with
    columns ``root_base`` and ``root_target``.
    """
    from neuronauts.dataset_builder import BoxCache
    from neuronauts.cave_root_mapping import map_roots_between_versions

    cache = BoxCache(args.cache_dir)
    records = cache.all_records()
    if not records:
        print(f"No cached boxes found in {args.cache_dir}")
        return 1

    print(f"Scanning {len(records)} cached boxes in {args.cache_dir} …")
    all_roots: set[int] = set()
    for rec in records:
        try:
            _vol, synapses = cache.load(rec)
        except Exception as exc:
            print(f"  [W] failed to load {rec.box_hash}: {exc}")
            continue
        pre = np.asarray(synapses.pre_root_id, dtype=np.int64)
        post = np.asarray(synapses.post_root_id, dtype=np.int64)
        all_roots.update(pre.tolist())
        all_roots.update(post.tolist())

    if not all_roots:
        print("No root IDs found in cached synapse tables.")
        return 1

    print(f"Collected {len(all_roots):,} unique root IDs at base v{args.base_version}.")
    mapping = map_roots_between_versions(
        root_ids=all_roots,
        old_version=args.base_version,
        new_version=args.target_version,
        verbose=getattr(args, "verbose", False),
    )

    # Basic stats
    n_total = len(mapping)
    n_same = sum(1 for b, t in mapping.items() if b == t and t != 0)
    n_changed = sum(1 for b, t in mapping.items() if t != 0 and b != t)
    n_gone = sum(1 for _b, t in mapping.items() if t == 0)
    print(
        f"[root-mapping] v{args.base_version} → v{args.target_version}: "
        f"{n_total:,} roots, {n_same:,} same, {n_changed:,} changed, {n_gone:,} gone"
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("root_base\troot_target\n")
        for b, t in mapping.items():
            fh.write(f"{b}\t{t}\n")

    print(f"Wrote root-ID mapping to {out_path}")
    return 0


def cmd_analyze_root_remap(args: argparse.Namespace) -> int:
    """Analyze a precomputed root remap TSV (root_base -> root_target).

    This is intended to understand the "merge graph" induced by remapping
    roots between materializations, without calling CAVE at runtime.
    """
    import csv
    import heapq
    from collections import Counter

    path = Path(args.root_remap_tsv)
    if not path.exists():
        raise SystemExit(f"--root-remap-tsv not found: {path}")

    n_rows = 0
    n_gone = 0
    n_survived = 0
    n_unchanged = 0
    n_changed = 0

    # target_id -> number of base roots mapping into it (in-degree / merge-size)
    target_counts: dict[int, int] = {}

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None or not {"root_base", "root_target"}.issubset(reader.fieldnames):
            raise SystemExit(
                f"TSV must include columns 'root_base' and 'root_target'. Found: {reader.fieldnames}"
            )

        for row in reader:
            b = int(row["root_base"])
            t = int(row["root_target"])
            n_rows += 1

            if t == 0:
                n_gone += 1
                continue

            n_survived += 1
            if b == t:
                n_unchanged += 1
            else:
                n_changed += 1

            target_counts[t] = target_counts.get(t, 0) + 1

    n_unique_surviving_targets = len(target_counts)
    if n_unique_surviving_targets == 0:
        print("No surviving targets (all root_target are 0).")
        return 1

    in_degree_hist: Counter[int] = Counter(target_counts.values())
    max_in_degree = max(in_degree_hist)

    top_targets = heapq.nlargest(args.top_k_targets, target_counts.items(), key=lambda kv: kv[1])
    top_target_captured_bases = sum(c for _t, c in top_targets)

    def _pct(x: int, denom: int) -> float:
        return (x / denom) if denom > 0 else 0.0

    print(f"Loaded {path}")
    print(f"Total base roots (rows): {n_rows:,}")
    print(
        "Survival / churn: "
        f"survived={n_survived:,} ({_pct(n_survived, n_rows):.2%}), "
        f"gone={n_gone:,} ({_pct(n_gone, n_rows):.2%})"
    )
    print(
        f"Unchanged={n_unchanged:,} ({_pct(n_unchanged, n_rows):.2%}), "
        f"changed={n_changed:,} ({_pct(n_changed, n_rows):.2%})"
    )
    print(
        f"Surviving target roots: {n_unique_surviving_targets:,}; "
        f"avg in-degree={n_survived / n_unique_surviving_targets:.3f}"
    )
    print(f"Merge distribution (in-degree = base-roots per surviving target): max={max_in_degree:,}")

    # Print a compact histogram for the lower part of the distribution.
    max_k = max(1, min(int(args.max_hist_k), max_in_degree))
    print(f"Histogram for in_degree=1..{max_k}:")
    for k in range(1, max_k + 1):
        if k not in in_degree_hist:
            continue
        n_targets_k = in_degree_hist[k]
        frac_targets_k = n_targets_k / n_unique_surviving_targets
        print(f"  in_degree={k}: {n_targets_k:,} targets ({frac_targets_k:.2%})")

    if max_in_degree > max_k:
        n_tail_targets = sum(cnt for deg, cnt in in_degree_hist.items() if deg > max_k)
        print(f"  ... plus in_degree>{max_k}: {n_tail_targets:,} targets")

    print(f"Top-{args.top_k_targets} largest merges (target_id -> in_degree):")
    for t, c in top_targets:
        print(f"  {t} -> {c:,}")
    print(
        f"Roots captured by top-{args.top_k_targets} targets: "
        f"{top_target_captured_bases:,} / {n_survived:,} ({_pct(top_target_captured_bases, n_survived):.2%})"
    )

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        stats = {
            "root_remap_tsv": str(path),
            "n_mapping_rows": n_rows,
            "n_gone": n_gone,
            "n_survived": n_survived,
            "n_unchanged": n_unchanged,
            "n_changed": n_changed,
            "n_unique_surviving_targets": n_unique_surviving_targets,
            "avg_in_degree_on_survivors": n_survived / n_unique_surviving_targets,
            "max_in_degree": max_in_degree,
            "top_targets": [{"root_target": t, "in_degree": c} for (t, c) in top_targets],
        }
        with json_path.open("w", encoding="utf-8") as out_fh:
            json.dump(stats, out_fh, indent=2)
        print(f"Wrote JSON stats to {json_path}")

    return 0


def _remap_root_array(arr: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Fast remap using unique + searchsorted (avoids per-element Python lookups)."""
    # np.unique sorts, which lets searchsorted work.
    roots = np.unique(arr.astype(np.int64, copy=False))
    mapped_roots = np.array([mapping.get(int(r), 0) for r in roots], dtype=np.int64)
    idx = np.searchsorted(roots, arr.astype(np.int64, copy=False))
    return mapped_roots[idx]


def cmd_remap_cache_roots(args: argparse.Namespace) -> int:
    """Remap a cached dataset and write a new cache with updated labels.

    This is intended for the case where you changed label spaces via root-ID
    remapping and want per-box derived supervision (n_positive_pairs and
    cached synapse root arrays) to reflect the target materialization.
    """
    from neuronauts.dataset_builder import BoxCache, count_positive_pairs

    cache_in = BoxCache(args.cache_dir)
    cache_out = BoxCache(args.out_cache_dir)

    records = cache_in.all_records()
    if not records:
        print(f"No cached boxes found in {args.cache_dir}")
        return 1

    # Load mapping table.
    import csv

    mapping: dict[int, int] = {}
    with open(args.root_remap_tsv, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None or not {"root_base", "root_target"}.issubset(reader.fieldnames):
            raise SystemExit(
                "--root-remap-tsv must include columns 'root_base' and 'root_target'. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            b = int(row["root_base"])
            t = int(row["root_target"])
            mapping[b] = t
    print(
        f"Remapping cache roots using {len(mapping):,} root_base→root_target entries "
        f"(v{args.base_version} → v{args.target_version})…"
    )

    n_written = 0
    n_skipped = 0
    for rec in records:
        # If already written, skip.
        if cache_out.contains(rec.to_spec()):
            n_skipped += 1
            continue

        try:
            volume_chunk, synapses = cache_in.load(rec)
        except Exception as exc:
            print(f"  [W] failed to load {rec.box_hash}: {exc}")
            continue

        pre_arr = np.asarray(synapses.pre_root_id, dtype=np.int64)
        post_arr = np.asarray(synapses.post_root_id, dtype=np.int64)

        pre_mapped = _remap_root_array(pre_arr, mapping)
        post_mapped = _remap_root_array(post_arr, mapping)
        keep_mask = (pre_mapped != 0) & (post_mapped != 0)

        if not np.any(keep_mask):
            print(f"  [I] box {rec.box_hash[:8]} has no synapses after remap; skipping")
            continue

        # Mask synapse fields consistently.
        pre_pt = np.asarray(synapses.pre_pt)[keep_mask]
        post_pt = np.asarray(synapses.post_pt)[keep_mask]
        syn_id = np.asarray(synapses.synapse_id)[keep_mask]

        def _mask_or_none(arr):
            if arr is None:
                return None
            return np.asarray(arr)[keep_mask]

        synapses_mapped = synapses.__class__(
            pre_pt=pre_pt,
            post_pt=post_pt,
            pre_root_id=pre_mapped[keep_mask],
            post_root_id=post_mapped[keep_mask],
            synapse_id=syn_id,
            pre_seg_id=_mask_or_none(synapses.pre_seg_id),
            post_seg_id=_mask_or_none(synapses.post_seg_id),
        )

        n_pos = count_positive_pairs(synapses_mapped)

        spec = rec.to_spec()
        if rec.has_volume:
            cache_out.save(
                spec,
                volume_chunk,
                synapses_mapped,
                n_positive_pairs=n_pos,
                root_id_version=args.target_version,
            )
        else:
            cache_out.save_synapse_only(
                spec,
                synapses_mapped,
                n_positive_pairs=n_pos,
                root_id_version=args.target_version,
            )

        n_written += 1
        if n_written % 10 == 0:
            print(f"  remapped {n_written} boxes…")

    print(
        f"Remap-cache done: wrote {n_written} boxes, skipped {n_skipped} already-present boxes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
