"""Chunkedgraph edit-history pipeline for CellGNN hard training pairs.

Proofreader edits (merges and splits) in the MICrONS chunkedgraph represent
**human-resolved ambiguities** — exactly the hard cases where the CellGNN
needs the most supervision.  This module fetches edit logs and converts them
into positive (merge) and hard-negative (split) synapse pairs.

Usage
-----
Standalone (builds a training-pairs TSV from a cache)::

    python -m neuronauts.edit_history build-pairs \\
        --cache-dir data/proofread \\
        --datastack minnie65_public \\
        --version 1718 \\
        --output run_logs/edit_pairs.tsv

Programmatic::

    from neuronauts.edit_history import (
        fetch_edit_log,
        edits_to_synapse_pairs,
        build_edit_pairs_for_cache,
    )
    pairs = build_edit_pairs_for_cache(cache, version=1718)

The pairs can then be injected into ``cell_graph_train_step`` or grammar
training as high-weight examples.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .dataset_builder import BoxCache, BoxRecord
    from .fetch import SynapseTable


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EditOperation:
    """A single proofreader merge or split operation."""
    operation: str          # "merge" or "split"
    before_root_ids: tuple[int, ...]   # root IDs before the edit
    after_root_ids: tuple[int, ...]    # root IDs after the edit
    timestamp: str | None = None


@dataclass(frozen=True)
class EditPair:
    """A synapse pair derived from a proofreader edit.

    ``label=1`` means the edit confirmed same-cell (merge).
    ``label=0`` means the edit confirmed different-cell (split).
    """
    synapse_i: int
    synapse_j: int
    label: int              # 1 = same cell (merge), 0 = different cell (split)
    role: str               # "pre" or "post"
    source_root_a: int      # root ID of synapse_i at the edit version
    source_root_b: int      # root ID of synapse_j at the edit version
    edit_type: str           # "merge" or "split"


# ---------------------------------------------------------------------------
# CAVE edit-log fetcher
# ---------------------------------------------------------------------------

def fetch_edit_log(
    root_id: int,
    *,
    datastack: str = "minnie65_public",
    version: int = 1718,
    token: str | None = None,
) -> list[EditOperation]:
    """Fetch the edit history for a root ID from the chunkedgraph.

    Returns a list of EditOperation objects representing merges and splits.
    Requires network access to the CAVE API.

    Parameters
    ----------
    root_id : int
        Root ID to query (at the given materialization version).
    datastack : str
        CAVE datastack name.
    version : int
        Materialization version.
    token : str or None
        Optional CAVE auth token.
    """
    try:
        from caveclient import CAVEclient
    except ImportError as exc:
        raise ImportError("pip install caveclient") from exc

    client = CAVEclient(datastack, auth_token=token) if token else CAVEclient(datastack)
    client.version = int(version)

    operations = []
    try:
        changelog = client.chunkedgraph.get_tabular_changelog(root_id)
        if changelog is not None and len(changelog) > 0:
            for _, row in changelog.iterrows():
                op_type = str(row.get("operation_type", row.get("is_merge", "")))
                # Normalize: CAVE returns is_merge=True/False or operation_type
                if op_type in ("True", "true", "1", "merge"):
                    op = "merge"
                elif op_type in ("False", "false", "0", "split"):
                    op = "split"
                else:
                    continue

                before = row.get("before_root_ids", [])
                after = row.get("after_root_ids", [])
                if not isinstance(before, (list, tuple, np.ndarray)):
                    before = [before] if before else []
                if not isinstance(after, (list, tuple, np.ndarray)):
                    after = [after] if after else []

                operations.append(EditOperation(
                    operation=op,
                    before_root_ids=tuple(int(r) for r in before if int(r) > 0),
                    after_root_ids=tuple(int(r) for r in after if int(r) > 0),
                    timestamp=str(row.get("timestamp", None)),
                ))
    except Exception:
        # API may fail for some root IDs; return empty
        pass

    return operations


# ---------------------------------------------------------------------------
# Convert edits to synapse pairs
# ---------------------------------------------------------------------------

def edits_to_synapse_pairs(
    edits: list[EditOperation],
    synapses: "SynapseTable",
    role: str,
) -> list[EditPair]:
    """Convert edit operations to training pairs using synapse root-ID membership.

    For each edit:
    - **Merge**: the before_root_ids were separate but should be one cell.
      Any synapse pair (i, j) where i belonged to one before-root and j to
      another is a positive (same-cell) pair.
    - **Split**: the after_root_ids were one but should be separate.
      Any synapse pair (i, j) where i belongs to one after-root and j to
      another is a hard negative (different-cell) pair.

    Parameters
    ----------
    edits : list of EditOperation
    synapses : SynapseTable
    role : "pre" or "post"
    """
    if role == "pre":
        root_ids = np.asarray(synapses.pre_root_id, dtype=np.int64)
    else:
        root_ids = np.asarray(synapses.post_root_id, dtype=np.int64)

    # Build root -> synapse indices lookup
    root_to_synapses: dict[int, list[int]] = {}
    for i, rid in enumerate(root_ids.tolist()):
        if rid > 0:
            root_to_synapses.setdefault(rid, []).append(i)

    pairs: list[EditPair] = []

    for edit in edits:
        if edit.operation == "merge":
            # Before-roots that were separate are now confirmed same-cell
            groups = [
                root_to_synapses.get(rid, [])
                for rid in edit.before_root_ids
                if rid in root_to_synapses
            ]
            # Cross-group pairs are positive
            for gi in range(len(groups)):
                for gj in range(gi + 1, len(groups)):
                    for si in groups[gi][:10]:  # cap to avoid combinatorial explosion
                        for sj in groups[gj][:10]:
                            pairs.append(EditPair(
                                synapse_i=min(si, sj),
                                synapse_j=max(si, sj),
                                label=1,
                                role=role,
                                source_root_a=int(root_ids[si]),
                                source_root_b=int(root_ids[sj]),
                                edit_type="merge",
                            ))

        elif edit.operation == "split":
            # After-roots were one but should be separate — hard negatives
            groups = [
                root_to_synapses.get(rid, [])
                for rid in edit.after_root_ids
                if rid in root_to_synapses
            ]
            for gi in range(len(groups)):
                for gj in range(gi + 1, len(groups)):
                    for si in groups[gi][:10]:
                        for sj in groups[gj][:10]:
                            pairs.append(EditPair(
                                synapse_i=min(si, sj),
                                synapse_j=max(si, sj),
                                label=0,
                                role=role,
                                source_root_a=int(root_ids[si]),
                                source_root_b=int(root_ids[sj]),
                                edit_type="split",
                            ))

    return pairs


def build_edit_pairs_for_box(
    synapses: "SynapseTable",
    *,
    datastack: str = "minnie65_public",
    version: int = 1718,
    token: str | None = None,
    max_roots_to_query: int = 50,
    seed: int = 42,
) -> list[EditPair]:
    """Fetch edit history for roots in a synapse table and build training pairs.

    Queries the chunkedgraph for edit logs of the most common root IDs in the
    box, then converts to synapse-level training pairs.
    """
    rng = np.random.default_rng(seed)
    all_pairs: list[EditPair] = []

    for role, root_ids in [
        ("pre", synapses.pre_root_id),
        ("post", synapses.post_root_id),
    ]:
        root_arr = np.asarray(root_ids, dtype=np.int64)
        unique_roots = [int(r) for r in np.unique(root_arr) if int(r) > 0]

        # Prioritize roots with more synapses (more likely to have edits)
        from collections import Counter
        counts = Counter(root_arr.tolist())
        unique_roots.sort(key=lambda r: counts.get(r, 0), reverse=True)

        if len(unique_roots) > max_roots_to_query:
            unique_roots = unique_roots[:max_roots_to_query]

        for rid in unique_roots:
            edits = fetch_edit_log(
                rid,
                datastack=datastack,
                version=version,
                token=token,
            )
            if edits:
                pairs = edits_to_synapse_pairs(edits, synapses, role)
                all_pairs.extend(pairs)

    return all_pairs


def build_edit_pairs_for_cache(
    cache: "BoxCache",
    *,
    datastack: str = "minnie65_public",
    version: int = 1718,
    token: str | None = None,
    max_roots_per_box: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> list[EditPair]:
    """Build edit-derived training pairs across all boxes in a cache.

    Parameters
    ----------
    cache : BoxCache
    datastack : CAVE datastack
    version : materialization version
    token : optional CAVE auth token
    max_roots_per_box : max roots to query per box
    seed : RNG seed
    verbose : print progress

    Returns
    -------
    list of EditPair across all boxes.
    """
    all_pairs: list[EditPair] = []
    records = cache.all_records()

    for i, rec in enumerate(records):
        try:
            _, synapses = cache.load(rec)
        except Exception:
            continue

        if verbose:
            print(f"  [{i+1}/{len(records)}] {rec.box_hash[:8]} ({rec.n_synapses} syn) …", end=" ", flush=True)

        pairs = build_edit_pairs_for_box(
            synapses,
            datastack=datastack,
            version=version,
            token=token,
            max_roots_to_query=max_roots_per_box,
            seed=seed + i,
        )
        all_pairs.extend(pairs)

        if verbose:
            n_pos = sum(1 for p in pairs if p.label == 1)
            n_neg = sum(1 for p in pairs if p.label == 0)
            print(f"+{n_pos} merge, -{n_neg} split pairs")

    if verbose:
        total_pos = sum(1 for p in all_pairs if p.label == 1)
        total_neg = sum(1 for p in all_pairs if p.label == 0)
        print(f"Total: {len(all_pairs)} pairs ({total_pos} merge, {total_neg} split)")

    return all_pairs


# ---------------------------------------------------------------------------
# Inject edit pairs into CellGNN training
# ---------------------------------------------------------------------------

def edit_pairs_to_contrastive(
    pairs: list[EditPair],
    role: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Convert edit pairs to contrastive positive/negative lists for cell_graph_train_step.

    Returns (positive_pairs, negative_pairs) suitable for augmenting the
    standard contrastive pair sampling in ``_sample_contrastive_pairs``.
    """
    positive = []
    negative = []
    for p in pairs:
        if p.role != role:
            continue
        pair = (p.synapse_i, p.synapse_j)
        if p.label == 1:
            positive.append(pair)
        else:
            negative.append(pair)
    return positive, negative


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-pairs", help="Build edit-derived training pairs from a cache.")
    p_build.add_argument("--cache-dir", required=True)
    p_build.add_argument("--datastack", default="minnie65_public")
    p_build.add_argument("--version", type=int, default=1718)
    p_build.add_argument("--cave-token", default=None)
    p_build.add_argument("--max-roots-per-box", type=int, default=50)
    p_build.add_argument("--seed", type=int, default=42)
    p_build.add_argument("--output", default="run_logs/edit_pairs.tsv")
    p_build.add_argument("--verbose", action="store_true", default=True)
    p_build.add_argument("--quiet", dest="verbose", action="store_false")

    args = ap.parse_args(argv)

    if args.cmd == "build-pairs":
        from .dataset_builder import BoxCache

        cache = BoxCache(args.cache_dir)
        pairs = build_edit_pairs_for_cache(
            cache,
            datastack=args.datastack,
            version=args.version,
            token=args.cave_token,
            max_roots_per_box=args.max_roots_per_box,
            seed=args.seed,
            verbose=args.verbose,
        )

        # Write TSV
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            f.write("synapse_i\tsynapse_j\tlabel\trole\tsource_root_a\tsource_root_b\tedit_type\n")
            for p in pairs:
                f.write(
                    f"{p.synapse_i}\t{p.synapse_j}\t{p.label}\t{p.role}\t"
                    f"{p.source_root_a}\t{p.source_root_b}\t{p.edit_type}\n"
                )
        print(f"Wrote {len(pairs)} pairs to {args.output}")
        return 0

    raise SystemExit(f"unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
