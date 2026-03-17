#!/usr/bin/env python3
"""View and filter the shared Neuronauts research ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuronauts.experiment_driver import load_experiment_ledger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-path", default="run_logs/research_ledger.jsonl")
    parser.add_argument("--source", choices=["codex", "gemini"], default=None)
    parser.add_argument("--decision", default=None, help="Optional exact decision filter.")
    parser.add_argument("--target-file", default=None, help="Optional substring filter on target_file.")
    parser.add_argument("--min-val-f1", type=float, default=None)
    parser.add_argument("--min-holdout-f1", type=float, default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--sort-by",
        choices=["timestamp", "val_f1", "holdout_f1", "merge_accuracy", "atomicity_accuracy", "reranker_corr"],
        default="holdout_f1",
    )
    parser.add_argument("--ascending", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    return parser.parse_args()


def filter_entries(entries: list[dict[str, object]], args: argparse.Namespace) -> list[dict[str, object]]:
    filtered = entries
    if args.source is not None:
        filtered = [entry for entry in filtered if entry.get("source") == args.source]
    if args.decision is not None:
        filtered = [entry for entry in filtered if entry.get("decision") == args.decision]
    if args.target_file is not None:
        filtered = [entry for entry in filtered if args.target_file in str(entry.get("target_file", ""))]
    if args.min_val_f1 is not None:
        filtered = [entry for entry in filtered if float(entry.get("val_f1") or 0.0) >= args.min_val_f1]
    if args.min_holdout_f1 is not None:
        filtered = [entry for entry in filtered if float(entry.get("holdout_f1") or 0.0) >= args.min_holdout_f1]
    return filtered


def sort_entries(entries: list[dict[str, object]], *, sort_by: str, ascending: bool) -> list[dict[str, object]]:
    def key(entry: dict[str, object]) -> object:
        if sort_by == "timestamp":
            return str(entry.get("timestamp", ""))
        return float(entry.get(sort_by) or 0.0)

    return sorted(entries, key=key, reverse=not ascending)


def format_table(entries: list[dict[str, object]]) -> str:
    headers = [
        "timestamp",
        "source",
        "decision",
        "iter",
        "target",
        "val_f1",
        "holdout_f1",
        "merge_acc",
        "atomicity_acc",
        "reranker_corr",
        "note",
    ]
    rows = []
    for entry in entries:
        rows.append(
            [
                str(entry.get("timestamp", "")),
                str(entry.get("source", "")),
                str(entry.get("decision", "")),
                "" if entry.get("iteration") is None else str(entry.get("iteration")),
                str(entry.get("target_file", "")),
                f"{float(entry.get('val_f1') or 0.0):.4f}",
                f"{float(entry.get('holdout_f1') or 0.0):.4f}",
                f"{float(entry.get('merge_accuracy') or 0.0):.4f}",
                f"{float(entry.get('atomicity_accuracy') or 0.0):.4f}",
                f"{float(entry.get('reranker_corr') or 0.0):.4f}",
                str(entry.get("note", "")),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def render_row(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    lines = [render_row(headers), render_row(["-" * width for width in widths])]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    ledger_path = Path(args.ledger_path)
    entries = load_experiment_ledger(ledger_path)
    filtered = filter_entries(entries, args)
    ordered = sort_entries(filtered, sort_by=args.sort_by, ascending=args.ascending)
    limited = ordered[: max(0, args.limit)]

    if args.json:
        print(json.dumps(limited, indent=2))
        return 0

    print(f"ledger={ledger_path} total={len(entries)} matched={len(filtered)} shown={len(limited)}")
    if not limited:
        print("No matching entries.")
        return 0
    print(format_table(limited))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
