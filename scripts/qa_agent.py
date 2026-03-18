#!/usr/bin/env python3
"""Neuronauts QA/QC agent CLI.

Usage
-----
One-shot full audit::

    python scripts/qa_agent.py

Focus on a specific session::

    python scripts/qa_agent.py --session smoke

Watch mode (poll for changes every 30 s)::

    python scripts/qa_agent.py --watch

Watch with faster polling::

    python scripts/qa_agent.py --watch --interval 10

Verbose (include OK findings)::

    python scripts/qa_agent.py --verbose

Check a specific log file::

    python scripts/qa_agent.py --log run_logs/smoke/run_001.log

Check a specific metrics file::

    python scripts/qa_agent.py --metrics models/shared_grammar_smoke.metrics.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuronauts.qa_agent import (
    check_log_file,
    check_metrics_file,
    check_training_log,
    render_report,
    run_full_audit,
    run_incremental_audit,
    QAReport,
    Finding,
)
from datetime import datetime, timezone


def _print_report(report: QAReport, verbose: bool) -> int:
    print(render_report(report, verbose=verbose))
    return 1 if report.errors() else 0


def cmd_check(args: argparse.Namespace) -> int:
    run_root = Path(args.root).resolve()
    report = run_full_audit(run_root, session=args.session)
    return _print_report(report, args.verbose)


def cmd_watch(args: argparse.Namespace) -> int:
    run_root = Path(args.root).resolve()
    interval = args.interval
    since = 0.0

    print(f"  Watching {run_root}  (interval={interval}s, Ctrl-C to stop)")

    try:
        while True:
            report = run_incremental_audit(run_root, since_mtime=since)
            since = time.time()

            if report.findings:
                print(render_report(report, verbose=args.verbose))
            else:
                ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                print(f"  [{ts}] No new findings")

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n  QA agent stopped.")
        return 0


def cmd_log(args: argparse.Namespace) -> int:
    log_path = Path(args.log).resolve()
    findings = check_log_file(log_path)
    report = QAReport(timestamp=datetime.now(timezone.utc))
    report.add(findings)
    return _print_report(report, args.verbose)


def cmd_metrics(args: argparse.Namespace) -> int:
    metrics_path = Path(args.metrics).resolve()
    findings = check_metrics_file(metrics_path)
    report = QAReport(timestamp=datetime.now(timezone.utc))
    report.add(findings)
    return _print_report(report, args.verbose)


def cmd_training(args: argparse.Namespace) -> int:
    log_path = Path(args.training_log).resolve()
    findings = check_training_log(log_path)
    report = QAReport(timestamp=datetime.now(timezone.utc))
    report.add(findings)
    return _print_report(report, args.verbose)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root", default=".",
        help="Repo root directory (default: current directory)",
    )
    parser.add_argument(
        "--session", default=None,
        help="Focus audit on a specific session name under run_logs/",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show OK-level findings in addition to warnings/errors",
    )

    sub = parser.add_subparsers(dest="command")

    # watch
    p_watch = sub.add_parser("watch", help="Poll for new findings continuously")
    p_watch.add_argument("--interval", type=int, default=30,
                         help="Polling interval in seconds (default: 30)")
    p_watch.set_defaults(func=cmd_watch)

    # log
    p_log = sub.add_parser("log", help="Analyse a single run log file")
    p_log.add_argument("log", help="Path to run_NNN.log")
    p_log.set_defaults(func=cmd_log)

    # metrics
    p_metrics = sub.add_parser("metrics", help="Check a single .metrics.json file")
    p_metrics.add_argument("metrics", help="Path to *.metrics.json")
    p_metrics.set_defaults(func=cmd_metrics)

    # training
    p_training = sub.add_parser("training", help="Analyse a train_log.tsv file")
    p_training.add_argument("training_log", help="Path to train_log.tsv")
    p_training.set_defaults(func=cmd_training)

    # default to full check
    parser.set_defaults(func=cmd_check)

    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
