import argparse
import signal
from collections import Counter
from typing import Dict, Tuple

import pandas as pd
from caveclient import CAVEclient


def get_client(version: int = 1412) -> CAVEclient:
    client = CAVEclient("minnie65_public")
    client.version = version
    return client


def compute_synapse_counts(
    client: CAVEclient,
    chunk_size: int = 500_000,
    log_every: int = 10,
    max_retries: int = 5,
    timeout_seconds: int = 2,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Stream the synapse table and compute per-root pre/post synapse counts.

    Returns
    -------
    pre_counts, post_counts : dict
        Mapping from root id -> number of presynaptic / postsynaptic synapses.
    """
    syn_table = client.info.get_datastack_info()["synapse_table"]  # synapses_pni_2
    total = client.materialize.get_annotation_count(syn_table)

    pre_counts: Counter = Counter()
    post_counts: Counter = Counter()

    n_chunks = (total + chunk_size - 1) // chunk_size

    def _with_timeout(func, *args, **kwargs):
        def handler(_signum, _frame):
            raise TimeoutError(f"Operation exceeded {timeout_seconds} seconds")

        if timeout_seconds is not None and timeout_seconds > 0:
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout_seconds)
        else:
            old_handler = None

        try:
            return func(*args, **kwargs)
        finally:
            if timeout_seconds is not None and timeout_seconds > 0:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    for i, offset in enumerate(range(0, total, chunk_size), start=1):
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                df = _with_timeout(
                    client.materialize.query_table,
                    syn_table,
                    limit=chunk_size,
                    offset=offset,
                    select_columns=["pre_pt_root_id", "post_pt_root_id"],
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                print(
                    f"[synapses_pni_2] chunk {i}/{n_chunks} attempt {attempt}/{max_retries} "
                    f"failed with {type(exc).__name__}: {exc}"
                )
        else:
            raise RuntimeError(
                f"Failed to fetch chunk {i}/{n_chunks} after {max_retries} attempts"
            ) from last_err

        # value_counts returns a Series; update Counters incrementally
        pre_vc = df["pre_pt_root_id"].value_counts()
        post_vc = df["post_pt_root_id"].value_counts()

        pre_counts.update(pre_vc.to_dict())
        post_counts.update(post_vc.to_dict())

        if log_every and (i % log_every == 0 or i == n_chunks):
            processed = min(offset + chunk_size, total)
            frac = processed / total
            print(
                f"[synapses_pni_2] chunk {i}/{n_chunks} "
                f"({processed:,}/{total:,} synapses, {frac:.1%})"
            )

    return dict(pre_counts), dict(post_counts)


def get_soma_roots(client: CAVEclient) -> pd.Series:
    """
    Return a Series of unique pt_root_id values that correspond to nuclei (somata).
    """
    nuc_df = client.materialize.query_table(
        "nucleus_detection_v0",
        select_columns=["pt_root_id"],
    )
    # Some entries may have pt_root_id == 0 (no associated segment); drop those.
    soma_roots = nuc_df["pt_root_id"]
    soma_roots = soma_roots[soma_roots != 0].dropna().astype("int64").drop_duplicates()
    return soma_roots


def build_root_table(
    pre_counts: Dict[int, int],
    post_counts: Dict[int, int],
    soma_roots: pd.Series,
) -> pd.DataFrame:
    """
    Combine pre/post counts and soma information into a single table.
    """
    all_roots = set(pre_counts.keys()) | set(post_counts.keys())
    df = pd.DataFrame({"root_id": list(all_roots)})

    df["pre_synapse_count"] = df["root_id"].map(pre_counts).fillna(0).astype("int64")
    df["post_synapse_count"] = df["root_id"].map(post_counts).fillna(0).astype("int64")
    df["total_synapse_count"] = (
        df["pre_synapse_count"] + df["post_synapse_count"]
    ).astype("int64")

    soma_set = set(soma_roots.to_list())
    df["has_soma"] = df["root_id"].isin(soma_set)

    return df.sort_values("total_synapse_count", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-root synapse counts from MICrONS CAVE (materialization 1412) "
            "and flag which roots have a soma."
        )
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1412,
        help="CAVE materialization version (default: 1412).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Number of synapses to fetch per query (default: 500000).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of retries per chunk on network / server errors.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=2,
        help="Per-chunk timeout in seconds for CAVE queries (0 to disable).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="run_logs/synapse_root_counts_v1412.tsv",
        help="Output TSV path relative to repo root.",
    )
    args = parser.parse_args()

    client = get_client(version=args.version)

    pre_counts, post_counts = compute_synapse_counts(
        client,
        chunk_size=args.chunk_size,
        max_retries=args.max_retries,
        timeout_seconds=args.timeout_seconds,
    )
    soma_roots = get_soma_roots(client)

    root_df = build_root_table(pre_counts, post_counts, soma_roots)

    root_df.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(root_df)} roots to {args.output}")


if __name__ == "__main__":
    main()

