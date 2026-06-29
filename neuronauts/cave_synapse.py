"""Per-root synapse statistics pulled from MICrONS CAVE (minnie65_public).

Consolidates three former single-purpose scripts
(``cave_synapse_degrees_v1412``, ``cave_synapse_counts_v1412``,
``cave_synapse_degrees_1078_to_1412``) into one module with three flows, all
sharing ``get_client`` and ``fetch_soma_roots``:

* **degrees**     — read the materialized ``synapses_pni_2_in_out_degree`` table
                    at one version  (:func:`fetch_degree_table` +
                    :func:`build_degree_root_table`).
* **counts**      — stream ``synapses_pni_2`` and count pre/post per root
                    (:func:`compute_synapse_counts` +
                    :func:`build_counts_root_table`).
* **remap-1078**  — degrees at an old version mapped forward via the
                    chunkedgraph and aggregated  (:func:`fetch_degree_table_1078`,
                    :func:`map_roots_to_1412`, :func:`aggregate_to_1412`,
                    :func:`attach_soma_flag`).

CLI: ``python -m neuronauts.cave_synapse {degrees,counts,remap-1078} [...]``.
Requires the ``cave`` extra (``pip install -e ".[cave]"``) and a CAVE token.
"""

import argparse
import signal
from collections import Counter
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
from caveclient import CAVEclient


def get_client(version: int = 1412) -> CAVEclient:
    client = CAVEclient("minnie65_public")
    client.version = version
    return client


def fetch_soma_roots(client: CAVEclient) -> Set[int]:
    """Return the set of root_ids that have a soma.

    Prefers ``soma_counts`` if available; otherwise falls back to
    ``nucleus_detection_v0`` and uses ``pt_root_id != 0`` as "has soma".
    """
    tables = set(client.materialize.get_tables())

    if "soma_counts" in tables:
        print("Using 'soma_counts' to identify soma-bearing roots...")
        soma_df = client.materialize.query_table("soma_counts")
        cols = {c.lower(): c for c in soma_df.columns}
        if "pt_root_id" in cols:
            root_col = cols["pt_root_id"]
        elif "root_id" in cols:
            root_col = cols["root_id"]
        else:
            raise KeyError(
                "Could not find a root-id column in soma_counts; "
                f"available columns: {list(soma_df.columns)}"
            )
        roots = soma_df[root_col]
        roots = roots[roots != 0].dropna().astype("int64").unique()
        return set(roots.tolist())

    print(
        "Table 'soma_counts' not found; falling back to 'nucleus_detection_v0' "
        "to identify soma-bearing roots..."
    )
    nuc_df = client.materialize.query_table(
        "nucleus_detection_v0",
        select_columns=["pt_root_id"],
    )
    roots = nuc_df["pt_root_id"]
    roots = roots[roots != 0].dropna().astype("int64").unique()
    return set(roots.tolist())


# --------------------------------------------------------------------------- #
# Flow 1: materialized degree table at a single version
# --------------------------------------------------------------------------- #


def fetch_degree_table(client: CAVEclient) -> pd.DataFrame:
    """Fetch synapse in/out degree per root_id from CAVE.

    Expects a materialized table ``synapses_pni_2_in_out_degree`` at the given
    ``client.version``.
    """
    table_name = "synapses_pni_2_in_out_degree"
    print(f"Querying degree table '{table_name}' at version {client.version}...")
    df = client.materialize.query_table(table_name)

    cols = {c.lower(): c for c in df.columns}

    if "pt_root_id" in cols:
        root_col = cols["pt_root_id"]
    elif "root_id" in cols:
        root_col = cols["root_id"]
    else:
        raise KeyError(
            f"Could not find a root-id column in {table_name}; "
            f"available columns: {list(df.columns)}"
        )

    in_candidates = ["in_degree", "in_syn_count", "post_count", "postsyn_count"]
    out_candidates = ["out_degree", "out_syn_count", "pre_count", "presyn_count"]

    def pick(col_names: List[str]) -> str:
        for name in col_names:
            if name in cols:
                return cols[name]
        raise KeyError(
            f"Could not find any of {col_names} in {table_name}; "
            f"available columns: {list(df.columns)}"
        )

    in_col = pick(in_candidates)
    out_col = pick(out_candidates)

    print(
        f"Using columns: root_id={root_col}, "
        f"in_synapse_count={in_col}, out_synapse_count={out_col}"
    )

    deg_df = df[[root_col, in_col, out_col]].copy()
    deg_df.rename(
        columns={
            root_col: "root_id",
            in_col: "in_synapse_count",
            out_col: "out_synapse_count",
        },
        inplace=True,
    )

    deg_df["root_id"] = deg_df["root_id"].astype("int64")
    deg_df["in_synapse_count"] = deg_df["in_synapse_count"].fillna(0).astype("int64")
    deg_df["out_synapse_count"] = deg_df["out_synapse_count"].fillna(0).astype("int64")
    return deg_df


def build_degree_root_table(deg_df: pd.DataFrame, soma_roots: Set[int]) -> pd.DataFrame:
    """Total the in/out degrees, flag somata, sort by total descending."""
    df = deg_df.copy()
    df["total_synapse_count"] = (
        df["in_synapse_count"] + df["out_synapse_count"]
    ).astype("int64")

    df["has_soma"] = df["root_id"].isin(soma_roots)

    df = df.sort_values("total_synapse_count", ascending=False).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Flow 2: streaming pre/post synapse counts
# --------------------------------------------------------------------------- #


def compute_synapse_counts(
    client: CAVEclient,
    chunk_size: int = 500_000,
    log_every: int = 10,
    max_retries: int = 5,
    timeout_seconds: int = 2,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Stream the synapse table and compute per-root pre/post synapse counts.

    Returns ``(pre_counts, post_counts)`` mapping root id -> count.
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


def build_counts_root_table(
    pre_counts: Dict[int, int],
    post_counts: Dict[int, int],
    soma_roots: pd.Series,
) -> pd.DataFrame:
    """Combine pre/post counts and soma information into a single table."""
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


# --------------------------------------------------------------------------- #
# Flow 3: degrees at an old version mapped forward to a target version
# --------------------------------------------------------------------------- #


def fetch_degree_table_1078(client_old: CAVEclient) -> pd.DataFrame:
    """Fetch ``synapses_pni_2_in_out_degree`` at an archived version.

    Column names are normalized; the root-id column is kept as ``root_id_1078``
    to make the subsequent forward-mapping explicit.
    """
    table_name = "synapses_pni_2_in_out_degree"
    print(
        f"Querying degree table '{table_name}' at version {client_old.version} "
        "(this should be a relatively small table)..."
    )
    df = client_old.materialize.query_table(table_name)

    cols = {c.lower(): c for c in df.columns}

    if "pt_root_id" in cols:
        root_col = cols["pt_root_id"]
    elif "root_id" in cols:
        root_col = cols["root_id"]
    else:
        raise KeyError(
            f"Could not find a root-id column in {table_name}; "
            f"available columns: {list(df.columns)}"
        )

    in_candidates = ["in_degree", "in_syn_count", "post_count", "postsyn_count"]
    out_candidates = ["out_degree", "out_syn_count", "pre_count", "presyn_count"]

    def pick(col_names: List[str]) -> str:
        for name in col_names:
            if name in cols:
                return cols[name]
        raise KeyError(
            f"Could not find any of {col_names} in {table_name}; "
            f"available columns: {list(df.columns)}"
        )

    in_col = pick(in_candidates)
    out_col = pick(out_candidates)

    print(
        f"Using columns from old table: root_id={root_col}, "
        f"in_synapse_count={in_col}, out_synapse_count={out_col}"
    )

    deg_df = df[[root_col, in_col, out_col]].copy()
    deg_df.rename(
        columns={
            root_col: "root_id_1078",
            in_col: "in_synapse_count",
            out_col: "out_synapse_count",
        },
        inplace=True,
    )

    deg_df["root_id_1078"] = deg_df["root_id_1078"].astype("int64")
    deg_df["in_synapse_count"] = deg_df["in_synapse_count"].fillna(0).astype("int64")
    deg_df["out_synapse_count"] = deg_df["out_synapse_count"].fillna(0).astype("int64")
    return deg_df


def map_roots_to_1412(
    client_new: CAVEclient,
    old_roots: Iterable[int],
    chunk_size: int = 100_000,
) -> Dict[int, int]:
    """Map old roots to latest roots at ``client_new.version`` via chunkedgraph.

    Returns ``{old_root -> new_root}``; ``new_root`` may be 0 if the body was
    deleted.
    """
    old_roots_arr = np.array(list(old_roots), dtype=np.int64)
    mapping: Dict[int, int] = {}

    print(
        f"Mapping {len(old_roots_arr):,} roots to latest roots at "
        f"version {client_new.version} via chunkedgraph..."
    )

    for start in range(0, len(old_roots_arr), chunk_size):
        stop = min(start + chunk_size, len(old_roots_arr))
        batch = old_roots_arr[start:stop].tolist()
        latest = client_new.chunkedgraph.get_latest_roots(batch)
        mapping.update(dict(zip(batch, latest)))

    return mapping


def aggregate_to_1412(
    deg_df_1078: pd.DataFrame,
    mapping_1078_to_1412: Dict[int, int],
) -> pd.DataFrame:
    """Apply the root-id mapping, summing degrees of roots that merge."""
    df = deg_df_1078.copy()
    df["root_id_1412"] = df["root_id_1078"].map(mapping_1078_to_1412).astype("int64")

    df = df[df["root_id_1412"] != 0]

    grouped = (
        df.groupby("root_id_1412", as_index=False)[
            ["in_synapse_count", "out_synapse_count"]
        ]
        .sum()
        .rename(columns={"root_id_1412": "root_id"})
    )

    grouped["total_synapse_count"] = (
        grouped["in_synapse_count"] + grouped["out_synapse_count"]
    ).astype("int64")

    return grouped


def attach_soma_flag(
    deg_df_1412: pd.DataFrame,
    soma_roots_1412: Set[int],
) -> pd.DataFrame:
    df = deg_df_1412.copy()
    df["has_soma"] = df["root_id"].isin(soma_roots_1412)
    df = df.sort_values("total_synapse_count", ascending=False).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _main_degrees(args: argparse.Namespace) -> None:
    client = get_client(version=args.version)
    deg_df = fetch_degree_table(client)
    soma_roots = fetch_soma_roots(client)
    root_df = build_degree_root_table(deg_df, soma_roots)
    root_df.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(root_df)} roots to {args.output}")


def _main_counts(args: argparse.Namespace) -> None:
    client = get_client(version=args.version)
    pre_counts, post_counts = compute_synapse_counts(
        client,
        chunk_size=args.chunk_size,
        max_retries=args.max_retries,
        timeout_seconds=args.timeout_seconds,
    )
    soma_roots = pd.Series(sorted(fetch_soma_roots(client)))
    root_df = build_counts_root_table(pre_counts, post_counts, soma_roots)
    root_df.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(root_df)} roots to {args.output}")


def _main_remap(args: argparse.Namespace) -> None:
    client_old = get_client(version=args.old_version)
    client_new = get_client(version=args.new_version)
    deg_df_1078 = fetch_degree_table_1078(client_old)
    mapping = map_roots_to_1412(client_new, old_roots=deg_df_1078["root_id_1078"].unique())
    deg_df_1412 = aggregate_to_1412(deg_df_1078, mapping)
    soma_roots_1412 = fetch_soma_roots(client_new)
    result_df = attach_soma_flag(deg_df_1412, soma_roots_1412)
    result_df.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(result_df)} roots to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pull per-root synapse statistics from MICrONS CAVE (minnie65_public)."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_deg = sub.add_parser("degrees", help="Read the materialized in/out degree table at one version.")
    p_deg.add_argument("--version", type=int, default=1412, help="CAVE materialization version (default: 1412).")
    p_deg.add_argument("--output", type=str, default="run_logs/synapse_root_degrees_v1412.tsv")
    p_deg.set_defaults(func=_main_degrees)

    p_cnt = sub.add_parser("counts", help="Stream synapses_pni_2 and count pre/post per root.")
    p_cnt.add_argument("--version", type=int, default=1412)
    p_cnt.add_argument("--chunk-size", type=int, default=500_000)
    p_cnt.add_argument("--max-retries", type=int, default=5)
    p_cnt.add_argument("--timeout-seconds", type=int, default=2)
    p_cnt.add_argument("--output", type=str, default="run_logs/synapse_root_counts_v1412.tsv")
    p_cnt.set_defaults(func=_main_counts)

    p_rmp = sub.add_parser("remap-1078", help="Degrees at an old version, mapped forward and aggregated.")
    p_rmp.add_argument("--old-version", type=int, default=1078)
    p_rmp.add_argument("--new-version", type=int, default=1412)
    p_rmp.add_argument("--output", type=str, default="run_logs/synapse_root_degrees_v1078_to_1412.tsv")
    p_rmp.set_defaults(func=_main_remap)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
