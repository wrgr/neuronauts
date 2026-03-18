import argparse
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
from caveclient import CAVEclient


def get_client(version: int) -> CAVEclient:
    client = CAVEclient("minnie65_public")
    client.version = version
    return client


def fetch_degree_table_1078(client_1078: CAVEclient) -> pd.DataFrame:
    """
    Fetch synapse in/out degree per root_id from CAVE at materialization 1078.

    Uses the 'synapses_pni_2_in_out_degree' table, which is documented as
    available for archived versions.
    """
    table_name = "synapses_pni_2_in_out_degree"
    print(
        f"Querying degree table '{table_name}' at version {client_1078.version} "
        "(this should be a relatively small table)..."
    )
    df = client_1078.materialize.query_table(table_name)

    cols = {c.lower(): c for c in df.columns}

    # Root id column
    if "pt_root_id" in cols:
        root_col = cols["pt_root_id"]
    elif "root_id" in cols:
        root_col = cols["root_id"]
    else:
        raise KeyError(
            f"Could not find a root-id column in {table_name}; "
            f"available columns: {list(df.columns)}"
        )

    # In / out degree columns
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
        f"Using columns from 1078 table: root_id={root_col}, "
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
    client_1412: CAVEclient,
    old_roots: Iterable[int],
    chunk_size: int = 100_000,
) -> Dict[int, int]:
    """
    Use the chunkedgraph mapping to find latest roots at version 1412 for
    a collection of old roots from 1078.

    Returns a mapping {old_root_1078 -> new_root_1412}, where new_root_1412 may be 0
    if the body was deleted / no longer exists.
    """
    old_roots_arr = np.array(list(old_roots), dtype=np.int64)
    mapping: Dict[int, int] = {}

    print(
        f"Mapping {len(old_roots_arr):,} roots from 1078 to latest roots at "
        f"version {client_1412.version} via chunkedgraph..."
    )

    for start in range(0, len(old_roots_arr), chunk_size):
        stop = min(start + chunk_size, len(old_roots_arr))
        batch = old_roots_arr[start:stop].tolist()
        latest = client_1412.chunkedgraph.get_latest_roots(batch)
        mapping.update(dict(zip(batch, latest)))

    return mapping


def aggregate_to_1412(
    deg_df_1078: pd.DataFrame,
    mapping_1078_to_1412: Dict[int, int],
) -> pd.DataFrame:
    """
    Apply the root-id mapping to move degree counts from 1078 roots to
    1412 roots. If multiple 1078 roots map to the same 1412 root, their
    degrees are summed.
    """
    df = deg_df_1078.copy()
    df["root_id_1412"] = df["root_id_1078"].map(mapping_1078_to_1412).astype("int64")

    # Drop bodies that no longer exist (mapped to 0)
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


def fetch_soma_roots_1412(client_1412: CAVEclient) -> Set[int]:
    """
    Get set of root_ids at 1412 that have a soma.

    Uses nucleus_detection_v0.pt_root_id != 0 at version 1412.
    """
    print(
        f"Querying nucleus_detection_v0 at version {client_1412.version} "
        "to identify soma-bearing roots..."
    )
    nuc_df = client_1412.materialize.query_table(
        "nucleus_detection_v0", select_columns=["pt_root_id"]
    )
    roots = nuc_df["pt_root_id"]
    roots = roots[roots != 0].dropna().astype("int64").unique()
    return set(roots.tolist())


def attach_soma_flag(
    deg_df_1412: pd.DataFrame,
    soma_roots_1412: Set[int],
) -> pd.DataFrame:
    df = deg_df_1412.copy()
    df["has_soma"] = df["root_id"].isin(soma_roots_1412)
    df = df.sort_values("total_synapse_count", ascending=False).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use CAVE to pull synapse in/out degrees at materialization 1078, "
            "map root IDs forward to 1412 via the chunkedgraph, and attach "
            "soma flags from 1412 nucleus_detection_v0."
        )
    )
    parser.add_argument(
        "--old-version",
        type=int,
        default=1078,
        help="Materialization version that has synapses_pni_2_in_out_degree (default: 1078).",
    )
    parser.add_argument(
        "--new-version",
        type=int,
        default=1412,
        help="Target materialization version to map roots into (default: 1412).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="run_logs/synapse_root_degrees_v1078_to_1412.tsv",
        help="Output TSV path for the per-root synapse degree table.",
    )
    args = parser.parse_args()

    client_1078 = get_client(version=args.old_version)
    client_1412 = get_client(version=args.new_version)

    # 1) Degree table at old version
    deg_df_1078 = fetch_degree_table_1078(client_1078)

    # 2) Map roots to latest roots at new-version
    mapping_1078_to_1412 = map_roots_to_1412(
        client_1412,
        old_roots=deg_df_1078["root_id_1078"].unique(),
    )

    # 3) Aggregate degrees to 1412 root IDs
    deg_df_1412 = aggregate_to_1412(deg_df_1078, mapping_1078_to_1412)

    # 4) Attach soma flags from new-version nucleus table
    soma_roots_1412 = fetch_soma_roots_1412(client_1412)
    result_df = attach_soma_flag(deg_df_1412, soma_roots_1412)

    result_df.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(result_df)} roots to {args.output}")


if __name__ == "__main__":
    main()

