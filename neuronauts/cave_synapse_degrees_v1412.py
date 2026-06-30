import argparse
from typing import Set

import pandas as pd

try:
    from caveclient import CAVEclient
except ImportError:
    CAVEclient = object  # type: ignore[assignment,misc]


def get_client(version: int = 1412) -> CAVEclient:
    client = CAVEclient("minnie65_public")
    client.version = version
    return client


def fetch_degree_table(client: CAVEclient) -> pd.DataFrame:
    """
    Fetch synapse in/out degree per root_id from CAVE.

    This expects a materialized table named 'synapses_pni_2_in_out_degree'
    at the given client.version.
    """
    table_name = "synapses_pni_2_in_out_degree"
    print(f"Querying degree table '{table_name}' at version {client.version}...")
    df = client.materialize.query_table(table_name)

    # Try to normalize column names to a common schema
    cols = {c.lower(): c for c in df.columns}

    # Required root-id column
    if "pt_root_id" in cols:
        root_col = cols["pt_root_id"]
    elif "root_id" in cols:
        root_col = cols["root_id"]
    else:
        raise KeyError(
            f"Could not find a root-id column in {table_name}; "
            f"available columns: {list(df.columns)}"
        )

    # Reasonable guesses for in / out degree column names
    in_candidates = ["in_degree", "in_syn_count", "post_count", "postsyn_count"]
    out_candidates = ["out_degree", "out_syn_count", "pre_count", "presyn_count"]

    def pick(col_names: list[str]) -> str:
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

    # Ensure integer types
    deg_df["root_id"] = deg_df["root_id"].astype("int64")
    deg_df["in_synapse_count"] = deg_df["in_synapse_count"].fillna(0).astype("int64")
    deg_df["out_synapse_count"] = deg_df["out_synapse_count"].fillna(0).astype("int64")
    return deg_df


def fetch_soma_roots(client: CAVEclient) -> Set[int]:
    """
    Fetch set of root_ids that have a soma.

    Prefers 'soma_counts' if available; otherwise falls back to 'nucleus_detection_v0'
    and uses pt_root_id != 0 as 'has soma'.
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


def build_root_table(deg_df: pd.DataFrame, soma_roots: Set[int]) -> pd.DataFrame:
    df = deg_df.copy()
    df["total_synapse_count"] = (
        df["in_synapse_count"] + df["out_synapse_count"]
    ).astype("int64")

    soma_set = soma_roots
    df["has_soma"] = df["root_id"].isin(soma_set)

    df = df.sort_values("total_synapse_count", ascending=False).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use CAVE (minnie65_public, materialization 1412 by default) to pull "
            "per-root synapse in/out degrees and mark which roots have a soma."
        )
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1412,
        help="CAVE materialization version (default: 1412).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="run_logs/synapse_root_degrees_v1412.tsv",
        help="Output TSV path for the per-root synapse degree table.",
    )
    args = parser.parse_args()

    client = get_client(version=args.version)

    deg_df = fetch_degree_table(client)
    soma_roots = fetch_soma_roots(client)

    root_df = build_root_table(deg_df, soma_roots)

    root_df.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(root_df)} roots to {args.output}")


if __name__ == "__main__":
    main()

