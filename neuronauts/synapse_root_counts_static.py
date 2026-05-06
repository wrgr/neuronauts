import argparse
import os
from collections import Counter
from typing import Dict, Tuple

import pandas as pd
import requests


DEFAULT_STATIC_BASE = "https://storage.googleapis.com/mat_dbs/public/minnie65_phase3_v1"
# Fallback nucleus source (v117-era root IDs). Used only if the version-matched
# GCS file is unavailable; root IDs will not join cleanly with counts at later
# materializations.
STATIC_NUCLEUS_URL = (
    "https://bossdb-open-data.s3.amazonaws.com/"
    "iarpa_microns/minnie/minnie65/nucleus_detection/nucleus_detection_v0.csv"
)


def download_file(url: str, dest_path: str, chunk_size: int = 1_048_576) -> None:
    """
    Stream a large file from HTTP to disk.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        print(f"Downloading {url} -> {dest_path} ({total / 1e9:.2f} GB)")
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    frac = downloaded / total
                    print(
                        f"  {downloaded / 1e9:.2f} GB / {total / 1e9:.2f} GB "
                        f"({frac:.1%})",
                        end="\r",
                    )
        print()  # newline after progress


def ensure_static_files(
    version: int,
    static_dir: str,
) -> Tuple[str, str, str]:
    """
    Ensure synapse CSV, header CSV, and nucleus CSV exist locally.

    Returns
    -------
    syn_csv, header_csv, nucleus_csv : str
        Paths to local files.
    """
    base_dir = os.path.join(static_dir, f"v{version}")
    syn_csv = os.path.join(base_dir, "synapses_pni_2_v1_filtered_view.csv.gz")
    header_csv = os.path.join(base_dir, "synapses_pni_2_v1_filtered_view_header.csv")
    nucleus_csv = os.path.join(base_dir, "nucleus_detection_v0.csv")

    syn_url = (
        f"{DEFAULT_STATIC_BASE}/v{version}/synapses_pni_2_v1_filtered_view.csv.gz"
    )
    header_url = (
        f"{DEFAULT_STATIC_BASE}/v{version}/synapses_pni_2_v1_filtered_view_header.csv"
    )
    nucleus_data_url = (
        f"{DEFAULT_STATIC_BASE}/v{version}/nucleus_detection_v0_merged.csv.gz"
    )
    nucleus_header_url = (
        f"{DEFAULT_STATIC_BASE}/v{version}/nucleus_detection_v0_merged_header.csv"
    )

    if not os.path.exists(syn_csv):
        download_file(syn_url, syn_csv)
    else:
        print(f"Found existing synapse CSV: {syn_csv}")

    if not os.path.exists(header_csv):
        download_file(header_url, header_csv)
    else:
        print(f"Found existing synapse header CSV: {header_csv}")

    if not os.path.exists(nucleus_csv):
        # Try the version-matched GCS source first (root IDs join with counts).
        # The GCS file is gzipped + header-less, so combine the header CSV with
        # the decompressed data into a single uncompressed CSV with a header
        # row, matching the format `select_boxes_from_nucleus_table` expects.
        try:
            os.makedirs(base_dir, exist_ok=True)
            tmp_data_gz = os.path.join(base_dir, "_nucleus_data.csv.gz")
            tmp_header = os.path.join(base_dir, "_nucleus_header.csv")
            download_file(nucleus_data_url, tmp_data_gz)
            download_file(nucleus_header_url, tmp_header)
            header_df = pd.read_csv(tmp_header, header=None, names=["column", "type"])
            cols = header_df["column"].tolist()
            import gzip
            with gzip.open(tmp_data_gz, "rt") as src, open(nucleus_csv, "w") as dst:
                dst.write(",".join(cols) + "\n")
                for line in src:
                    dst.write(line)
            os.remove(tmp_data_gz)
            os.remove(tmp_header)
            print(f"Materialised version-matched nucleus CSV at v{version}: {nucleus_csv}")
        except requests.HTTPError as exc:
            print(
                f"Version-matched nucleus CSV not found on GCS for v{version} ({exc}). "
                "Falling back to BossDB v117-era nucleus."
            )
            download_file(STATIC_NUCLEUS_URL, nucleus_csv)
    else:
        print(f"Found existing nucleus CSV: {nucleus_csv}")

    return syn_csv, header_csv, nucleus_csv


def load_synapse_header(header_csv: str) -> Dict[int, str]:
    """
    Load the header mapping from the small header CSV file.
    """
    header_df = pd.read_csv(header_csv, header=None, names=["column", "type"])
    return dict(enumerate(header_df["column"].tolist()))


def compute_synapse_counts_from_csv(
    syn_csv: str,
    header_csv: str,
    chunksize: int = 1_000_000,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Stream the static synapse CSV and compute pre/post synapse counts per root.
    """
    col_map = load_synapse_header(header_csv)
    # Find indices for pre/post root id columns
    pre_col_name = "pre_pt_root_id"
    post_col_name = "post_pt_root_id"
    try:
        pre_idx = next(i for i, c in col_map.items() if c == pre_col_name)
        post_idx = next(i for i, c in col_map.items() if c == post_col_name)
    except StopIteration as exc:  # pragma: no cover - defensive
        raise KeyError(
            "Could not find pre_pt_root_id / post_pt_root_id in synapse header "
            f"{header_csv}"
        ) from exc

    usecols = [pre_idx, post_idx]

    pre_counts: Counter = Counter()
    post_counts: Counter = Counter()

    print(
        f"Streaming synapses from {syn_csv} with chunksize={chunksize:,} rows "
        f"using columns {pre_idx} ({pre_col_name}), {post_idx} ({post_col_name})"
    )

    for i, chunk in enumerate(
        pd.read_csv(
            syn_csv,
            compression="gzip",
            header=None,
            usecols=usecols,
            chunksize=chunksize,
        ),
        start=1,
    ):
        # Rename numeric columns to semantic names
        chunk.columns = [pre_col_name, post_col_name]

        pre_vc = chunk[pre_col_name].value_counts()
        post_vc = chunk[post_col_name].value_counts()

        pre_counts.update(pre_vc.to_dict())
        post_counts.update(post_vc.to_dict())

        if i % 10 == 0:
            processed = i * chunksize
            print(f"  processed ~{processed:,} synapse rows (chunk {i})")

    return dict(pre_counts), dict(post_counts)


def get_soma_roots_from_csv(nucleus_csv: str) -> pd.Series:
    """
    Load pt_root_id values from the static nucleus_detection_v0 CSV.
    """
    df = pd.read_csv(nucleus_csv, usecols=["pt_root_id"])
    series = df["pt_root_id"]
    series = series[series != 0].dropna().astype("int64").drop_duplicates()
    return series


def build_root_table(
    pre_counts: Dict[int, int],
    post_counts: Dict[int, int],
    soma_roots: pd.Series,
) -> pd.DataFrame:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download static MICrONS synapse/nucleus tables for a given archived "
            "materialization version and compute per-root synapse counts locally."
        )
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1078,
        help="Archived materialization version to use (e.g. 343, 661, 795, 943, 1078).",
    )
    parser.add_argument(
        "--static-dir",
        type=str,
        default="data/microns_static",
        help="Directory to store downloaded static CSV files.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="Number of rows per chunk when streaming the synapse CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="run_logs/synapse_root_counts_static.tsv",
        help="Output TSV path for the per-root synapse counts.",
    )
    args = parser.parse_args()

    syn_csv, header_csv, nucleus_csv = ensure_static_files(
        version=args.version,
        static_dir=args.static_dir,
    )

    pre_counts, post_counts = compute_synapse_counts_from_csv(
        syn_csv=syn_csv,
        header_csv=header_csv,
        chunksize=args.chunksize,
    )
    soma_roots = get_soma_roots_from_csv(nucleus_csv)

    root_df = build_root_table(pre_counts, post_counts, soma_roots)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    root_df.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(root_df)} roots to {args.output}")


if __name__ == "__main__":
    main()

