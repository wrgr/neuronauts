import argparse
import time
from typing import Dict, Iterable, List

import numpy as np

try:
    from caveclient import CAVEclient
except ImportError:
    CAVEclient = object  # type: ignore[assignment,misc]


def get_client(version: int) -> CAVEclient:
    """Construct a CAVEclient for minnie65_public at a given materialization."""
    client = CAVEclient("minnie65_public")
    client.version = version
    return client


def map_roots_between_versions(
    root_ids: Iterable[int],
    old_version: int,
    new_version: int,
    chunk_size: int = 100_000,
    verbose: bool = False,
) -> Dict[int, int]:
    """
    Map a collection of root IDs from one materialization version to another.

    Uses the chunkedgraph's get_latest_roots API at the target version.

    Parameters
    ----------
    root_ids
        Iterable of root IDs defined at `old_version`.
    old_version
        Materialization version where `root_ids` are valid.
    new_version
        Target materialization version to map into (e.g. 1412).
    chunk_size
        Number of IDs per chunk when calling get_latest_roots.
    verbose
        If True, print progress after each batch (count done, elapsed, rate).

    Returns
    -------
    mapping
        Dict mapping old root IDs to new root IDs. A value of 0 indicates
        the body no longer exists at the target version. Root IDs that are
        0 or negative are never sent to the API (they would cause 500s);
        0 is always mapped to 0 in the result when present in the input.
    """
    # Note: the chunkedgraph mapping is anchored at the target version.
    client = get_client(new_version)

    # Force to numpy array of int64; drop invalid IDs (0 and negative) so the API doesn't 500
    raw = list(root_ids)
    roots_arr = np.array(raw, dtype=np.int64)
    valid = roots_arr > 0
    roots_arr = roots_arr[valid]
    mapping: Dict[int, int] = {}
    if 0 in raw:
        mapping[0] = 0

    n_total = len(roots_arr)
    n_batches = (n_total + chunk_size - 1) // chunk_size
    print(
        f"Mapping {n_total:,} roots from v{old_version} to v{new_version} "
        "via chunkedgraph.get_latest_roots..."
    )
    t0 = time.perf_counter()

    for i, start in enumerate(range(0, len(roots_arr), chunk_size)):
        stop = min(start + chunk_size, len(roots_arr))
        batch = roots_arr[start:stop].tolist()
        latest = client.chunkedgraph.get_latest_roots(batch)
        mapping.update(dict(zip(batch, latest)))
        if verbose:
            elapsed = time.perf_counter() - t0
            done = stop
            rate = done / elapsed if elapsed > 0 else 0
            print(
                f"  batch {i + 1}/{n_batches}: {done:,} / {n_total:,} roots "
                f"({elapsed:.0f}s elapsed, ~{rate:,.0f} roots/s)"
            )

    return mapping


def main() -> None:
    """
    Small CLI helper:

    Example:
        python -m neuronauts.cave_root_mapping \\
          --old-version 117 --new-version 1412 \\
          --ids 864691135122603047 864691135771677771
    """
    parser = argparse.ArgumentParser(
        description=(
            "Map specific MICrONS root IDs from one materialization version "
            "to another using the chunkedgraph."
        )
    )
    parser.add_argument(
        "--old-version",
        type=int,
        required=True,
        help="Materialization version where the provided root IDs are valid.",
    )
    parser.add_argument(
        "--new-version",
        type=int,
        required=True,
        help="Target materialization version to map root IDs into.",
    )
    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        required=True,
        help="One or more root IDs (space-separated) to map.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of IDs per chunk when calling get_latest_roots.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress after each mapping batch.",
    )
    args = parser.parse_args()

    mapping = map_roots_between_versions(
        root_ids=args.ids,
        old_version=args.old_version,
        new_version=args.new_version,
        chunk_size=args.chunk_size,
        verbose=args.verbose,
    )

    for old_id in args.ids:
        new_id = mapping.get(old_id, 0)
        print(f"{old_id} -> {new_id}")


if __name__ == "__main__":
    main()

