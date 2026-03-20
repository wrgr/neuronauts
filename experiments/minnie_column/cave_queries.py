"""CAVE helpers for nucleus (and optional proofreading) queries."""

from __future__ import annotations

import pandas as pd

from .constants import DEFAULT_DATASTACK, NUCLEUS_TABLE, PROOFREADING_TABLE


def get_client(datastack: str, version: int, token: str | None = None):
    from caveclient import CAVEclient

    client = CAVEclient(datastack, auth_token=token) if token else CAVEclient(datastack)
    client.version = int(version)
    return client


def query_nuclei_in_bbox_nm(
    bbox_nm: tuple[tuple[int, int, int], tuple[int, int, int]],
    *,
    datastack: str = DEFAULT_DATASTACK,
    version: int = 1718,
    token: str | None = None,
) -> pd.DataFrame:
    """Query ``nucleus_detection_v0`` rows whose soma lies inside ``bbox_nm`` (nm).

    Returns columns ``id``, ``pt_root_id``, ``pt_position_x``, ``pt_position_y``, ``pt_position_z``
    (nm) when successful.
    """
    client = get_client(datastack, version, token=token)
    (x0, y0, z0), (x1, y1, z1) = bbox_nm
    bb = [[int(x0), int(y0), int(z0)], [int(x1), int(y1), int(z1)]]

    df = client.materialize.query_table(
        NUCLEUS_TABLE,
        bounding_box=bb,
        bounding_box_column="pt_position",
        split_positions=True,
        desired_resolution=[1, 1, 1],
        select_columns=["id", "pt_root_id", "pt_position"],
    )
    if df is None:
        return pd.DataFrame()
    return df


def query_proofread_for_roots(
    root_ids: list[int],
    *,
    datastack: str = DEFAULT_DATASTACK,
    version: int = 1718,
    token: str | None = None,
) -> pd.DataFrame:
    """Fetch proofreading rows for the given ``pt_root_id`` values (batched)."""
    if not root_ids:
        return pd.DataFrame()

    client = get_client(datastack, version, token=token)
    cols = [
        "pt_root_id",
        "status_axon",
        "status_dendrite",
        "strategy_axon",
        "strategy_dendrite",
        "valid_id",
    ]

    batch_size = 200
    chunks: list[pd.DataFrame] = []
    for i in range(0, len(root_ids), batch_size):
        part = root_ids[i : i + batch_size]
        df = client.materialize.query_table(
            PROOFREADING_TABLE,
            filter_in_dict={"pt_root_id": part},
            select_columns=cols,
        )
        if df is not None and len(df) > 0:
            chunks.append(df)

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["pt_root_id"]).copy()


def query_nuclei_by_nucleus_ids(
    nucleus_ids: list[int],
    *,
    datastack: str = DEFAULT_DATASTACK,
    version: int = 1718,
    token: str | None = None,
) -> pd.DataFrame:
    """Fetch ``nucleus_detection_v0`` rows for the given nucleus ``id`` values."""
    if not nucleus_ids:
        return pd.DataFrame()

    client = get_client(datastack, version, token=token)
    batch_size = 500
    chunks: list[pd.DataFrame] = []
    for i in range(0, len(nucleus_ids), batch_size):
        part = nucleus_ids[i : i + batch_size]
        df = client.materialize.query_table(
            NUCLEUS_TABLE,
            filter_in_dict={"id": part},
            split_positions=True,
            desired_resolution=[1, 1, 1],
            select_columns=["id", "pt_root_id", "pt_position"],
        )
        if df is not None and len(df) > 0:
            chunks.append(df)

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).copy()


def query_column_target_ids(
    *,
    table_name: str = "allen_v1_column_types_slanted_ref",
    datastack: str = DEFAULT_DATASTACK,
    version: int = 1718,
    token: str | None = None,
) -> list[int]:
    """Return nucleus ``target_id`` values from a column membership reference table.

    The default table tags nuclei in the Minnie V1 column (slanted typing reference).
    If the table is missing at ``version``, try ``--column-table`` alternatives
    (e.g. ``l5et_column``) per MICrONS release notes.
    """
    client = get_client(datastack, version, token=token)
    df = client.materialize.query_table(table_name, select_columns=["target_id"])
    if df is None or len(df) == 0 or "target_id" not in df.columns:
        return []
    s = df["target_id"].dropna()
    return [int(x) for x in s.astype("int64").unique().tolist()]