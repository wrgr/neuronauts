"""Visualisation helpers for Neuronauts scaffolded-grammar development.

All functions return a ``matplotlib.figure.Figure`` so callers can either
display it interactively (``plt.show()``) or save it with ``fig.savefig()``.
Matplotlib is treated as an optional dependency — import errors are caught
at the module boundary and re-raised with an actionable install message.

Three plotting families are provided:

1. **Scaffold plots** — show synapse positions coloured by seg_id or root_id
   so you can visually verify that scaffold groupings make sense before
   training.

2. **Bridge plots** — overlay Dijkstra bridge proposals on a synapse scatter,
   useful for checking that the bridge head is proposing plausible connections.

3. **F1-history plots** — line charts for tracking line-graph F1 across
   experiment ledger runs, helping the outer research loop stay grounded in
   terminal metrics.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _require_matplotlib():
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        return matplotlib, plt, Figure
    except ImportError as exc:
        raise ImportError(
            "pip install matplotlib  (or pip install -e .[viz])"
        ) from exc


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _id_to_color_array(ids: np.ndarray, cmap_name: str = "tab20") -> np.ndarray:
    """Map integer IDs to RGBA colours using a cyclic colourmap."""
    _, plt, _ = _require_matplotlib()
    unique_ids = np.unique(ids)
    cmap = plt.get_cmap(cmap_name, max(len(unique_ids), 1))
    id_to_idx = {int(uid): i for i, uid in enumerate(unique_ids)}
    colors = np.array([cmap(id_to_idx[int(i)] % cmap.N) for i in ids])
    return colors


# ---------------------------------------------------------------------------
# 1. Scaffold plots
# ---------------------------------------------------------------------------

def plot_scaffold_synapses(
    pre_pt: np.ndarray,
    post_pt: np.ndarray,
    pre_seg_id: np.ndarray | None = None,
    post_seg_id: np.ndarray | None = None,
    *,
    pre_root_id: np.ndarray | None = None,
    post_root_id: np.ndarray | None = None,
    title: str = "Scaffold synapses",
    projection: str = "xy",
    point_size: float = 18.0,
    alpha: float = 0.75,
) -> "matplotlib.figure.Figure":
    """2-D scatter of synapse positions coloured by seg_id (or root_id).

    The left panel shows pre-synaptic points and the right panel shows
    post-synaptic points.  Colours represent scaffold segment IDs when
    available, falling back to root IDs.  A grey dot is drawn when neither
    is available.

    Parameters
    ----------
    pre_pt, post_pt:
        Float arrays of shape ``[N, 3]`` (voxel coordinates).
    pre_seg_id, post_seg_id:
        Optional int64 arrays ``[N]``.  When present, colours by seg_id.
    pre_root_id, post_root_id:
        Optional fallback int64 arrays ``[N]``.  Used when seg_id is absent.
    title:
        Figure super-title.
    projection:
        Axis pair to display: ``"xy"``, ``"xz"``, or ``"yz"``.
    """
    _, plt, _ = _require_matplotlib()
    axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    ax0, ax1 = axis_map.get(projection, (0, 1))
    axis_labels = "xyz"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    for ax, pts, seg_ids, root_ids, label in (
        (axes[0], pre_pt, pre_seg_id, pre_root_id, "Pre-synaptic"),
        (axes[1], post_pt, post_seg_id, post_root_id, "Post-synaptic"),
    ):
        x = pts[:, ax0]
        y = pts[:, ax1]
        if seg_ids is not None:
            colors = _id_to_color_array(seg_ids)
            sc = ax.scatter(x, y, c=colors, s=point_size, alpha=alpha, linewidths=0)
            ax.set_title(f"{label} (coloured by seg_id)")
        elif root_ids is not None:
            colors = _id_to_color_array(root_ids)
            sc = ax.scatter(x, y, c=colors, s=point_size, alpha=alpha, linewidths=0)
            ax.set_title(f"{label} (coloured by root_id)")
        else:
            ax.scatter(x, y, c="grey", s=point_size, alpha=alpha, linewidths=0)
            ax.set_title(f"{label}")
        ax.set_xlabel(axis_labels[ax0])
        ax.set_ylabel(axis_labels[ax1])

    fig.tight_layout()
    return fig


def plot_scaffold_groups(
    pre_pt: np.ndarray,
    post_pt: np.ndarray,
    pre_seg_id: np.ndarray,
    post_seg_id: np.ndarray,
    pre_root_id: np.ndarray,
    post_root_id: np.ndarray,
    *,
    title: str = "Scaffold vs ground truth",
    projection: str = "xy",
    point_size: float = 18.0,
) -> "matplotlib.figure.Figure":
    """Side-by-side comparison of scaffold grouping vs ground-truth root_id.

    Left panel: coloured by seg_id (scaffold).
    Right panel: coloured by root_id (ground truth).

    This is useful to quantify and visualise how "noisy" the scaffold is
    relative to the true neurite identity.
    """
    _, plt, _ = _require_matplotlib()
    axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    ax0, ax1 = axis_map.get(projection, (0, 1))
    axis_labels = "xyz"

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title)

    for row, (pts, seg_ids, root_ids, side_label) in enumerate((
        (pre_pt, pre_seg_id, pre_root_id, "Pre"),
        (post_pt, post_seg_id, post_root_id, "Post"),
    )):
        x, y = pts[:, ax0], pts[:, ax1]
        for col, (ids, col_label) in enumerate((
            (seg_ids, "seg_id (scaffold)"),
            (root_ids, "root_id (ground truth)"),
        )):
            ax = axes[row, col]
            ax.scatter(x, y, c=_id_to_color_array(ids), s=point_size, alpha=0.75, linewidths=0)
            ax.set_title(f"{side_label} — {col_label}")
            ax.set_xlabel(axis_labels[ax0])
            ax.set_ylabel(axis_labels[ax1])

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Bridge proposal plots
# ---------------------------------------------------------------------------

def plot_bridge_proposals(
    pre_pt: np.ndarray,
    post_pt: np.ndarray,
    proposals: Sequence[tuple[int, int, float]],
    neuron_endpoint_pts: dict[int, np.ndarray],
    *,
    title: str = "Bridge proposals",
    projection: str = "xy",
    max_proposals: int = 20,
    point_size: float = 12.0,
) -> "matplotlib.figure.Figure":
    """Overlay Dijkstra bridge proposals on a synapse scatter.

    Parameters
    ----------
    pre_pt, post_pt:
        Synapse positions, shape ``[N, 3]``.
    proposals:
        List of ``(neuron_id_a, neuron_id_b, cost)`` as returned by
        ``_propose_bridges``.  Only the cheapest ``max_proposals`` are drawn.
    neuron_endpoint_pts:
        Mapping ``neuron_id → np.ndarray [K, 3]`` of path points.  The first
        and last points are used as the bridge endpoints to draw.
    title:
        Figure title.
    projection:
        Axis pair: ``"xy"``, ``"xz"``, or ``"yz"``.
    max_proposals:
        Maximum number of bridge arcs to overlay.
    """
    _, plt, _ = _require_matplotlib()
    axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    ax0, ax1 = axis_map.get(projection, (0, 1))
    axis_labels = "xyz"

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(title)

    # Background synapse scatter.
    all_pts = np.vstack([pre_pt, post_pt])
    ax.scatter(all_pts[:, ax0], all_pts[:, ax1],
               c="lightgrey", s=point_size, alpha=0.5, linewidths=0, zorder=1)

    # Neuron path clouds.
    for nid, pts in neuron_endpoint_pts.items():
        if len(pts) == 0:
            continue
        ax.scatter(pts[:, ax0], pts[:, ax1],
                   s=point_size * 0.8, alpha=0.3, linewidths=0, zorder=2)

    # Bridge arcs.
    sorted_proposals = sorted(proposals, key=lambda p: p[2])[:max_proposals]
    n = max(len(sorted_proposals), 1)
    for rank, (nid_a, nid_b, cost) in enumerate(sorted_proposals):
        pts_a = neuron_endpoint_pts.get(nid_a)
        pts_b = neuron_endpoint_pts.get(nid_b)
        if pts_a is None or pts_b is None or len(pts_a) == 0 or len(pts_b) == 0:
            continue
        xa, ya = float(pts_a[-1, ax0]), float(pts_a[-1, ax1])
        xb, yb = float(pts_b[0, ax0]), float(pts_b[0, ax1])
        alpha = max(0.2, 1.0 - rank / n)
        ax.annotate(
            "", xy=(xb, yb), xytext=(xa, ya),
            arrowprops=dict(arrowstyle="->", color="crimson", alpha=alpha, lw=1.5),
            zorder=3,
        )
        mid_x = (xa + xb) / 2
        mid_y = (ya + yb) / 2
        ax.text(mid_x, mid_y, f"{cost:.1f}", fontsize=6, color="crimson",
                alpha=alpha, ha="center", zorder=4)

    ax.set_xlabel(axis_labels[ax0])
    ax.set_ylabel(axis_labels[ax1])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. F1-history plots
# ---------------------------------------------------------------------------

def plot_f1_history(
    run_ids: Sequence[str],
    f1_values: Sequence[float],
    *,
    holdout_f1_values: Sequence[float] | None = None,
    title: str = "Line-graph F1 over research cycles",
    highlight_best: bool = True,
    figsize: tuple[float, float] = (10, 4),
) -> "matplotlib.figure.Figure":
    """Line chart of selection-set F1 across consecutive experiment runs.

    Parameters
    ----------
    run_ids:
        Sequence of run labels (e.g. commit hashes or timestamps).
    f1_values:
        Corresponding selection-set line-graph F1 scores.
    holdout_f1_values:
        Optional holdout F1 scores plotted as a dashed line.
    highlight_best:
        If ``True``, mark the run with the best selection-set F1 with a star.
    """
    _, plt, _ = _require_matplotlib()

    n = len(run_ids)
    xs = list(range(n))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xs, list(f1_values), "o-", color="steelblue", label="selection F1", linewidth=2)

    if holdout_f1_values is not None:
        ax.plot(xs, list(holdout_f1_values), "s--", color="darkorange",
                label="holdout F1", linewidth=1.5, alpha=0.85)

    if highlight_best and n > 0:
        best_idx = int(np.argmax(list(f1_values)))
        ax.scatter([best_idx], [f1_values[best_idx]], marker="*",
                   s=200, color="gold", zorder=5, label=f"best ({f1_values[best_idx]:.3f})")

    ax.set_xticks(xs)
    ax.set_xticklabels(list(run_ids), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Line-graph F1")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_f1_history_from_ledger(
    ledger_path: str,
    *,
    max_runs: int = 40,
    title: str = "F1 history from research ledger",
) -> "matplotlib.figure.Figure":
    """Load a ``research_ledger.jsonl`` file and plot the F1 history.

    Each line in the ledger must contain at least ``run_id`` and ``val_f1``
    fields.  Lines without ``val_f1`` are skipped.

    Parameters
    ----------
    ledger_path:
        Path to a JSON-lines ledger file.
    max_runs:
        Limit the chart to the most recent ``max_runs`` entries.
    """
    import json

    run_ids: list[str] = []
    f1s: list[float] = []
    holdout_f1s: list[float] = []

    with open(ledger_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("val_f1") is None:
                continue
            run_ids.append(str(entry.get("run_id", "?")))
            f1s.append(float(entry["val_f1"]))
            holdout_f1s.append(float(entry.get("holdout_f1") or 0.0))

    # Keep only the most recent max_runs entries.
    if len(run_ids) > max_runs:
        run_ids = run_ids[-max_runs:]
        f1s = f1s[-max_runs:]
        holdout_f1s = holdout_f1s[-max_runs:]

    has_holdout = any(v > 0 for v in holdout_f1s)
    return plot_f1_history(
        run_ids,
        f1s,
        holdout_f1_values=holdout_f1s if has_holdout else None,
        title=title,
    )


# ---------------------------------------------------------------------------
# 4. Segment-purity diagnostic
# ---------------------------------------------------------------------------

def plot_scaffold_purity(
    seg_ids: np.ndarray,
    root_ids: np.ndarray,
    *,
    role: str = "pre",
    title: str | None = None,
    figsize: tuple[float, float] = (8, 4),
) -> "matplotlib.figure.Figure":
    """Bar chart of scaffold purity: fraction of synapses in each seg_id whose
    root_id is the majority root.

    A perfectly clean scaffold has purity = 1.0 for every segment.
    Under-merges show up as segments where multiple root_ids are mixed.

    Parameters
    ----------
    seg_ids:
        Int64 array ``[N]``.
    root_ids:
        Int64 array ``[N]``.
    role:
        Label prefix (``"pre"`` or ``"post"``).
    """
    _, plt, _ = _require_matplotlib()

    unique_segs = np.unique(seg_ids)
    purities = []
    sizes = []
    for seg in unique_segs:
        mask = seg_ids == seg
        roots_in_seg = root_ids[mask]
        majority_count = int(np.bincount(
            np.searchsorted(np.unique(roots_in_seg), roots_in_seg)
        ).max())
        purities.append(majority_count / int(mask.sum()))
        sizes.append(int(mask.sum()))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title or f"Scaffold purity ({role})")

    axes[0].bar(range(len(purities)), sorted(purities, reverse=True),
                color="steelblue", alpha=0.8)
    axes[0].axhline(1.0, color="grey", linestyle=":", linewidth=1)
    axes[0].set_xlabel("segment rank")
    axes[0].set_ylabel("purity (majority root fraction)")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Per-segment purity")

    axes[1].hist(purities, bins=20, range=(0, 1.01), color="coral", alpha=0.8, edgecolor="white")
    axes[1].set_xlabel("purity")
    axes[1].set_ylabel("segment count")
    axes[1].set_title("Purity distribution")

    fig.tight_layout()
    return fig


__all__ = [
    "plot_scaffold_synapses",
    "plot_scaffold_groups",
    "plot_bridge_proposals",
    "plot_f1_history",
    "plot_f1_history_from_ledger",
    "plot_scaffold_purity",
]
