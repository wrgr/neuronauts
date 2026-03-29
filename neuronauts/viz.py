"""Visualisation helpers for Neuronauts scaffolded-grammar development.

All functions return a ``matplotlib.figure.Figure`` so callers can either
display it interactively (``plt.show()``) or save it with ``fig.savefig()``.
Matplotlib is treated as an optional dependency — import errors are caught
at the module boundary and re-raised with an actionable install message.

Six plotting families are provided:

1. **Scaffold plots** — synapse positions coloured by seg_id or root_id.
2. **Bridge plots** — Dijkstra bridge proposals overlaid on synapse scatter.
3. **F1-history plots** — line-graph F1 across experiments or epochs.
4. **Segment-purity diagnostic** — per-segment scaffold quality bars.
5. **CellGNN plots** — inferred cell labels and quality scores.
6. **Training dashboard** — loss curves, merge probability histograms.
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


# ---------------------------------------------------------------------------
# 5. CellGNN cell-label and quality plots
# ---------------------------------------------------------------------------

def plot_cell_labels(
    pre_pt: np.ndarray,
    post_pt: np.ndarray,
    pre_labels: np.ndarray,
    post_labels: np.ndarray,
    *,
    pre_root_id: np.ndarray | None = None,
    post_root_id: np.ndarray | None = None,
    title: str = "CellGNN inferred cells vs ground truth",
    projection: str = "xy",
    point_size: float = 18.0,
) -> "matplotlib.figure.Figure":
    """Side-by-side comparison of CellGNN cell assignments vs ground truth.

    Left column: coloured by inferred cell label.
    Right column: coloured by ground-truth root_id (if provided).
    """
    _, plt, _ = _require_matplotlib()
    axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    ax0, ax1 = axis_map.get(projection, (0, 1))
    axis_labels = "xyz"

    has_truth = pre_root_id is not None and post_root_id is not None
    n_cols = 2 if has_truth else 1

    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 10))
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(title)

    for row, (pts, labels, root_ids, side) in enumerate((
        (pre_pt, pre_labels, pre_root_id, "Pre"),
        (post_pt, post_labels, post_root_id, "Post"),
    )):
        x, y = pts[:, ax0], pts[:, ax1]
        ax = axes[row, 0]
        ax.scatter(x, y, c=_id_to_color_array(labels),
                   s=point_size, alpha=0.75, linewidths=0)
        n_cells = len(np.unique(labels))
        ax.set_title(f"{side} — inferred ({n_cells} cells)")
        ax.set_xlabel(axis_labels[ax0])
        ax.set_ylabel(axis_labels[ax1])

        if has_truth and root_ids is not None:
            ax2 = axes[row, 1]
            ax2.scatter(x, y, c=_id_to_color_array(root_ids),
                        s=point_size, alpha=0.75, linewidths=0)
            n_true = len(np.unique(root_ids))
            ax2.set_title(f"{side} — ground truth ({n_true} cells)")
            ax2.set_xlabel(axis_labels[ax0])
            ax2.set_ylabel(axis_labels[ax1])

    fig.tight_layout()
    return fig


def plot_cell_quality(
    quality_scores: dict[int, float],
    *,
    title: str = "Cell quality scores",
    threshold: float = 0.5,
    figsize: tuple[float, float] = (10, 4),
) -> "matplotlib.figure.Figure":
    """Bar chart and histogram of per-cell quality scores.

    Left panel: bar chart sorted by quality (descending).
    Right panel: distribution histogram with threshold line.

    Parameters
    ----------
    quality_scores:
        Mapping ``cell_id → quality`` from ``score_cell_quality()``.
    threshold:
        Quality threshold drawn as a vertical/horizontal reference line.
    """
    _, plt, _ = _require_matplotlib()

    cell_ids = sorted(quality_scores.keys(), key=lambda k: quality_scores[k], reverse=True)
    values = [quality_scores[k] for k in cell_ids]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title)

    # Bar chart
    colors = ["steelblue" if v >= threshold else "coral" for v in values]
    axes[0].bar(range(len(values)), values, color=colors, alpha=0.8)
    axes[0].axhline(threshold, color="grey", linestyle=":", linewidth=1,
                     label=f"threshold={threshold}")
    axes[0].set_xlabel("cell rank")
    axes[0].set_ylabel("quality score")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"Per-cell quality ({sum(1 for v in values if v >= threshold)}/{len(values)} pass)")
    axes[0].legend(fontsize=8)

    # Histogram
    axes[1].hist(values, bins=20, range=(0, 1.01), color="steelblue",
                 alpha=0.8, edgecolor="white")
    axes[1].axvline(threshold, color="coral", linestyle="--", linewidth=1.5,
                     label=f"threshold={threshold}")
    axes[1].set_xlabel("quality score")
    axes[1].set_ylabel("cell count")
    axes[1].set_title("Quality distribution")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Training dashboard
# ---------------------------------------------------------------------------

def plot_training_history(
    history: dict[str, list[float]],
    *,
    title: str = "Training history",
    figsize: tuple[float, float] = (12, 4),
) -> "matplotlib.figure.Figure":
    """Multi-panel line chart of training metrics over epochs.

    Accepts the history dict returned by ``train_cell_gnn()`` or loaded
    from ``cell_gnn_history.json``.  Common keys: ``train_loss``,
    ``train_pos_sim``, ``train_neg_sim``, ``val_loss``.

    Parameters
    ----------
    history:
        Mapping ``metric_name → [value_epoch_1, value_epoch_2, ...]``.
    """
    _, plt, _ = _require_matplotlib()

    # Group metrics into panels
    loss_keys = [k for k in history if "loss" in k.lower()]
    sim_keys = [k for k in history if "sim" in k.lower()]
    f1_keys = [k for k in history if "f1" in k.lower()]
    other_keys = [k for k in history if k not in loss_keys + sim_keys + f1_keys]

    panels = []
    if loss_keys:
        panels.append(("Loss", loss_keys))
    if sim_keys:
        panels.append(("Similarity", sim_keys))
    if f1_keys:
        panels.append(("F1", f1_keys))
    if other_keys:
        panels.append(("Other", other_keys))
    if not panels:
        panels = [("Metrics", list(history.keys()))]

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0], figsize[1]))
    if n == 1:
        axes = [axes]
    fig.suptitle(title)

    for ax, (panel_title, keys) in zip(axes, panels):
        for key in keys:
            values = history[key]
            ax.plot(range(1, len(values) + 1), values, "o-", label=key,
                    markersize=3, linewidth=1.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel(panel_title.lower())
        ax.set_title(panel_title)
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    return fig


def plot_merge_probabilities(
    probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
    title: str = "Merge candidate probabilities",
    figsize: tuple[float, float] = (8, 4),
) -> "matplotlib.figure.Figure":
    """Histogram of merge candidate probabilities with accept/reject regions.

    Parameters
    ----------
    probabilities:
        List of ``CandidateMerge.probability`` values from a single box.
    threshold:
        Decision boundary separating accept (right) from reject (left).
    """
    _, plt, _ = _require_matplotlib()

    probs = np.asarray(probabilities, dtype=np.float64)
    n_accept = int(np.sum(probs >= threshold))
    n_reject = len(probs) - n_accept

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(probs, bins=30, range=(0, 1), color="steelblue",
            alpha=0.8, edgecolor="white")
    ax.axvline(threshold, color="coral", linestyle="--", linewidth=2,
               label=f"threshold={threshold}")

    # Shade regions
    ax.axvspan(0, threshold, alpha=0.05, color="coral")
    ax.axvspan(threshold, 1, alpha=0.05, color="steelblue")

    ax.set_xlabel("merge probability")
    ax.set_ylabel("candidate count")
    ax.set_title(f"{title}  (accept={n_accept}, reject={n_reject})")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    return fig


def plot_evaluation_summary(
    results: dict,
    *,
    title: str = "Evaluation summary",
    figsize: tuple[float, float] = (10, 5),
) -> "matplotlib.figure.Figure":
    """Summary dashboard from ``evaluate_results.json``.

    Shows GNN F1/precision/recall and optional baseline comparison
    as grouped bars.

    Parameters
    ----------
    results:
        Dict loaded from ``evaluate_results.json`` (as written by
        ``cmd_evaluate``).  Expected keys: ``gnn``, optional ``baseline``.
    """
    _, plt, _ = _require_matplotlib()

    gnn = results.get("gnn", {})
    baseline = results.get("baseline")

    metrics = ["f1_mean", "precision_mean", "recall_mean"]
    labels = ["F1", "Precision", "Recall"]
    gnn_vals = [gnn.get(m, 0.0) for m in metrics]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    width = 0.35

    bars_gnn = ax.bar(x - (width / 2 if baseline else 0), gnn_vals,
                       width, label="CellGNN", color="steelblue", alpha=0.85)

    if baseline:
        baseline_vals = [baseline.get(m, 0.0) for m in metrics]
        bars_base = ax.bar(x + width / 2, baseline_vals,
                            width, label="Grammar baseline", color="coral", alpha=0.85)

    # Value labels on bars
    for bar in bars_gnn:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    if baseline:
        for bar in bars_base:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("score")
    ax.set_title(f"{title}  ({results.get('n_boxes', '?')} boxes, {results.get('split', '?')} split)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    if baseline:
        delta = gnn.get("f1_mean", 0) - baseline.get("f1_mean", 0)
        ax.text(0.98, 0.02, f"Delta F1: {delta:+.4f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, fontweight="bold",
                color="green" if delta > 0 else "red")

    fig.tight_layout()
    return fig


__all__ = [
    # scaffold
    "plot_scaffold_synapses",
    "plot_scaffold_groups",
    "plot_scaffold_purity",
    # bridge
    "plot_bridge_proposals",
    # F1 history
    "plot_f1_history",
    "plot_f1_history_from_ledger",
    # CellGNN
    "plot_cell_labels",
    "plot_cell_quality",
    # training dashboard
    "plot_training_history",
    "plot_merge_probabilities",
    "plot_evaluation_summary",
]
