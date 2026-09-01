"""Figures derived from an ``ExperimentRecord`` -- one form per data job.

The figures are generic over the record structure so a new experiment gets
pictures without new plotting code:

  metric panels     each sweep table -> small multiples, one panel per metric,
                    horizontal bars in one hue; the extreme row is labelled
  operating points  a table with precision + recall columns -> recall vs
                    precision scatter, every rule named, baseline and chosen
                    rule emphasised (the picture that shows whether a sweep
                    found a usable operating point)
  grid heatmaps     row names like ``r5_cone90`` -> a 2-D heatmap per metric,
                    one hue, light->dark
  pair counts       ``pair_counts.{tp,fp,fn,tn}`` -> share-of-pairs stacked
                    bars, which exposes class imbalance directly
  percentiles       any ``name.<q>`` scalar group -> a percentile curve

Colour and mark rules follow the project data-viz conventions: one hue for
magnitude, fixed categorical slots (never cycled), hairline solid grids, thin
marks, a legend whenever two or more series share an axis, and text in ink
tokens rather than series colour.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from neuronauts.report.registry import ExperimentRecord, Table, fmt, is_number

# --- palette (validated categorical order; sequential = blue ramp) ---------
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
              "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, DEEMPH = "#e1e0d9", "#c3c2b7", "#c3c2b7"

_PRIORITY = [
    "pair_precision", "pair_recall", "cross_lineage_split_recall",
    "perfect_roots", "macro_root_pair_f1", "median_components_per_root",
    "merge_precision", "merge_recall", "recall_all_true_pairs",
    "recall_l2_covered_true_pairs", "candidate_pairs", "panel_size.median",
    "ari", "circuit_f1", "circuit_precision", "circuit_recall", "erl_um",
    "largest_cluster", "n_atoms", "n_with_geom", "total_l2_nodes",
    "coord_coverage", "caliber_coverage", "elapsed_min", "n_l2", "n_edges",
]
_GRID_RE = re.compile(r"^([a-zA-Z]+)(\d+(?:\.\d+)?)_([a-zA-Z]+)(\d+(?:\.\d+)?)$")


@dataclass
class FigureSpec:
    path: Path
    caption: str
    check: str          # the pass/fail question a reader should ask of it


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titlesize": 9.5,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "legend.frameon": False, "legend.fontsize": 8,
    })
    return plt


def _style(ax, *, grid_axis: str = "x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def pick_metrics(table: Table, limit: int = 6) -> list[str]:
    numeric = table.numeric_columns()
    chosen = [c for c in _PRIORITY if c in numeric]
    for c in numeric:
        if len(chosen) >= limit:
            break
        if c not in chosen and not c.startswith("pair_counts.") \
                and "sha" not in c and "quantiles" not in c:
            chosen.append(c)
    return chosen[:limit]


def _values(table: Table, col: str) -> np.ndarray:
    return np.array([v if is_number(v) else np.nan for v in table.column(col)],
                    dtype=float)


# ---------------------------------------------------------------------------
# figure families
# ---------------------------------------------------------------------------

def fig_metric_panels(table: Table, metrics: list[str], out: Path,
                      title: str) -> Optional[Path]:
    plt = _mpl()
    metrics = [m for m in metrics if not np.all(np.isnan(_values(table, m)))]
    if not metrics or not table.rows:
        return None
    n_rows, n_panels = len(table.rows), len(metrics)
    fig, axes = plt.subplots(
        1, n_panels, sharey=True,
        figsize=(max(2.4, 2.3 * n_panels) + 1.2, 0.28 * n_rows + 1.4))
    axes = np.atleast_1d(axes)
    y = np.arange(n_rows)[::-1]
    for ax, metric in zip(axes, metrics):
        vals = _values(table, metric)
        ax.barh(y, np.nan_to_num(vals), height=0.55, color=CATEGORICAL[0],
                edgecolor=SURFACE, linewidth=1)
        _style(ax)
        ax.set_title(metric)
        finite = np.isfinite(vals)
        if finite.any():
            i = int(np.nanargmax(vals))
            ax.text(vals[i], y[i], " " + fmt(float(vals[i])), va="center",
                    ha="left", fontsize=8, color=INK2)
            top = float(np.nanmax(vals))
            ax.set_xlim(0, top * 1.28 if top > 0 else 1)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(table.rows, fontsize=8)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out)


def fig_operating_points(table: Table, out: Path, title: str, *,
                         x_col: str, y_col: str,
                         baseline: Optional[str] = None,
                         chosen: Optional[str] = None,
                         label_col: Optional[str] = None) -> Optional[Path]:
    plt = _mpl()
    x, y = _values(table, x_col), _values(table, y_col)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    handles = {}
    # Label selectively: emphasised rows always, others only when they sit
    # clear of a label already placed (axes are 0-1, so data distance is
    # axis fraction). Piled-up labels in a corner are noise, not information.
    emphasised = {chosen, baseline}
    order = sorted(range(len(table.rows)),
                   key=lambda i: (table.rows[i] not in emphasised, i))
    placed: list[np.ndarray] = []
    hidden = 0
    for i in order:
        row = table.rows[i]
        if not ok[i]:
            continue
        if row == chosen:
            kind, color, z = "chosen rule", CATEGORICAL[0], 3
        elif row == baseline:
            kind, color, z = "baseline", INK, 2
        else:
            kind, color, z = "other rules", DEEMPH, 1
        h = ax.scatter(x[i], y[i], s=64, color=color, edgecolor=SURFACE,
                       linewidth=1.5, zorder=z)
        handles.setdefault(kind, h)
        p = np.array([x[i], y[i]])
        if row in emphasised or all(np.linalg.norm(p - q) >= 0.06 for q in placed):
            ax.annotate(row, (x[i], y[i]), xytext=(5, 4), textcoords="offset points",
                        fontsize=7.5, color=INK2 if kind != "other rules" else MUTED)
            placed.append(p)
        else:
            hidden += 1
    if hidden:
        ax.text(0.99, 0.02, f"{hidden} overlapping rules unlabelled (see table)",
                transform=ax.transAxes, ha="right", fontsize=7.5, color=MUTED)
    _style(ax, grid_axis="both")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(min(0, np.nanmin(y[ok])) - 0.03, 1.05)
    ax.set_title(title, loc="left", color=INK)
    if len(handles) > 1:
        ax.legend(handles.values(), handles.keys(), loc="lower left")
    fig.tight_layout()
    return _save(fig, out)


def _parse_grid(rows: list[str]):
    parsed = [_GRID_RE.match(r) for r in rows]
    if not rows or not all(parsed):
        return None
    a_name, b_name = parsed[0].group(1), parsed[0].group(3)
    if any(m.group(1) != a_name or m.group(3) != b_name for m in parsed):
        return None
    a_vals = sorted({float(m.group(2)) for m in parsed})
    b_vals = sorted({float(m.group(4)) for m in parsed})
    idx = {r: (a_vals.index(float(m.group(2))), b_vals.index(float(m.group(4))))
           for r, m in zip(rows, parsed)}
    return a_name, a_vals, b_name, b_vals, idx


def fig_grid_heatmap(table: Table, metric: str, out: Path,
                     title: str) -> Optional[Path]:
    grid = _parse_grid(table.rows)
    if grid is None:
        return None
    plt = _mpl()
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    a_name, a_vals, b_name, b_vals, idx = grid
    mat = np.full((len(a_vals), len(b_vals)), np.nan)
    for row in table.rows:
        v = table.value(row, metric)
        if is_number(v):
            i, j = idx[row]
            mat[i, j] = v
    if np.all(np.isnan(mat)):
        return None
    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    finite = mat[np.isfinite(mat)]
    use_log = finite.max() > 0 and finite.max() / max(finite[finite > 0].min(), 1) > 100 \
        if (finite > 0).any() else False
    norm = LogNorm(max(finite[finite > 0].min(), 1), finite.max()) if use_log else None
    fig, ax = plt.subplots(figsize=(1.0 * len(b_vals) + 2.2, 0.55 * len(a_vals) + 1.6))
    im = ax.imshow(np.where(np.isfinite(mat), mat, np.nan), cmap=cmap, norm=norm,
                   aspect="auto", origin="lower")
    ax.set_xticks(range(len(b_vals)), [f"{v:g}" for v in b_vals])
    ax.set_yticks(range(len(a_vals)), [f"{v:g}" for v in a_vals])
    ax.set_xlabel(b_name)
    ax.set_ylabel(a_name)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    vmax = finite.max()
    for i in range(len(a_vals)):
        for j in range(len(b_vals)):
            v = mat[i, j]
            if not np.isfinite(v):
                continue
            frac = (np.log(max(v, 1)) / np.log(max(vmax, 2))) if use_log else \
                (v / vmax if vmax else 0)
            label = fmt(int(v)) if float(v).is_integer() else fmt(float(v))
            ax.text(j, i, label, ha="center", va="center", fontsize=7.5,
                    color="white" if frac > 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=7.5)
    ax.set_title(f"{title}: {metric}", loc="left", color=INK)
    fig.tight_layout()
    return _save(fig, out)


def fig_pair_counts(table: Table, out: Path, title: str) -> Optional[Path]:
    cols = [f"pair_counts.{k}" for k in ("tp", "fp", "fn", "tn")]
    if not all(c in table.columns for c in cols):
        return None
    plt = _mpl()
    counts = np.stack([_values(table, c) for c in cols], axis=1)
    counts = np.nan_to_num(counts)
    total = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    n = len(table.rows)
    fig, ax = plt.subplots(figsize=(7.2, 0.3 * n + 1.5))
    y = np.arange(n)[::-1]
    left = np.zeros(n)
    labels = ["same-lineage kept (tp)", "cross-lineage kept (fp)",
              "same-lineage split (fn)", "cross-lineage split (tn)"]
    for k in range(4):
        ax.barh(y, share[:, k], left=left, height=0.55, color=CATEGORICAL[k],
                edgecolor=SURFACE, linewidth=2, label=labels[k])
        left += share[:, k]
    _style(ax)
    ax.set_yticks(y, table.rows, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of evaluated pairs")
    ax.set_title(title, loc="left", color=INK)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    return _save(fig, out)


def fig_percentiles(series: dict[str, dict[float, float]], out: Path,
                    title: str, ylabel: str = "") -> Optional[Path]:
    if not series:
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for k, (name, pts) in enumerate(list(series.items())[:8]):
        q = np.array(list(pts.keys()))
        v = np.array(list(pts.values()))
        ax.plot(q, v, color=CATEGORICAL[k % 8], linewidth=2, marker="o",
                markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=name, solid_joinstyle="round")
    _style(ax, grid_axis="y")
    ax.set_xlabel("percentile")
    ax.set_ylabel(ylabel)
    vals = np.concatenate([np.array(list(p.values())) for p in series.values()])
    if (vals > 0).all() and vals.max() / vals.min() > 50:
        ax.set_yscale("log")
    ax.set_title(title, loc="left", color=INK)
    if len(series) > 1:
        ax.legend()
    fig.tight_layout()
    return _save(fig, out)


def fig_compare_tables(tables: list[Table], series: list[str], metrics: list[str],
                       out: Path, title: str, x_label: str) -> Optional[Path]:
    """Sibling sweeps (same rows, same columns) as one line per sibling.

    This is the bake-off picture: e.g. four checkpoints' threshold sweeps on
    one axis per metric, so the reader sees whether any checkpoint separates
    from the rest rather than reading four tables.
    """
    if len(tables) < 2 or not metrics:
        return None
    plt = _mpl()
    try:
        xs = [np.array([float(r) for r in t.rows]) for t in tables]
        numeric_x = True
    except ValueError:
        xs = [np.arange(len(t.rows), dtype=float) for t in tables]
        numeric_x = False
    n = len(metrics)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol + 0.6, 2.6 * nrow + 0.9),
                             squeeze=False)
    for k, metric in enumerate(metrics):
        ax = axes[k // ncol][k % ncol]
        for s, (t, x) in enumerate(zip(tables[:8], xs)):
            y = _values(t, metric)
            ax.plot(x, y, color=CATEGORICAL[s % 8], linewidth=2, marker="o",
                    markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                    label=series[s], solid_joinstyle="round")
        _style(ax, grid_axis="y")
        ax.set_title(metric)
        ax.set_xlabel(x_label)
        if not numeric_x:
            ax.set_xticks(xs[0], tables[0].rows, fontsize=7)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, out)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _chosen_rule(rec: ExperimentRecord) -> Optional[str]:
    for key, val in rec.raw.items():
        if key.startswith("best_") and isinstance(val, dict) and isinstance(val.get("rule"), str):
            return val["rule"]
    return None


def _baseline_row(table: Table) -> Optional[str]:
    return next((r for r in table.rows
                 if r in ("atomic", "baseline", "untouched", "none")), None)


def figures_for_record(rec: ExperimentRecord, out_dir: str | Path,
                       max_tables: int = 6) -> list[FigureSpec]:
    """Every figure the record supports, written under ``out_dir``."""
    out_dir = Path(out_dir)
    specs: list[FigureSpec] = []
    tag = rec.id.replace("/", "_")
    tables = rec.top_tables[:max_tables]
    if not tables:
        tables = rec.tables[:max_tables]

    for table in tables:
        if len(table.rows) < 3:
            continue            # two bars are a table, not a chart
        safe = re.sub(r"[^A-Za-z0-9]+", "_", table.name).strip("_")
        metrics = pick_metrics(table)
        p = fig_metric_panels(table, metrics, out_dir / f"{tag}_{safe}_panels.png",
                              f"{rec.id} · {table.name}")
        if p:
            specs.append(FigureSpec(
                p, f"`{table.name}`: {', '.join(metrics)} per row.",
                "Does the best row on each metric agree with the evaluation note?"))

        rec_col = next((c for c in table.numeric_columns() if c.endswith("recall")
                        and "split" not in c), None)
        prec_col = next((c for c in table.numeric_columns() if c.endswith("precision")), None)
        if rec_col and prec_col and len(table.rows) >= 3:
            p = fig_operating_points(
                table, out_dir / f"{tag}_{safe}_operating.png",
                f"{rec.id} · operating points", x_col=rec_col, y_col=prec_col,
                baseline=_baseline_row(table), chosen=_chosen_rule(rec))
            if p:
                specs.append(FigureSpec(
                    p, f"`{table.name}`: {rec_col} vs {prec_col}, one point per row.",
                    "Is any point in the top-right corner? If not, no rule "
                    "in the sweep is both precise and complete."))

        if _parse_grid(table.rows):
            for metric in pick_metrics(table, limit=3):
                p = fig_grid_heatmap(table, metric,
                                     out_dir / f"{tag}_{safe}_{metric.replace('.', '_')}_grid.png",
                                     f"{rec.id} · {table.name}")
                if p:
                    specs.append(FigureSpec(
                        p, f"`{table.name}` as a grid: {metric}.",
                        "Does the metric vary smoothly with both parameters?"))

        p = fig_pair_counts(table, out_dir / f"{tag}_{safe}_pairs.png",
                            f"{rec.id} · pair outcomes per row")
        if p:
            specs.append(FigureSpec(
                p, f"`{table.name}`: share of tp/fp/fn/tn pairs per row.",
                "How large is the class imbalance the pair F1 is hiding?"))

    # sibling sub-tables: the same leaf table under every row of one parent
    groups: dict[tuple[str, str], list[Table]] = {}
    for t in rec.tables:
        if t.parent and t.leaf:
            groups.setdefault((t.parent, t.leaf), []).append(t)
    for (parent, leaf), members in groups.items():
        first = members[0]
        members = [m for m in members
                   if m.rows == first.rows and m.columns == first.columns]
        if len(members) < 2:
            continue
        safe = re.sub(r"[^A-Za-z0-9]+", "_", f"{parent}_{leaf}").strip("_")
        names = [m.parent_row or m.name for m in members]
        p = fig_compare_tables(members, names, pick_metrics(first),
                               out_dir / f"{tag}_{safe}_compare.png",
                               f"{rec.id} · {leaf} per {parent}", x_label=leaf)
        if p:
            specs.append(FigureSpec(
                p, f"`{parent}.*.{leaf}`: one line per {parent} entry.",
                "Does any line separate from the others, or do they all fail together?"))

    series = rec.percentile_series()
    for prefix, pts in series.items():
        safe = re.sub(r"[^A-Za-z0-9]+", "_", prefix).strip("_")
        p = fig_percentiles({prefix: pts}, out_dir / f"{tag}_{safe}_pct.png",
                            f"{rec.id} · {prefix}", ylabel=prefix)
        if p:
            specs.append(FigureSpec(
                p, f"Percentile curve of `{prefix}` (log axis when the range spans more than 50×).",
                "Where does the tail start, and is a threshold there defensible?"))
    return specs
