"""
Neuronauts v2 performance dashboard.
Run with: .venv/bin/python dashboard/app.py
Then open http://localhost:5050

Primary training path (v2): scripts/train.py
  build-dataset → train (grammar + GAT)
Legacy v1 pipeline: run_research_cycle (export_merge, export_topology, …)
"""
import glob
import json
import os
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

PIPELINE_STEPS = [
    "export_merge",
    "export_topology",
    "train_shared",
    "export_assembly",
    "train_reranker",
    "validate_selection",
    "validate_holdout",
]

STEP_LABELS = {
    "export_merge": "Export Merge Dataset",
    "export_topology": "Export Topology Dataset",
    "train_shared": "Train Shared Grammar",
    "export_assembly": "Export Assembly Dataset",
    "train_reranker": "Train Reranker",
    "validate_selection": "Validate Selection",
    "validate_holdout": "Validate Holdout",
}


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def read_ledger():
    ledger_path = PROJECT_ROOT / "run_logs" / "research_ledger.jsonl"
    entries = []
    if ledger_path.exists():
        with open(ledger_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return entries


def read_iteration_cycles():
    """Return per-iteration research_cycle_summary.json data, keyed by iter label."""
    results = {}
    # Search all depths under codex_optimize iteration dirs
    patterns = [
        str(PROJECT_ROOT / "run_logs" / "codex_optimize" / "iteration_*" / "research_cycle" / "research_cycle_summary.json"),
        str(PROJECT_ROOT / "run_logs" / "codex_optimize" / "iteration_*" / "accepted_loop" / "research_cycle_summary.json"),
        str(PROJECT_ROOT / "run_logs" / "codex_optimize" / "iteration_*" / "research_cycle_summary.json"),
        str(PROJECT_ROOT / "run_logs" / "codex_optimize" / "baseline_cycle" / "research_cycle_summary.json"),
    ]
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            p = Path(path)
            # Find the iteration_XXX parent
            iter_name = None
            for part in p.parts:
                if part.startswith("iteration_") or part == "baseline_cycle":
                    iter_name = part
                    break
            if iter_name and iter_name not in results:
                data = _load_json(path)
                if data:
                    results[iter_name] = data
    return results


def read_model_metrics():
    metrics = {}
    for path in sorted(glob.glob(str(PROJECT_ROOT / "models" / "*.metrics.json"))):
        name = Path(path).stem  # e.g. "shared_grammar_smoke.metrics"
        data = _load_json(path)
        if data:
            metrics[name] = data
    return metrics


def read_latest_cycle():
    """Return the most recent research_cycle_summary.json available."""
    candidates = [
        PROJECT_ROOT / "run_logs" / "latest" / "research_cycle_summary.json",
    ]
    # Also check most recent codex_optimize iteration
    pattern = str(
        PROJECT_ROOT / "run_logs" / "codex_optimize" / "iteration_*" / "research_cycle" / "research_cycle_summary.json"
    )
    paths = sorted(glob.glob(pattern))
    if paths:
        candidates.insert(0, Path(paths[-1]))

    for c in candidates:
        data = _load_json(c)
        if data:
            return data
    return None


def read_v2_train_log():
    """Read v2 training log (scripts/train.py) if present."""
    path = PROJECT_ROOT / "run_logs" / "train_log.tsv"
    if not path.exists():
        return None
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().strip().splitlines()
        if len(lines) < 2:
            return None
        header = lines[0].split("\t")
        for line in lines[1:]:
            vals = line.split("\t")
            if len(vals) >= len(header):
                rows.append(dict(zip(header, vals)))
    except Exception:
        return None
    return {"header": header, "rows": rows} if rows else None


def detect_running():
    """Heuristic: check if any stdout log or v2 train log was written in the last 60 s."""
    patterns = [
        str(PROJECT_ROOT / "run_logs" / "latest" / "*.stdout.log"),
        str(PROJECT_ROOT / "run_logs" / "codex_optimize" / "iteration_*" / "accepted_loop" / "*.stdout.log"),
    ]
    paths_to_check = list(glob.glob(pattern) for pattern in patterns)
    train_log = PROJECT_ROOT / "run_logs" / "train_log.tsv"
    if train_log.exists():
        paths_to_check.append([str(train_log)])
    now = time.time()
    for path_list in paths_to_check:
        for log in path_list:
            try:
                if Path(log).exists() and now - os.path.getmtime(log) < 60:
                    return True
            except OSError:
                pass
    return False


def read_dataset_stats():
    """Return shape/balance stats for current training datasets (no heavy loading)."""
    stats = {}
    datasets = {
        "merge": PROJECT_ROOT / "data" / "merge_dataset_smoke.npz",
        "topology": PROJECT_ROOT / "data" / "topology_dataset_smoke.npz",
        "assembly": PROJECT_ROOT / "data" / "assembly_ranking_smoke.npz",
    }
    for name, path in datasets.items():
        if not path.exists():
            continue
        try:
            import numpy as np
            d = np.load(path, allow_pickle=False)
            entry = {}
            if "y" in d.files:
                y = d["y"]
                entry["n"] = int(len(y))
                entry["pos_rate"] = float(y.mean()) if y.dtype != object else None
                entry["n_pos"] = int(y.sum()) if y.dtype != object else None
                entry["n_neg"] = int((1 - y).sum()) if y.dtype != object else None
            if "y_f1" in d.files:
                y = d["y_f1"]
                entry["n"] = int(len(y))
                entry["mean_f1"] = float(y.mean())
                entry["max_f1"] = float(y.max())
            if "left_x" in d.files:
                entry["seq_len"] = int(d["left_x"].shape[1])
                entry["feat_dim"] = int(d["left_x"].shape[2])
            if "x" in d.files:
                shape = d["x"].shape
                entry["shape"] = list(shape)
            stats[name] = entry
        except Exception as e:
            stats[name] = {"error": str(e)}
    return stats


def read_training_losses():
    """Extract per-iteration training loss components from research_cycle_summary files."""
    cycles = read_iteration_cycles()
    rows = []
    for iter_name in sorted(cycles.keys()):
        data = cycles[iter_name]
        sm = data.get("shared_training_metrics", {})
        last = sm.get("last_step", {})
        rm = data.get("reranker_metrics", {})
        rows.append({
            "iteration": iter_name,
            "loss": last.get("loss"),
            "merge_loss": last.get("merge_loss"),
            "atomicity_loss": last.get("atomicity_loss"),
            "merge_accuracy": sm.get("merge_accuracy"),
            "atomicity_accuracy": sm.get("atomicity_accuracy"),
            "n_merge": sm.get("n_merge"),
            "n_topology": sm.get("n_topology"),
            "reranker_corr": rm.get("corr"),
            "reranker_mse": rm.get("mse"),
        })
    return rows


def build_step_timeline():
    """
    Build a list of {iteration, step, status, returncode} records
    from all iteration cycles, useful for the step-status heatmap.
    """
    cycles = read_iteration_cycles()
    rows = []
    for iter_name in sorted(cycles.keys()):
        data = cycles[iter_name]
        steps = data.get("steps", {})
        for step in PIPELINE_STEPS:
            info = steps.get(step, {})
            rc = info.get("returncode", None)
            if rc is None:
                status = "missing"
            elif rc == 0:
                status = "ok"
            else:
                status = "fail"
            rows.append({
                "iteration": iter_name,
                "step": step,
                "status": status,
                "returncode": rc,
            })
    return rows


def _fig_to_png_response(fig):
    """Render a matplotlib figure to a Flask PNG response."""
    import io
    from flask import Response
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110,
                facecolor=fig.get_facecolor())
    buf.seek(0)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return Response(buf.read(), mimetype="image/png")


def _dark_style():
    """Return a context manager that applies a dark plot style."""
    import matplotlib.pyplot as plt
    return plt.style.context({
        "axes.facecolor": "#1a1d27",
        "figure.facecolor": "#1a1d27",
        "axes.edgecolor": "#2e3350",
        "axes.labelcolor": "#8892a4",
        "xtick.color": "#8892a4",
        "ytick.color": "#8892a4",
        "text.color": "#e2e8f0",
        "axes.titlecolor": "#e2e8f0",
        "grid.color": "#2e3350",
        "axes.grid": True,
    })


@app.route("/api/plot/merge_paths.png")
def plot_merge_paths():
    """
    Grid of merge-pair path samples. Each cell shows the left (blue) and
    right (orange) agent paths for one example. Label shown in title:
    MERGE (green border) or NO-MERGE (red border).
    """
    import random
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    seed = int(request.args.get("seed", 0))
    rng = np.random.default_rng(seed)

    path = PROJECT_ROOT / "data" / "merge_dataset_smoke.npz"
    if not path.exists():
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No data", ha="center"); return _fig_to_png_response(fig)

    d = np.load(path)
    lx, rx, y, lmask, rmask = d["left_x"], d["right_x"], d["y"], d["left_mask"], d["right_mask"]

    # Sample 6 merge + 6 no-merge
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_each = min(6, len(pos_idx), len(neg_idx))
    chosen_pos = rng.choice(pos_idx, n_each, replace=False)
    chosen_neg = rng.choice(neg_idx, n_each, replace=False)
    indices = np.concatenate([chosen_pos, chosen_neg])
    labels_chosen = np.concatenate([np.ones(n_each, dtype=int), np.zeros(n_each, dtype=int)])

    cols = 4
    rows = int(np.ceil(len(indices) / cols))
    with _dark_style():
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6))
        fig.suptitle("Merge Dataset — Sample Path Pairs (XY projection)", fontsize=11, y=1.01)
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for i, (idx, lbl) in enumerate(zip(indices, labels_chosen)):
            ax = axes_flat[i]
            lpath = lx[idx][lmask[idx].astype(bool)]   # valid steps only
            rpath = rx[idx][rmask[idx].astype(bool)]

            ax.scatter(lpath[:, 0], lpath[:, 1], c="#3b82f6", s=8, alpha=0.85, label="Left" if i == 0 else "")
            ax.scatter(rpath[:, 0], rpath[:, 1], c="#f97316", s=8, alpha=0.85, label="Right" if i == 0 else "")
            # Mark endpoints
            if len(lpath): ax.scatter([lpath[-1, 0]], [lpath[-1, 1]], c="#3b82f6", s=40, marker="^", zorder=5)
            if len(rpath): ax.scatter([rpath[0, 0]], [rpath[0, 1]], c="#f97316", s=40, marker="v", zorder=5)

            color = "#22c55e" if lbl == 1 else "#ef4444"
            label_str = "MERGE" if lbl == 1 else "NO MERGE"
            ax.set_title(f"#{idx} — {label_str}", fontsize=8, color=color)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.5)
            ax.set_xticks([]); ax.set_yticks([])

        for j in range(len(indices), len(axes_flat)):
            axes_flat[j].set_visible(False)

        handles = [
            mpatches.Patch(color="#3b82f6", label="Left path"),
            mpatches.Patch(color="#f97316", label="Right path"),
        ]
        fig.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.3)
        fig.tight_layout()

    return _fig_to_png_response(fig)


@app.route("/api/plot/merge_features.png")
def plot_merge_features():
    """
    Diagnostic feature plots for the merge dataset:
    left panel — endpoint distance by label (should be lower for MERGE=1)
    right panel — path length distribution by label
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path = PROJECT_ROOT / "data" / "merge_dataset_smoke.npz"
    if not path.exists():
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No data", ha="center"); return _fig_to_png_response(fig)

    d = np.load(path)
    lx, rx, y = d["left_x"], d["right_x"], d["y"]
    lmask, rmask = d["left_mask"], d["right_mask"]

    # Endpoint distance: distance between last valid left step and first valid right step
    end_dists = []
    path_lens_l, path_lens_r = [], []
    for i in range(len(y)):
        lvalid = lx[i][lmask[i].astype(bool)]
        rvalid = rx[i][rmask[i].astype(bool)]
        if len(lvalid) and len(rvalid):
            end_dists.append(float(np.linalg.norm(lvalid[-1] - rvalid[0])))
        else:
            end_dists.append(np.nan)
        path_lens_l.append(lmask[i].sum())
        path_lens_r.append(rmask[i].sum())

    end_dists = np.array(end_dists)
    path_lens = (np.array(path_lens_l) + np.array(path_lens_r)) / 2.0

    pos = y == 1
    neg = y == 0

    with _dark_style():
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
        fig.suptitle("Merge Dataset — Feature Diagnostics", fontsize=11)

        # Endpoint distance histogram
        ax = axes[0]
        bins = np.linspace(0, np.nanpercentile(end_dists, 99), 30)
        ax.hist(end_dists[pos], bins=bins, alpha=0.75, color="#22c55e", label=f"MERGE (n={pos.sum()})", density=True)
        ax.hist(end_dists[neg], bins=bins, alpha=0.65, color="#ef4444", label=f"NO-MERGE (n={neg.sum()})", density=True)
        ax.set_xlabel("Endpoint distance (vox)")
        ax.set_ylabel("Density")
        ax.set_title("Path Endpoint Distance by Label")
        ax.legend(fontsize=8)

        # Path length (valid steps) by label
        ax = axes[1]
        bins2 = np.arange(0, lx.shape[1] + 2)
        ax.hist(path_lens[pos], bins=bins2, alpha=0.75, color="#22c55e", label="MERGE", density=True)
        ax.hist(path_lens[neg], bins=bins2, alpha=0.65, color="#ef4444", label="NO-MERGE", density=True)
        ax.set_xlabel("Mean valid path length (steps)")
        ax.set_ylabel("Density")
        ax.set_title("Path Length by Label")
        ax.legend(fontsize=8)

        # Scatter: endpoint dist vs path len, coloured by label
        ax = axes[2]
        ax.scatter(path_lens[pos], end_dists[pos], c="#22c55e", s=14, alpha=0.7, label="MERGE")
        ax.scatter(path_lens[neg], end_dists[neg], c="#ef4444", s=14, alpha=0.6, label="NO-MERGE")
        ax.set_xlabel("Mean path length")
        ax.set_ylabel("Endpoint distance (vox)")
        ax.set_title("Length vs Distance")
        ax.legend(fontsize=8)

        fig.tight_layout()

    return _fig_to_png_response(fig)


@app.route("/api/plot/topology_branches.png")
def plot_topology_branches():
    """Sample topology branch path pairs, coloured by atomic label."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    seed = int(request.args.get("seed", 0))
    rng = np.random.default_rng(seed)

    path = PROJECT_ROOT / "data" / "topology_dataset_smoke.npz"
    if not path.exists():
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No data", ha="center"); return _fig_to_png_response(fig)

    d = np.load(path)
    bx = d["branch_x"]        # [N, max_paths, 2, 3] — (path_idx, branch_idx, step, coord)
    y = d["y"]                  # [N]
    mask = d["branch_mask"]     # [N, max_paths]
    bmask = d["branch_sequence_mask"]  # [N, max_paths, 2]

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_each = min(4, len(pos_idx), len(neg_idx))
    chosen = np.concatenate([rng.choice(pos_idx, n_each, replace=False),
                             rng.choice(neg_idx, n_each, replace=False)])
    labels = np.concatenate([np.ones(n_each, int), np.zeros(n_each, int)])

    cols = 4
    rows = int(np.ceil(len(chosen) / cols))
    with _dark_style():
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.5))
        fig.suptitle("Topology Dataset — Branch Path Samples (XY)", fontsize=11, y=1.01)
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        branch_colors = ["#06b6d4", "#f59e0b", "#a855f7", "#22c55e"]

        for i, (idx, lbl) in enumerate(zip(chosen, labels)):
            ax = axes_flat[i]
            n_valid_paths = int(mask[idx].sum())
            for p in range(n_valid_paths):
                for b in range(2):
                    if bmask[idx, p, b]:
                        pts = bx[idx, p, b]   # [3]
                        # branch_x is (step, coord) not a sequence here — it's per-path endpoints
                        # Plot as a single point with offset jitter for visibility
                        ax.scatter([pts[0]], [pts[1]],
                                   c=branch_colors[b % len(branch_colors)], s=20,
                                   alpha=0.85, marker="o" if b == 0 else "s")

            color = "#22c55e" if lbl == 1 else "#ef4444"
            label_str = "ATOMIC" if lbl == 1 else "NON-ATOMIC"
            ax.set_title(f"#{idx} — {label_str}", fontsize=8, color=color)
            for spine in ax.spines.values():
                spine.set_edgecolor(color); spine.set_linewidth(1.5)
            ax.set_xticks([]); ax.set_yticks([])

        for j in range(len(chosen), len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.tight_layout()

    return _fig_to_png_response(fig)


@app.route("/api/plot/assembly_dist.png")
def plot_assembly_dist():
    """Assembly hypothesis F1 score distribution + reranker score correlation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path = PROJECT_ROOT / "data" / "assembly_ranking_smoke.npz"
    if not path.exists():
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No data", ha="center"); return _fig_to_png_response(fig)

    d = np.load(path)
    y_f1 = d["y_f1"]       # [N] — true F1 per hypothesis
    y_best = d["y_best"]   # [N] — 1 if this is the best hypothesis per group

    # Try to load reranker weights and compute predicted scores
    reranker_path = PROJECT_ROOT / "models" / "assembly_reranker_smoke.npz"
    pred_scores = None
    if reranker_path.exists():
        try:
            r = np.load(reranker_path)
            w = r["weights"] if "weights" in r.files else r[r.files[0]]
            b = r["bias"] if "bias" in r.files else np.zeros(1)
            x = d["x"].astype(np.float32)
            if w.ndim == 1:
                pred_scores = x @ w + float(b)
            elif w.ndim == 2:
                pred_scores = (x @ w).squeeze() + float(b)
        except Exception:
            pass

    with _dark_style():
        ncols = 3 if pred_scores is not None else 2
        fig, axes = plt.subplots(1, ncols, figsize=(ncols * 4, 3.5))
        fig.suptitle("Assembly Dataset — Hypothesis F1 Distribution", fontsize=11)

        # Histogram of F1 scores
        ax = axes[0]
        ax.hist(y_f1, bins=20, color="#3b82f6", alpha=0.8, edgecolor="#2e3350")
        ax.axvline(float(y_f1.mean()), color="#f59e0b", linewidth=1.5, linestyle="--", label=f"mean={y_f1.mean():.3f}")
        ax.axvline(float(y_f1.max()), color="#22c55e", linewidth=1.5, linestyle="--", label=f"max={y_f1.max():.3f}")
        ax.set_xlabel("Hypothesis F1")
        ax.set_ylabel("Count")
        ax.set_title("F1 Distribution")
        ax.legend(fontsize=8)

        # Best vs non-best split
        ax = axes[1]
        best_f1 = y_f1[y_best == 1]
        nonbest_f1 = y_f1[y_best == 0]
        bins = np.linspace(0, y_f1.max() + 0.01, 20)
        ax.hist(nonbest_f1, bins=bins, alpha=0.7, color="#ef4444", label=f"Non-best (n={len(nonbest_f1)})", density=True)
        ax.hist(best_f1, bins=bins, alpha=0.75, color="#22c55e", label=f"Best (n={len(best_f1)})", density=True)
        ax.set_xlabel("F1")
        ax.set_ylabel("Density")
        ax.set_title("Best vs Non-best Hypotheses")
        ax.legend(fontsize=8)

        # Reranker predicted vs true F1
        if pred_scores is not None:
            ax = axes[2]
            ax.scatter(pred_scores, y_f1, c=np.where(y_best == 1, "#22c55e", "#8892a4"),
                      s=20, alpha=0.8)
            ax.set_xlabel("Reranker score")
            ax.set_ylabel("True F1")
            ax.set_title("Reranker vs True F1")
            corr = float(np.corrcoef(pred_scores, y_f1)[0, 1])
            ax.text(0.05, 0.92, f"ρ={corr:.4f}", transform=ax.transAxes, fontsize=9, color="#f59e0b")

        fig.tight_layout()

    return _fig_to_png_response(fig)


@app.route("/api/plot/purity.png")
def plot_purity():
    """
    Scaffold purity plot using the merge dataset.
    We treat each unique left_x group (by label cluster) as a synthetic segment.
    Uses viz.plot_scaffold_purity with data synthesized from the merge dataset.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    path = PROJECT_ROOT / "data" / "merge_dataset_smoke.npz"
    if not path.exists():
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No data", ha="center"); return _fig_to_png_response(fig)

    d = np.load(path)
    lx, y, lmask = d["left_x"], d["y"], d["left_mask"]

    # Build synthetic seg_ids and root_ids from the merge labels.
    # Each example i belongs to a "segment" derived from its y label.
    # Use a simple cluster: assign seg_id = i (unique segment per example),
    # root_id = 0 for no-merge examples, 1 for merge examples.
    # This lets us see purity across a set of segments.
    seg_ids = np.arange(len(y), dtype=np.int64)
    root_ids = y.astype(np.int64)

    try:
        from neuronauts.viz import plot_scaffold_purity
        with _dark_style():
            fig = plot_scaffold_purity(seg_ids, root_ids, role="merge dataset",
                                       title="Merge Dataset — Segment/Root Purity (synthetic)",
                                       figsize=(10, 4))
            # Re-apply dark background since viz.py creates its own figure
            fig.set_facecolor("#1a1d27")
            for ax in fig.axes:
                ax.set_facecolor("#1a1d27")
                ax.tick_params(colors="#8892a4")
    except Exception as e:
        with _dark_style():
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(0.5, 0.5, f"Could not render purity plot:\n{e}",
                    ha="center", va="center", wrap=True)

    return _fig_to_png_response(fig)


@app.route("/api/plot/f1_history.png")
def plot_f1_history_route():
    """F1 history from ledger using viz.plot_f1_history_from_ledger."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    ledger_path = PROJECT_ROOT / "run_logs" / "research_ledger.jsonl"
    if not ledger_path.exists():
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No ledger", ha="center"); return _fig_to_png_response(fig)

    try:
        from neuronauts.viz import plot_f1_history_from_ledger
        with _dark_style():
            fig = plot_f1_history_from_ledger(str(ledger_path), max_runs=40,
                                              title="F1 History (from viz.py)")
            fig.set_facecolor("#1a1d27")
            for ax in fig.axes:
                ax.set_facecolor("#1a1d27")
    except Exception as e:
        with _dark_style():
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, str(e), ha="center", va="center", wrap=True)

    return _fig_to_png_response(fig)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    ledger = read_ledger()
    models = read_model_metrics()
    latest_cycle = read_latest_cycle()
    step_timeline = build_step_timeline()
    is_running = detect_running()

    dataset_stats = read_dataset_stats()
    training_losses = read_training_losses()
    v2_train_log = read_v2_train_log()

    return jsonify({
        "ledger": ledger,
        "models": models,
        "latest_cycle": latest_cycle,
        "step_timeline": step_timeline,
        "dataset_stats": dataset_stats,
        "training_losses": training_losses,
        "v2_train_log": v2_train_log,
        "is_running": is_running,
        "pipeline_steps": PIPELINE_STEPS,
        "step_labels": STEP_LABELS,
        "server_time": time.time(),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5050, use_reloader=True)
