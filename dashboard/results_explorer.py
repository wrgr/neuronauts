"""NeuronautS Results Explorer — Streamlit dashboard.

Usage
-----
  streamlit run dashboard/results_explorer.py

Or with a pre-loaded bundle:
  streamlit run dashboard/results_explorer.py -- --bundle /tmp/my_bundle.json

Expects a result bundle produced by:
  python scripts/spatial_variance.py --dual-side --balanced-dual \\
    --save-bundle /tmp/my_bundle.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make repo importable when run from the dashboard/ directory.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.neuroglancer_export import (
    _PALETTE,
    _nm_to_vox,
    bundle_to_neuroglancer_url,
    build_neuroglancer_state,
    neuroglass_instructions,
    state_to_neuroglancer_url,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuronautS Explorer",
    page_icon="🧠",
    layout="wide",
)

# ── Bundle loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading bundle …")
def load_bundle(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _sidebar_load() -> dict | None:
    st.sidebar.title("🧠 NeuronautS Explorer")
    st.sidebar.markdown("---")

    # Check CLI arg first.
    default_path = ""
    if len(sys.argv) > 1:
        import argparse
        p = argparse.ArgumentParser(add_help=False)
        p.add_argument("--bundle", default="")
        known, _ = p.parse_known_args()
        default_path = known.bundle

    path_input = st.sidebar.text_input(
        "Bundle path",
        value=default_path or st.session_state.get("bundle_path", ""),
        placeholder="/tmp/neuronauts_bundle.json",
    )
    uploaded = st.sidebar.file_uploader("Or upload bundle", type=["json"])

    bundle = None
    if uploaded is not None:
        bundle = json.loads(uploaded.read())
        st.sidebar.success(f"Loaded from upload ({len(bundle.get('bboxes', {}))} bboxes)")
    elif path_input and Path(path_input).exists():
        bundle = load_bundle(path_input)
        st.session_state["bundle_path"] = path_input
        st.sidebar.success(f"Loaded: {Path(path_input).name}")
    elif path_input:
        st.sidebar.error("File not found.")

    if bundle:
        meta = bundle.get("meta", {})
        st.sidebar.markdown("**Run info**")
        st.sidebar.json({
            "date": meta.get("timestamp", "?")[:10],
            "train": meta.get("train_regions", "?"),
            "version": f"v{meta.get('version', '?')}",
            "dual_side": meta.get("dual_side", False),
        })

    return bundle


# ── Helpers ───────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "ari": "ARI",
    "merge_p": "merge_P",
    "merge_r": "merge_R",
    "over": "over_merge",
    "under": "under_merge",
    "conn_f1": "conn_F1",
    "conn_p": "conn_P",
    "conn_r": "conn_R",
    "dual_f1": "dual_F1",
    "dual_p": "dual_P",
    "dual_r": "dual_R",
    "lg_pre_f1": "lg_pre_F1",
    "lg_and_f1": "lg_and_F1",
    "lg_post_f1": "lg_post_F1",
    "syn_attr_acc": "syn_attr_acc",
}


def _neuron_table(neurons: dict) -> pd.DataFrame:
    rows = []
    for cid, n in neurons.items():
        m = n.get("metrics", {})
        rows.append({
            "cluster": cid,
            "true_root_id": n.get("true_root_id", 0),
            "n_synapses": n.get("n_synapses", 0),
            "cable_um": round(m.get("cable_length_um", 0), 1),
            "branch_pts": m.get("n_branch_points", 0),
            "is_tree": m.get("is_tree", False),
            "tortuosity": round(m.get("tortuosity", float("nan")), 2),
            "caliber_um": round(m.get("mean_caliber_um", float("nan")), 3),
            "has_soma": n.get("has_soma", False),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("n_synapses", ascending=False).reset_index(drop=True)
    return df


def _skeleton_figure(neurons: dict, selected_ids: list[str],
                     synapses: dict | None = None) -> go.Figure:
    fig = go.Figure()

    for i, cid in enumerate(selected_ids):
        n = neurons.get(str(cid), neurons.get(cid))
        if n is None:
            continue
        verts = n.get("vertices_nm", [])
        edges = n.get("edges", [])
        colour = _PALETTE[i % len(_PALETTE)]
        true_id = n.get("true_root_id", 0)
        label = f"cluster {cid}" + (f" [gt:{true_id}]" if true_id else "")

        if verts and edges:
            v = np.array(verts, dtype=float) / 1000  # → µm
            xs, ys, zs = [], [], []
            for e in edges:
                a, b = v[e[0]], v[e[1]]
                xs += [float(a[0]), float(b[0]), None]
                ys += [float(a[1]), float(b[1]), None]
                zs += [float(a[2]), float(b[2]), None]
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=colour, width=3),
                name=label,
                hoverinfo="name",
            ))
        elif verts:
            # Cloud fragment (no edges) — show as scatter points
            v = np.array(verts, dtype=float) / 1000
            fig.add_trace(go.Scatter3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                mode="markers",
                marker=dict(color=colour, size=2),
                name=label + " (cloud)",
            ))

    # Synapse scatter
    if synapses and synapses.get("positions_nm"):
        pos = np.array(synapses["positions_nm"], dtype=float) / 1000
        pred = synapses.get("pred_cluster", [0] * len(pos))
        sel_set = {str(c) for c in selected_ids} | {int(c) for c in selected_ids if str(c).isdigit()}
        mask = np.array([str(p) in sel_set or p in sel_set for p in pred])
        if mask.any():
            fig.add_trace(go.Scatter3d(
                x=pos[mask, 0], y=pos[mask, 1], z=pos[mask, 2],
                mode="markers",
                marker=dict(color="yellow", size=3, symbol="circle"),
                name="Synapses",
                hoverinfo="name",
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title="x (µm)",
            yaxis_title="y (µm)",
            zaxis_title="z (µm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(x=0, y=1),
        height=650,
    )
    return fig


# ── Tab: Summary ──────────────────────────────────────────────────────────────

def _tab_summary(bundle: dict) -> None:
    in_col = bundle.get("in_col_results", [])
    if not in_col:
        st.info("No in-column results in bundle.")
        return

    # Build display table
    metrics = ["ari", "merge_p", "merge_r", "over", "under",
               "conn_f1", "conn_p", "conn_r", "dual_f1", "lg_and_f1", "lg_post_f1"]

    rows = []
    for r in in_col:
        if "error" in r:
            rows.append({"Location": r["name"], **{METRIC_LABELS.get(m, m): "ERR" for m in metrics}})
        else:
            row = {"Location": r.get("name", "?")}
            for m in metrics:
                v = r.get(m, float("nan"))
                row[METRIC_LABELS.get(m, m)] = round(v, 3) if isinstance(v, float) else v
            rows.append(row)

    df = pd.DataFrame(rows).set_index("Location")
    st.subheader("In-column metrics")
    st.dataframe(
        df.style.background_gradient(cmap="RdYlGn", axis=0, subset=[
            c for c in df.columns if c in {"ARI", "merge_P", "conn_F1", "dual_F1", "lg_and_F1"}
        ]).format("{:.3f}", na_rep="—"),
        use_container_width=True,
    )

    # Bar chart comparison
    chart_metric = st.selectbox(
        "Compare metric",
        [METRIC_LABELS[m] for m in metrics if METRIC_LABELS[m] in df.columns],
        index=0,
    )
    fig = go.Figure(go.Bar(
        x=df.index.tolist(),
        y=df[chart_metric].tolist(),
        marker_color=["#4363d8"] * len(df),
        text=[f"{v:.3f}" for v in df[chart_metric].fillna(0)],
        textposition="outside",
    ))
    fig.update_layout(
        yaxis=dict(range=[0, 1.05], title=chart_metric),
        height=350, margin=dict(t=20, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Calibration
    cal = bundle.get("calibration", {})
    if cal:
        c1, c2 = st.columns(2)
        c1.metric("Temperature T", f"{cal.get('T', float('nan')):.4f}")
        c2.metric("ECE (train-A)", f"{cal.get('ece_train', float('nan')):.4f}")

    # OOC
    ooc = bundle.get("ooc_results", [])
    if ooc:
        st.subheader("Out-of-column (shape plausibility)")
        ooc_df = pd.DataFrame([
            {k: v for k, v in r.items() if k != "error"}
            for r in ooc if "error" not in r
        ])
        if not ooc_df.empty:
            st.dataframe(ooc_df.round(3), use_container_width=True)


# ── Tab: Bbox Explorer ────────────────────────────────────────────────────────

def _tab_explore(bundle: dict) -> str | None:
    """Returns the selected bbox_name (for use in other tabs)."""
    bboxes = bundle.get("bboxes", {})
    in_col = bundle.get("in_col_results", [])

    bbox_names = list(bboxes.keys())
    if not bbox_names:
        st.warning("Bundle contains no per-bbox skeleton data (was --save-bundle used?)")
        return None

    selected = st.selectbox("Select test bbox", bbox_names)
    st.session_state["selected_bbox"] = selected

    bbox_data = bboxes[selected]
    neurons = bbox_data.get("neurons", {})

    # Metrics card
    row = next((r for r in in_col if r.get("name") == selected), {})
    if row:
        cols = st.columns(5)
        for col, (k, label) in zip(cols, [
            ("ari", "ARI"), ("merge_p", "merge_P"), ("conn_f1", "conn_F1"),
            ("dual_f1", "dual_F1"), ("lg_and_f1", "AND_F1"),
        ]):
            v = row.get(k, float("nan"))
            col.metric(label, f"{v:.3f}" if isinstance(v, float) else str(v))

        cols2 = st.columns(5)
        for col, (k, label) in zip(cols2, [
            ("merge_r", "merge_R"), ("over", "over_merge"), ("under", "under_merge"),
            ("syn_attr_acc", "syn_attr"), ("lg_post_f1", "post_F1"),
        ]):
            v = row.get(k, float("nan"))
            col.metric(label, f"{v:.3f}" if isinstance(v, float) else str(v))

    # Neuron table
    st.subheader(f"Neurons in {selected}  ({len(neurons)} clusters)")
    df = _neuron_table(neurons)
    if not df.empty:
        event = st.dataframe(
            df.style.format({
                "cable_um": "{:.1f}", "tortuosity": "{:.2f}",
                "caliber_um": "{:.3f}",
            }),
            use_container_width=True,
            selection_mode="multi-row",
            on_select="rerun",
            key=f"neuron_table_{selected}",
        )
        # Store selected cluster IDs in session state
        sel_rows = event.selection.get("rows", []) if hasattr(event, "selection") else []
        if sel_rows:
            st.session_state["selected_clusters"] = df.iloc[sel_rows]["cluster"].tolist()
        elif "selected_clusters" not in st.session_state:
            # Default: top-5 by synapse count
            st.session_state["selected_clusters"] = df["cluster"].head(5).tolist()

    return selected


# ── Tab: 3D Skeleton ──────────────────────────────────────────────────────────

def _tab_skeleton(bundle: dict, bbox_name: str | None) -> None:
    if bbox_name is None:
        st.info("Select a bbox in the Explorer tab first.")
        return

    bboxes = bundle.get("bboxes", {})
    bbox_data = bboxes.get(bbox_name, {})
    neurons = bbox_data.get("neurons", {})
    synapses = bbox_data.get("synapses", {})

    all_ids = list(neurons.keys())
    default_ids = st.session_state.get("selected_clusters", all_ids[:5])

    selected_ids = st.multiselect(
        "Neurons to display",
        options=all_ids,
        default=[c for c in default_ids if c in all_ids],
        format_func=lambda c: (
            f"cluster {c}  "
            f"[{neurons[c].get('n_synapses', 0)} syn, "
            f"{neurons[c].get('metrics', {}).get('cable_length_um', 0):.0f} µm]"
        ),
    )
    if not selected_ids:
        st.info("Select at least one neuron above.")
        return

    show_syn = st.checkbox("Show synapse positions", value=True)
    fig = _skeleton_figure(neurons, selected_ids, synapses if show_syn else None)
    st.plotly_chart(fig, use_container_width=True)

    # Per-neuron stats
    if len(selected_ids) == 1:
        n = neurons.get(selected_ids[0], {})
        m = n.get("metrics", {})
        st.markdown(f"**Cluster {selected_ids[0]}**  |  "
                    f"True neuron: `{n.get('true_root_id', '?')}`  |  "
                    f"Synapses: {n.get('n_synapses', 0)}  |  "
                    f"Soma: {'✓' if n.get('has_soma') else '—'}")
        cols = st.columns(4)
        cols[0].metric("Cable", f"{m.get('cable_length_um', 0):.1f} µm")
        cols[1].metric("Branch pts", m.get("n_branch_points", 0))
        cols[2].metric("Tortuosity", f"{m.get('tortuosity', 0):.2f}")
        cols[3].metric("Mean caliber", f"{m.get('mean_caliber_um', 0):.3f} µm")

    # Save selected for Neuroglass tab
    st.session_state["neuroglass_clusters"] = selected_ids


# ── Tab: Neuroglass ───────────────────────────────────────────────────────────

def _tab_neuroglass(bundle: dict, bbox_name: str | None) -> None:
    st.subheader("Open in Neuroglass")
    st.markdown(
        "Neuroglass can open any Neuroglancer link. Generate a URL below, then "
        "import it at [app.neuroglass.com](https://app.neuroglass.com) via "
        "**Import → Neuroglancer Link**."
    )

    if bbox_name is None:
        st.info("Select a bbox in the Explorer tab first.")
        return

    bboxes = bundle.get("bboxes", {})
    bbox_data = bboxes.get(bbox_name, {})
    neurons = bbox_data.get("neurons", {})
    all_ids = list(neurons.keys())

    default_ids = st.session_state.get("neuroglass_clusters",
                                       st.session_state.get("selected_clusters", all_ids[:5]))

    col1, col2 = st.columns([3, 1])
    with col1:
        cluster_ids = st.multiselect(
            "Neurons to include",
            options=all_ids,
            default=[c for c in default_ids if c in all_ids],
            format_func=lambda c: f"cluster {c} [{neurons[c].get('n_synapses', 0)} syn]",
        )
    with col2:
        include_syn = st.checkbox("Include synapses", value=True)
        base_url = st.selectbox(
            "Viewer",
            ["https://neuroglancer-demo.appspot.com",
             "https://neuromancer-seung-import.appspot.com"],
        )

    if st.button("🔗 Generate Neuroglancer URL", type="primary"):
        if not cluster_ids:
            st.warning("Select at least one neuron.")
        else:
            state = build_neuroglancer_state(
                bbox_data,
                cluster_ids=cluster_ids,
                include_synapses=include_syn,
            )
            url = state_to_neuroglancer_url(state, base_url=base_url)
            st.session_state["ng_url"] = url
            url_chars = len(url)
            st.info(f"URL length: {url_chars:,} characters  "
                    f"({'⚠ may be too long for some browsers' if url_chars > 500_000 else '✓ OK'})")

    if "ng_url" in st.session_state:
        url = st.session_state["ng_url"]
        st.markdown("### Neuroglancer URL")
        st.code(url[:2000] + ("..." if len(url) > 2000 else ""), language=None)

        col_a, col_b = st.columns(2)
        with col_a:
            st.link_button("Open in Neuroglancer", url)
        with col_b:
            st.markdown("**Import into Neuroglass:**")
            st.markdown(
                "1. Open [app.neuroglass.com](https://app.neuroglass.com)\n"
                "2. Click **Import** → **Neuroglancer Link**\n"
                "3. Paste the URL above"
            )

        # JSON preview for debugging
        with st.expander("View state JSON"):
            state = build_neuroglancer_state(
                bbox_data, cluster_ids=cluster_ids, include_synapses=include_syn,
            )
            # Truncate annotations for display
            preview = dict(state)
            preview["layers"] = [
                {**lyr, "annotations": lyr.get("annotations", [])[:3]}
                for lyr in preview.get("layers", [])
            ]
            st.json(preview)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    bundle = _sidebar_load()

    if bundle is None:
        st.title("🧠 NeuronautS Results Explorer")
        st.markdown(
            "Load a result bundle produced by:\n"
            "```bash\n"
            "python scripts/spatial_variance.py \\\n"
            "  --dual-side --balanced-dual \\\n"
            "  --save-bundle /tmp/neuronauts_bundle.json\n"
            "```\n"
            "Enter the path in the sidebar or upload the file."
        )
        return

    tab_summary, tab_explore, tab_skel, tab_ng = st.tabs([
        "📊 Summary", "🔍 Bbox Explorer", "🧠 3D Skeleton", "🌐 Neuroglass"
    ])

    with tab_summary:
        _tab_summary(bundle)

    with tab_explore:
        bbox_name = _tab_explore(bundle)

    # Use session state to pass bbox_name across tabs
    bbox_name = st.session_state.get("selected_bbox")

    with tab_skel:
        _tab_skeleton(bundle, bbox_name)

    with tab_ng:
        _tab_neuroglass(bundle, bbox_name)


if __name__ == "__main__":
    main()
