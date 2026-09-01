"""NeuronautS evaluation dashboard.

Launch:
    streamlit run dashboard/streamlit_app.py

Load a result bundle produced by:
    python scripts/spatial_variance.py --save-bundle results.json ...
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from neuroglancer_export import bundle_to_neuroglancer_url, neuroglass_instructions

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuronautS Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Bundle loading ────────────────────────────────────────────────────────────
@st.cache_data
def load_bundle(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_from_upload(uploaded) -> dict:
    return json.load(uploaded)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🧠 NeuronautS")
st.sidebar.markdown("---")

uploaded = st.sidebar.file_uploader("Upload result bundle (.json)", type=["json"])
bundle_path = st.sidebar.text_input("Or enter bundle path")

bundle: dict | None = None
if uploaded is not None:
    try:
        bundle = _load_from_upload(uploaded)
    except Exception as e:
        st.sidebar.error(f"Failed to parse: {e}")
elif bundle_path:
    try:
        bundle = load_bundle(bundle_path)
    except Exception as e:
        st.sidebar.error(f"Failed to load: {e}")

if bundle is None:
    st.title("NeuronautS Evaluation Dashboard")
    st.info(
        "**Load a result bundle to explore results.**\n\n"
        "Generate one with:\n"
        "```bash\n"
        "python scripts/spatial_variance.py \\\n"
        "  --checkpoint /tmp/neuronauts_dual_trained.pt \\\n"
        "  --dual-side --save-bundle results.json\n"
        "```"
    )
    st.stop()

# ── Meta ──────────────────────────────────────────────────────────────────────
meta = bundle.get("meta", {})
st.sidebar.markdown("**Run info**")
st.sidebar.write(f"Version: v{meta.get('version', '?')}")
st.sidebar.write(f"Regions: {', '.join(meta.get('train_regions', []))}")
ts = meta.get("timestamp", "")[:19]
if ts:
    st.sidebar.write(f"Run: {ts}")
ckpt = meta.get("checkpoint", "")
if ckpt:
    st.sidebar.write(f"Checkpoint: `{Path(ckpt).name}`")

calib = bundle.get("calibration", {})
if calib.get("T"):
    st.sidebar.write(f"Temp T={calib['T']:.4f}  ECE={calib.get('ece_train', float('nan')):.4f}")

# ── Tab layout ────────────────────────────────────────────────────────────────
tab_summary, tab_explore, tab_ooc = st.tabs(["Summary", "Region Explorer", "Out-of-Column"])

# ════════════════════════════════════════════════════════════════════════════
# Tab 1 — Summary metrics table
# ════════════════════════════════════════════════════════════════════════════
with tab_summary:
    st.header("In-column results")

    in_col = [r for r in bundle.get("in_col_results", []) if "error" not in r]
    if not in_col:
        st.warning("No in-column results in this bundle.")
    else:
        df = pd.DataFrame(in_col).set_index("name")

        # Which columns to show in the main table
        MAIN_COLS = [
            "ari", "merge_p", "merge_r", "over", "under", "fk",
            "n_merges_pred", "n_splits_pred", "n_true_merges", "n_output_cands",
            "syn_pre_min", "syn_pre_max", "syn_pre_med",
            "conn_f1", "conn_f1_undir",
            "dual_f1", "dual_f1_undir",
            "lg_and_f1", "lg_post_f1",
            "cable_med",
        ]
        show_cols = [c for c in MAIN_COLS if c in df.columns]
        int_cols = {"syn_pre_min", "syn_pre_max", "syn_pre_med", "cable_med",
                    "n_merges_pred", "n_splits_pred", "n_true_merges", "n_output_cands"}
        fmt = {c: "{:.3f}" for c in show_cols if c not in int_cols}
        fmt.update({c: "{:.0f}" for c in show_cols if c in int_cols})

        st.dataframe(
            df[show_cols].style.format(fmt, na_rep="—"),
            use_container_width=True,
        )

        # Sparkline-style bar chart for key metrics
        plot_metric = st.selectbox(
            "Plot metric",
            [c for c in ("ari", "conn_f1", "dual_f1", "lg_and_f1", "merge_p", "over", "under",
                         "n_merges_pred", "n_splits_pred", "n_output_cands")
             if c in df.columns],
            index=0,
        )
        fig_bar = px.bar(
            df.reset_index(), x="name", y=plot_metric,
            range_y=[0, 1], title=plot_metric,
            color=plot_metric, color_continuous_scale="RdYlGn",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Scale metrics
    scale_cols = [c for c in ("n_merges_pred", "n_splits_pred", "n_true_merges",
                               "n_output_cands") if in_col and c in df.columns]
    if scale_cols:
        st.subheader("Scale metrics — merge/split decisions per region")
        st.caption(
            "n_merges: edges the model chose to merge  ·  "
            "n_splits: edges the model chose to split  ·  "
            "n_true_merges: ground-truth same-neuron edges  ·  "
            "out_cands: distinct predicted clusters"
        )
        scale_df = df[scale_cols].copy()
        scale_df.columns = [c.replace("n_", "").replace("_pred", "").replace("_cands", "_candidates")
                            for c in scale_cols]
        st.dataframe(scale_df.style.format("{:.0f}", na_rep="—"), use_container_width=True)

        fig_scale = px.bar(
            df[scale_cols].reset_index().melt(id_vars="name", var_name="metric", value_name="count"),
            x="name", y="count", color="metric", barmode="group",
            title="Merge/split decisions at scale",
        )
        st.plotly_chart(fig_scale, use_container_width=True)

    # Synapse-per-fragment distribution summary
    if in_col and "syn_pre_min" in df.columns:
        st.subheader("Synapse count per fragment — pre side")
        dist_df = df[["syn_pre_min", "syn_pre_med", "syn_pre_max"]].rename(columns={
            "syn_pre_min": "min", "syn_pre_med": "median", "syn_pre_max": "max"
        })
        st.dataframe(dist_df.style.format("{:.0f}"), use_container_width=True)

    if in_col and "syn_post_med" in df.columns:
        st.subheader("Synapse count per fragment — post side (dual-mode)")
        dist_df_post = df[[c for c in ("syn_post_min", "syn_post_med", "syn_post_max")
                            if c in df.columns]].rename(columns={
            "syn_post_min": "min", "syn_post_med": "median", "syn_post_max": "max"
        })
        st.dataframe(dist_df_post.style.format("{:.0f}"), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# Tab 2 — Per-region explorer: 3D synapses + skeleton + neuron table + Neuroglass
# ════════════════════════════════════════════════════════════════════════════
with tab_explore:
    bboxes = bundle.get("bboxes", {})
    if not bboxes:
        st.warning("No bbox data — re-run with --save-bundle.")
        st.stop()

    selected_bbox = st.selectbox("Region", list(bboxes.keys()))
    bbox_data = bboxes[selected_bbox]
    synapses = bbox_data.get("synapses", {})
    neurons = bbox_data.get("neurons", {})

    col_vis, col_info = st.columns([2, 1])

    with col_vis:
        positions = synapses.get("positions_nm", [])
        if positions:
            pos_arr = np.array(positions, dtype=float)
            clus = np.array(synapses.get("pred_cluster", [0] * len(positions)))
            pre_root = np.array(synapses.get("pre_root_id", [0] * len(positions)))

            color_by = st.radio("Colour by", ["Predicted cluster", "True pre-root"],
                                horizontal=True)
            color_val = clus.astype(str) if color_by == "Predicted cluster" \
                        else pre_root.astype(str)

            fig3d = go.Figure()
            unique_colors = np.unique(color_val)
            palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24

            for i, uid in enumerate(unique_colors[:48]):
                mask = color_val == uid
                fig3d.add_trace(go.Scatter3d(
                    x=pos_arr[mask, 0] / 1000,
                    y=pos_arr[mask, 1] / 1000,
                    z=pos_arr[mask, 2] / 1000,
                    mode="markers",
                    marker=dict(size=2, color=palette[i % len(palette)]),
                    name=uid,
                    showlegend=len(unique_colors) <= 20,
                ))

            fig3d.update_layout(
                title=f"Synapse cloud — {selected_bbox}",
                scene=dict(
                    xaxis_title="X (µm)", yaxis_title="Y (µm)", zaxis_title="Z (µm)"
                ),
                height=520,
                margin=dict(l=0, r=0, b=0, t=40),
            )
            st.plotly_chart(fig3d, use_container_width=True)

        # Skeleton viewer
        if neurons and st.checkbox("Show skeletons (top-20 by synapse count)"):
            fig_sk = go.Figure()
            sorted_neurons = sorted(neurons.items(),
                                    key=lambda kv: -kv[1].get("n_synapses", 0))
            palette = px.colors.qualitative.Dark24
            for i, (cid, n) in enumerate(sorted_neurons[:20]):
                verts = n.get("vertices_nm", [])
                edges = n.get("edges", [])
                if not verts or not edges:
                    continue
                v = np.array(verts, dtype=float)
                xs, ys, zs = [], [], []
                for e in edges:
                    xs += [v[e[0], 0] / 1000, v[e[1], 0] / 1000, None]
                    ys += [v[e[0], 1] / 1000, v[e[1], 1] / 1000, None]
                    zs += [v[e[0], 2] / 1000, v[e[1], 2] / 1000, None]
                fig_sk.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs, mode="lines",
                    line=dict(color=palette[i % len(palette)], width=2),
                    name=f"c{cid} ({n.get('n_synapses', 0)} syn)",
                ))
            fig_sk.update_layout(
                title=f"Skeletons — {selected_bbox}",
                scene=dict(xaxis_title="X (µm)", yaxis_title="Y (µm)", zaxis_title="Z (µm)"),
                height=520, margin=dict(l=0, r=0, b=0, t=40),
            )
            st.plotly_chart(fig_sk, use_container_width=True)

    with col_info:
        # Neuroglass link
        st.subheader("Open in Neuroglass")
        ng_url = bundle_to_neuroglancer_url(bundle, selected_bbox, max_neurons=20)
        if ng_url:
            st.markdown(neuroglass_instructions(ng_url))
            st.text_area("Neuroglancer URL", ng_url, height=80, key="ng_url_area")
        else:
            st.info("No skeleton data available for this region.")

        # Neuron stats table
        st.subheader(f"Neurons ({len(neurons)})")
        if neurons:
            rows = []
            for cid, n in neurons.items():
                m = n.get("metrics", {})
                rows.append({
                    "cluster": cid,
                    "n_syn": n.get("n_synapses", 0),
                    "cable_µm": round(m.get("cable_length_um", 0)),
                    "branches": m.get("n_branch_points", 0),
                    "tree": "✓" if m.get("is_tree") else "✗",
                    "soma": "✓" if n.get("has_soma") else "",
                    "true_root": str(n.get("true_root_id", "")),
                })
            df_n = pd.DataFrame(rows).sort_values("n_syn", ascending=False)
            st.dataframe(
                df_n.set_index("cluster"),
                use_container_width=True,
                height=420,
            )

            # Synapse count histogram
            syn_counts = [n.get("n_synapses", 0) for n in neurons.values()]
            fig_hist = px.histogram(
                x=syn_counts, nbins=30,
                labels={"x": "Synapses per cluster"},
                title="Synapse count distribution",
            )
            fig_hist.update_layout(height=200, margin=dict(l=0, r=0, b=30, t=40))
            st.plotly_chart(fig_hist, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# Tab 3 — Out-of-column results
# ════════════════════════════════════════════════════════════════════════════
with tab_ooc:
    st.header("Out-of-column results")

    ooc = bundle.get("ooc_results", [])
    good_ooc = [r for r in ooc if "error" not in r]
    err_ooc = [r for r in ooc if "error" in r]

    if not good_ooc:
        st.info("No out-of-column results in this bundle.")
    else:
        df_ooc = pd.DataFrame(good_ooc).set_index("name")
        OOC_COLS = ["cable_med", "max_path_med", "tort_med", "is_tree", "n_neurons",
                    "syn_pre_min", "syn_pre_max", "syn_pre_med"]
        show_ooc = [c for c in OOC_COLS if c in df_ooc.columns]
        fmt_ooc = {c: "{:.0f}" for c in ("cable_med", "max_path_med",
                                          "syn_pre_min", "syn_pre_max", "syn_pre_med")}
        fmt_ooc.update({c: "{:.3f}" for c in ("tort_med", "is_tree")
                        if c in show_ooc and c not in fmt_ooc})
        st.dataframe(
            df_ooc[show_ooc].style.format(fmt_ooc, na_rep="—"),
            use_container_width=True,
        )

    if err_ooc:
        st.subheader("Errors")
        for r in err_ooc:
            st.error(f"{r['name']}: {r['error']}")
