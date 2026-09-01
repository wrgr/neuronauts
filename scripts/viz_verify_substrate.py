"""Visual verification of each substrate stage.

Numbers alone have not been enough on this project, so every stage gets a
picture with an explicit pass/fail question attached:

  fig 1  region + proofread somas    -- do GT somas sit inside the region, and
                                        do the two coordinate frames agree?
  fig 2  one arbor, coloured by v117 -- does the L2 graph look like a neuron,
                                        and is each v117 atom a *contiguous*
                                        piece of it? Scattered colours would
                                        mean the lineage mapping is scrambled.
  fig 3  soma check                  -- does the nucleus position land on a
                                        dense blob of L2 nodes?
  fig 4  atom statistics             -- fragment sizes and per-cell counts.

Fig 2 is the load-bearing one. If v117 atoms were wrong they would appear as
salt-and-pepper colour noise over the arbor rather than as connected runs.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuronauts.data import lineage as L  # noqa: E402
from neuronauts.harness.substrate import (  # noqa: E402
    fetch_l2_coords, fetch_l2_graphs, fetch_v117_map, load_proofread_table,
    region_bounds, select_cells,
)

OUT = Path("results/figures")
PROJ = [(0, 1, "x", "y"), (0, 2, "x", "z")]


def _style(ax, xl, yl, title=""):
    ax.set_xlabel(f"{xl} (um)")
    ax.set_ylabel(f"{yl} (um)")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.15, lw=0.4)


def fig_region(sel_all, sel_tier, lo, hi, centre_um, side_um, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (i, j, xl, yl) in zip(axes, PROJ):
        a = sel_all[["x_nm", "y_nm", "z_nm"]].to_numpy() / 1000
        b = sel_tier[["x_nm", "y_nm", "z_nm"]].to_numpy() / 1000
        ax.scatter(a[:, i], a[:, j], s=9, c="0.75", label=f"all proofread ({len(a)})")
        ax.scatter(b[:, i], b[:, j], s=18, c="crimson",
                   label=f"gold in region ({len(b)})")
        r = plt.Rectangle((lo[i] / 1000, lo[j] / 1000),
                          (hi[i] - lo[i]) / 1000, (hi[j] - lo[j]) / 1000,
                          fill=False, ec="navy", lw=1.8, ls="--",
                          label="region")
        ax.add_patch(r)
        _style(ax, xl, yl)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Stage 1 - region and ground-truth somas "
                 f"({side_um:g} um cube @ {centre_um} um)\n"
                 "CHECK: red somas inside the dashed box; frames agree",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)


def _atom_palette(atoms, seed=0):
    """60 distinct colours, shuffled so neighbours rarely collide.

    tab20 alone repeats every 20, which made several distinct atoms render as
    one blue mass and hid exactly the structure the figure is meant to show.
    """
    cols = []
    for name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(name)
        cols.extend(cmap(i / 19.0) for i in range(20))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cols))
    return {f: cols[order[i % len(cols)]] for i, f in enumerate(atoms)}


def atom_contiguity(frag, edges, min_nodes=5):
    """Connected components each v117 atom forms in the L2 adjacency graph.

    If the lineage mapping is right an atom is a physically connected piece of
    neurite, so it should form ~1 component. Scrambling the labels destroys
    that, which is the control this is compared against.
    """
    n = len(frag)
    adj = [[] for _ in range(n)]
    for a, b in edges.tolist():
        adj[a].append(b)
        adj[b].append(a)

    out = {}
    for atom in np.unique(frag[frag > 0]):
        idx = np.flatnonzero(frag == atom)
        if len(idx) < min_nodes:
            continue
        member = set(idx.tolist())
        seen, comps = set(), 0
        for s in idx.tolist():
            if s in seen:
                continue
            comps += 1
            stack = [s]
            seen.add(s)
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v in member and v not in seen:
                        seen.add(v)
                        stack.append(v)
        out[int(atom)] = (comps, len(idx))
    return out


def fig_arbor(pos, edges, frag, soma_nm, cell_id, path, min_nodes=5):
    """Arbor coloured by v117 atom. Contiguity here is the real check."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    counts = collections.Counter(frag[frag > 0].tolist())
    big = [f for f, c in counts.most_common() if c >= min_nodes]
    colour = _atom_palette(big)

    for ax, (i, j, xl, yl) in zip(axes, PROJ):
        # Real L2 adjacency as grey wiring underneath.
        seg = np.stack([pos[edges[:, 0]][:, [i, j]],
                        pos[edges[:, 1]][:, [i, j]]], axis=1) / 1000
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(seg, colors="0.82", linewidths=0.45,
                                         zorder=1))
        unres = frag == 0
        if unres.any():
            ax.scatter(pos[unres][:, i] / 1000, pos[unres][:, j] / 1000,
                       s=3, c="0.6", marker="x", zorder=2,
                       label=f"no v117 ({int(unres.sum())})")
        for f in big:
            m = frag == f
            ax.scatter(pos[m][:, i] / 1000, pos[m][:, j] / 1000, s=7,
                       color=colour[f], zorder=3)
        small = np.isin(frag, list(big), invert=True) & (frag > 0)
        if small.any():
            ax.scatter(pos[small][:, i] / 1000, pos[small][:, j] / 1000,
                       s=4, c="0.45", zorder=2)
        ax.scatter([soma_nm[i] / 1000], [soma_nm[j] / 1000], s=190,
                   marker="*", c="red", ec="k", lw=0.7, zorder=5, label="soma")
        _style(ax, xl, yl)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Stage 2 - proofread cell {cell_id}: L2 arbor coloured by v117 atom\n"
        f"{len(pos):,} L2 nodes | {len(edges):,} real adjacency edges | "
        f"{len(counts)} v117 atoms ({len(big)} with >={min_nodes} nodes)\n"
        "CHECK: colours form connected runs (lineage OK), not salt-and-pepper",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)


def fig_soma(pos, soma_nm, cell_id, path, win_um=25.0):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    w = win_um * 1000
    for ax, (i, j, xl, yl) in zip(axes, PROJ):
        m = (np.abs(pos[:, i] - soma_nm[i]) < w) & (np.abs(pos[:, j] - soma_nm[j]) < w)
        ax.hexbin(pos[m][:, i] / 1000, pos[m][:, j] / 1000, gridsize=38,
                  cmap="viridis", mincnt=1)
        ax.scatter([soma_nm[i] / 1000], [soma_nm[j] / 1000], s=230, marker="*",
                   c="red", ec="w", lw=1.2, zorder=5)
        _style(ax, xl, yl, f"L2 node density near nucleus ({int(m.sum())} nodes)")
    fig.suptitle(f"Stage 3 - soma check, cell {cell_id}\n"
                 "CHECK: red star sits on the density peak (nucleus table and "
                 "L2 geometry agree)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)


def fig_contiguity(real, scrambled, path):
    """Real vs label-scrambled contiguity: the quantitative lineage check."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    bins = np.arange(0.5, 14.5, 1.0)
    axes[0].hist([real, scrambled], bins=bins, label=["real v117", "scrambled control"],
                 color=["seagreen", "0.7"], ec="w")
    axes[0].set_xlabel("connected components per atom")
    axes[0].set_ylabel("atoms")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=9)
    axes[0].set_title("Lower is better; 1 = perfectly contiguous")

    r1 = float(np.mean(np.asarray(real) == 1))
    s1 = float(np.mean(np.asarray(scrambled) == 1))
    axes[1].bar(["real v117", "scrambled"], [r1, s1],
                color=["seagreen", "0.7"], ec="k", lw=0.6)
    axes[1].set_ylabel("fraction of atoms that are a single component")
    axes[1].set_ylim(0, 1.05)
    for i, v in enumerate([r1, s1]):
        axes[1].text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=11)
    axes[1].set_title(f"real median={int(np.median(real))}  "
                      f"scrambled median={int(np.median(scrambled))}")
    for a in axes:
        a.grid(alpha=0.2, lw=0.4)
    fig.suptitle("Stage 5 - is the v117 lineage mapping real?\n"
                 "CHECK: real atoms are near-contiguous; scrambled labels are "
                 "not. If these match, the mapping is meaningless.", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)


def fig_stats(per_cell_frags, all_frag_sizes, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].hist(all_frag_sizes, bins=np.logspace(0, np.log10(max(all_frag_sizes) + 1), 34),
                 color="steelblue", ec="w")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("L2 nodes per v117 atom")
    axes[0].set_ylabel("count")
    axes[0].set_title("Atom size distribution")

    axes[1].hist(per_cell_frags, bins=18, color="darkorange", ec="w")
    axes[1].set_xlabel("v117 atoms per proofread cell")
    axes[1].set_ylabel("cells")
    axes[1].set_title(f"False-split degree\nmedian={int(np.median(per_cell_frags))}")

    ks = [1, 2, 3, 5, 10, 20, 50]
    kept = [int((np.asarray(all_frag_sizes) >= k).sum()) for k in ks]
    axes[2].plot(ks, kept, "o-", color="seagreen")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("min L2 nodes")
    axes[2].set_ylabel("atoms kept")
    axes[2].set_title("Effect of the size filter")
    for a in axes:
        a.grid(alpha=0.2, lw=0.4)
    fig.suptitle("Stage 4 - atom statistics  |  CHECK: many atoms per cell "
                 "= real false-split signal to recover", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=1822)
    ap.add_argument("--centre-um", type=float, nargs=3, default=[663, 591, 860])
    ap.add_argument("--side-um", type=float, default=200.0)
    ap.add_argument("--n-cells", type=int, default=8)
    ap.add_argument("--detail-cells", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cache", default="data/substrate/viz_check")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    token = L.DEFAULT_TOKEN
    cache = Path(args.cache)
    lo, hi, seg = region_bounds(args.centre_um, args.side_um)

    df = load_proofread_table(args.version)
    sel = select_cells(df, args.centre_um, args.side_um, "gold")
    print(f"gold proofread cells in region: {len(sel)}", flush=True)
    fig_region(df, sel, lo, hi, args.centre_um, args.side_um,
               OUT / "01_region_somas.png")
    print(f"  wrote {OUT}/01_region_somas.png", flush=True)

    cells = sel.head(args.n_cells)
    roots = [int(r) for r in cells["pt_root_id"]]
    soma = {int(r.pt_root_id): np.array([r.x_nm, r.y_nm, r.z_nm])
            for _, r in cells.iterrows()}

    print("fetching L2 adjacency ...", flush=True)
    graphs = fetch_l2_graphs(roots, seg, cache / "l2_graphs", args.workers)
    pool = np.unique(np.concatenate([np.unique(e) for e in graphs.values()]))
    print(f"  {len(graphs)} cells, {len(pool):,} L2 nodes", flush=True)

    coords = fetch_l2_coords(pool, token, cache / "l2_coords.npz", verbose=False)
    v117 = fetch_v117_map(pool, token, cache / "l2_v117.npz", verbose=False)
    print(f"  coords {len(coords)/len(pool):.1%} | "
          f"v117 {sum(1 for v in v117.values() if v)/len(pool):.1%}", flush=True)

    per_cell, all_sizes = [], []
    contig_real, contig_scram = [], []
    rng = np.random.default_rng(0)
    for n, (root, edges) in enumerate(graphs.items()):
        ids = np.unique(edges)
        ids = np.array([i for i in ids.tolist() if i in coords], np.uint64)
        if len(ids) < 10:
            continue
        idx = {int(v): k for k, v in enumerate(ids.tolist())}
        pos = np.stack([coords[int(i)] for i in ids])
        frag = np.array([v117.get(int(i), 0) for i in ids], np.uint64)
        ei = np.array([(idx[int(a)], idx[int(b)]) for a, b in edges.tolist()
                       if int(a) in idx and int(b) in idx], np.int32)

        c = collections.Counter(frag[frag > 0].tolist())
        per_cell.append(len(c))
        all_sizes.extend(c.values())

        # Quantitative lineage check, with a scrambled-label null.
        contig_real.extend(v[0] for v in atom_contiguity(frag, ei).values())
        shuffled = frag.copy()
        nz = np.flatnonzero(shuffled > 0)
        shuffled[nz] = shuffled[rng.permutation(nz)]
        contig_scram.extend(v[0] for v in atom_contiguity(shuffled, ei).values())

        if n < args.detail_cells:
            fig_arbor(pos, ei, frag, soma[root], root,
                      OUT / f"02_arbor_{root}.png")
            print(f"  wrote {OUT}/02_arbor_{root}.png", flush=True)
            fig_soma(pos, soma[root], root, OUT / f"03_soma_{root}.png")
            print(f"  wrote {OUT}/03_soma_{root}.png", flush=True)

    fig_stats(per_cell, all_sizes, OUT / "04_atom_stats.png")
    print(f"  wrote {OUT}/04_atom_stats.png", flush=True)
    fig_contiguity(contig_real, contig_scram, OUT / "05_contiguity.png")
    print(f"  wrote {OUT}/05_contiguity.png", flush=True)

    print(f"\natoms/cell: median={int(np.median(per_cell))} "
          f"range={min(per_cell)}-{max(per_cell)}")
    print(f"atom sizes: median={int(np.median(all_sizes))} "
          f"max={int(max(all_sizes))}")
    r1 = float(np.mean(np.asarray(contig_real) == 1))
    s1 = float(np.mean(np.asarray(contig_scram) == 1))
    print(f"\nlineage contiguity ({len(contig_real)} atoms >=5 nodes):")
    print(f"  real v117 single-component : {r1:.1%} "
          f"(median {int(np.median(contig_real))} components)")
    print(f"  scrambled control          : {s1:.1%} "
          f"(median {int(np.median(contig_scram))} components)")
    print("  -> a large gap means the v117 mapping tracks real physical "
          "objects, not noise")


if __name__ == "__main__":
    main()
