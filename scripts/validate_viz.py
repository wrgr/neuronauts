#!/usr/bin/env python3
"""Grammar validation visualizer — Neuroglancer edition.

Loads a cached val box, runs the grammar on candidate synapse pairs, and
opens a Neuroglancer viewer showing TP / FP / FN pairs as colored line
annotations overlaid on the MICrONS EM volume, CAVE segmentation, and
neuron meshes.

Colour key
----------
    Green  — True Positive  (predicted connected, ground truth connected)
    Red    — False Positive  (predicted connected, actually different neurons)
    Orange — False Negative  (predicted disconnected, actually same neuron)
    Grey   — True Negative   (hidden by default; use --show-tn to include)

Usage
-----
Basic — open the first val box::

    python scripts/validate_viz.py \\
        --cache-dir data/boxes30 \\
        --grammar-path models/shared_grammar_real.pt

Show all boxes and pick by index::

    python scripts/validate_viz.py --cache-dir data/boxes30 --list-boxes

Specific box, show true negatives too::

    python scripts/validate_viz.py \\
        --cache-dir data/boxes30 \\
        --grammar-path models/shared_grammar_real.pt \\
        --box-idx 2 \\
        --show-tn

Requirements::

    pip install neuroglancer caveclient
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# MICrONS / CAVE layer URLs
# ---------------------------------------------------------------------------
# Use HTTPS so the browser can load tiles; gs:// is not directly accessible.
EM_SOURCE  = "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em"
SEG_SOURCE = "graphene://https://minnie65-proofreading.zetta.ai/segmentation/1.0/minnie65_8x8x40"

# MIP-0 voxel size (nm). Cached synapse pre_pt are box-relative MIP-2 voxels.
EM_VOX_NM = (8.0, 8.0, 40.0)
MIP2_TO_MIP0 = (4, 4, 1)  # one MIP-2 voxel = 4×4×1 in MIP-0


# ---------------------------------------------------------------------------
# Pair generation (all pairs, not capped)
# ---------------------------------------------------------------------------

def _root_groups(root_ids: np.ndarray) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for idx, r in enumerate(root_ids.tolist()):
        groups.setdefault(int(r), []).append(idx)
    return groups


def generate_all_candidate_pairs(
    synapses,
    *,
    max_positive_pairs: int = 200,
    max_negative_pairs: int = 200,
    max_roots_for_negatives: int = 80,
    seed: int = 0,
) -> list[tuple[int, int, int]]:
    """Return (left_idx, right_idx, label) for candidate synapse pairs.

    Uses pre-side root IDs only (post-side would double the pairs).
    Label 1 = same neuron, 0 = different neuron.
    """
    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int, int]] = []

    groups = _root_groups(synapses.pre_root_id)

    # Positives — pairs within the same root
    pos_groups = [(r, idx) for r, idx in groups.items() if len(idx) >= 2]
    rng.shuffle(pos_groups)  # type: ignore[arg-type]
    for root_id, indices in pos_groups:
        if len(pairs) >= max_positive_pairs:
            break
        arr = list(indices)
        rng.shuffle(arr)  # type: ignore[arg-type]
        for k in range(0, len(arr) - 1, 2):
            pairs.append((arr[k], arr[k + 1], 1))
            if len(pairs) >= max_positive_pairs:
                break

    n_pos = len(pairs)

    # Negatives — pairs across different roots
    neg_candidates = [(r, idx) for r, idx in groups.items() if len(idx) >= 1]
    if len(neg_candidates) > max_roots_for_negatives:
        chosen = rng.choice(len(neg_candidates), size=max_roots_for_negatives, replace=False)
        neg_candidates = [neg_candidates[i] for i in chosen]

    neg_pairs: list[tuple[float, int, int]] = []
    pts = synapses.pre_pt.astype(np.float64)
    for i in range(len(neg_candidates)):
        for j in range(i + 1, len(neg_candidates)):
            ra, ia = neg_candidates[i]
            rb, ib = neg_candidates[j]
            ca = pts[ia].mean(axis=0)
            cb = pts[ib].mean(axis=0)
            dist = float(np.linalg.norm(ca - cb))
            neg_pairs.append((dist, ia[0], ib[0]))

    neg_pairs.sort(key=lambda x: x[0])
    for _, li, ri in neg_pairs[:max_negative_pairs]:
        pairs.append((li, ri, 0))

    print(
        f"  Candidate pairs: {n_pos} positive, {len(pairs) - n_pos} negative "
        f"(from {len(groups)} root groups)"
    )
    return pairs


# ---------------------------------------------------------------------------
# Grammar scoring
# ---------------------------------------------------------------------------

def score_pairs(
    grammar_model,
    synapses,
    pairs: list[tuple[int, int, int]],
    device,
) -> list[tuple[int, int, int, int, float]]:
    """Score each pair with the grammar model.

    Returns list of (left_idx, right_idx, label, pred, score).
    """
    import torch
    from neuronauts.merge_dataset import _sequence_from_points

    results = []
    grammar_model.eval()

    for left_idx, right_idx, label in pairs:
        # For visualization we just need a simple geometric path per synapse.
        # Construct a 2-point path using the two synapse positions; this yields
        # a single-step (edge_len, radius, curvature) sequence compatible with
        # the trained grammar.
        left_pts = synapses.pre_pt[[left_idx, right_idx]]
        right_pts = synapses.pre_pt[[right_idx, left_idx]]

        left_seq = _sequence_from_points(left_pts)
        right_seq = _sequence_from_points(right_pts)

        if len(left_seq) == 0 or len(right_seq) == 0:
            continue

        with torch.no_grad():
            lx = torch.from_numpy(left_seq[None]).float().to(device)
            rx = torch.from_numpy(right_seq[None]).float().to(device)
            lm = torch.zeros((1, lx.shape[1]), dtype=torch.bool, device=device)
            rm = torch.zeros((1, rx.shape[1]), dtype=torch.bool, device=device)
            logit = grammar_model.score_merge(lx, lm, rx, rm)
            score = float(torch.sigmoid(logit).cpu())

        pred = int(score >= 0.5)
        results.append((left_idx, right_idx, label, pred, score))

    return results


# ---------------------------------------------------------------------------
# Neuroglancer setup
# ---------------------------------------------------------------------------

def _box_rel_mip2_to_global_mip0_vox(box_record, pt_mip2: np.ndarray) -> list[float]:
    """Convert box-relative MIP-2 voxel coords to global MIP-0 voxels.

    Cached synapse pre_pt are in box-relative MIP-2 voxels (from fetch_synapses).
    Neuroglancer EM is in global MIP-0 voxel space.
    """
    origin_nm = np.array(box_record.center_nm, dtype=np.float64) - (
        box_record.side_um * 1000.0 / 2.0
    )
    origin_mip0 = origin_nm / np.array(EM_VOX_NM, dtype=np.float64)
    pt = np.asarray(pt_mip2, dtype=np.float64).reshape(3)
    global_mip0 = origin_mip0 + pt * np.array(MIP2_TO_MIP0, dtype=np.float64)
    return [float(global_mip0[i]) for i in range(3)]


COLORS = {
    "tp": "#00cc44",   # green
    "fp": "#ff2222",   # red
    "fn": "#ff8800",   # orange
    "tn": "#888888",   # grey
}

OUTCOME_LABEL = {
    "tp": "TP (connected, predicted connected)",
    "fp": "FP (different neurons, predicted connected)",
    "fn": "FN (connected, predicted disconnected)",
    "tn": "TN (different neurons, predicted disconnected)",
}


def launch_neuroglancer(
    synapses,
    scored: list[tuple[int, int, int, int, float]],
    box_record,
    *,
    volume_chunk=None,
    show_tn: bool = False,
    top_k: int | None = None,
) -> str:
    """Build and launch a Neuroglancer viewer.  Returns the viewer URL."""
    try:
        import neuroglancer
    except ImportError as exc:
        raise SystemExit("pip install neuroglancer") from exc

    neuroglancer.set_server_bind_address("127.0.0.1")
    viewer = neuroglancer.Viewer()

    # Classify pairs
    outcomes: dict[str, list[tuple[int, int, float]]] = {
        "tp": [], "fp": [], "fn": [], "tn": []
    }
    for li, ri, label, pred, score in scored:
        if label == 1 and pred == 1:
            outcomes["tp"].append((li, ri, score))
        elif label == 0 and pred == 1:
            outcomes["fp"].append((li, ri, score))
        elif label == 1 and pred == 0:
            outcomes["fn"].append((li, ri, score))
        else:
            outcomes["tn"].append((li, ri, score))

    # Which root segments to show meshes for
    involved_roots: set[int] = set()
    for key in ("tp", "fp", "fn"):
        for li, ri, _ in outcomes[key]:
            involved_roots.add(int(synapses.pre_root_id[li]))
            involved_roots.add(int(synapses.pre_root_id[ri]))

    # Cached pre_pt are box-relative MIP-2 voxels (see fetch_synapses).
    pts = synapses.pre_pt

    with viewer.txn() as s:
        # Coordinate space: global MIP-0 voxels (matches the EM precomputed source).
        s.dimensions = neuroglancer.CoordinateSpace(
            names=["x", "y", "z"],
            units=["vox", "vox", "vox"],
            scales=[1, 1, 1],
        )

        # EM image layer (BossDB HTTPS so the browser can load tiles)
        s.layers["em"] = neuroglancer.ImageLayer(source=EM_SOURCE)

        # Fallback: when this box has cached EM, show it as a local volume so
        # we always have visible tiles even if the remote source fails to load.
        if (
            volume_chunk is not None
            and getattr(volume_chunk.data, "size", 0) > 0
        ):
            origin_nm = np.array(box_record.center_nm, dtype=np.float64) - (
                box_record.side_um * 1000.0 / 2.0
            )
            origin_mip0 = (origin_nm / np.array(EM_VOX_NM, dtype=np.float64)).astype(
                np.int64
            )
            # Cached volume is MIP-2; one MIP-2 voxel = 4×4×1 in MIP-0 voxel space.
            vol_dims = neuroglancer.CoordinateSpace(
                names=["x", "y", "z"],
                units=["vox", "vox", "vox"],
                scales=[4, 4, 1],
            )
            vol = neuroglancer.LocalVolume(
                volume=volume_chunk.data,
                dimensions=vol_dims,
                voxel_offset=origin_mip0.tolist(),
                volume_type="uint8",
            )
            s.layers["em_cached"] = neuroglancer.ImageLayer(source=vol)

        # Segmentation + meshes layer
        seg_layer = neuroglancer.SegmentationLayer(source=SEG_SOURCE)
        # ``segments`` is a VisibleSegments set; add IDs one by one.
        for seg_id in involved_roots:
            seg_layer.segments.add(int(seg_id))
        s.layers["seg"] = seg_layer

        # Synapse pre-point annotations (box-rel MIP-2 → global MIP-0)
        syn_annotations = []
        for i in range(len(pts)):
            syn_annotations.append(
                neuroglancer.PointAnnotation(
                    id=f"syn_{i}",
                    point=_box_rel_mip2_to_global_mip0_vox(box_record, pts[i]),
                )
            )
        s.layers["synapses"] = neuroglancer.LocalAnnotationLayer(
            dimensions=s.dimensions,
            annotations=syn_annotations,
            shader="""
void main() {
  setColor(vec4(0.2, 0.6, 1.0, 0.8));
  setPointMarkerSize(6.0);
}
""",
        )

        # Pair annotation layers — one layer per outcome type
        keys_to_show = ["tp", "fp", "fn"] + (["tn"] if show_tn else [])
        for key in keys_to_show:
            pair_list = outcomes[key]
            if top_k is not None:
                # Sort by confidence (TP/FP: highest score first; FN: lowest)
                if key in ("tp", "fp"):
                    pair_list = sorted(pair_list, key=lambda x: -x[2])[:top_k]
                else:
                    pair_list = sorted(pair_list, key=lambda x: x[2])[:top_k]

            line_annotations = []
            for ann_id, (li, ri, score) in enumerate(pair_list):
                line_annotations.append(
                    neuroglancer.LineAnnotation(
                        id=f"{key}_{ann_id}",
                        point_a=_box_rel_mip2_to_global_mip0_vox(box_record, pts[li]),
                        point_b=_box_rel_mip2_to_global_mip0_vox(box_record, pts[ri]),
                    )
                )

            color = COLORS[key]
            s.layers[f"pairs_{key}"] = neuroglancer.LocalAnnotationLayer(
                dimensions=s.dimensions,
                annotations=line_annotations,
                shader=f"""
void main() {{
  setColor(vec4({_hex_to_rgba(color)}, 0.9));
  setLineWidth(2.5);
}}
""",
            )

        # Navigate to box centre (global MIP-0 voxels)
        centre_mip0 = [
            float(box_record.center_nm[i]) / EM_VOX_NM[i] for i in range(3)
        ]
        s.position = centre_mip0
        s.cross_section_scale = 1.0  # 1 voxel per pixel at base resolution

    return viewer.get_viewer_url()


def _hex_to_rgba(hex_color: str) -> str:
    """Convert '#rrggbb' to 'r, g, b' floats for GLSL."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r/255:.3f}, {g/255:.3f}, {b/255:.3f}"


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def print_stats(
    scored: list[tuple[int, int, int, int, float]],
    synapses,
) -> None:
    labels = [s[2] for s in scored]
    preds  = [s[3] for s in scored]
    scores = [s[4] for s in scored]

    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)

    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
    acc  = (tp + tn) / len(scored) if scored else float("nan")

    pos_scores = [s for l, s in zip(labels, scores) if l == 1]
    neg_scores = [s for l, s in zip(labels, scores) if l == 0]

    print(f"\n{'='*55}")
    print(f"  Pairs evaluated : {len(scored)}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Accuracy  : {acc:.3f}")
    print(f"  Precision : {prec:.3f}")
    print(f"  Recall    : {rec:.3f}")
    print(f"  F1        : {f1:.3f}")
    print(f"  Mean score (pos) : {np.mean(pos_scores):.3f}" if pos_scores else "  No positive pairs")
    print(f"  Mean score (neg) : {np.mean(neg_scores):.3f}" if neg_scores else "  No negative pairs")

    root_counts = Counter(synapses.pre_root_id.tolist())
    print(f"\n  Synapses in box  : {len(synapses.pre_pt)}")
    print(f"  Unique pre roots : {len(root_counts)}")
    print(f"  Largest cluster  : {max(root_counts.values())} synapses")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", default="data/boxes30")
    p.add_argument("--grammar-path", default="models/shared_grammar_real.pt")
    p.add_argument("--box-idx", type=int, default=0,
                   help="Index into the val split (0 = first val box).")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-positive-pairs", type=int, default=200)
    p.add_argument("--max-negative-pairs", type=int, default=200)
    p.add_argument("--show-tn", action="store_true",
                   help="Include true-negative (grey) lines in the viewer.")
    p.add_argument("--top-k", type=int, default=None,
                   help="Limit each outcome layer to the top-K most confident pairs.")
    p.add_argument("--list-boxes", action="store_true",
                   help="Print available boxes and exit.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    import torch
    from neuronauts.dataset_builder import load_dataset
    from neuronauts.shared_grammar_model import load_shared_grammar_model

    device = torch.device("cpu")

    # ── Load dataset ──────────────────────────────────────────────────────
    cache, all_records = load_dataset(args.cache_dir)
    if not all_records:
        print(f"No boxes in {args.cache_dir}. Run build-dataset first.")
        return 1

    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_records)   # type: ignore[arg-type]
    n_val = max(1, int(len(all_records) * args.val_fraction))
    val_records = all_records[:n_val]
    train_records = all_records[n_val:]

    if args.list_boxes:
        print(f"\n{'IDX':>4}  {'SPLIT':>6}  {'SYNAPSES':>9}  {'POS_PAIRS':>10}  BOX_HASH")
        for i, r in enumerate(val_records):
            print(f"{i:>4}  {'val':>6}  {r.n_synapses:>9}  {r.n_positive_pairs:>10}  {r.box_hash[:12]}")
        print(f"\n  {len(val_records)} val boxes, {len(train_records)} train boxes")
        return 0

    if args.box_idx >= len(val_records):
        print(f"--box-idx {args.box_idx} out of range (only {len(val_records)} val boxes). "
              f"Use --list-boxes to see available indices.")
        return 1

    record = val_records[args.box_idx]
    print(f"\nBox: {record.box_hash[:16]}  side={record.side_um}µm  "
          f"n_synapses={record.n_synapses}  pos_pairs={record.n_positive_pairs}")

    # ── Load grammar ──────────────────────────────────────────────────────
    gpath = Path(args.grammar_path)
    if not gpath.exists():
        print(f"Grammar checkpoint not found: {gpath}")
        return 1
    print(f"Loading grammar from {gpath} …")
    grammar_model = load_shared_grammar_model(str(gpath)).to(device)
    grammar_model.eval()

    # ── Load box ──────────────────────────────────────────────────────────
    volume_chunk, synapses = cache.load(record)

    # ── Generate and score pairs ──────────────────────────────────────────
    print("Generating candidate pairs …")
    pairs = generate_all_candidate_pairs(
        synapses,
        max_positive_pairs=args.max_positive_pairs,
        max_negative_pairs=args.max_negative_pairs,
    )

    print("Scoring pairs with grammar model …")
    t0 = time.time()
    scored = score_pairs(grammar_model, synapses, pairs, device)
    print(f"  Scored {len(scored)} pairs in {time.time()-t0:.1f}s")

    # ── Stats ─────────────────────────────────────────────────────────────
    print_stats(scored, synapses)

    # ── Launch Neuroglancer ───────────────────────────────────────────────
    print("Launching Neuroglancer viewer …")
    url = launch_neuroglancer(
        synapses,
        scored,
        record,
        volume_chunk=volume_chunk,
        show_tn=args.show_tn,
        top_k=args.top_k,
    )

    print(f"\nOpen this URL in your browser:\n\n  {url}\n")
    print("Colour key:")
    for key, desc in OUTCOME_LABEL.items():
        if key == "tn" and not args.show_tn:
            continue
        print(f"  {COLORS[key]}  {desc}")
    print("\nPress Ctrl-C to exit.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
