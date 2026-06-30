"""Skeleton GNN: graph-level DNA encoder from raw skeleton vertices.

Replaces the path-sampling TreeDNAEncoder with a GNN that operates directly
on the skeleton graph.  No hand-crafted features — the model sees raw
(centroid-normalised x, y, z, radius) at each vertex and learns what to
attend to.

Node features : (x-cx, y-cy, z-cz, radius)  — 4 scalars, translation-invariant
Edge features : edge length (nm)              — 1 scalar
Readout       : mean + max pool across all vertices → linear → [output_dim]

The GNN uses the same message-passing pattern as CellGNN (src‖dst‖edge →
message → scatter_mean → residual update), so it reuses familiar code paths.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..schemas import Fragment


def fragment_to_tensors(fragment: Fragment, device: str = "cpu"):
    """Convert a Fragment skeleton to GNN input tensors.

    Both edge directions are included so every node receives messages from
    all its neighbours.

    Returns
    -------
    (node_feat [V,4], edge_src [E*2], edge_dst [E*2], edge_feat [E*2,1])
    all as torch tensors on *device*.
    """
    import torch

    verts = fragment.vertices_nm.astype(np.float32)
    centroid = verts.mean(axis=0)
    xyz = verts - centroid
    radii = fragment.radius_nm.reshape(-1, 1).astype(np.float32)
    node_feat = np.concatenate([xyz, radii], axis=1)  # [V, 4]

    edges = fragment.edges
    if len(edges) == 0 or len(verts) < 2:
        src = np.zeros(0, dtype=np.int64)
        dst = np.zeros(0, dtype=np.int64)
        lengths = np.zeros((0, 1), dtype=np.float32)
    else:
        u, v = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
        src = np.concatenate([u, v])
        dst = np.concatenate([v, u])
        edge_len = np.linalg.norm(verts[u] - verts[v], axis=1, keepdims=True)
        lengths = np.concatenate([edge_len, edge_len]).astype(np.float32)

    return (
        torch.from_numpy(node_feat).to(device),
        torch.from_numpy(src).long().to(device),
        torch.from_numpy(dst).long().to(device),
        torch.from_numpy(lengths).to(device),
    )


class SkeletonGNN:
    """GNN that maps a skeleton graph to a single [output_dim] DNA embedding.

    Factory pattern (returns an nn.Module instance) consistent with CellGNN.

    Parameters
    ----------
    node_input_dim : int
        Input feature dimension per vertex (default 4: x,y,z,radius).
    edge_input_dim : int
        Input feature dimension per edge (default 1: edge length).
    d_model : int
        Hidden width throughout the GNN.
    n_layers : int
        Message-passing rounds.
    dropout : float
        Dropout rate.
    output_dim : int
        DNA embedding dimension.
    """

    def __new__(
        cls,
        *,
        node_input_dim: int = 4,
        edge_input_dim: int = 1,
        d_model: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        output_dim: int = 32,
    ):
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError as exc:
            raise ImportError("pip install torch") from exc

        class _SkeletonGNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.d_model = d_model
                self.output_dim = output_dim

                self.node_proj = nn.Linear(node_input_dim, d_model)
                self.edge_proj = nn.Linear(edge_input_dim, d_model)

                self.msg_linears = nn.ModuleList([
                    nn.Linear(d_model * 3, d_model) for _ in range(n_layers)
                ])
                self.upd_linears = nn.ModuleList([
                    nn.Linear(d_model * 2, d_model) for _ in range(n_layers)
                ])
                self.norms = nn.ModuleList([
                    nn.LayerNorm(d_model) for _ in range(n_layers)
                ])
                self.drop = nn.Dropout(dropout)

                # mean + max readout → output
                self.output_proj = nn.Linear(d_model * 2, output_dim)

            def forward(self, node_feat, edge_src, edge_dst, edge_feat):
                """
                Parameters
                ----------
                node_feat : Tensor [V, node_input_dim]
                edge_src  : Tensor [E] int64
                edge_dst  : Tensor [E] int64
                edge_feat : Tensor [E, edge_input_dim]

                Returns
                -------
                Tensor [output_dim]  — single graph-level embedding (unnormalised)
                """
                V = node_feat.size(0)
                h = self.node_proj(node_feat)   # [V, d_model]

                if edge_src.numel() > 0:
                    e = self.edge_proj(edge_feat)   # [E, d_model]
                    for msg_lin, upd_lin, norm in zip(
                        self.msg_linears, self.upd_linears, self.norms
                    ):
                        src_h = h[edge_src]
                        dst_h = h[edge_dst]
                        msgs = F.relu(msg_lin(torch.cat([src_h, dst_h, e], dim=-1)))
                        msgs = self.drop(msgs)

                        agg = torch.zeros(V, self.d_model, device=h.device, dtype=h.dtype)
                        cnt = torch.zeros(V, 1, device=h.device, dtype=h.dtype)
                        agg.scatter_add_(0, edge_dst.unsqueeze(1).expand_as(msgs), msgs)
                        cnt.scatter_add_(0, edge_dst.unsqueeze(1),
                                         torch.ones(edge_dst.size(0), 1,
                                                    device=h.device, dtype=h.dtype))
                        agg = agg / cnt.clamp_min(1.0)

                        h = norm(h + F.relu(upd_lin(torch.cat([h, agg], dim=-1))))

                mean_pool = h.mean(dim=0)         # [d_model]
                max_pool = h.max(dim=0).values    # [d_model]
                graph_emb = torch.cat([mean_pool, max_pool], dim=-1)  # [2*d_model]
                return self.output_proj(graph_emb)  # [output_dim]

        return _SkeletonGNN()


def encode_fragments_gnn(
    gnn,
    fragments: Sequence[Fragment],
    *,
    device: str = "cpu",
) -> list[Fragment]:
    """Run SkeletonGNN on each Fragment; return copies with dna= filled.

    Parameters
    ----------
    gnn : SkeletonGNN (nn.Module)
    fragments : list of Fragment (vertices_nm, edges, radius_nm must be set)
    device : torch device string

    Returns
    -------
    New Fragment list with dna set to L2-normalised [output_dim] float32 arrays.
    """
    import torch
    import torch.nn.functional as F

    gnn.eval()
    gnn = gnn.to(device)
    result: list[Fragment] = []

    with torch.no_grad():
        for frag in fragments:
            nf, es, ed, ef = fragment_to_tensors(frag, device)
            emb = gnn(nf, es, ed, ef)
            emb = F.normalize(emb, p=2, dim=-1)
            dna = emb.cpu().numpy().astype(np.float32)
            result.append(Fragment(
                fragment_id=frag.fragment_id,
                region_id=frag.region_id,
                base_root_id=frag.base_root_id,
                vertices_nm=frag.vertices_nm,
                edges=frag.edges,
                endpoints_nm=frag.endpoints_nm,
                radius_nm=frag.radius_nm,
                synapse_indices=frag.synapse_indices,
                dna=dna,
            ).validate())

    return result


def train_skeleton_gnn(
    gnn,
    fragment_lists: list[list[Fragment]],
    *,
    n_epochs: int = 80,
    lr: float = 1e-3,
    margin: float = 1.0,   # kept for API compatibility; unused (see temperature)
    temperature: float = 0.1,
    device: str = "cpu",
    root_label_map: dict[int, set[int]] | None = None,
    log_every: int = 10,
) -> dict:
    """NT-Xent (SimCLR) contrastive training for SkeletonGNN.

    Same interface as train_dna_encoder — drop-in replacement.

    Each step builds a batch of N positive pairs (2N unique fragment encodings).
    Loss is NT-Xent:
        L = -log(exp(sim(a,p)/τ) / Σ_{k≠a} exp(sim(a,k)/τ))
    Unlike triplet losses (cosine or L2), NT-Xent has non-vanishing gradients
    at collapse — when all cosine similarities equal 1, the softmax denominator
    ≠ numerator (ratio = 1/(2N-1) < 1), so loss = log(2N-1) > 0 with a clear
    gradient direction.  Temperature τ=0.1 amplifies gradient magnitude even
    when positive/negative sims are close.
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    gnn = gnn.to(device)
    opt = torch.optim.Adam(gnn.parameters(), lr=lr)

    all_frags = [f for fl in fragment_lists for f in fl]

    # Group fragments by label_root
    group_to_frags: dict[int, list[Fragment]] = {}
    for frag in all_frags:
        rid = frag.base_root_id
        if root_label_map is not None:
            labels = root_label_map.get(rid, set())
            if len(labels) != 1:
                continue
            key = next(iter(labels))
        else:
            key = rid
        group_to_frags.setdefault(key, []).append(frag)

    groups = [(k, v) for k, v in group_to_frags.items() if len(v) >= 1]
    if len(groups) < 2:
        raise ValueError("Need ≥2 neuron groups with ≥1 clean fragment each")

    history: dict[str, list[float]] = {"loss": [], "pos_cos": [], "neg_cos": []}

    tensor_cache_cpu = [fragment_to_tensors(f, "cpu") for f in all_frags]
    frag_to_idx = {id(f): i for i, f in enumerate(all_frags)}

    # Build fragment→neuron label map for false-negative masking in NT-Xent.
    # When multiple fragments of the same neuron appear in one batch, they must
    # be excluded from each other's denominator (they are same-neuron positives,
    # not true negatives).  Without this mask, conflicting gradient signals
    # cancel out and cause collapse regardless of loss formulation.
    frag_to_neuron: dict[int, int] = {}
    for frag in all_frags:
        rid = frag.base_root_id
        if root_label_map is not None:
            labels = root_label_map.get(rid, set())
            if len(labels) == 1:
                frag_to_neuron[frag_to_idx[id(frag)]] = next(iter(labels))
        else:
            frag_to_neuron[frag_to_idx[id(frag)]] = rid

    n_pos_pairs = 128   # N positive pairs → 2N embeddings per batch
    pos_groups = [(k, v) for k, v in groups if len(v) >= 2]

    for epoch in range(1, n_epochs + 1):
        gnn.train()
        opt.zero_grad()

        # Sample N positive pairs; each fragment appears at most once
        # so the batch is exactly 2N distinct embeddings.
        pairs: list[tuple[int, int]] = []
        used: set[int] = set()
        for _ in range(n_pos_pairs * 4):  # oversample to fill quota
            if len(pairs) >= n_pos_pairs or not pos_groups:
                break
            gk, gfrags = pos_groups[int(rng.integers(len(pos_groups)))]
            ia, ib = rng.choice(len(gfrags), size=2, replace=False)
            ai = frag_to_idx[id(gfrags[int(ia)])]
            bi = frag_to_idx[id(gfrags[int(ib)])]
            if ai in used or bi in used:
                continue
            used.add(ai); used.add(bi)
            pairs.append((ai, bi))

        if not pairs:
            history["loss"].append(0.0)
            history["pos_cos"].append(0.0)
            history["neg_cos"].append(0.0)
            continue

        N = len(pairs)
        # Encode: anchors first, then positives (each is unique)
        a_global = [ai for ai, _ in pairs]
        b_global = [bi for _, bi in pairs]
        all_global = a_global + b_global  # length 2N

        mini_embs = []
        for g_idx in all_global:
            nf, es, ed, ef = tensor_cache_cpu[g_idx]
            mini_embs.append(gnn(nf.to(device), es.to(device),
                                 ed.to(device), ef.to(device)))
        z = torch.stack(mini_embs, dim=0)     # [2N, D]
        z = F.normalize(z, p=2, dim=-1)       # onto unit sphere

        # NT-Xent similarity matrix: [2N, 2N] / temperature
        sim = (z @ z.T) / temperature

        # False-negative mask: same-neuron fragments in the batch must be
        # excluded from each other's NT-Xent denominator (they are true positives
        # of one another, not negatives).  Self-similarity is always excluded.
        # The actual positive pair (i, i+N) is excluded from the mask so it
        # stays in the numerator.
        neuron_ids = torch.tensor(
            [frag_to_neuron.get(g, -g) for g in all_global],
            dtype=torch.long, device=device)
        same_neuron = neuron_ids.unsqueeze(0) == neuron_ids.unsqueeze(1)  # [2N, 2N]
        # Mark the designated positive pair so we don't mask it out
        pos_pair_mask = torch.zeros(2 * N, 2 * N, dtype=torch.bool, device=device)
        for i in range(N):
            pos_pair_mask[i, i + N] = True
            pos_pair_mask[i + N, i] = True
        eye = torch.eye(2 * N, dtype=torch.bool, device=device)
        exclude = eye | (same_neuron & ~pos_pair_mask)
        sim = sim.masked_fill(exclude, -1e9)

        # For anchor i (i < N): positive at i+N; for positive i+N: anchor at i
        targets = torch.cat([
            torch.arange(N, 2 * N, dtype=torch.long, device=device),
            torch.arange(0, N, dtype=torch.long, device=device),
        ])
        loss = F.cross_entropy(sim, targets)

        # Monitor: mean cosine sim of positive pairs vs a random in-batch negative
        sim_pos = (z[:N] * z[N:]).sum(dim=-1)
        neg_perm = (torch.arange(N, device=device) + 1) % N
        sim_neg = (z[:N] * z[N:][neg_perm]).sum(dim=-1)
        pos_sims = sim_pos.detach().cpu().tolist()
        neg_sims = sim_neg.detach().cpu().tolist()

        loss.backward()
        opt.step()

        lv = float(loss.item())
        pc = float(np.mean(pos_sims)) if pos_sims else 0.0
        nc = float(np.mean(neg_sims)) if neg_sims else 0.0
        history["loss"].append(lv)
        history["pos_cos"].append(pc)
        history["neg_cos"].append(nc)

        if log_every > 0 and epoch % log_every == 0:
            print(f"  epoch {epoch:3d}: loss={lv:.4f}  pos_cos={pc:.3f}  neg_cos={nc:.3f}")

    gnn.eval()
    return history
