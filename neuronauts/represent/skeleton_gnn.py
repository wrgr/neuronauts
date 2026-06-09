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
    margin: float = 1.0,
    device: str = "cpu",
    root_label_map: dict[int, set[int]] | None = None,
    log_every: int = 10,
) -> dict:
    """Contrastive training for SkeletonGNN.

    Same interface as train_dna_encoder — drop-in replacement.

    Positives: fragment pairs with the same label_root.
    Negatives: fragment pairs with different label roots.
    Loss: cosine contrastive (pull positives, push negatives past margin).
    """
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(0)
    gnn = gnn.to(device)
    opt = torch.optim.Adam(gnn.parameters(), lr=lr)

    all_frags = [f for fl in fragment_lists for f in fl]

    # Group fragments by label_root (same logic as train_dna_encoder)
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

    group_keys = [k for k, _ in groups]
    history: dict[str, list[float]] = {"loss": [], "pos_cos": [], "neg_cos": []}

    # Pre-compute tensors for all fragments (skeleton graphs don't change)
    frag_tensors = {
        frag.base_root_id if root_label_map is None else next(iter(root_label_map.get(frag.base_root_id, {frag.base_root_id}))): None
        for frag in all_frags
    }
    # Actually index by fragment object identity for simplicity
    frag_list_flat = all_frags
    tensor_cache = [fragment_to_tensors(f, device) for f in frag_list_flat]
    frag_to_idx = {id(f): i for i, f in enumerate(frag_list_flat)}

    for epoch in range(1, n_epochs + 1):
        gnn.train()
        opt.zero_grad()

        # Encode all fragments
        embeddings = []
        for tensors in tensor_cache:
            nf, es, ed, ef = tensors
            emb = gnn(nf, es, ed, ef)
            embeddings.append(emb)
        embs = torch.stack(embeddings, dim=0)          # [N, output_dim]
        embs_norm = F.normalize(embs, p=2, dim=-1)

        # Sample positive pairs
        n_pairs = 256
        pos_pairs, neg_pairs = [], []

        pos_groups = [(k, v) for k, v in groups if len(v) >= 2]
        for _ in range(n_pairs):
            if not pos_groups:
                break
            gk, frags = pos_groups[int(rng.integers(len(pos_groups)))]
            ia, ib = rng.choice(len(frags), size=2, replace=False)
            pos_pairs.append((frag_to_idx[id(frags[int(ia)])],
                               frag_to_idx[id(frags[int(ib)])]))

        for _ in range(n_pairs):
            ga_i, gb_i = rng.choice(len(groups), size=2, replace=False)
            ga_k, ga_f = groups[int(ga_i)]
            gb_k, gb_f = groups[int(gb_i)]
            ia = int(rng.integers(len(ga_f)))
            ib = int(rng.integers(len(gb_f)))
            neg_pairs.append((frag_to_idx[id(ga_f[ia])],
                               frag_to_idx[id(gb_f[ib])]))

        if not pos_pairs and not neg_pairs:
            history["loss"].append(0.0)
            history["pos_cos"].append(0.0)
            history["neg_cos"].append(0.0)
            continue

        loss = torch.tensor(0.0, device=device)
        pos_sims, neg_sims = [], []

        if pos_pairs:
            src = torch.tensor([p[0] for p in pos_pairs], dtype=torch.long, device=device)
            dst = torch.tensor([p[1] for p in pos_pairs], dtype=torch.long, device=device)
            sim = (embs_norm[src] * embs_norm[dst]).sum(dim=-1)
            loss = loss + (1.0 - sim).mean()
            pos_sims = sim.detach().cpu().tolist()

        if neg_pairs:
            src = torch.tensor([p[0] for p in neg_pairs], dtype=torch.long, device=device)
            dst = torch.tensor([p[1] for p in neg_pairs], dtype=torch.long, device=device)
            sim = (embs_norm[src] * embs_norm[dst]).sum(dim=-1)
            loss = loss + F.relu(sim - (1.0 - margin)).mean()
            neg_sims = sim.detach().cpu().tolist()

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
