"""Per-object synapse polarity: the grammar's one hard constraint.

An axon does not continue into a dendrite. Where both sides of a candidate join
carry synapses, polarity is a constraint rather than a score -- 98% of objects
with any synapse are pure one way or the other (175,715 presynaptic, 97,797
postsynaptic, 5,563 mixed), so disagreement is strong evidence against the join.

The limitation is the same one EXP-071 identified: only 279,075 of 910,888
objects carry a synapse at all, so about 69% of candidates have no polarity
signal. The connective cable between fragments is exactly the population that
synapse-anchored evidence cannot see. Polarity can veto a join; it cannot
propose one.

EXP-063 measured polarity at held-out AUC 0.914 for frankenmerge detection --
the strongest single feature family there, ahead of the published shape
detector at 0.875 -- and no EXP-075 geometric term used it.

    python scripts/build_object_polarity.py
"""
from pathlib import Path
import numpy as np

R = Path("/Users/wgray13/projects/neuronauts")
OUT = R / "data/external/object_polarity.npz"


def main():
    p = np.load(R / "data/substrate/c100um/population.npz", allow_pickle=False)
    pre, post = p["syn_atom_pre"], p["syn_atom_post"]
    ids = np.unique(np.concatenate([pre, post]))
    ids = ids[ids > 0]

    def counts(arr):
        a = arr[arr > 0]
        u, c = np.unique(a, return_counts=True)
        out = np.zeros(len(ids), dtype=np.int64)
        i = np.searchsorted(ids, u)
        ok = (i < len(ids)) & (ids[np.clip(i, 0, len(ids) - 1)] == u)
        out[i[ok]] = c[ok]
        return out

    npre, npost = counts(pre), counts(post)
    tot = npre + npost
    frac_pre = np.where(tot > 0, npre / np.maximum(tot, 1), np.nan).astype(np.float32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, object_id=ids.astype(np.uint64), n_pre=npre, n_post=npost, frac_pre=frac_pre)
    print(f"{len(pre):,} synapses -> {len(ids):,} objects with polarity")
    print(f"  presynaptic  {(frac_pre > 0.9).sum():,}")
    print(f"  postsynaptic {(frac_pre < 0.1).sum():,}")
    print(f"  mixed        {((frac_pre >= 0.1) & (frac_pre <= 0.9)).sum():,}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
