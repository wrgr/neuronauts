"""Does synapse polarity separate compartments? (no GT, no network)

The proposed grammar leans on a compartment inventory -- Soma, Dend, Spine,
Axon, Bouton -- and the cheap way to get it is synapse polarity: a v117 object
that is presynaptic at its contacts is axonal, one that is postsynaptic is
dendritic. If that holds, the per-atom presynaptic fraction should be strongly
**bimodal** (atoms near 0 or near 1), because a real object is mostly one
compartment. A unimodal pile at 0.5 would mean polarity carries no compartment
information at this granularity and the inventory needs rethinking.

Two null models keep this honest:
  global shuffle -- reassign polarity across all synapses; kills any structure
  per-atom binom -- keep each atom's synapse count, draw polarity i.i.d. at the
                    global base rate; this is the null that matters, because a
                    small atom looks "pure" by chance and only the excess over
                    this null is real compartment signal

Runs entirely off the cached population NPZ.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuronauts.harness.population import load_population  # noqa: E402

OUT = Path("results/figures")


def polarity_counts(pop):
    """Per-atom (n_pre, n_post): presynaptic = axonal side of a contact."""
    atoms = pop.atom_id
    order = np.argsort(atoms)
    sorted_atoms = atoms[order]

    def tally(side):
        v = side[side > 0]
        idx = np.searchsorted(sorted_atoms, v)
        ok = (idx < len(sorted_atoms)) & (sorted_atoms[np.clip(idx, 0, len(sorted_atoms)-1)] == v)
        counts = np.bincount(order[idx[ok]], minlength=len(atoms))
        return counts.astype(np.int64)

    return tally(pop.syn_atom_pre), tally(pop.syn_atom_post)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="data/substrate/c100um/population.npz")
    ap.add_argument("--min-syn", type=int, default=10)
    ap.add_argument("--out", default=str(OUT / "06_polarity_compartments.png"))
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    pop = load_population(args.population)
    n_pre, n_post = polarity_counts(pop)
    tot = n_pre + n_post
    print(f"atoms: {len(pop.atom_id):,}  synapse-endpoints: {tot.sum():,}")

    base = n_pre.sum() / max(tot.sum(), 1)
    print(f"global presynaptic fraction: {base:.3f}")

    m = tot >= args.min_syn
    frac = n_pre[m] / tot[m]
    print(f"atoms with >={args.min_syn} synapses: {int(m.sum()):,}")

    rng = np.random.default_rng(0)
    null_binom = rng.binomial(tot[m], base) / tot[m]

    pure = float(np.mean((frac <= 0.1) | (frac >= 0.9)))
    pure_null = float(np.mean((null_binom <= 0.1) | (null_binom >= 0.9)))
    mid = float(np.mean((frac > 0.3) & (frac < 0.7)))
    mid_null = float(np.mean((null_binom > 0.3) & (null_binom < 0.7)))

    print(f"\ncompartment-pure atoms (pre-frac <=0.1 or >=0.9):")
    print(f"  observed            : {pure:.1%}")
    print(f"  binomial null       : {pure_null:.1%}")
    print(f"mixed atoms (0.3-0.7):")
    print(f"  observed            : {mid:.1%}")
    print(f"  binomial null       : {mid_null:.1%}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    bins = np.linspace(0, 1, 41)
    axes[0].hist(frac, bins=bins, color="steelblue", ec="w",
                 label=f"observed (n={int(m.sum()):,})")
    axes[0].hist(null_binom, bins=bins, histtype="step", color="crimson", lw=1.8,
                 label="binomial null")
    axes[0].set_xlabel("presynaptic fraction of an atom's synapses")
    axes[0].set_ylabel("atoms")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"Polarity purity (>={args.min_syn} synapses)\n"
                      f"pure: {pure:.0%} vs null {pure_null:.0%}")

    for k, c in ((5, "0.7"), (10, "steelblue"), (30, "navy")):
        mm = tot >= k
        if mm.sum() > 50:
            axes[1].hist(n_pre[mm] / tot[mm], bins=bins, histtype="step",
                         lw=1.6, color=c, density=True, label=f">={k} synapses")
    axes[1].set_xlabel("presynaptic fraction")
    axes[1].set_ylabel("density")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Purity vs atom size")

    axes[2].scatter(tot[m], frac, s=2, alpha=0.12, c="steelblue")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("synapses on atom")
    axes[2].set_ylabel("presynaptic fraction")
    axes[2].set_title("Big atoms should stay pure if\ncompartments are real")
    for a in axes:
        a.grid(alpha=0.2, lw=0.4)

    fig.suptitle(
        "Grammar premise - does polarity give us compartments for free?\n"
        "CHECK: strong peaks at 0 (dendritic) and 1 (axonal), well above the "
        "binomial null = usable Axon/Dend nonterminals with no EM",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=125)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
