"""EXP-059 — do the old metric entry points still return the old numbers?

`neuronauts/metrics/` replaced four independent implementations, and the old
call sites were rewritten as delegating shims. That is only safe if the shims
return what they returned before. A refactor that silently changes a metric is
worse than four implementations, because every historical number becomes
incomparable without anyone noticing.

This checks agreement numerically rather than by inspection, on randomly
generated partitions spanning the shapes that break these functions: perfect
agreement, total disagreement, a single giant cluster, all singletons, heavy
class imbalance, and unlabelled items present.

Where a difference is deliberate it is asserted as such rather than waved
through. Two are known and intended:

* **Undefined ratios are NaN, not 1.0.** ``global_merge.eval.benchmark``
  returned precision 1.0 when nothing was merged, which is why "untouched
  v117" scored a perfect precision in EXP-052 and why EXP-058's do-nothing rung
  reports no precision at all. The shim keeps the old constant for callers that
  depend on it; the package returns NaN. Both are checked, and the divergence
  is recorded here rather than hidden.
* **Abstention.** ``treestitch`` drops an abstained observation entirely;
  ``partition_metrics`` keeps it as a singleton under ``pred_ignore``. The shim
  drops before delegating, so the numbers match.

    uv run python -m neuronauts.experiments.exp059_metric_agreement
"""

from __future__ import annotations

import numpy as np

from neuronauts.experiments._runner import Context, Outcome, Spec, main
from neuronauts.metrics import partition_metrics
from neuronauts.metrics.partition import adjusted_rand_index as ari_new

TOL = 1e-9
N_CASES = 200
SEED = 0

SPEC = Spec(
    id="EXP-059",
    title="Metric agreement",
    question="Do the delegating shims return what the four original "
             "implementations returned?",
    criterion=f"every shared quantity agrees to {TOL:g} across {N_CASES} "
              f"randomly generated partitions, or the difference is a "
              f"deliberate convention change asserted by name",
    requires_ran=["EXP-058"],
    inputs=[],
    params={"tol": TOL, "n_cases": N_CASES, "seed": SEED},
)


def _cases(rng, n: int):
    """Partition shapes that break these functions, not just easy ones."""
    for i in range(n):
        m = rng.integers(20, 400)
        kind = i % 8
        true = rng.integers(1, max(2, m // 5), size=m)
        if kind == 0:                      # perfect agreement
            pred = true.copy()
        elif kind == 1:                    # everything in one cluster
            pred = np.ones(m, np.int64)
        elif kind == 2:                    # all singletons
            pred = np.arange(m)
        elif kind == 3:                    # independent of truth
            pred = rng.integers(1, max(2, m // 5), size=m)
        elif kind == 4:                    # heavy imbalance
            true = np.where(rng.random(m) < 0.95, 1, rng.integers(2, 9, size=m))
            pred = rng.integers(1, 4, size=m)
        elif kind == 5:                    # unlabelled items present
            true = rng.integers(0, max(2, m // 6), size=m)
            pred = rng.integers(1, max(2, m // 6), size=m)
        elif kind == 6:                    # one true class only
            true = np.ones(m, np.int64)
            pred = rng.integers(1, 5, size=m)
        else:                              # near-perfect with a few errors
            pred = true.copy()
            flip = rng.choice(m, size=max(1, m // 20), replace=False)
            pred[flip] = rng.integers(1, max(2, m // 5), size=len(flip))
        yield np.asarray(pred, np.int64), np.asarray(true, np.int64)


def run(ctx: Context) -> Outcome:
    from neuronauts.global_merge.eval import benchmark as gm
    from treestitch import partition as ts

    rng = np.random.default_rng(SEED)
    checks: dict[str, dict] = {}
    diffs: list[str] = []

    def record(name, a, b, *, note=""):
        d = checks.setdefault(name, {"n": 0, "max_abs_diff": 0.0, "note": note})
        d["n"] += 1
        if a is None or b is None:
            return
        if np.isnan(a) and np.isnan(b):
            return
        if np.isnan(a) != np.isnan(b):
            d["max_abs_diff"] = float("inf")
            diffs.append(f"{name}: NaN mismatch ({a} vs {b})")
            return
        d["max_abs_diff"] = max(d["max_abs_diff"], abs(float(a) - float(b)))

    for pred, true in _cases(rng, N_CASES):
        new = partition_metrics(pred, true, ignore=0)

        # 1. treestitch.partition.evaluate_partition
        old = ts.evaluate_partition(pred, true, ignore_label=0)
        for k in ("ari", "homogeneity", "completeness", "v_measure",
                  "n_clusters_pred", "n_clusters_true", "n_nodes"):
            if k in old and k in new:
                record(f"treestitch.evaluate_partition.{k}", old[k], new[k])

        # 2. global_merge.eval.benchmark, keyed on string labels
        pm = {str(i): str(v) for i, v in enumerate(pred)}
        tm = {str(i): str(v) for i, v in enumerate(true)}
        gmm = gm.compute_pairwise_partition_metrics(pm, tm)
        record("benchmark.ari", gmm["ari"], ari_new(pred, true))
        record("benchmark.adjusted_rand_index",
               gm.adjusted_rand_index([str(v) for v in true],
                                      [str(v) for v in pred]),
               ari_new(pred, true))

        # merge_P/R: same pair counts, but benchmark evaluates every item
        # (no ignore) and returns 1.0 rather than NaN for an empty denominator.
        full = partition_metrics(pred, true, ignore=None)
        p_new = full["pair_precision"]
        p_old = gmm["merge_P"]
        if np.isnan(p_new) and p_old == 1.0:
            record("benchmark.merge_P (undefined -> 1.0 vs NaN)", 0.0, 0.0,
                   note="deliberate: benchmark returns 1.0 when nothing is "
                        "merged; metrics returns NaN. This is why untouched "
                        "v117 scored perfect precision in EXP-052.")
        else:
            record("benchmark.merge_P", p_old, p_new)
        r_old, r_new = gmm["merge_R"], full["pair_recall"]
        if np.isnan(r_new) and r_old == 1.0:
            record("benchmark.merge_R (undefined -> 1.0 vs NaN)", 0.0, 0.0,
                   note="deliberate, as above")
        else:
            record("benchmark.merge_R", r_old, r_new)

    real = {k: v for k, v in checks.items()
            if v["max_abs_diff"] > TOL and not v["note"]}
    deliberate = {k: v for k, v in checks.items() if v["note"]}

    for k, v in sorted(checks.items()):
        flag = "" if k not in real else "  <-- DISAGREES"
        print(f"  {k:<52} n={v['n']:>4}  max|diff|="
              f"{v['max_abs_diff']:.3g}{flag}", flush=True)
    if deliberate:
        print("\n  deliberate convention changes, asserted not waved through:")
        for k, v in deliberate.items():
            print(f"    {k}\n      {v['note']}", flush=True)

    return Outcome(
        passed=not real,
        observed={
            "n_cases": N_CASES,
            "n_quantities_checked": len(checks),
            "n_disagreeing": len(real),
            "max_abs_diff": round(max((v["max_abs_diff"] for v in checks.values()
                                       if not v["note"]), default=0.0), 12),
            "n_deliberate_differences": len(deliberate),
        },
        population={"partition_shapes": 8, "seed": SEED},
        tables={"agreement": checks},
        note=(f"all {len(checks)} shared quantities agree to {TOL:g} across "
              f"{N_CASES} partitions; {len(deliberate)} deliberate convention "
              f"changes recorded" if not real else
              f"{len(real)} quantities disagree: {', '.join(sorted(real))}"),
    )


if __name__ == "__main__":
    raise SystemExit(main(SPEC, run))
