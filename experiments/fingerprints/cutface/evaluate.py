"""CONTRIBUTION 4 -- how to evaluate (honestly), and diagnose.

Three metrics that must be kept separate -- conflating them is how a hash looks
better than it is:

* **Panel recall (location).**  `panel_recall_status` classifies each real site:
  is the true partner even *present* in the candidate panel?  Distinguishes a
  genuine miss from a "not-a-split" non-error and from out-of-radius.  This is the
  ceiling on everything downstream; report it, don't hide it by dropping sites.
* **Correction top-1 (given candidates).**  `combiner_top1` -- of the sites whose
  partner is present, how often the ranker picks it.  The headline number, but
  conditional: multiply by panel recall for the deployed yield.
* **Abstention (deployed).**  `abstention_curve` runs over the FULL population
  (partner-absent sites kept) with a label-blind accept/abstain threshold, giving
  a precision-coverage-recall trade-off.  This is the honest, deployable view:
  a proofreader that takes no action when unsure is legitimate; cherry-picking by
  label is not.

`collect_sites` / `collect_sites_with_abstain` build the per-site feature tables
the combiner trains/evaluates on.  For introspection, the residual-error montage
(query | true partner | wrong pick) lives in ``archive/diagnose_residual_errors``.
"""

from __future__ import annotations

# location: panel recall + miss taxonomy (present / absent / not_a_split / ...)
from .measure_panel_recall import (
    site_partner_status as panel_recall_status,
    _nearest_same_root_nm as nearest_same_root_nm,
)

# correction: per-site feature tables + combiner top-1
from .train_combiner import (
    collect as collect_sites,
    evaluate as combiner_top1,
)

# deployed: full-population abstention sweep (precision / coverage / recall)
from .train_combiner_abstain import (
    collect_all as collect_sites_with_abstain,
    site_features_all as combiner_features_keep_absent,
    evaluate_abstain as abstention_curve,
)

# raw-cosine band re-id baseline (training-free)
from .v117_artifact_bands import evaluate_bands
from .train_band_encoders import evaluate_bands_learned

__all__ = [
    "panel_recall_status", "nearest_same_root_nm",
    "collect_sites", "combiner_top1",
    "collect_sites_with_abstain", "combiner_features_keep_absent", "abstention_curve",
    "evaluate_bands", "evaluate_bands_learned",
]
