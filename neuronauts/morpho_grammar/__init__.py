"""Deprecated: this package moved to ``attic/morpho_grammar/``.

Kept as an import shim so existing code and notebooks keep working. Importing
anything from here emits a :class:`DeprecationWarning` and loads the module
from the attic.

**Why it was retired (2026-09-01).** The results these engines produced are not
real-data evidence, verified by direct inspection rather than inference:

* 15 of the 26 modules draw random numbers at runtime and **none** contains a
  ``torch.load`` or a checkpoint path -- the published accuracies are those of
  randomly initialised models.
* Every benchmark that consumed them (``attic/benchmarks_semi_synthetic/``)
  builds its test world with ``treestitch.worldbuild.frankenmerge_adjacent`` at
  45-50% on synthetically cut skeletons, and two of them generate the neurons
  themselves from random walks.
* ``results/exp051_evaluation.md`` reached the same conclusion independently for
  EXP-049 and for the SANTIAGO infiller.

**How code here comes back.** Not by being imported, but by winning EXP-069 in
``docs/consolidation_plan.md`` §6: the same engine run on the real harness
substrate, under the EXP-064 fixed-panel protocol, with a trained grammar. The
individual *ideas* (tangent flow, caliber continuity, conservation priors) are
expected to return sooner as features in the EXP-064 scorer bake-off.

See ``attic/README.md``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.warn(
    "neuronauts.morpho_grammar has moved to attic/morpho_grammar/ and is "
    "retired: its engines load no trained checkpoint and every benchmark that "
    "used them ran on synthetic damage. See attic/README.md and "
    "docs/consolidation_plan.md section 1.4.",
    DeprecationWarning,
    stacklevel=2,
)

# Resolve submodules (neuronauts.morpho_grammar.santiago_v2_grammar, ...) from
# the attic, so the modules' own `from neuronauts.morpho_grammar.X import Y`
# imports keep resolving to one consistent set of objects.
__path__ = [str(Path(__file__).resolve().parents[2] / "attic" / "morpho_grammar")]
