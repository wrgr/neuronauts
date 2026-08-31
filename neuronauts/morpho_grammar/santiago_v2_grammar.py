"""Observable-only segment and cell-class induction for SANTIAGO v2.

The functions in this module deliberately do not accept annotation or ground-truth
cell-type fields.  Coarse cell class is an inferred grammar context, not an input
label; benchmark strata are applied only after assembly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _number(observables: Mapping[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in observables:
            try:
                return float(observables[name])
            except (TypeError, ValueError):
                pass
    return default


def type_segment_v2(observables: Mapping[str, Any]) -> str:
    """Infer Soma/Axon/Dendrite/Glia from measurable fragment properties."""
    radius = _number(observables, "mean_radius_nm", "radius_nm", "r")
    max_radius = _number(observables, "max_radius_nm", default=radius)
    n_pre = _number(observables, "n_pre")
    n_post = _number(observables, "n_post")
    if max_radius >= 500.0 or radius >= 450.0:
        return "Soma"
    if bool(observables.get("is_glia", False)) or (
        n_pre + n_post == 0 and radius >= 180.0
    ):
        return "Glia"
    if n_pre > n_post * 1.35 or (radius < 110.0 and n_pre >= n_post):
        return "Axon"
    return "Dendrite"


def induce_cell_type_from_observables(observables: Mapping[str, Any]) -> str:
    """Infer ``Pyramidal`` or ``Interneuron`` without consulting GT labels.

    The rule combines outgoing synapse fraction, bouton density and caliber.
    It is intentionally transparent so leakage audits can inspect every input.
    """
    n_pre = _number(observables, "n_pre")
    n_post = _number(observables, "n_post")
    pre_ratio = n_pre / max(n_pre + n_post, 1.0)
    bouton_density = _number(observables, "bouton_density", "boutons_per_um")
    max_caliber = _number(observables, "max_radius_nm", "soma_radius_nm")
    inhibitory_evidence = 0
    inhibitory_evidence += pre_ratio >= 0.60
    inhibitory_evidence += bouton_density >= 0.10
    inhibitory_evidence += 0.0 < max_caliber < 1_400.0
    return "Interneuron" if inhibitory_evidence >= 2 else "Pyramidal"


def derive_expected_lhs_v2(
    parent_symbol: str,
    observables: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return legal child productions given blind, observable morphology."""
    parent = parent_symbol.strip().upper()
    if parent in {"[SOMA]", "SOMA"}:
        if induce_cell_type_from_observables(observables) == "Interneuron":
            return ("<AspinyDendriteTree>", "<DenseAxonPlexus>")
        return ("<ApicalTree>", "<BasalTree>", "<AxonTree>")
    kind = type_segment_v2(observables)
    return {
        "Axon": ("<AxonContinuation>",),
        "Dendrite": ("<DendriteContinuation>",),
        "Glia": ("<GlialProcess>",),
        "Soma": ("<SomaContinuation>",),
    }[kind]


__all__ = [
    "derive_expected_lhs_v2",
    "induce_cell_type_from_observables",
    "type_segment_v2",
]
