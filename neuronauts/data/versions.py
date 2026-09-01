"""Single source of truth for materialization versions and coordinate frames.

Why this module exists
----------------------
Version defaults were previously scattered across the codebase with three
different values (117 / 1412 / 1718), and coordinate frames were assumed
per-call-site. Both classes of drift have caused real, measured damage:

- ``neuronauts/data/lineage.py`` warned that **v1412 is expired**; this was
  confirmed live on 2026-09-01 — the server reports available versions
  ``[117, 943, 1300, 1507, 1621, 1718, 1822]`` with **no 1412**. Any module
  still defaulting to 1412 silently resolves nothing.
- A stale ``(8, 8, 40)`` voxel assumption once placed ~93% of seeded box
  centres outside the EM volume (see ``neuronauts/dataset_builder.py``).

Import these constants; do not re-declare them.

The version pair
----------------
``BASE_VERSION`` (117) is the segmentation being corrected — the "before"
state whose merge and split errors we want to detect.

``LABEL_VERSION`` (1718) supplies supervision: a v117 root's label is the set
of ``LABEL_VERSION`` roots its supervoxels resolve to. Two v117 roots sharing a
label root were split apart by the segmentation and merged by a proofreader
(a *true merge pair*); one v117 root spanning two label roots is a
*frankenmerge*.

1718 is chosen over the newer 1822 because the trustworthy Phase 2.3–2.12
results were derived against it, so tooling and findings stay comparable.
1822 carries more accumulated proofreading and is the natural next bump — do
that deliberately, as ``bench_v2``, not incidentally.
"""

from __future__ import annotations

from typing import Optional

# --------------------------------------------------------------------------
# Materialization versions
# --------------------------------------------------------------------------

#: The segmentation whose errors we are correcting (the "before" state).
BASE_VERSION: int = 117

#: The proofread materialization supplying ground-truth labels (the "after").
LABEL_VERSION: int = 1718

#: Newer version available; a deliberate future bump, not a drop-in default.
CANDIDATE_NEXT_LABEL_VERSION: int = 1822

#: Known-expired versions. Defaulting to one of these resolves nothing.
EXPIRED_VERSIONS: frozenset[int] = frozenset({1412})

DATASTACK: str = "minnie65_public"
SYNAPSE_TABLE: str = "synapses_pni_2"


# --------------------------------------------------------------------------
# Coordinate frames
# --------------------------------------------------------------------------

#: Voxel size of the synapse table (``synapses_pni_2``) in nm.
#: Positions in that table are voxel indices; multiply by this to get nm.
SYNAPSE_VOXEL_NM: tuple[float, float, float] = (4.0, 4.0, 40.0)

#: Voxel size of the EM imagery at its base mip, in nm.
EM_VOXEL_NM: tuple[float, float, float] = (4.0, 4.0, 40.0)

#: Voxel size used by the static nucleus CSV export, in nm. NOT the same as
#: the synapse table; mixing the two is the "93% of boxes outside the volume"
#: bug. Always convert explicitly.
NUCLEUS_CSV_VOXEL_NM: tuple[float, float, float] = (8.0, 8.0, 40.0)


class VersionContractError(RuntimeError):
    """Raised when the pinned versions disagree with what the server offers."""


def nm_to_voxel(
    xyz_nm, voxel_nm: tuple[float, float, float] = SYNAPSE_VOXEL_NM
):
    """Convert nm coordinates to voxel indices in *voxel_nm*'s frame.

    Accepts a 3-tuple or any array-like broadcastable against ``(..., 3)``.
    """
    import numpy as np

    arr = np.asarray(xyz_nm, dtype=float)
    return arr / np.asarray(voxel_nm, dtype=float)


def voxel_to_nm(
    xyz_voxel, voxel_nm: tuple[float, float, float] = SYNAPSE_VOXEL_NM
):
    """Convert voxel indices in *voxel_nm*'s frame to nm coordinates."""
    import numpy as np

    arr = np.asarray(xyz_voxel, dtype=float)
    return arr * np.asarray(voxel_nm, dtype=float)


def verify_version_contract(
    base: int = BASE_VERSION,
    label: int = LABEL_VERSION,
    token: Optional[str] = None,
) -> dict:
    """Assert the pinned versions are actually available on the server.

    Fails closed. A dataset build that silently proceeds against a missing or
    expired materialization produces labels that look fine and mean nothing —
    exactly the class of failure this project is recovering from.

    Returns a provenance dict suitable for stamping into a manifest.

    Raises:
        VersionContractError: if either version is expired or unavailable.
    """
    from neuronauts.data import lineage

    if base in EXPIRED_VERSIONS:
        raise VersionContractError(f"BASE_VERSION {base} is known-expired.")
    if label in EXPIRED_VERSIONS:
        raise VersionContractError(
            f"LABEL_VERSION {label} is known-expired (v1412 was removed "
            "server-side; use 1718)."
        )

    available = lineage.list_versions(token) if token else lineage.list_versions()
    if not available:
        raise VersionContractError(
            "Could not list materialization versions — refusing to proceed "
            "rather than guess. Check the CAVE token and network."
        )
    missing = [v for v in (base, label) if v not in available]
    if missing:
        raise VersionContractError(
            f"Pinned version(s) {missing} not available. Server offers "
            f"{sorted(available)}."
        )

    base_ts = lineage.version_timestamp(base)
    label_ts = lineage.version_timestamp(label)
    if base_ts is None or label_ts is None:
        raise VersionContractError(
            f"Could not resolve timestamps (base={base_ts}, label={label_ts})."
        )
    if label_ts <= base_ts:
        raise VersionContractError(
            f"LABEL_VERSION {label} (ts {label_ts}) is not newer than "
            f"BASE_VERSION {base} (ts {base_ts}); supervision would be empty "
            "or inverted."
        )

    return {
        "datastack": DATASTACK,
        "synapse_table": SYNAPSE_TABLE,
        "base_version": base,
        "label_version": label,
        "base_timestamp": base_ts,
        "label_timestamp": label_ts,
        "available_versions": sorted(available),
        "synapse_voxel_nm": list(SYNAPSE_VOXEL_NM),
    }
