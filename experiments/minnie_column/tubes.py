"""Axis-aligned tube approximations as bboxes in nanometers."""

from __future__ import annotations

from neuronauts.fetch import RealBoxSpec


def tube_bbox_nm_from_soma(
    center_nm: tuple[int, int, int],
    *,
    radius_xy_um: float,
    z_half_extent_um: float,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Build a bbox that approximates a vertical "tube": wide in xy, extent in z.

    Parameters
    ----------
    center_nm
        Soma or arbor reference center (nm).
    radius_xy_um
        Half-size in x and y in **microns** (cube side = 2 * radius_xy_um).
    z_half_extent_um
        Half-height in z (microns): box z extent = 2 * z_half_extent_um.
    """
    cx, cy, cz = (int(center_nm[0]), int(center_nm[1]), int(center_nm[2]))
    r_nm = int(float(radius_xy_um) * 1000.0)
    hz_nm = int(float(z_half_extent_um) * 1000.0)
    return (
        (cx - r_nm, cy - r_nm, cz - hz_nm),
        (cx + r_nm, cy + r_nm, cz + hz_nm),
    )


def real_box_spec_for_tube(
    center_nm: tuple[int, int, int],
    *,
    radius_xy_um: float,
    z_half_extent_um: float,
    mip: int = 2,
) -> RealBoxSpec:
    """Return a :class:`RealBoxSpec` whose **cube** side fits the xy diameter.

    Note: :class:`RealBoxSpec` is a cube; for non-cube tubes use ``tube_bbox_nm_from_soma``
    and call ``fetch_synapses(bbox_nm, ...)`` directly with the returned **non-cube** bbox.
    """
    side_um = float(radius_xy_um) * 2.0
    return RealBoxSpec(center_nm=center_nm, side_um=side_um, mip=mip)


def fetch_bbox_for_tube(
    center_nm: tuple[int, int, int],
    *,
    radius_xy_um: float,
    z_half_extent_um: float,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Preferred non-cube bbox for CAVE synapse queries."""
    return tube_bbox_nm_from_soma(
        center_nm,
        radius_xy_um=radius_xy_um,
        z_half_extent_um=z_half_extent_um,
    )


__all__ = [
    "tube_bbox_nm_from_soma",
    "real_box_spec_for_tube",
    "fetch_bbox_for_tube",
]
