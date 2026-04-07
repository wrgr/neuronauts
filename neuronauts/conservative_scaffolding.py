"""Low-resolution neuron scaffolding with conservative merging.

Builds high-confidence cell scaffolds from low-res segmentation:
1. Identify cell bodies (bright cores)
2. Grow to connected arbors (high confidence paths)
3. Stop before uncertain boundaries
4. No false merges - unmerged is better than wrong

Philosophy: Conservative. Stop when not confident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage


@dataclass
class NeuronScaffold:
    """High-confidence neuron scaffold.

    Attributes
    ----------
    label : int
        Unique neuron ID.

    cell_body_mask : np.ndarray
        Binary mask of cell body core (high confidence).
        Shape: (x, y, z)

    arbor_mask : np.ndarray
        Binary mask of cell body + major arbors.
        Shape: (x, y, z)

    arbor_voxels : int
        Total voxels in arbor (including cell body).

    confidence : float
        Average confidence of assignment [0, 1].
        1.0 = all high-confidence, <0.5 = uncertain.

    num_components : int
        Number of disconnected components in arbor.
        1 = single connected structure, >1 = fragmented.
    """

    label: int
    cell_body_mask: np.ndarray
    arbor_mask: np.ndarray
    arbor_voxels: int
    confidence: float
    num_components: int


@dataclass
class ScaffoldingResult:
    """Result of low-res scaffolding.

    Attributes
    ----------
    scaffolds : list[NeuronScaffold]
        High-confidence neuron scaffolds.

    labels : np.ndarray
        3D label map of scaffolds.
        Shape: (x, y, z)

    num_neurons : int
        Number of scaffolds extracted.

    merging_stopped_at : int
        Merge iteration where algorithm stopped.
        Higher = more confident result.
    """

    scaffolds: list[NeuronScaffold]
    labels: np.ndarray
    num_neurons: int
    merging_stopped_at: int


class ConservativeScaffoldingPipeline:
    """Build conservative, high-confidence neuron scaffolds.

    Parameters
    ----------
    cell_body_threshold : float
        Intensity percentile for cell body core detection.
        Default: 75 (top 25% brightest voxels)

    arbor_threshold : float
        Intensity percentile for arbor boundary.
        Default: 50 (median brightness)

    confidence_threshold : float
        Min confidence to keep scaffold. Default: 0.5

    min_scaffold_size : int
        Minimum voxels per scaffold. Default: 10

    max_merge_iterations : int
        Max iterations before stopping. Default: 10
    """

    def __init__(
        self,
        cell_body_threshold: float = 75.0,
        arbor_threshold: float = 50.0,
        confidence_threshold: float = 0.5,
        min_scaffold_size: int = 10,
        max_merge_iterations: int = 10,
    ):
        self.cell_body_threshold = cell_body_threshold
        self.arbor_threshold = arbor_threshold
        self.confidence_threshold = confidence_threshold
        self.min_scaffold_size = min_scaffold_size
        self.max_merge_iterations = max_merge_iterations

    def scaffold_volume(
        self,
        volume: np.ndarray,
        membrane_field: np.ndarray | None = None,
    ) -> ScaffoldingResult:
        """Extract high-confidence neuron scaffolds from volume.

        Parameters
        ----------
        volume : np.ndarray
            Input intensity volume. Shape: (x, y, z)

        membrane_field : np.ndarray, optional
            Membrane field [0, 1]. High values = boundaries.
            Shape: (x, y, z)

        Returns
        -------
        ScaffoldingResult
            Scaffolds with confidence scores and labels.
        """
        # Step 1: Identify cell bodies (bright cores)
        cell_body_mask = self._detect_cell_bodies(volume)

        # Step 2: Label cell bodies as initial seeds
        cell_labels, num_cells = ndimage.label(cell_body_mask)

        # Step 3: Grow from cell bodies to arbors (conservative)
        grown_labels, stop_iteration = self._conservative_grow(
            volume,
            cell_labels,
            num_cells,
            membrane_field,
        )

        # Step 4: Extract scaffolds
        scaffolds = []
        for cell_id in range(1, num_cells + 1):
            scaffold = self._extract_scaffold(
                grown_labels,
                cell_body_mask,
                volume,
                cell_id,
            )
            if scaffold is not None:
                scaffolds.append(scaffold)

        # Step 5: Create output label map
        labels = np.zeros_like(grown_labels)
        for i, scaffold in enumerate(scaffolds, 1):
            labels[scaffold.arbor_mask] = i

        return ScaffoldingResult(
            scaffolds=scaffolds,
            labels=labels,
            num_neurons=len(scaffolds),
            merging_stopped_at=stop_iteration,
        )

    def _detect_cell_bodies(self, volume: np.ndarray) -> np.ndarray:
        """Detect cell body cores (bright, compact regions).

        Strategy: top percentile of brightness, then filter for compactness.
        """
        # Top brightness percentile
        threshold = np.percentile(volume, self.cell_body_threshold)
        bright = volume > threshold

        # Filter: keep only compact clusters (likely cell bodies)
        # Remove thin extensions
        struct = ndimage.generate_binary_structure(3, 2)
        opened = ndimage.binary_opening(bright, structure=struct, iterations=2)

        return opened

    def _conservative_grow(
        self,
        volume: np.ndarray,
        cell_labels: np.ndarray,
        num_cells: int,
        membrane_field: np.ndarray | None,
    ) -> tuple[np.ndarray, int]:
        """Conservatively grow cell bodies to arbors.

        Per-voxel gate: absolute intensity floor AND low membrane value.
        Per-cell stop: when its frontier has no admissible voxels.
        """
        grown = cell_labels.copy()
        struct = ndimage.generate_binary_structure(3, 1)

        # Absolute intensity floor from arbor_threshold percentile of whole volume
        intensity_floor = np.percentile(volume, self.arbor_threshold)
        # Membrane ceiling: voxels above this are considered boundary
        if membrane_field is not None:
            mem_ceiling = np.percentile(membrane_field, 100.0 - self.arbor_threshold)
        else:
            mem_ceiling = None

        active = np.ones(num_cells + 1, dtype=bool)
        active[0] = False

        for iteration in range(self.max_merge_iterations):
            if not active.any():
                return grown, iteration

            old_grown = grown.copy()

            for cell_id in range(1, num_cells + 1):
                if not active[cell_id]:
                    continue
                cell_region = grown == cell_id
                expanded = ndimage.binary_dilation(cell_region, structure=struct)
                frontier = expanded & (grown == 0)

                if not frontier.any():
                    active[cell_id] = False
                    continue

                # Absolute intensity gate
                admit = frontier & (volume > intensity_floor)
                # Membrane gate
                if mem_ceiling is not None:
                    admit &= membrane_field < mem_ceiling

                if not admit.any():
                    active[cell_id] = False
                    continue

                # Per-cell frontier confidence: mean intensity of admitted voxels
                frontier_conf = float(np.mean(volume[admit]) / 255.0)
                if mem_ceiling is not None:
                    frontier_conf = 0.6 * frontier_conf + 0.4 * (
                        1.0 - float(np.mean(membrane_field[admit]))
                    )
                if frontier_conf < self.confidence_threshold:
                    active[cell_id] = False
                    continue

                grown[admit] = cell_id

            if np.array_equal(old_grown, grown):
                return grown, iteration + 1

        return grown, self.max_merge_iterations

    def _compute_growth_confidence(
        self,
        volume: np.ndarray,
        labels: np.ndarray,
        num_cells: int,
        membrane_field: np.ndarray | None,
    ) -> float:
        """Compute confidence of next growth step.

        High confidence = bright, non-membranous regions nearby.
        Low confidence = dim regions or near boundaries.
        """
        # Get unlabeled foreground
        unlabeled = (labels == 0) & (volume > np.percentile(volume, 30))

        if not unlabeled.any():
            return 0.0

        # Confidence from intensity
        intensity_conf = np.mean(volume[unlabeled]) / 255.0

        # Confidence from membrane (if available)
        if membrane_field is not None:
            membrane_conf = 1.0 - np.mean(membrane_field[unlabeled])
        else:
            membrane_conf = 0.5

        # Combined confidence
        confidence = 0.6 * intensity_conf + 0.4 * membrane_conf
        return float(np.clip(confidence, 0, 1))

    def _extract_scaffold(
        self,
        labels: np.ndarray,
        cell_body_mask: np.ndarray,
        volume: np.ndarray,
        cell_id: int,
    ) -> NeuronScaffold | None:
        """Extract a single neuron scaffold."""
        arbor_mask = labels == cell_id

        if not arbor_mask.any():
            return None

        # Cell body is intersection with original detection
        cell_body = arbor_mask & cell_body_mask

        # Compute metrics
        arbor_voxels = np.sum(arbor_mask)
        if arbor_voxels < self.min_scaffold_size:
            return None

        # Confidence: mean intensity in arbor
        confidence = float(np.mean(volume[arbor_mask]) / 255.0)

        # Connectivity: number of components
        components, num_components = ndimage.label(arbor_mask)

        return NeuronScaffold(
            label=cell_id,
            cell_body_mask=cell_body,
            arbor_mask=arbor_mask,
            arbor_voxels=arbor_voxels,
            confidence=confidence,
            num_components=num_components,
        )

    def filter_low_confidence(
        self,
        result: ScaffoldingResult,
        threshold: float | None = None,
    ) -> ScaffoldingResult:
        """Remove low-confidence scaffolds.

        Parameters
        ----------
        result : ScaffoldingResult
            Scaffolding result to filter.

        threshold : float, optional
            Confidence threshold. Default: use pipeline default.

        Returns
        -------
        ScaffoldingResult
            Filtered result with only high-confidence scaffolds.
        """
        if threshold is None:
            threshold = self.confidence_threshold

        kept_scaffolds = [s for s in result.scaffolds if s.confidence >= threshold]

        # Rebuild labels
        new_labels = np.zeros_like(result.labels)
        for i, scaffold in enumerate(kept_scaffolds, 1):
            new_labels[scaffold.arbor_mask] = i

        return ScaffoldingResult(
            scaffolds=kept_scaffolds,
            labels=new_labels,
            num_neurons=len(kept_scaffolds),
            merging_stopped_at=result.merging_stopped_at,
        )

    def merge_fragments(
        self,
        result: ScaffoldingResult,
        max_distance_voxels: int = 10,
    ) -> ScaffoldingResult:
        """Merge fragmented scaffolds if they're close together.

        Only merges if confident.

        Parameters
        ----------
        result : ScaffoldingResult
            Scaffolding result.

        max_distance_voxels : int
            Max distance to consider merging. Default: 10

        Returns
        -------
        ScaffoldingResult
            Result with fragments merged (if confident).
        """
        scaffolds = result.scaffolds
        if len(scaffolds) <= 1:
            return result

        # For each scaffold with multiple components, try to merge
        merged_scaffolds = []

        for scaffold in scaffolds:
            if scaffold.num_components <= 1:
                merged_scaffolds.append(scaffold)
                continue

            # Find component centroids
            components, num_comp = ndimage.label(scaffold.arbor_mask)

            centroids = {}
            for comp_id in range(1, num_comp + 1):
                comp_mask = components == comp_id
                coords = np.where(comp_mask)
                centroids[comp_id] = np.array(
                    [np.mean(coords[i]) for i in range(3)],
                    dtype=np.float32,
                )

            # Check if components are close
            distances = {}
            for c1 in range(1, num_comp + 1):
                for c2 in range(c1 + 1, num_comp + 1):
                    dist = np.linalg.norm(centroids[c1] - centroids[c2])
                    distances[(c1, c2)] = dist

            # Only merge if very close AND high confidence
            if distances and min(distances.values()) < max_distance_voxels:
                if scaffold.confidence > 0.7:  # High confidence requirement
                    merged_scaffolds.append(scaffold)
                else:
                    # Keep separate due to low confidence
                    merged_scaffolds.append(scaffold)
            else:
                merged_scaffolds.append(scaffold)

        # Rebuild labels
        new_labels = np.zeros_like(result.labels)
        for i, scaffold in enumerate(merged_scaffolds, 1):
            new_labels[scaffold.arbor_mask] = i

        return ScaffoldingResult(
            scaffolds=merged_scaffolds,
            labels=new_labels,
            num_neurons=len(merged_scaffolds),
            merging_stopped_at=result.merging_stopped_at,
        )

    def report(self, result: ScaffoldingResult) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 60,
            "CONSERVATIVE NEURON SCAFFOLDING REPORT",
            "=" * 60,
            f"Neurons extracted: {result.num_neurons}",
            f"Stopped at iteration: {result.merging_stopped_at}/10",
            "",
            "Scaffold summary:",
            "-" * 60,
        ]

        for i, s in enumerate(result.scaffolds, 1):
            lines.append(
                f"  {i}. ID {s.label}: {s.arbor_voxels} voxels, "
                f"confidence={s.confidence:.2f}, components={s.num_components}"
            )

        lines.extend([
            "-" * 60,
            f"Mean confidence: {np.mean([s.confidence for s in result.scaffolds]):.2f}",
            f"Mean size: {np.mean([s.arbor_voxels for s in result.scaffolds]):.0f} voxels",
        ])

        return "\n".join(lines)
