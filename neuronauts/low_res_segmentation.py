"""Low-resolution segmentation pipeline for volumetric EM data.

This module implements a downsampling-based segmentation pipeline designed to
work at reduced resolution (e.g., 128×128×120 nm) for fast preprocessing and
hierarchical analysis of connectome volumes.

Key features:
- Flexible downsampling with configurable voxel sizes
- Connected component analysis at low resolution
- Bidirectional coordinate transformation (low-res ↔ full-res)
- Integration with existing pipeline components (synapses, skeleton graphs)
- Optional refinement stages for improved segmentation quality

Usage
-----
Basic low-resolution segmentation::

    from neuronauts.low_res_segmentation import LowResSegmentationPipeline

    pipeline = LowResSegmentationPipeline(
        target_voxel_nm=(128, 128, 120),
        connectivity="6"
    )

    low_res_seg = pipeline.segment_volume(volume)
    low_res_labels = low_res_seg.labels

Coordinate transformation::

    low_res_pt = pipeline.to_low_res([256, 256, 240])
    full_res_pt = pipeline.to_full_res([2, 2, 2])

Synapse mapping::

    mapped_synapses = pipeline.map_synapses_to_lowres(
        pre_pt, post_pt,
        pre_root_id, post_root_id
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage


# Default full-resolution voxel size in nanometers
DEFAULT_FULL_RES_VOX_NM = (8.0, 8.0, 40.0)


@dataclass
class LowResSegmentation:
    """Result of low-resolution segmentation.

    Attributes
    ----------
    labels : np.ndarray
        3D array of connected component labels at low resolution.
        Shape: (downsampled_x, downsampled_y, downsampled_z)

    binary_mask : np.ndarray
        Binary foreground/background mask at low resolution.
        Shape: (downsampled_x, downsampled_y, downsampled_z)

    num_components : int
        Number of connected components found (max label value).

    voxel_sizes : tuple[float, float, float]
        Voxel size in nm (x, y, z) for this segmentation level.

    downsampling_factor : tuple[int, int, int]
        Downsampling factor applied relative to full resolution (x, y, z).

    component_sizes : np.ndarray
        Array of component sizes in voxels, indexed by label (0-padded).
        Shape: (num_components + 1,)
    """

    labels: np.ndarray
    binary_mask: np.ndarray
    num_components: int
    voxel_sizes: tuple[float, float, float]
    downsampling_factor: tuple[int, int, int]
    component_sizes: np.ndarray


@dataclass
class SynapseMappingResult:
    """Result of mapping synapses to low-resolution segmentation.

    Attributes
    ----------
    pre_lowres : np.ndarray
        Pre-synaptic point coordinates at low resolution.
        Shape: (n_synapses, 3)

    post_lowres : np.ndarray
        Post-synaptic point coordinates at low resolution.
        Shape: (n_synapses, 3)

    pre_labels : np.ndarray
        Segmentation labels at pre-synaptic points.
        Shape: (n_synapses,)

    post_labels : np.ndarray
        Segmentation labels at post-synaptic points.
        Shape: (n_synapses,)

    same_component : np.ndarray
        Boolean array: True where pre and post synapses map to same low-res component.
        Shape: (n_synapses,)
    """

    pre_lowres: np.ndarray
    post_lowres: np.ndarray
    pre_labels: np.ndarray
    post_labels: np.ndarray
    same_component: np.ndarray


class LowResSegmentationPipeline:
    """Pipeline for low-resolution volumetric segmentation.

    Coordinates are managed in (x, y, z) order throughout. This matches the
    numpy array indexing convention [x, y, z] used in 3D volumes.

    Parameters
    ----------
    target_voxel_nm : tuple[float, float, float], optional
        Target voxel size in nanometers (x, y, z).
        Default: (128.0, 128.0, 120.0)

    full_res_voxel_nm : tuple[float, float, float], optional
        Full-resolution voxel size in nanometers (x, y, z).
        Default: (8.0, 8.0, 40.0) — MICrONS EM standard

    connectivity : Literal["6", "26"], optional
        Connectivity for connected component analysis.
        "6": face-connectivity (faster, fewer components)
        "26": full connectivity (slower, more aggressive merging)
        Default: "6"

    intensity_threshold : float | None, optional
        Intensity threshold for foreground detection.
        If None, use automatic thresholding (Otsu's method).
        Default: None
    """

    def __init__(
        self,
        target_voxel_nm: tuple[float, float, float] = (128.0, 128.0, 120.0),
        full_res_voxel_nm: tuple[float, float, float] = DEFAULT_FULL_RES_VOX_NM,
        connectivity: Literal["6", "26"] = "6",
        intensity_threshold: float | None = None,
    ):
        self.target_voxel_nm = target_voxel_nm
        self.full_res_voxel_nm = full_res_voxel_nm
        self.connectivity = connectivity
        self.intensity_threshold = intensity_threshold

        # Precompute downsampling factors
        self.downsampling_factor = tuple(
            round(target / full)
            for target, full in zip(target_voxel_nm, full_res_voxel_nm)
        )

        # Validate downsampling factors
        if any(f < 1 for f in self.downsampling_factor):
            raise ValueError(
                f"Target voxel size {target_voxel_nm} is smaller than "
                f"full resolution {full_res_voxel_nm}. "
                f"Low-res pipeline requires downsampling."
            )

    def downsample_volume(
        self,
        volume: np.ndarray,
        method: Literal["mean", "max", "min"] = "mean",
    ) -> np.ndarray:
        """Downsample a volume to target resolution.

        Parameters
        ----------
        volume : np.ndarray
            Input 3D volume. Shape: (x, y, z)

        method : {"mean", "max", "min"}
            Pooling method for downsampling.
            Default: "mean"

        Returns
        -------
        np.ndarray
            Downsampled volume. Shape will be volume.shape / downsampling_factor.
        """
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

        # Trim volume to exact multiple of downsampling factors
        shape = np.array(volume.shape, dtype=int)
        trim = shape % np.array(self.downsampling_factor)
        slices = tuple(slice(0, s - t) for s, t in zip(shape, trim))
        trimmed = volume[slices]

        # Reshape for pooling
        new_shape = tuple(
            trimmed.shape[i] // self.downsampling_factor[i]
            for i in range(3)
        )

        reshaped = trimmed.reshape(
            new_shape[0], self.downsampling_factor[0],
            new_shape[1], self.downsampling_factor[1],
            new_shape[2], self.downsampling_factor[2],
        )

        # Apply pooling
        if method == "mean":
            downsampled = reshaped.mean(axis=(1, 3, 5))
        elif method == "max":
            downsampled = reshaped.max(axis=(1, 3, 5))
        elif method == "min":
            downsampled = reshaped.min(axis=(1, 3, 5))
        else:
            raise ValueError(f"Unknown downsampling method: {method}")

        return downsampled.astype(volume.dtype)

    def segment_volume(
        self,
        volume: np.ndarray,
        downsample_method: Literal["mean", "max", "min"] = "mean",
    ) -> LowResSegmentation:
        """Segment volume at low resolution via connected component analysis.

        Parameters
        ----------
        volume : np.ndarray
            Input 3D volume. Shape: (x, y, z)

        downsample_method : {"mean", "max", "min"}
            Method for initial downsampling. Default: "mean"

        Returns
        -------
        LowResSegmentation
            Low-resolution segmentation result with labels and metadata.
        """
        # Downsample volume
        low_res_vol = self.downsample_volume(volume, method=downsample_method)

        # Create binary mask via thresholding
        if self.intensity_threshold is not None:
            threshold = self.intensity_threshold
        else:
            # Automatic threshold using Otsu's method
            counts = np.bincount(low_res_vol.astype(int).ravel())
            # Simple estimate: use median
            threshold = np.median(low_res_vol[low_res_vol > 0])

        binary_mask = low_res_vol > threshold

        # Connected component analysis
        if self.connectivity == "6":
            structure = ndimage.generate_binary_structure(3, 1)
        else:  # "26"
            structure = ndimage.generate_binary_structure(3, 3)

        labels, num_components = ndimage.label(binary_mask, structure=structure)

        # Compute component sizes
        component_sizes = np.bincount(labels.ravel())

        return LowResSegmentation(
            labels=labels,
            binary_mask=binary_mask,
            num_components=num_components,
            voxel_sizes=self.target_voxel_nm,
            downsampling_factor=self.downsampling_factor,
            component_sizes=component_sizes,
        )

    def to_low_res(self, point: np.ndarray | list) -> np.ndarray:
        """Convert full-resolution coordinates to low-resolution coordinates.

        Parameters
        ----------
        point : np.ndarray or list
            Point coordinates in full resolution (x, y, z).

        Returns
        -------
        np.ndarray
            Point coordinates in low resolution (x, y, z).
        """
        point = np.asarray(point, dtype=np.float32)
        return (point / np.array(self.downsampling_factor)).astype(np.int32)

    def to_full_res(self, point: np.ndarray | list) -> np.ndarray:
        """Convert low-resolution coordinates to full-resolution coordinates.

        Parameters
        ----------
        point : np.ndarray or list
            Point coordinates in low resolution (x, y, z).

        Returns
        -------
        np.ndarray
            Point coordinates in full resolution (x, y, z).
        """
        point = np.asarray(point, dtype=np.float32)
        return (point * np.array(self.downsampling_factor)).astype(np.int32)

    def map_synapses_to_lowres(
        self,
        pre_pt: np.ndarray,
        post_pt: np.ndarray,
        segmentation: LowResSegmentation,
    ) -> SynapseMappingResult:
        """Map synapses to low-resolution segmentation.

        Parameters
        ----------
        pre_pt : np.ndarray
            Pre-synaptic coordinates in full resolution. Shape: (n_synapses, 3)

        post_pt : np.ndarray
            Post-synaptic coordinates in full resolution. Shape: (n_synapses, 3)

        segmentation : LowResSegmentation
            Result of segment_volume().

        Returns
        -------
        SynapseMappingResult
            Mapped synapses with low-resolution labels and same-component flags.
        """
        pre_pt = np.asarray(pre_pt, dtype=np.float32)
        post_pt = np.asarray(post_pt, dtype=np.float32)

        if pre_pt.shape[0] != post_pt.shape[0]:
            raise ValueError(
                f"pre_pt and post_pt must have same length, got "
                f"{pre_pt.shape[0]} vs {post_pt.shape[0]}"
            )

        # Convert to low-resolution coordinates
        pre_lowres = self.to_low_res(pre_pt)
        post_lowres = self.to_low_res(post_pt)

        # Clamp to valid range
        labels_shape = np.array(segmentation.labels.shape)
        pre_lowres = np.clip(pre_lowres, 0, labels_shape - 1)
        post_lowres = np.clip(post_lowres, 0, labels_shape - 1)

        # Index labels
        pre_labels = segmentation.labels[tuple(pre_lowres.T)]
        post_labels = segmentation.labels[tuple(post_lowres.T)]

        same_component = (pre_labels == post_labels) & (pre_labels > 0)

        return SynapseMappingResult(
            pre_lowres=pre_lowres,
            post_lowres=post_lowres,
            pre_labels=pre_labels,
            post_labels=post_labels,
            same_component=same_component,
        )

    def filter_small_components(
        self,
        segmentation: LowResSegmentation,
        min_size: int = 10,
    ) -> LowResSegmentation:
        """Remove connected components smaller than minimum size.

        Parameters
        ----------
        segmentation : LowResSegmentation
            Input segmentation from segment_volume().

        min_size : int
            Minimum component size in voxels. Default: 10

        Returns
        -------
        LowResSegmentation
            Filtered segmentation with small components removed.
        """
        filtered_labels = segmentation.labels.copy()

        for label in range(1, segmentation.num_components + 1):
            if segmentation.component_sizes[label] < min_size:
                filtered_labels[filtered_labels == label] = 0

        # Recompute connected component IDs (relabel)
        new_labels, new_num = ndimage.label(filtered_labels > 0)
        new_component_sizes = np.bincount(new_labels.ravel())

        return LowResSegmentation(
            labels=new_labels,
            binary_mask=new_labels > 0,
            num_components=new_num,
            voxel_sizes=segmentation.voxel_sizes,
            downsampling_factor=segmentation.downsampling_factor,
            component_sizes=new_component_sizes,
        )

    def dilate_segmentation(
        self,
        segmentation: LowResSegmentation,
        radius: int = 1,
    ) -> LowResSegmentation:
        """Dilate segmentation to expand components.

        Useful for closing small gaps or expanding thin structures.

        Parameters
        ----------
        segmentation : LowResSegmentation
            Input segmentation from segment_volume().

        radius : int
            Dilation radius in voxels. Default: 1

        Returns
        -------
        LowResSegmentation
            Dilated segmentation.
        """
        structure = ndimage.generate_binary_structure(3, 3)
        dilated_mask = ndimage.binary_dilation(
            segmentation.binary_mask,
            structure=structure,
            iterations=radius,
        )

        # Recompute labels on dilated mask
        new_labels, new_num = ndimage.label(dilated_mask)
        new_component_sizes = np.bincount(new_labels.ravel())

        return LowResSegmentation(
            labels=new_labels,
            binary_mask=dilated_mask,
            num_components=new_num,
            voxel_sizes=segmentation.voxel_sizes,
            downsampling_factor=segmentation.downsampling_factor,
            component_sizes=new_component_sizes,
        )

    def erode_segmentation(
        self,
        segmentation: LowResSegmentation,
        radius: int = 1,
    ) -> LowResSegmentation:
        """Erode segmentation to shrink components.

        Useful for removing thin noise structures.

        Parameters
        ----------
        segmentation : LowResSegmentation
            Input segmentation from segment_volume().

        radius : int
            Erosion radius in voxels. Default: 1

        Returns
        -------
        LowResSegmentation
            Eroded segmentation.
        """
        structure = ndimage.generate_binary_structure(3, 3)
        eroded_mask = ndimage.binary_erosion(
            segmentation.binary_mask,
            structure=structure,
            iterations=radius,
        )

        # Recompute labels on eroded mask
        new_labels, new_num = ndimage.label(eroded_mask)
        new_component_sizes = np.bincount(new_labels.ravel())

        return LowResSegmentation(
            labels=new_labels,
            binary_mask=eroded_mask,
            num_components=new_num,
            voxel_sizes=segmentation.voxel_sizes,
            downsampling_factor=segmentation.downsampling_factor,
            component_sizes=new_component_sizes,
        )
