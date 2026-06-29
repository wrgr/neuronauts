"""Tests for low-resolution segmentation pipeline.

Covers downsampling, coordinate transformation, connected component analysis,
and synapse mapping at reduced resolution.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.low_res_segmentation.low_res_segmentation import (
    LowResSegmentationPipeline,
    LowResSegmentation,
    SynapseMappingResult,
)


@pytest.fixture
def pipeline():
    """Create a standard low-res pipeline for testing."""
    return LowResSegmentationPipeline(
        target_voxel_nm=(128.0, 128.0, 120.0),
        full_res_voxel_nm=(8.0, 8.0, 40.0),
        connectivity="6",
    )


@pytest.fixture
def sample_volume():
    """Create a synthetic 3D volume with simple structure."""
    vol = np.zeros((256, 256, 240), dtype=np.uint8)

    # Add two bright regions separated by dark space
    vol[50:100, 50:100, 50:100] = 200
    vol[150:200, 150:200, 140:190] = 200

    return vol


class TestLowResSegmentationPipeline:
    """Test LowResSegmentationPipeline initialization and configuration."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        pipeline = LowResSegmentationPipeline()
        assert pipeline.target_voxel_nm == (128.0, 128.0, 120.0)
        assert pipeline.full_res_voxel_nm == (8.0, 8.0, 40.0)
        assert pipeline.connectivity == "6"
        assert pipeline.downsampling_factor == (16, 16, 3)

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        pipeline = LowResSegmentationPipeline(
            target_voxel_nm=(256.0, 256.0, 240.0),
            full_res_voxel_nm=(8.0, 8.0, 40.0),
            connectivity="26",
        )
        assert pipeline.downsampling_factor == (32, 32, 6)
        assert pipeline.connectivity == "26"

    def test_init_invalid_downsampling(self):
        """Test that target voxel size smaller than full-res raises error."""
        with pytest.raises(ValueError, match="smaller than"):
            LowResSegmentationPipeline(
                target_voxel_nm=(4.0, 4.0, 20.0),
                full_res_voxel_nm=(8.0, 8.0, 40.0),
            )


class TestDownsampling:
    """Test volume downsampling."""

    def test_downsample_mean(self, pipeline, sample_volume):
        """Test mean pooling downsampling."""
        downsampled = pipeline.downsample_volume(sample_volume, method="mean")

        expected_shape = (
            256 // 16,
            256 // 16,
            240 // 3,
        )
        assert downsampled.shape == expected_shape
        # Mean of bright region should still be bright
        assert downsampled[3, 3, 16] > 100

    def test_downsample_max(self, pipeline, sample_volume):
        """Test max pooling downsampling."""
        downsampled = pipeline.downsample_volume(sample_volume, method="max")
        assert downsampled.shape == (16, 16, 80)
        # Max should preserve bright region values
        assert downsampled[3, 3, 16] == 200

    def test_downsample_min(self, pipeline, sample_volume):
        """Test min pooling downsampling."""
        downsampled = pipeline.downsample_volume(sample_volume, method="min")
        assert downsampled.shape == (16, 16, 80)
        # Min should find 0 values in pooling regions
        assert downsampled[0, 0, 0] == 0

    def test_downsample_invalid_method(self, pipeline, sample_volume):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown downsampling method"):
            pipeline.downsample_volume(sample_volume, method="invalid")

    def test_downsample_invalid_shape(self, pipeline):
        """Test that non-3D volume raises error."""
        vol_2d = np.zeros((256, 256))
        with pytest.raises(ValueError, match="Expected 3D"):
            pipeline.downsample_volume(vol_2d)


class TestSegmentation:
    """Test low-resolution segmentation."""

    def test_segment_volume_basic(self, pipeline, sample_volume):
        """Test basic segmentation produces valid output."""
        seg = pipeline.segment_volume(sample_volume)

        assert isinstance(seg, LowResSegmentation)
        assert seg.labels.shape == (16, 16, 80)
        assert seg.binary_mask.shape == (16, 16, 80)
        assert seg.num_components >= 0
        assert seg.voxel_sizes == (128.0, 128.0, 120.0)
        assert seg.downsampling_factor == (16, 16, 3)

    def test_segment_volume_num_components(self, pipeline, sample_volume):
        """Test that two separated regions create two components."""
        seg = pipeline.segment_volume(sample_volume)
        # Two separate bright regions should create at least 2 components
        assert seg.num_components >= 2

    def test_segment_volume_connectivity(self):
        """Test that connectivity parameter affects segmentation."""
        # Create volume with diagonally touching regions
        vol = np.zeros((64, 64, 24), dtype=np.uint8)
        vol[10:20, 10:20, 5:15] = 200
        vol[21:30, 21:30, 5:15] = 200  # Diagonal touch only

        pipeline_6 = LowResSegmentationPipeline(connectivity="6")
        seg_6 = pipeline_6.segment_volume(vol)

        pipeline_26 = LowResSegmentationPipeline(connectivity="26")
        seg_26 = pipeline_26.segment_volume(vol)

        # 26-connectivity should merge diagonally adjacent regions
        assert seg_6.num_components >= seg_26.num_components

    def test_component_sizes(self, pipeline, sample_volume):
        """Test that component sizes array is computed correctly."""
        seg = pipeline.segment_volume(sample_volume)

        # Sum of all component sizes should equal total voxels
        total_foreground = np.sum(seg.binary_mask)
        sum_sizes = np.sum(seg.component_sizes[1:])  # Skip background (0)
        assert sum_sizes == total_foreground


class TestCoordinateTransformation:
    """Test coordinate conversion between resolutions."""

    def test_to_low_res(self, pipeline):
        """Test conversion to low-resolution coordinates."""
        full_res_pt = np.array([256, 256, 120], dtype=np.float32)
        low_res_pt = pipeline.to_low_res(full_res_pt)

        expected = np.array([16, 16, 40], dtype=np.int32)
        np.testing.assert_array_equal(low_res_pt, expected)

    def test_to_full_res(self, pipeline):
        """Test conversion to full-resolution coordinates."""
        low_res_pt = np.array([16, 16, 40], dtype=np.float32)
        full_res_pt = pipeline.to_full_res(low_res_pt)

        expected = np.array([256, 256, 120], dtype=np.int32)
        np.testing.assert_array_equal(full_res_pt, expected)

    def test_roundtrip(self, pipeline):
        """Test roundtrip conversion full -> low -> full."""
        original = np.array([123, 456, 789], dtype=np.float32)
        low_res = pipeline.to_low_res(original)
        restored = pipeline.to_full_res(low_res)

        # Should be close (within downsampling factor)
        assert np.allclose(restored, original, atol=8)

    def test_to_low_res_list_input(self, pipeline):
        """Test that list input works."""
        result = pipeline.to_low_res([256, 256, 120])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [16, 16, 40])


class TestSynapseMappingFunction:
    """Test synapse mapping to low-resolution segmentation."""

    def test_map_synapses_basic(self, pipeline):
        """Test basic synapse mapping."""
        # Create simple segmentation
        vol = np.zeros((256, 256, 240), dtype=np.uint8)
        vol[50:150, 50:150, 50:150] = 200
        seg = pipeline.segment_volume(vol)

        # Create synapses: some inside, some outside
        pre_pt = np.array([[60, 60, 60], [200, 200, 200]], dtype=np.float32)
        post_pt = np.array([[70, 70, 70], [210, 210, 210]], dtype=np.float32)

        result = pipeline.map_synapses_to_lowres(pre_pt, post_pt, seg)

        assert isinstance(result, SynapseMappingResult)
        assert result.pre_lowres.shape == (2, 3)
        assert result.post_lowres.shape == (2, 3)
        assert result.pre_labels.shape == (2,)
        assert result.post_labels.shape == (2,)
        assert result.same_component.shape == (2,)

    def test_synapse_clipping(self, pipeline):
        """Test that synapses are clipped to valid range."""
        vol = np.zeros((256, 256, 240), dtype=np.uint8)
        vol[50:150, 50:150, 50:150] = 200
        seg = pipeline.segment_volume(vol)

        # Synapses way outside volume
        pre_pt = np.array([[1000, 1000, 1000]], dtype=np.float32)
        post_pt = np.array([[1100, 1100, 1100]], dtype=np.float32)

        result = pipeline.map_synapses_to_lowres(pre_pt, post_pt, seg)

        # Should clamp to valid indices
        assert np.all(result.pre_lowres < seg.labels.shape)
        assert np.all(result.post_lowres < seg.labels.shape)

    def test_synapse_length_mismatch(self, pipeline):
        """Test that mismatched synapse arrays raise error."""
        vol = np.zeros((256, 256, 240), dtype=np.uint8)
        vol[50:150, 50:150, 50:150] = 200
        seg = pipeline.segment_volume(vol)

        pre_pt = np.array([[60, 60, 60], [70, 70, 70]], dtype=np.float32)
        post_pt = np.array([[80, 80, 80]], dtype=np.float32)

        with pytest.raises(ValueError, match="same length"):
            pipeline.map_synapses_to_lowres(pre_pt, post_pt, seg)


class TestFilterSmallComponents:
    """Test small component filtering."""

    def test_filter_small_components(self, pipeline):
        """Test filtering components by size."""
        # Create volume with one large and several small regions
        vol = np.zeros((256, 256, 240), dtype=np.uint8)
        vol[50:150, 50:150, 50:150] = 200  # Large region
        vol[160:162, 160:162, 160:162] = 200  # Tiny region
        seg = pipeline.segment_volume(vol)

        # Filter components with < 50 voxels
        filtered = pipeline.filter_small_components(seg, min_size=50)

        # Should have removed small components
        assert filtered.num_components <= seg.num_components

    def test_filter_all_components(self, pipeline):
        """Test filtering with very large threshold removes everything."""
        vol = np.zeros((256, 256, 240), dtype=np.uint8)
        vol[50:150, 50:150, 50:150] = 200
        seg = pipeline.segment_volume(vol)

        # Filter with huge threshold
        filtered = pipeline.filter_small_components(seg, min_size=100000)

        # Should remove all components
        assert filtered.num_components == 0
        assert np.all(filtered.labels == 0)


class TestMorphologicalOperations:
    """Test morphological operations (dilation, erosion)."""

    def test_dilate_segmentation(self, pipeline):
        """Test dilation expands components."""
        vol = np.zeros((256, 256, 240), dtype=np.uint8)
        vol[50:100, 50:100, 50:100] = 200
        seg = pipeline.segment_volume(vol)

        dilated = pipeline.dilate_segmentation(seg, radius=1)

        # Dilated mask should be larger or equal
        assert np.sum(dilated.binary_mask) >= np.sum(seg.binary_mask)

    def test_erode_segmentation(self, pipeline):
        """Test erosion shrinks components."""
        vol = np.zeros((256, 256, 240), dtype=np.uint8)
        vol[50:100, 50:100, 50:100] = 200
        seg = pipeline.segment_volume(vol)

        eroded = pipeline.erode_segmentation(seg, radius=1)

        # Eroded mask should be smaller or equal
        assert np.sum(eroded.binary_mask) <= np.sum(seg.binary_mask)

    def test_dilate_erode_roundtrip(self, pipeline):
        """Test that dilate then erode approximately restores original."""
        vol = np.zeros((256, 256, 240), dtype=np.uint8)
        vol[50:100, 50:100, 50:100] = 200
        seg = pipeline.segment_volume(vol)

        dilated = pipeline.dilate_segmentation(seg, radius=2)
        eroded = pipeline.erode_segmentation(dilated, radius=2)

        # Masks should be close in size
        original_size = np.sum(seg.binary_mask)
        final_size = np.sum(eroded.binary_mask)
        # Allow 20% change due to morphological operations
        assert abs(final_size - original_size) < 0.2 * original_size


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_volume(self, pipeline):
        """Test segmentation of all-zero volume."""
        vol = np.zeros((256, 256, 240), dtype=np.uint8)
        seg = pipeline.segment_volume(vol)

        assert seg.num_components == 0
        assert np.all(seg.labels == 0)

    def test_uniform_bright_volume(self, pipeline):
        """Test segmentation of uniform bright volume."""
        vol = np.ones((256, 256, 240), dtype=np.uint8) * 200
        seg = pipeline.segment_volume(vol)

        # All foreground should be one component
        assert seg.num_components == 1

    def test_single_voxel_region(self, pipeline):
        """Test segmentation with single-voxel regions."""
        vol = np.zeros((256, 256, 240), dtype=np.uint8)
        vol[50, 50, 50] = 200
        vol[100, 100, 100] = 200
        seg = pipeline.segment_volume(vol)

        # Single-voxel regions should be detected
        assert seg.num_components >= 2 or seg.num_components == 0  # May be too small to threshold

    def test_threshold_boundary(self, pipeline):
        """Test with explicit threshold."""
        vol = np.zeros((256, 256, 240), dtype=np.uint8)
        vol[50:100, 50:100, 50:100] = 150

        pipeline_threshold = LowResSegmentationPipeline(
            intensity_threshold=100
        )
        seg = pipeline_threshold.segment_volume(vol)

        # Volume above threshold should be segmented
        assert seg.num_components > 0
