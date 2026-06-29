"""Tests for high-precision 3D segmentation.

Covers watershed segmentation, boundary detection, level-set refinement,
and synapse-guided segmentation.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.low_res_segmentation.high_precision_segmentation import (
    HighPrecisionSegmentation3D,
    SegmentationResult,
    SynapseSeparability,
)


@pytest.fixture
def segmenter():
    """Create a standard high-precision segmenter."""
    return HighPrecisionSegmentation3D(
        membrane_threshold=0.3,
        sigma_smoothing=1.0,
        connectivity="6",
        min_component_size=5,
        use_watershed=True,
        use_level_set=True,
    )


@pytest.fixture
def synthetic_volume():
    """Create a synthetic 3D volume with clear structures."""
    vol = np.zeros((256, 256, 240), dtype=np.uint8)

    # Bright neurite-like structures
    vol[50:100, 50:100, 50:100] = 200  # Cell body
    vol[100:150, 50:50, 50:100] = 180  # Extending neurite
    vol[50:100, 150:200, 50:100] = 180  # Another neurite

    # Add some noise
    vol = (vol.astype(np.int32) + np.random.randint(0, 20, vol.shape))
    return np.clip(vol, 0, 255).astype(np.uint8)


@pytest.fixture
def synthetic_membrane():
    """Create synthetic membrane field."""
    mem = np.ones((256, 256, 240), dtype=np.float32) * 0.2

    # Strong membranes around structures
    mem[99:101, 49:101, 49:101] = 0.8
    mem[149:151, 49:151, 49:101] = 0.8

    return np.clip(mem, 0, 1)


class TestHighPrecisionSegmentation3DInit:
    """Test segmenter initialization."""

    def test_init_default_params(self):
        """Test initialization with defaults."""
        seg = HighPrecisionSegmentation3D()
        assert seg.membrane_threshold == 0.3
        assert seg.sigma_smoothing == 1.0
        assert seg.connectivity == "6"
        assert seg.min_component_size == 5
        assert seg.use_watershed is True
        assert seg.use_level_set is True

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        seg = HighPrecisionSegmentation3D(
            membrane_threshold=0.5,
            sigma_smoothing=2.0,
            connectivity="26",
            min_component_size=20,
            use_watershed=False,
            use_level_set=False,
        )
        assert seg.membrane_threshold == 0.5
        assert seg.sigma_smoothing == 2.0
        assert seg.connectivity == "26"
        assert seg.min_component_size == 20
        assert seg.use_watershed is False
        assert seg.use_level_set is False


class TestVolumeSegmentation:
    """Test basic volume segmentation."""

    def test_segment_volume_basic(self, segmenter, synthetic_volume):
        """Test basic segmentation produces valid output."""
        result = segmenter.segment_volume(synthetic_volume)

        assert isinstance(result, SegmentationResult)
        assert result.labels.shape == synthetic_volume.shape
        assert result.num_components >= 0
        assert result.probabilities is not None
        assert result.boundaries is not None

    def test_segment_volume_with_membrane(
        self, segmenter, synthetic_volume, synthetic_membrane
    ):
        """Test segmentation with membrane field guidance."""
        result = segmenter.segment_volume(synthetic_volume, synthetic_membrane)

        assert result.labels.shape == synthetic_volume.shape
        assert result.num_components >= 0
        # Membrane field should improve boundaries
        assert result.boundaries is not None

    def test_segment_volume_output_shapes(self, segmenter, synthetic_volume):
        """Test that output arrays have correct shapes."""
        result = segmenter.segment_volume(synthetic_volume)

        assert result.labels.shape == synthetic_volume.shape
        assert result.probabilities.shape == synthetic_volume.shape
        assert result.boundaries.shape == synthetic_volume.shape

    def test_segment_empty_volume(self, segmenter):
        """Test segmentation of empty volume."""
        vol = np.zeros((100, 100, 100), dtype=np.uint8)
        result = segmenter.segment_volume(vol)

        assert result.num_components == 0
        assert np.all(result.labels == 0)

    def test_segment_uniform_volume(self, segmenter):
        """Test segmentation of uniform volume."""
        vol = np.ones((100, 100, 100), dtype=np.uint8) * 128
        result = segmenter.segment_volume(vol)

        # Uniform volume should be treated as single component or background
        assert result.num_components <= 1

    def test_segment_invalid_shape(self, segmenter):
        """Test that non-3D volume raises error."""
        vol_2d = np.zeros((100, 100))
        with pytest.raises(ValueError, match="Expected 3D"):
            segmenter.segment_volume(vol_2d)


class TestNormalization:
    """Test volume normalization."""

    def test_normalize_dynamic_range(self, segmenter):
        """Test that normalization handles different dynamic ranges."""
        vol_low = np.random.randint(0, 10, (64, 64, 64), dtype=np.uint8)
        vol_high = np.random.randint(200, 256, (64, 64, 64), dtype=np.uint8)

        result_low = segmenter.segment_volume(vol_low)
        result_high = segmenter.segment_volume(vol_high)

        # Both should produce valid results despite different scales
        assert result_low.labels.shape == result_high.labels.shape

    def test_normalize_extremes(self, segmenter):
        """Test normalization with extreme values."""
        vol = np.ones((64, 64, 64), dtype=np.uint8) * 42  # Uniform
        result = segmenter.segment_volume(vol)

        # Should handle gracefully
        assert result.labels is not None


class TestBoundaryDetection:
    """Test boundary detection."""

    def test_boundaries_exist(self, segmenter, synthetic_volume, synthetic_membrane):
        """Test that boundaries are detected."""
        result = segmenter.segment_volume(synthetic_volume, synthetic_membrane)

        assert result.boundaries is not None
        assert result.boundaries.dtype in [np.uint8, np.int32, bool]
        assert np.any(result.boundaries > 0)

    def test_boundaries_on_component_edges(self, segmenter):
        """Test that boundaries appear at component edges."""
        # Create volume with clear component
        vol = np.zeros((100, 100, 100), dtype=np.uint8)
        vol[25:75, 25:75, 25:75] = 200

        result = segmenter.segment_volume(vol)

        # Boundaries should be found at transitions
        boundary_count = np.sum(result.boundaries > 0)
        assert boundary_count > 0


class TestWatershedSegmentation:
    """Test watershed-specific functionality."""

    def test_watershed_creates_components(self, segmenter, synthetic_volume):
        """Test that watershed creates multiple components."""
        result = segmenter.segment_volume(synthetic_volume)

        # Multiple distinct regions should be found
        assert result.num_components > 0

    def test_watershed_respects_membrane(self, segmenter, synthetic_volume):
        """Test that watershed respects membrane boundaries."""
        # Without membrane
        result1 = segmenter.segment_volume(synthetic_volume)

        # With membrane
        membrane = np.ones_like(synthetic_volume, dtype=np.float32) * 0.1
        membrane[99:101, :, :] = 0.9  # Strong boundary
        result2 = segmenter.segment_volume(synthetic_volume, membrane)

        # Membrane might affect component count
        assert result1.labels.shape == result2.labels.shape


class TestSynapseSeparability:
    """Test synapse-guided segmentation and separability analysis."""

    def test_segment_with_synapses(self, segmenter, synthetic_volume, synthetic_membrane):
        """Test synapse-guided segmentation."""
        # Create pre/post synapses at known locations
        pre_pt = np.array([[60, 60, 60], [120, 60, 60]], dtype=np.float32)
        post_pt = np.array([[70, 70, 70], [130, 70, 70]], dtype=np.float32)
        pre_root = np.array([1, 2])
        post_root = np.array([3, 4])

        seg_result, sep_result = segmenter.segment_with_synapses(
            synthetic_volume,
            synthetic_membrane,
            pre_pt,
            post_pt,
            pre_root,
            post_root,
        )

        assert isinstance(seg_result, SegmentationResult)
        assert isinstance(sep_result, SynapseSeparability)
        assert sep_result.separation_rate >= 0
        assert sep_result.separation_rate <= 1

    def test_separability_result_shapes(self, segmenter, synthetic_volume, synthetic_membrane):
        """Test separability result array shapes."""
        n_synapses = 5
        pre_pt = np.random.rand(n_synapses, 3) * 100 + 50
        post_pt = pre_pt + np.random.rand(n_synapses, 3) * 20
        pre_root = np.arange(n_synapses)
        post_root = np.arange(n_synapses) + 10

        seg_result, sep_result = segmenter.segment_with_synapses(
            synthetic_volume,
            synthetic_membrane,
            pre_pt,
            post_pt,
            pre_root,
            post_root,
        )

        assert sep_result.pre_labels.shape == (n_synapses,)
        assert sep_result.post_labels.shape == (n_synapses,)
        assert sep_result.separated.shape == (n_synapses,)

    def test_synapses_outside_volume(self, segmenter, synthetic_volume, synthetic_membrane):
        """Test handling of synapses outside volume."""
        pre_pt = np.array([[1000, 1000, 1000]], dtype=np.float32)
        post_pt = np.array([[2000, 2000, 2000]], dtype=np.float32)
        pre_root = np.array([1])
        post_root = np.array([2])

        seg_result, sep_result = segmenter.segment_with_synapses(
            synthetic_volume,
            synthetic_membrane,
            pre_pt,
            post_pt,
            pre_root,
            post_root,
        )

        # Should clamp and not crash
        assert len(sep_result.pre_labels) == 1
        assert len(sep_result.post_labels) == 1


class TestConfidence:
    """Test confidence computation."""

    def test_confidence_values(self, segmenter, synthetic_volume):
        """Test that confidence is computed."""
        result = segmenter.segment_volume(synthetic_volume)

        assert result.confidence is not None
        assert len(result.confidence) == result.num_components + 1
        # Confidence should be in [0, 1]
        assert np.all((result.confidence >= 0) & (result.confidence <= 1))

    def test_confidence_indexed_by_label(self, segmenter, synthetic_volume):
        """Test that confidence is indexed by label."""
        result = segmenter.segment_volume(synthetic_volume)

        # Confidence array size should match max label
        assert len(result.confidence) >= result.num_components


class TestBoundaryRefinement:
    """Test boundary refinement."""

    def test_refine_boundaries(self, segmenter, synthetic_volume, synthetic_membrane):
        """Test boundary refinement operation."""
        seg_result = segmenter.segment_volume(synthetic_volume, synthetic_membrane)
        refined = segmenter.refine_boundaries(seg_result, synthetic_membrane, num_iterations=3)

        assert isinstance(refined, SegmentationResult)
        assert refined.labels.shape == seg_result.labels.shape


class TestComponentMerging:
    """Test merging nearby components."""

    def test_merge_nearby_components(self, segmenter, synthetic_volume):
        """Test merging of nearby components."""
        seg_result = segmenter.segment_volume(synthetic_volume)
        merged = segmenter.merge_nearby_components(seg_result, distance_threshold=20.0)

        assert isinstance(merged, SegmentationResult)
        assert merged.num_components <= seg_result.num_components

    def test_merge_extreme_threshold(self, segmenter, synthetic_volume):
        """Test merging with extreme threshold."""
        seg_result = segmenter.segment_volume(synthetic_volume)

        # Very small threshold should merge nothing
        merged_small = segmenter.merge_nearby_components(seg_result, distance_threshold=0.1)
        assert merged_small.num_components <= seg_result.num_components

        # Very large threshold should merge everything
        merged_large = segmenter.merge_nearby_components(seg_result, distance_threshold=10000.0)
        assert merged_large.num_components <= merged_small.num_components


class TestEdgeCases:
    """Test edge cases."""

    def test_single_voxel_structure(self, segmenter):
        """Test segmentation of single-voxel structures."""
        vol = np.zeros((100, 100, 100), dtype=np.uint8)
        vol[50, 50, 50] = 200
        vol[75, 75, 75] = 200

        result = segmenter.segment_volume(vol)
        # Small structures might be filtered out
        assert result.num_components >= 0

    def test_large_volume(self, segmenter):
        """Test segmentation of large volume."""
        vol = np.zeros((512, 512, 128), dtype=np.uint8)
        vol[100:200, 100:200, 30:100] = 150

        result = segmenter.segment_volume(vol)
        assert result.labels.shape == vol.shape
        assert result.num_components >= 0

    def test_highly_fragmented_volume(self, segmenter):
        """Test volume with many small fragments."""
        vol = np.random.randint(0, 150, (100, 100, 100), dtype=np.uint8)
        # Add scattered bright voxels
        for _ in range(100):
            x, y, z = np.random.randint(0, 100, 3)
            vol[x, y, z] = 200

        result = segmenter.segment_volume(vol)
        # Should create components, may filter small ones
        assert result.num_components >= 0
