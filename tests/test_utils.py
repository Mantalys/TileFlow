"""Tests for utility functions."""

import warnings

import numpy as np
import pytest

from tileflow.utils import estimate_memory_usage, validate_overlap, validate_tile_size


class TestValidation:
    """Test validation functions."""

    def test_validate_tile_size_valid(self):
        """Test valid tile size passes."""
        # Should not raise
        validate_tile_size((64, 64), (256, 256))
        validate_tile_size((128, 128), (128, 128))  # Equal size

    def test_validate_tile_size_too_large(self):
        """Test tile size larger than image."""
        with pytest.raises(ValueError, match="cannot exceed image height"):
            validate_tile_size((300, 64), (256, 256))

        with pytest.raises(ValueError, match="cannot exceed image width"):
            validate_tile_size((64, 300), (256, 256))

    def test_validate_tile_size_warnings(self):
        """Test warnings for very small tiles."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_tile_size((16, 16), (256, 256))

            assert len(w) == 1
            assert "may impact performance" in str(w[0].message)

    def test_validate_overlap_valid(self):
        """Test valid overlap passes."""
        validate_overlap((8, 8), (64, 64))
        validate_overlap((0, 0), (64, 64))  # Zero overlap

    def test_validate_overlap_too_large(self):
        """Test overlap validation."""
        with pytest.raises(ValueError, match="less than half tile height"):
            validate_overlap((40, 8), (64, 64))

        with pytest.raises(ValueError, match="less than half tile width"):
            validate_overlap((8, 40), (64, 64))

    def test_validate_overlap_negative(self):
        """Test negative overlap rejection."""
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_overlap((-1, 8), (64, 64))


class TestMemoryEstimation:
    """Test memory usage estimation."""

    def test_basic_estimation(self):
        """Test basic memory estimation."""
        result = estimate_memory_usage(
            image_shape=(1024, 1024), tile_size=(128, 128), overlap=(0, 0)
        )

        assert "original_image_mb" in result
        assert "peak_memory_mb" in result
        assert "total_tiles" in result
        assert "processing_mode" in result

        assert result["processing_mode"] == "direct"
        assert result["total_tiles"] == 64  # 8x8 grid
        assert result["original_image_mb"] > 0

    def test_chunked_estimation(self):
        """Test memory estimation with chunking."""
        result = estimate_memory_usage(
            image_shape=(2048, 2048), tile_size=(64, 64), overlap=(4, 4), chunk_size=(512, 512)
        )

        assert result["processing_mode"] == "chunked"
        assert (
            result["peak_memory_mb"] < result["original_image_mb"] * 2
        )  # Should be more efficient

    def test_different_dtypes(self):
        """Test estimation with different data types."""
        result_float32 = estimate_memory_usage(
            image_shape=(1024, 1024), tile_size=(128, 128), overlap=(0, 0), dtype=np.float32
        )

        result_float64 = estimate_memory_usage(
            image_shape=(1024, 1024), tile_size=(128, 128), overlap=(0, 0), dtype=np.float64
        )

        # float64 should use twice as much memory
        assert result_float64["original_image_mb"] == result_float32["original_image_mb"] * 2
