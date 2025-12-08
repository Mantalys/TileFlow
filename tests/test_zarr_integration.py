"""Tests for Zarr integration."""

import numpy as np
import pytest

# Try to import zarr, skip tests if not available
try:
    import zarr

    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False

from tileflow.backends import ZarrStreamable, as_streamable
from tileflow.model import TileFlow


@pytest.mark.skipif(not ZARR_AVAILABLE, reason="zarr not available")
class TestZarrStreamable:
    """Test ZarrStreamable backend."""

    def test_initialization(self):
        """Test ZarrStreamable initialization."""
        # Create in-memory zarr array
        arr = zarr.zeros((100, 200), dtype=np.float32)
        streamable = ZarrStreamable(arr)

        assert streamable.shape == (100, 200)
        assert streamable.dtype == np.float32

    def test_initialization_errors(self):
        """Test initialization error handling."""
        with pytest.raises(TypeError):
            ZarrStreamable("not an array")

        # 1D array should fail
        arr_1d = zarr.zeros(100)
        with pytest.raises(ValueError, match="at least 2D"):
            ZarrStreamable(arr_1d)

    def test_getitem(self):
        """Test data extraction."""
        # Create test data
        data = np.arange(100 * 200).reshape(100, 200).astype(np.float32)
        arr = zarr.array(data)
        streamable = ZarrStreamable(arr)

        # Extract a region
        result = streamable[10:20, 50:60]
        expected = data[10:20, 50:60]
        np.testing.assert_array_equal(result, expected)

    def test_setitem(self):
        """Test data modification."""
        arr = zarr.zeros((100, 200), dtype=np.float32)
        streamable = ZarrStreamable(arr)

        # Set a region
        test_data = np.ones((10, 10)) * 42
        streamable[10:20, 50:60] = test_data

        # Verify it was set
        result = streamable[10:20, 50:60]
        np.testing.assert_array_equal(result, test_data)

    def test_create_output(self):
        """Test output creation."""
        arr = zarr.zeros((100, 200), dtype=np.float32)
        streamable = ZarrStreamable(arr)

        output = streamable.create_output((50, 100), dtype=np.uint8)
        assert isinstance(output, ZarrStreamable)
        assert output.shape == (50, 100)
        assert output.dtype == np.uint8

    def test_array_access(self):
        """Test underlying array access."""
        arr = zarr.zeros((100, 200))
        streamable = ZarrStreamable(arr)
        assert streamable.array is arr


@pytest.mark.skipif(not ZARR_AVAILABLE, reason="zarr not available")
class TestZarrIntegration:
    """Test TileFlow integration with Zarr."""

    def test_as_streamable_conversion(self):
        """Test zarr array conversion."""
        arr = zarr.zeros((100, 200), dtype=np.float32)
        streamable = as_streamable(arr)
        assert isinstance(streamable, ZarrStreamable)
        assert streamable.shape == (100, 200)

    def test_tileflow_with_zarr(self):
        """Test TileFlow processing with zarr arrays."""
        # Create test zarr array
        data = np.random.rand(100, 100).astype(np.float32)
        zarr_array = zarr.array(data)

        # Simple doubling function
        def double_func(x):
            return x * 2

        # Process with TileFlow
        processor = TileFlow(tile_size=(32, 32), overlap=(4, 4))
        processor.configure(function=double_func)
        result = processor.run(zarr_array)

        # Verify result
        expected = data * 2
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_zarr_chunked_processing(self):
        """Test chunked processing with zarr."""
        # Create larger zarr array
        data = np.random.rand(200, 200).astype(np.float32)
        zarr_array = zarr.array(data, chunks=(64, 64))

        def add_one(x):
            return x + 1

        # Process with chunking
        processor = TileFlow(
            tile_size=(32, 32), overlap=(4, 4), chunk_size=(80, 80), chunk_overlap=(8, 8)
        )
        processor.configure(function=add_one)
        result = processor.run(zarr_array)

        # Chunked processing returns None to save memory for massive images
        assert result is None


class TestMultiChannelSupport:
    """Test support for multi-channel images."""

    def test_3d_numpy_processing(self):
        """Test processing 3D (C, H, W) numpy arrays."""
        # Create 3-channel test image
        data = np.random.rand(3, 64, 64).astype(np.float32)

        def channel_mean(x):
            """Average across channels."""
            if x.ndim == 3:
                return np.mean(x, axis=0, keepdims=True)
            return x

        processor = TileFlow(tile_size=(32, 32), overlap=(4, 4))
        processor.configure(function=channel_mean)
        result = processor.run(data)

        assert result.shape == (1, 64, 64)
        expected = np.mean(data, axis=0, keepdims=True)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    @pytest.mark.skipif(not ZARR_AVAILABLE, reason="zarr not available")
    def test_3d_zarr_processing(self):
        """Test processing 3D zarr arrays."""
        # Create 3-channel zarr array
        data = np.random.rand(3, 64, 64).astype(np.float32)
        zarr_array = zarr.array(data)

        def select_channel(x):
            """Select first channel only."""
            if x.ndim == 3:
                return x[0:1]  # Keep channel dimension
            return x

        processor = TileFlow(tile_size=(32, 32), overlap=(4, 4))
        processor.configure(function=select_channel)
        result = processor.run(zarr_array)

        assert result.shape == (1, 64, 64)
        expected = data[0:1]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_channel_processing_function(self):
        """Test function that processes specific channels."""
        # Create RGB image
        rgb_data = np.random.rand(3, 100, 100).astype(np.float32)

        def rgb_to_grayscale(x):
            """Convert RGB to grayscale using standard weights."""
            if x.ndim == 3 and x.shape[0] == 3:
                # Standard RGB to grayscale conversion
                weights = np.array([0.299, 0.587, 0.114]).reshape(3, 1, 1)
                gray = np.sum(x * weights, axis=0, keepdims=True)
                return gray
            return x

        processor = TileFlow(tile_size=(32, 32), overlap=(4, 4))
        processor.configure(function=rgb_to_grayscale)
        result = processor.run(rgb_data)

        assert result.shape == (1, 100, 100)

        # Verify grayscale conversion
        weights = np.array([0.299, 0.587, 0.114]).reshape(3, 1, 1)
        expected = np.sum(rgb_data * weights, axis=0, keepdims=True)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
