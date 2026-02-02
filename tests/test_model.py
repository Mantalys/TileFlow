"""Tests for main TileFlow model/processor."""

import numpy as np
import pytest

from tileflow.backends import NumpyStreamable
from tileflow.callback import TileFlowCallback
from tileflow.model import TileFlow


class TestTileFlow:
    """Test main TileFlow processor."""

    def test_initialization(self):
        """Test processor initialization."""
        processor = TileFlow(tile_size=(64, 64), overlap=(8, 8))

        assert processor.tile_size == (64, 64)
        assert processor.overlap == (8, 8)
        assert processor.chunk_size is None
        assert not processor._configured

    def test_initialization_with_chunks(self):
        """Test initialization with chunking."""
        processor = TileFlow(
            tile_size=(32, 32), overlap=(4, 4), chunk_size=(128, 128), chunk_overlap=(16, 16)
        )

        assert processor.chunk_size == (128, 128)
        assert processor.chunk_overlap == (16, 16)

    def test_validation_errors(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="must be positive"):
            TileFlow(tile_size=(0, 64))

        with pytest.raises(ValueError, match="must be non-negative"):
            TileFlow(tile_size=(64, 64), overlap=(-1, 4))

        with pytest.raises(ValueError, match="must be a tuple"):
            TileFlow(tile_size=(64,))  # Wrong length

        with pytest.raises(ValueError, match="must contain integers"):
            TileFlow(tile_size=(64.5, 64))

    def test_overlap_validation(self):
        """Test overlap validation against tile size."""
        with pytest.raises(ValueError, match="less than half"):
            TileFlow(tile_size=(64, 64), overlap=(40, 20))  # overlap too large

    def test_configuration(self):
        """Test processor configuration."""
        processor = TileFlow(tile_size=(32, 32))

        def simple_func(x):
            return x * 2

        processor.configure(function=simple_func)
        assert processor._configured
        assert processor._processor == simple_func

    def test_configuration_errors(self):
        """Test configuration validation."""
        processor = TileFlow(tile_size=(32, 32))

        with pytest.raises(TypeError, match="must be callable"):
            processor.configure(function="not_callable")

        with pytest.raises(TypeError, match="must be callable"):
            processor.configure(function=lambda x: x, chunk_function="not_callable")

    def test_process_requires_configuration(self):
        """Test that processing requires configuration."""
        processor = TileFlow(tile_size=(32, 32))
        image = np.zeros((64, 64))

        with pytest.raises(RuntimeError, match="must be configured"):
            processor.run(image)

    def test_simple_processing(self):
        """Test basic image processing."""
        processor = TileFlow(tile_size=(32, 32), overlap=(4, 4))
        processor.configure(function=lambda x: x * 2)

        image = np.ones((64, 64))
        result = processor.run(image)

        assert result.shape == (64, 64)
        assert np.allclose(result, 2.0)

    def test_chunked_processing(self):
        """Test chunked processing for large images."""
        processor = TileFlow(
            tile_size=(16, 16), overlap=(2, 2), chunk_size=(32, 32), chunk_overlap=(4, 4)
        )
        processor.configure(function=lambda x: x + 1)

        image = np.zeros((64, 64))
        result = processor.run(image)

        # Chunked processing returns None to save memory for massive images
        assert result is None

    def test_simple_processing_multiply(self):
        """Test simple processing with multiplication."""
        processor = TileFlow(tile_size=(32, 32))
        processor.configure(function=lambda x: x * 3)

        image = np.ones((64, 64))
        result = processor.run(image)

        assert result.shape == (64, 64)
        assert np.allclose(result, 3.0)

    def test_processor_name(self):
        """Test processor naming."""
        processor = TileFlow(tile_size=(32, 32), name="TestProcessor")
        assert processor.name == "TestProcessor"

        # Test default name
        processor2 = TileFlow(tile_size=(32, 32))
        assert processor2.name == "TileFlow"

    def test_basic_functionality(self):
        """Test basic processor functionality."""
        processor = TileFlow(tile_size=(32, 32))
        processor.configure(function=lambda x: x + 10)

        image = np.zeros((64, 64))
        result = processor.run(image)
        assert np.allclose(result, 10.0)

    def test_processing_no_callbacks(self):
        """Test processing when no callbacks are used."""
        processor = TileFlow(tile_size=(32, 32))
        processor.configure(function=lambda x: x * 2)

        image = np.ones((64, 64))
        result = processor.run(image)  # No callbacks provided

        assert result.shape == (64, 64)
        assert np.allclose(result, 2.0)

    def test_dtype_handling(self):
        """Test processing with different data types."""
        processor = TileFlow(tile_size=(32, 32))
        processor.configure(function=lambda x: x.astype(np.float32) * 1.5)

        # Create a larger image to trigger optimizations
        image = np.ones((128, 128), dtype=np.float64)  # Use float64 to test optimization
        result = processor.run(image)

        assert result.shape == (128, 128)
        assert result.dtype == np.float32  # Should be optimized to float32
        assert np.allclose(result, 1.5)

    def test_with_callbacks(self):
        """Test processing with callbacks."""
        processor = TileFlow(tile_size=(32, 32))
        processor.configure(function=lambda x: x)

        callback_calls = []

        class TestCallback(TileFlowCallback):
            def on_tile_end(self, tile, tile_index, total_tiles):
                callback_calls.append(f"tile_{tile_index}")

            def on_chunk_end(self, chunk_index, total_chunks, chunk_shape):
                callback_calls.append(f"chunk_{chunk_index}")

            def on_processing_end(self, stats):
                callback_calls.append("complete")

        image = np.zeros((64, 64))
        processor.run(image, callbacks=[TestCallback()])

        assert "complete" in callback_calls
        assert len([c for c in callback_calls if c.startswith("tile_")]) == 4  # 2x2 grid

    def test_return_tiles(self):
        """Test returning individual tiles."""
        processor = TileFlow(tile_size=(32, 32))
        processor.configure(function=lambda x: x)

        image = np.zeros((64, 64))
        regions = processor.run(image, return_tiles=True)

        assert isinstance(regions, list)
        assert len(regions) == 4  # 2x2 grid
        assert all(hasattr(r, "tile_spec") for r in regions)

    def test_tile_size_validation_against_image(self):
        """Test tile size validation against actual image."""
        processor = TileFlow(tile_size=(100, 100))
        processor.configure(function=lambda x: x)

        small_image = np.zeros((50, 50))
        with pytest.raises(ValueError, match="cannot exceed image"):
            processor.run(small_image)

    def test_summary_output(self, capsys):
        """Test summary output."""
        processor = TileFlow(tile_size=(64, 64), overlap=(8, 8), name="TestProcessor")
        processor.summary()

        captured = capsys.readouterr()
        assert "TestProcessor" in captured.out
        assert "64, 64" in captured.out
        assert "8, 8" in captured.out
        assert "Configured:     False" in captured.out

    def test_streamable_input(self):
        """Test processing with streamable input."""
        processor = TileFlow(tile_size=(32, 32))
        processor.configure(function=lambda x: x * 3)

        array = np.ones((64, 64))
        streamable = NumpyStreamable(array)
        result = processor.run(streamable)

        assert result.shape == (64, 64)
        assert np.allclose(result, 3.0)
