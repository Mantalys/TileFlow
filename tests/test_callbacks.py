"""Tests for enhanced callback system."""

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from tileflow.callback import (
    CodeCarbonTracker,
    CompositeCallback,
    MemoryTracker,
    MetricsCallback,
    ProcessingStats,
    ProgressCallback,
    TileFlowCallback,
)
from tileflow.core import BBox, BoundaryEdges, ProcessedTile, TileGeometry, TilePosition, TileSpec
from tileflow.model import TileFlow


class TestProcessingStats:
    """Test ProcessingStats container."""

    def test_initialization(self):
        """Test stats initialization."""
        stats = ProcessingStats()
        assert stats.start_time is None
        assert stats.end_time is None
        assert stats.total_tiles == 0
        assert stats.processed_tiles == 0

    def test_elapsed_time_calculation(self):
        """Test elapsed time calculation."""
        stats = ProcessingStats()
        stats.start_time = 1000.0
        stats.end_time = 1005.5
        assert stats.elapsed_time == 5.5

    def test_tiles_per_second_calculation(self):
        """Test processing rate calculation."""
        stats = ProcessingStats()
        stats.start_time = 1000.0
        stats.end_time = 1010.0  # 10 seconds
        stats.processed_tiles = 50
        assert stats.tiles_per_second == 5.0


class TestTileFlowCallback:
    """Test enhanced base callback class."""

    def test_all_methods_optional(self):
        """Test that all callback methods are optional."""
        callback = TileFlowCallback()
        stats = ProcessingStats()

        # All these should not raise errors
        callback.on_processing_start(stats)
        callback.on_processing_end(stats)
        callback.on_processing_error(Exception("test"), stats)
        callback.on_chunk_start(0, 1, (64, 64))
        callback.on_chunk_end(0, 1, (64, 64))

        # Create dummy tile for testing
        bbox = BBox(0, 0, 32, 32)
        geometry = TileGeometry(core=bbox, halo=bbox)
        position = TilePosition((0, 0), BoundaryEdges(True, True, True, True))
        spec = TileSpec(geometry=geometry, position=position)
        tile = ProcessedTile(spec, np.zeros((32, 32)))

        callback.on_tile_start(tile, 0, 1)
        callback.on_tile_end(tile, 0, 1)

    def test_legacy_compatibility(self):
        """Test backward compatibility with old callback methods."""
        callback = TileFlowCallback()

        # Create dummy tile
        bbox = BBox(0, 0, 32, 32)
        geometry = TileGeometry(core=bbox, halo=bbox)
        position = TilePosition((0, 0), BoundaryEdges(True, True, True, True))
        spec = TileSpec(geometry=geometry, position=position)
        tile = ProcessedTile(spec, np.zeros((32, 32)))

        # Legacy methods should work without errors
        callback.on_tile_processed(tile, 0, 1)
        callback.on_chunk_processed(tile, 0, 1)
        callback.on_processing_complete([tile])


class TestProgressCallback:
    """Test enhanced progress callback."""

    def test_initialization(self):
        """Test progress callback initialization."""
        callback = ProgressCallback(verbose=False, show_rate=False)
        assert not callback.verbose
        assert not callback.show_rate

    def test_progress_tracking(self, capsys):
        """Test progress display."""
        callback = ProgressCallback(verbose=True, show_rate=False)
        stats = ProcessingStats()
        stats.total_tiles = 5

        callback.on_processing_start(stats)
        captured = capsys.readouterr()
        assert "Starting processing: 5 tiles" in captured.out

    def test_rate_calculation(self, capsys):
        """Test processing rate display."""
        callback = ProgressCallback(verbose=True, show_rate=True)

        # Create dummy tile
        bbox = BBox(0, 0, 32, 32)
        geometry = TileGeometry(core=bbox, halo=bbox)
        position = TilePosition((0, 0), BoundaryEdges(True, True, True, True))
        spec = TileSpec(geometry=geometry, position=position)
        tile = ProcessedTile(spec, np.zeros((32, 32)))

        # Start timing
        stats = ProcessingStats()
        callback.on_processing_start(stats)

        # Simulate some processing time
        time.sleep(0.01)
        callback.on_tile_end(tile, 0, 10)

        captured = capsys.readouterr()
        assert "tiles/sec" in captured.out

    def test_silent_mode(self, capsys):
        """Test silent mode produces no output."""
        callback = ProgressCallback(verbose=False)
        stats = ProcessingStats()

        callback.on_processing_start(stats)
        callback.on_processing_end(stats)

        captured = capsys.readouterr()
        assert captured.out == ""


class TestMemoryTracker:
    """Test memory tracking callback."""

    def test_initialization(self):
        """Test memory tracker initialization."""
        tracker = MemoryTracker(detailed=True)
        assert tracker.detailed
        assert not tracker._tracking_started

    def test_memory_tracking_lifecycle(self):
        """Test complete memory tracking lifecycle."""
        tracker = MemoryTracker(detailed=False)
        stats = ProcessingStats()

        # Start tracking
        tracker.on_processing_start(stats)
        assert tracker._tracking_started

        # Get initial stats
        initial_stats = tracker.get_memory_stats()
        assert "baseline_memory_bytes" in initial_stats

        # Simulate tile processing
        bbox = BBox(0, 0, 32, 32)
        geometry = TileGeometry(core=bbox, halo=bbox)
        position = TilePosition((0, 0), BoundaryEdges(True, True, True, True))
        spec = TileSpec(geometry=geometry, position=position)
        tile = ProcessedTile(spec, np.zeros((32, 32)))

        tracker.on_tile_end(tile, 0, 1)

        # End tracking
        tracker.on_processing_end(stats)

        final_stats = tracker.get_memory_stats()
        assert len(final_stats["memory_per_tile_bytes"]) == 1

    def test_bytes_formatting(self):
        """Test human-readable bytes formatting."""
        assert MemoryTracker._format_bytes(500) == "500.0 B"
        assert MemoryTracker._format_bytes(1500) == "1.5 KB"
        assert MemoryTracker._format_bytes(1500000) == "1.4 MB"

    def test_error_handling(self):
        """Test error handling stops tracking."""
        tracker = MemoryTracker()
        stats = ProcessingStats()

        tracker.on_processing_start(stats)
        assert tracker._tracking_started

        tracker.on_processing_error(Exception("test"), stats)
        assert not tracker._tracking_started


class TestCodeCarbonTracker:
    """Test energy consumption tracking."""

    def test_initialization_without_codecarbon(self):
        """Test initialization when codecarbon is not available."""
        with patch("builtins.__import__", side_effect=ImportError):
            tracker = CodeCarbonTracker(detailed=True)
            assert not tracker._available

    def test_initialization_with_codecarbon(self):
        """Test initialization when codecarbon is available."""
        tracker = CodeCarbonTracker(project_name="test", output_dir="/tmp")
        assert tracker.project_name == "test"
        assert tracker.output_dir == "/tmp"

    def test_tracking_without_codecarbon_installed(self):
        """Test graceful handling when codecarbon is not installed."""
        tracker = CodeCarbonTracker()
        tracker._available = False

        stats = ProcessingStats()
        # Should not raise errors
        tracker.on_processing_start(stats)
        tracker.on_processing_end(stats)
        tracker.on_processing_error(Exception("test"), stats)

    def test_emissions_data_structure(self):
        """Test emissions data structure."""
        tracker = CodeCarbonTracker()
        data = tracker.get_emissions_data()
        assert isinstance(data, dict)


class TestCompositeCallback:
    """Test composite callback functionality."""

    def test_initialization(self):
        """Test composite callback creation."""
        cb1 = ProgressCallback(verbose=False)
        cb2 = MemoryTracker()
        composite = CompositeCallback([cb1, cb2])

        assert len(composite.callbacks) == 2

    def test_method_delegation(self):
        """Test that methods are called on all callbacks."""
        mock1 = Mock(spec=TileFlowCallback)
        mock2 = Mock(spec=TileFlowCallback)
        composite = CompositeCallback([mock1, mock2])

        stats = ProcessingStats()
        composite.on_processing_start(stats)

        mock1.on_processing_start.assert_called_once_with(stats)
        mock2.on_processing_start.assert_called_once_with(stats)

    def test_error_handling_in_callbacks(self, capsys):
        """Test error handling when individual callbacks fail."""
        failing_callback = Mock(spec=TileFlowCallback)
        failing_callback.on_processing_start.side_effect = Exception("Callback error")

        working_callback = Mock(spec=TileFlowCallback)
        composite = CompositeCallback([failing_callback, working_callback])

        stats = ProcessingStats()
        composite.on_processing_start(stats)

        # Working callback should still be called
        working_callback.on_processing_start.assert_called_once()

        # Error should be logged
        captured = capsys.readouterr()
        assert "Callback error" in captured.out

    def test_legacy_method_delegation(self):
        """Test legacy method compatibility."""
        mock = Mock(spec=TileFlowCallback)
        composite = CompositeCallback([mock])

        # Create dummy tile
        bbox = BBox(0, 0, 32, 32)
        geometry = TileGeometry(core=bbox, halo=bbox)
        position = TilePosition((0, 0), BoundaryEdges(True, True, True, True))
        spec = TileSpec(geometry=geometry, position=position)
        tile = ProcessedTile(spec, np.zeros((32, 32)))

        composite.on_tile_processed(tile, 0, 1)
        mock.on_tile_processed.assert_called_once()


class TestMetricsCallback:
    """Test comprehensive metrics collection."""

    def test_initialization(self):
        """Test metrics callback initialization."""
        callback = MetricsCallback(verbose=False)
        assert not callback.verbose
        assert isinstance(callback.stats, ProcessingStats)

    def test_timing_collection(self):
        """Test tile timing collection."""
        callback = MetricsCallback(verbose=False)

        # Create dummy tile
        bbox = BBox(0, 0, 32, 32)
        geometry = TileGeometry(core=bbox, halo=bbox)
        position = TilePosition((0, 0), BoundaryEdges(True, True, True, True))
        spec = TileSpec(geometry=geometry, position=position)
        tile = ProcessedTile(spec, np.zeros((32, 32)))

        callback.on_tile_start(tile, 0, 1)
        time.sleep(0.01)  # Simulate processing
        callback.on_tile_end(tile, 0, 1)

        assert len(callback._tile_times) == 1
        assert callback._tile_times[0] > 0

    def test_metrics_data_export(self):
        """Test detailed metrics data export."""
        callback = MetricsCallback()
        stats = ProcessingStats()
        stats.start_time = 1000.0
        stats.end_time = 1005.0
        stats.processed_tiles = 10

        callback.stats = stats
        callback._tile_times = [0.1, 0.2, 0.15]

        metrics = callback.get_detailed_metrics()
        assert metrics["total_time_s"] == 5.0
        assert metrics["tiles_processed"] == 10
        assert metrics["tiles_per_second"] == 2.0
        assert abs(metrics["average_tile_time_s"] - 0.15) < 1e-10


class TestCallbackIntegration:
    """Test callback integration with TileFlow."""

    def test_single_callback_integration(self):
        """Test TileFlow with single callback."""
        processor = TileFlow(tile_size=(32, 32))
        processor.configure(function=lambda x: x * 2)

        progress_callback = ProgressCallback(verbose=False)
        image = np.ones((64, 64))

        result = processor.run(image, callbacks=[progress_callback])
        assert np.allclose(result, 2.0)

    def test_multiple_callbacks_integration(self):
        """Test TileFlow with multiple callbacks."""
        processor = TileFlow(tile_size=(32, 32))
        processor.configure(function=lambda x: x + 1)

        progress = ProgressCallback(verbose=False)
        memory = MemoryTracker(detailed=False)
        metrics = MetricsCallback(verbose=False)

        image = np.zeros((64, 64))
        result = processor.run(image, callbacks=[progress, memory, metrics])

        assert np.allclose(result, 1.0)
        assert metrics.stats.processed_tiles == 4  # 2x2 grid

    def test_composite_callback_convenience(self):
        """Test using CompositeCallback directly."""
        processor = TileFlow(tile_size=(32, 32))
        processor.configure(function=lambda x: x)

        callbacks = [ProgressCallback(verbose=False), MemoryTracker()]
        composite = CompositeCallback(callbacks)

        image = np.ones((64, 64))
        result = processor.run(image, callbacks=[composite])
        assert result.shape == (64, 64)

    def test_error_propagation_with_callbacks(self):
        """Test error handling with callbacks active."""
        processor = TileFlow(tile_size=(32, 32))

        def failing_function(x):
            raise ValueError("Processing failed")

        processor.configure(function=failing_function)

        progress = ProgressCallback(verbose=False)
        image = np.ones((64, 64))

        with pytest.raises(ValueError, match="Processing failed"):
            processor.run(image, callbacks=[progress])

    def test_chunked_processing_with_callbacks(self):
        """Test chunked processing with callbacks."""
        processor = TileFlow(
            tile_size=(16, 16), overlap=(2, 2), chunk_size=(32, 32), chunk_overlap=(4, 4)
        )
        processor.configure(function=lambda x: x * 3)

        metrics = MetricsCallback(verbose=False)
        image = np.ones((64, 64))

        result = processor.run(image, callbacks=[metrics])
        # Chunked processing returns None to save memory for massive images
        assert result is None
        assert metrics.stats.processed_chunks > 0

    def test_return_tiles_with_callbacks(self):
        """Test tile return with callbacks."""
        processor = TileFlow(tile_size=(32, 32))
        processor.configure(function=lambda x: x)

        progress = ProgressCallback(verbose=False)
        image = np.ones((64, 64))

        tiles = processor.run(image, callbacks=[progress], return_tiles=True)
        assert isinstance(tiles, list)
        assert len(tiles) == 4


class TestCodeCarbonTrackerMocked:
    """Test CodeCarbon tracker functionality without external dependencies."""

    def test_energy_tracking_without_codecarbon(self, capsys):
        """Test energy tracking when codecarbon is not available."""
        tracker = CodeCarbonTracker(detailed=True)
        # Force unavailable for testing
        tracker._available = False

        stats = ProcessingStats()
        stats.processed_tiles = 10
        stats.input_shape = (1000, 1000)

        # These should work without errors when unavailable
        tracker.on_processing_start(stats)
        tracker.on_processing_end(stats)
        tracker.on_processing_error(Exception("test"), stats)

        # No emissions should be tracked
        data = tracker.get_emissions_data()
        assert data == {}

    def test_manual_emissions_simulation(self, capsys):
        """Test emissions display with manual data."""
        tracker = CodeCarbonTracker(detailed=True)

        # Simulate availability and manual emissions data
        tracker._available = True
        tracker._emissions_data = {
            "emissions_kg": 0.001234,
            "processing_time_s": 5.0,
            "total_tiles": 10,
            "input_shape": (1000, 1000),
            "tile_size": (128, 128),
        }

        stats = ProcessingStats()
        stats.processed_tiles = 10
        stats.input_shape = (1000, 1000)

        # Manually trigger display (simulate _tracker.stop() return)
        tracker._tracker = Mock()
        tracker._tracker.stop = Mock(return_value=0.001234)
        tracker.on_processing_end(stats)

        captured = capsys.readouterr()
        assert "Carbon Footprint Summary" in captured.out
        assert "CO₂ emissions" in captured.out


class TestCallbackFactory:
    """Test convenient callback factory patterns."""

    def test_create_monitoring_suite(self):
        """Test creating a complete monitoring suite."""

        def create_monitoring_callbacks(verbose=True):
            return [
                ProgressCallback(verbose=verbose),
                MemoryTracker(detailed=verbose),
                MetricsCallback(verbose=verbose),
                CodeCarbonTracker(detailed=verbose),
            ]

        callbacks = create_monitoring_callbacks(verbose=False)
        assert len(callbacks) == 4

        # Test they can be used together
        processor = TileFlow(tile_size=(32, 32))
        processor.configure(function=lambda x: x)

        image = np.ones((64, 64))
        result = processor.run(image, callbacks=callbacks)
        assert result.shape == (64, 64)

    def test_create_performance_callbacks(self):
        """Test creating performance-focused callbacks."""

        def create_performance_suite():
            return [MetricsCallback(verbose=True), MemoryTracker(detailed=True)]

        callbacks = create_performance_suite()
        assert len(callbacks) == 2
        assert isinstance(callbacks[0], MetricsCallback)
        assert isinstance(callbacks[1], MemoryTracker)
