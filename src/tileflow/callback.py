"""Enhanced callback system for TileFlow processing events.

This module provides a comprehensive callback framework for monitoring and tracking
various aspects of tile-based image processing, including progress, performance,
energy consumption, and memory usage.
"""

from abc import ABC
import time
import tracemalloc
from typing import Any

import numpy as np

from tileflow.core import ProcessedTile


class ProcessingStats:
    """Container for processing statistics and metrics."""

    def __init__(self):
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.total_tiles: int = 0
        self.processed_tiles: int = 0
        self.total_chunks: int = 0
        self.processed_chunks: int = 0
        self.input_shape: tuple | None = None
        self.output_shape: tuple | None = None
        self.tile_size: tuple | None = None
        self.overlap: tuple | None = None
        self.chunk_size: tuple | None = None

    @property
    def elapsed_time(self) -> float | None:
        """Total elapsed processing time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def tiles_per_second(self) -> float | None:
        """Processing rate in tiles per second."""
        if self.elapsed_time and self.processed_tiles > 0:
            return self.processed_tiles / self.elapsed_time
        return None


class TileFlowCallback(ABC):
    """Enhanced base class for TileFlow processing callbacks.
    
    Provides a comprehensive event system with lifecycle management,
    error handling, and rich context information. All methods are
    optional - subclasses only need to implement relevant events.
    """

    def on_processing_start(self, stats: ProcessingStats) -> None:
        """Called when processing begins.
        
        Parameters
        ----------
        stats : ProcessingStats
            Processing configuration and initial statistics
        """
        pass

    def on_processing_end(self, stats: ProcessingStats) -> None:
        """Called when processing completes successfully.
        
        Parameters
        ----------
        stats : ProcessingStats
            Final processing statistics and metrics
        """
        pass

    def on_processing_error(self, error: Exception, stats: ProcessingStats) -> None:
        """Called when processing encounters an error.
        
        Parameters
        ----------
        error : Exception
            The exception that occurred
        stats : ProcessingStats
            Current processing statistics
        """
        pass

    def on_chunk_start(self, chunk_index: int, total_chunks: int, chunk_shape: tuple) -> None:
        """Called when chunk processing begins.
        
        Parameters
        ----------
        chunk_index : int
            Index of current chunk (0-based)
        total_chunks : int
            Total number of chunks to process
        chunk_shape : tuple
            Shape of current chunk
        """
        pass

    def on_chunk_end(self, chunk_index: int, total_chunks: int, chunk_shape: tuple) -> None:
        """Called when chunk processing completes.
        
        Parameters
        ----------
        chunk_index : int
            Index of completed chunk (0-based)
        total_chunks : int
            Total number of chunks
        chunk_shape : tuple
            Shape of completed chunk
        """
        pass

    def on_tile_start(self, tile: ProcessedTile, tile_index: int, total_tiles: int) -> None:
        """Called when tile processing begins.
        
        Parameters
        ----------
        tile : ProcessedTile
            Tile specification being processed
        tile_index : int
            Index of current tile (0-based)
        total_tiles : int
            Total number of tiles to process
        """
        pass

    def on_tile_end(self, tile: ProcessedTile, tile_index: int, total_tiles: int) -> None:
        """Called when tile processing completes.
        
        Parameters
        ----------
        tile : ProcessedTile
            Completed processed tile
        tile_index : int
            Index of completed tile (0-based)
        total_tiles : int
            Total number of tiles
        """
        pass

    # Legacy methods for backward compatibility
    def on_tile_processed(self, tile: ProcessedTile, tile_index: int, total_tiles: int) -> None:
        """Legacy method - use on_tile_end instead."""
        self.on_tile_end(tile, tile_index, total_tiles)

    def on_chunk_processed(self, tile: ProcessedTile, chunk_index: int, total_chunks: int) -> None:
        """Legacy method - use on_chunk_end instead."""
        self.on_chunk_end(chunk_index, total_chunks, getattr(tile, "shape", ()))

    def on_processing_complete(self, tiles: list[ProcessedTile]) -> None:
        """Legacy method - use on_processing_end instead."""
        stats = ProcessingStats()
        stats.processed_tiles = len(tiles)
        self.on_processing_end(stats)


class ProgressCallback(TileFlowCallback):
    """Enhanced progress tracking with detailed statistics."""

    def __init__(self, verbose: bool = True, show_rate: bool = True):
        self.verbose = verbose
        self.show_rate = show_rate
        self._start_time: float | None = None

    def on_processing_start(self, stats: ProcessingStats) -> None:
        """Initialize progress tracking."""
        if self.verbose:
            print(f"🚀 Starting processing: {stats.total_tiles} tiles")
            if stats.input_shape:
                print(f"   Input shape: {stats.input_shape}")
            if stats.tile_size:
                print(f"   Tile size: {stats.tile_size}")
        self._start_time = time.time()

    def on_tile_end(self, tile: ProcessedTile, tile_index: int, total_tiles: int) -> None:
        """Display tile progress with rate information."""
        if not self.verbose:
            return

        progress = ((tile_index + 1) / total_tiles) * 100
        message = f"   Tile {tile_index + 1}/{total_tiles} ({progress:.1f}%)"

        if self.show_rate and self._start_time:
            elapsed = time.time() - self._start_time
            if elapsed > 0:
                rate = (tile_index + 1) / elapsed
                message += f" - {rate:.1f} tiles/sec"

        print(message)

    def on_chunk_end(self, chunk_index: int, total_chunks: int, chunk_shape: tuple) -> None:
        """Display chunk progress."""
        if self.verbose and total_chunks > 1:
            progress = ((chunk_index + 1) / total_chunks) * 100
            print(f"   Chunk {chunk_index + 1}/{total_chunks} ({progress:.1f}%)")

    def on_processing_end(self, stats: ProcessingStats) -> None:
        """Display completion summary."""
        if self.verbose:
            message = f"✅ Processing complete! {stats.processed_tiles} tiles"
            if stats.elapsed_time:
                message += f" in {stats.elapsed_time:.2f}s"
            if stats.tiles_per_second:
                message += f" ({stats.tiles_per_second:.1f} tiles/sec)"
            print(message)

    def on_processing_error(self, error: Exception, stats: ProcessingStats) -> None:
        """Display error information."""
        if self.verbose:
            print(f"❌ Processing failed after {stats.processed_tiles} tiles: {error}")


class MemoryTracker(TileFlowCallback):
    """Track memory usage during processing.
    
    Monitors peak memory usage, memory per tile, and provides
    detailed memory statistics.
    """

    def __init__(self, detailed: bool = False):
        self.detailed = detailed
        self._baseline_memory: int = 0
        self._peak_memory: int = 0
        self._memory_per_tile: list[int] = []
        self._tracking_started: bool = False

    def on_processing_start(self, stats: ProcessingStats) -> None:
        """Start memory tracking."""
        if not self._tracking_started:
            tracemalloc.start()
            self._tracking_started = True

        # Get baseline memory usage
        current, peak = tracemalloc.get_traced_memory()
        self._baseline_memory = current
        self._peak_memory = peak

        if self.detailed:
            print(f"🔍 Memory tracking started - Baseline: {self._format_bytes(current)}")

    def on_tile_end(self, tile: ProcessedTile, tile_index: int, total_tiles: int) -> None:
        """Track memory usage per tile."""
        if not self._tracking_started:
            return

        current, peak = tracemalloc.get_traced_memory()
        self._peak_memory = max(self._peak_memory, peak)
        self._memory_per_tile.append(current)

        if self.detailed:
            memory_delta = current - self._baseline_memory
            print(f"   Memory: {self._format_bytes(current)} "
                  f"(+{self._format_bytes(memory_delta)} from baseline)")

    def on_processing_end(self, stats: ProcessingStats) -> None:
        """Display final memory statistics."""
        if not self._tracking_started:
            return

        current, _ = tracemalloc.get_traced_memory()
        peak_delta = self._peak_memory - self._baseline_memory
        final_delta = current - self._baseline_memory

        print("📊 Memory Usage Summary:")
        print(f"   Peak memory: {self._format_bytes(self._peak_memory)} "
              f"(+{self._format_bytes(peak_delta)} from baseline)")
        print(f"   Final memory: {self._format_bytes(current)} "
              f"(+{self._format_bytes(final_delta)} from baseline)")

        if self._memory_per_tile:
            avg_memory = np.mean(self._memory_per_tile)
            print(f"   Average per tile: {self._format_bytes(avg_memory)}")

    def on_processing_error(self, error: Exception, stats: ProcessingStats) -> None:
        """Stop tracking on error."""
        if self._tracking_started:
            tracemalloc.stop()
            self._tracking_started = False

    def get_memory_stats(self) -> dict[str, Any]:
        """Get detailed memory statistics."""
        if not self._tracking_started:
            return {}

        current, _ = tracemalloc.get_traced_memory()
        return {
            "baseline_memory_bytes": self._baseline_memory,
            "peak_memory_bytes": self._peak_memory,
            "current_memory_bytes": current,
            "memory_per_tile_bytes": self._memory_per_tile.copy(),
            "peak_delta_bytes": self._peak_memory - self._baseline_memory,
            "average_per_tile_bytes": np.mean(self._memory_per_tile) if self._memory_per_tile else 0
        }

    @staticmethod
    def _format_bytes(bytes_value: int) -> str:
        """Format bytes in human readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_value < 1024:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024
        return f"{bytes_value:.1f} TB"


class CodeCarbonTracker(TileFlowCallback):
    """Track energy consumption during processing using CodeCarbon.
    
    Requires codecarbon to be installed: pip install codecarbon
    """

    def __init__(self,
                 project_name: str = "tileflow-processing",
                 output_dir: str = "./carbon_logs",
                 detailed: bool = False):
        self.project_name = project_name
        self.output_dir = output_dir
        self.detailed = detailed
        self._tracker = None
        self._emissions_data: dict[str, Any] = {}

        try:
            from codecarbon import EmissionsTracker
            self._tracker_class = EmissionsTracker
            self._available = True
        except ImportError:
            self._available = False
            if detailed:
                print("⚠️ CodeCarbon not available. Install with: pip install codecarbon")

    def on_processing_start(self, stats: ProcessingStats) -> None:
        """Start energy tracking."""
        if not self._available:
            return

        self._tracker = self._tracker_class(
            project_name=self.project_name,
            output_dir=self.output_dir,
            save_to_file=True,
            log_level="WARNING"  # Reduce noise
        )
        self._tracker.start()

        if self.detailed:
            print(f"🌱 Carbon tracking started - Project: {self.project_name}")

    def on_processing_end(self, stats: ProcessingStats) -> None:
        """Stop tracking and display energy statistics."""
        if not self._available or not self._tracker:
            return

        emissions = self._tracker.stop()

        # Store emissions data
        self._emissions_data = {
            "emissions_kg": emissions,
            "processing_time_s": stats.elapsed_time or 0,
            "total_tiles": stats.processed_tiles,
            "input_shape": stats.input_shape,
            "tile_size": stats.tile_size
        }

        print("🌍 Carbon Footprint Summary:")
        print(f"   CO₂ emissions: {emissions:.6f} kg")
        if stats.processed_tiles > 0:
            emissions_per_tile = emissions / stats.processed_tiles
            print(f"   Per tile: {emissions_per_tile:.8f} kg CO₂")

        if stats.input_shape and len(stats.input_shape) >= 2:
            total_pixels = np.prod(stats.input_shape[-2:])  # H * W
            emissions_per_mpixel = (emissions * 1000000) / total_pixels
            print(f"   Per megapixel: {emissions_per_mpixel:.6f} kg CO₂")

    def on_processing_error(self, error: Exception, stats: ProcessingStats) -> None:
        """Stop tracking on error."""
        if self._available and self._tracker:
            try:
                self._tracker.stop()
            except Exception:
                pass  # Ignore tracker errors during error handling

    def get_emissions_data(self) -> dict[str, Any]:
        """Get detailed emissions data."""
        return self._emissions_data.copy()


class CompositeCallback(TileFlowCallback):
    """Compose multiple callbacks into a single callback.
    
    Allows combining multiple monitoring callbacks (progress, memory, carbon)
    into a unified interface.
    """

    def __init__(self, callbacks: list[TileFlowCallback]):
        self.callbacks = callbacks
        # Cache whether we have active listeners for performance
        self._has_tile_listeners = any(
            hasattr(cb, 'on_tile_start') and callable(getattr(cb, 'on_tile_start'))
            for cb in callbacks
        )
        self._has_chunk_listeners = any(
            hasattr(cb, 'on_chunk_start') and callable(getattr(cb, 'on_chunk_start'))
            for cb in callbacks
        )

    def has_tile_listeners(self) -> bool:
        """Check if any callbacks are interested in tile events."""
        return self._has_tile_listeners

    def has_chunk_listeners(self) -> bool:
        """Check if any callbacks are interested in chunk events."""
        return self._has_chunk_listeners

    def _call_all(self, method_name: str, *args, **kwargs) -> None:
        """Call method on all callbacks, catching and logging errors."""
        for callback in self.callbacks:
            try:
                method = getattr(callback, method_name, None)
                if method and callable(method):
                    method(*args, **kwargs)
            except Exception as e:
                print(f"⚠️ Callback error in {callback.__class__.__name__}.{method_name}: {e}")

    def on_processing_start(self, stats: ProcessingStats) -> None:
        self._call_all("on_processing_start", stats)

    def on_processing_end(self, stats: ProcessingStats) -> None:
        self._call_all("on_processing_end", stats)

    def on_processing_error(self, error: Exception, stats: ProcessingStats) -> None:
        self._call_all("on_processing_error", error, stats)

    def on_chunk_start(self, chunk_index: int, total_chunks: int, chunk_shape: tuple) -> None:
        self._call_all("on_chunk_start", chunk_index, total_chunks, chunk_shape)

    def on_chunk_end(self, chunk_index: int, total_chunks: int, chunk_shape: tuple) -> None:
        self._call_all("on_chunk_end", chunk_index, total_chunks, chunk_shape)

    def on_tile_start(self, tile: ProcessedTile, tile_index: int, total_tiles: int) -> None:
        self._call_all("on_tile_start", tile, tile_index, total_tiles)

    def on_tile_end(self, tile: ProcessedTile, tile_index: int, total_tiles: int) -> None:
        self._call_all("on_tile_end", tile, tile_index, total_tiles)

    # Legacy compatibility
    def on_tile_processed(self, tile: ProcessedTile, tile_index: int, total_tiles: int) -> None:
        self._call_all("on_tile_processed", tile, tile_index, total_tiles)

    def on_chunk_processed(self, tile: ProcessedTile, chunk_index: int, total_chunks: int) -> None:
        self._call_all("on_chunk_processed", tile, chunk_index, total_chunks)

    def on_processing_complete(self, tiles: list[ProcessedTile]) -> None:
        self._call_all("on_processing_complete", tiles)


class MetricsCallback(TileFlowCallback):
    """Comprehensive metrics collection callback.
    
    Combines progress tracking, timing, and basic system metrics
    in a single convenient callback.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.stats = ProcessingStats()
        self._tile_times: list[float] = []
        self._chunk_times: list[float] = []

    def on_processing_start(self, stats: ProcessingStats) -> None:
        """Initialize metrics collection."""
        self.stats = stats
        self.stats.start_time = time.time()

        if self.verbose:
            print("📈 Metrics collection started")

    def on_tile_start(self, tile: ProcessedTile, tile_index: int, total_tiles: int) -> None:
        """Track tile start time."""
        self._tile_start_time = time.time()

    def on_tile_end(self, tile: ProcessedTile, tile_index: int, total_tiles: int) -> None:
        """Track tile completion and timing."""
        if hasattr(self, "_tile_start_time"):
            tile_duration = time.time() - self._tile_start_time
            self._tile_times.append(tile_duration)

    def on_processing_end(self, stats: ProcessingStats) -> None:
        """Display comprehensive metrics summary."""
        stats.end_time = time.time()

        if self.verbose:
            print("📊 Processing Metrics:")
            print(f"   Total time: {stats.elapsed_time:.2f}s")
            print(f"   Tiles processed: {stats.processed_tiles}")
            if stats.tiles_per_second:
                print(f"   Processing rate: {stats.tiles_per_second:.1f} tiles/sec")

            if self._tile_times:
                avg_tile_time = np.mean(self._tile_times)
                min_tile_time = np.min(self._tile_times)
                max_tile_time = np.max(self._tile_times)
                print(f"   Tile timing - avg: {avg_tile_time:.3f}s, "
                      f"min: {min_tile_time:.3f}s, max: {max_tile_time:.3f}s")

    def get_detailed_metrics(self) -> dict[str, Any]:
        """Get comprehensive metrics data."""
        return {
            "total_time_s": self.stats.elapsed_time,
            "tiles_processed": self.stats.processed_tiles,
            "tiles_per_second": self.stats.tiles_per_second,
            "tile_times_s": self._tile_times.copy(),
            "chunk_times_s": self._chunk_times.copy(),
            "average_tile_time_s": np.mean(self._tile_times) if self._tile_times else 0,
            "input_shape": self.stats.input_shape,
            "output_shape": self.stats.output_shape
        }
