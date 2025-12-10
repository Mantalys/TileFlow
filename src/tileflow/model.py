"""Tile-based image processing engine.

This module implements TileFlow's main processing pipeline:
- TileFlow: Main processor class with configure/run interface
- Direct tiling: Process images tile by tile
- Hierarchical chunking: Handle massive images through chunk → tile processing
- Multi-dimensional support: Handle CHW, CHWD, and arbitrary array shapes
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from tileflow.backends import Streamable, as_streamable
from tileflow.callback import CompositeCallback, ProcessingStats, TileFlowCallback
from tileflow.core import ProcessedTile, TileSpec
from tileflow.reconstruction import reconstruct
from tileflow.tiling import GridSpec
from tileflow.utils import validate_overlap, validate_tile_size
from abc import abstractmethod


class MaskedStreamable:
    @abstractmethod
    def read_raster(self, level: int, channels: int | list[int]) -> np.ndarray:
        pass

    @abstractmethod    
    def get_shape_hw(self) -> tuple[int, int]:
        pass

    @abstractmethod
    def read_mask_region(self, level: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        pass

    @abstractmethod
    def read_region(self, level: int, channels:int | list[int], y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        pass

class TileFlowMasked:
    def __init__(
        self,
        tile_size: tuple[int, int],
        tile_overlap: tuple[int, int] = (0, 0),
        chunk_size: tuple[int, int] | None = None,
        chunk_overlap: tuple[int, int] = (0, 0),
        optimize=True,
    ):
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._processor = None
        self._chunk_processor = None
        self._configured = False
        self.level = None
        self.channels = None
        self.optimize = optimize

    def configure(self, level: str, channels: list[int], function: Callable, chunk_function: Callable) -> None:
        if not callable(function):
            raise TypeError("function must be callable")
        if chunk_function is not None and not callable(chunk_function):
            raise TypeError("chunk_function must be callable")

        self._processor = function
        self._chunk_processor = chunk_function
        self._configured = True
        self.level = level
        self.channels = channels

    def run(self, streamable: MaskedStreamable) -> Any:
        if not self._configured:
            raise RuntimeError(
                f"Processor must be configured before use. Call processor.configure(function=fn)"
            )
        # Implementation of run method for masked streamable
        if self.chunk_size is not None:
            result = self._process_by_chunks(streamable)
        else:
            # use only tiles
            # array = streamable.get_level(self.level)[self.channels : self.channels + 1][0]
            array = streamable.read_raster(self.level, self.channels)
            mask = streamable.read_mask_region(self.level, 0, array.shape[0], 0, array.shape[1])
            result = self._process_by_tiles(array, mask)
        return result

    def _process_by_tiles(
        self, array: np.ndarray, mask: np.ndarray | None = None, return_tiles: bool = False
    ) -> np.ndarray | list[ProcessedTile]:
        """Process with direct tiling (no chunking)."""

        region_shape = array.shape
        grid_spec = GridSpec(size=self.tile_size, overlap=self.tile_overlap)
        tile_specs = list(grid_spec.build_grid(region_shape))

        tiles: list[ProcessedTile] = []
        for tile_spec in tile_specs:
            x0, x1 = tile_spec.geometry.halo.x0, tile_spec.geometry.halo.x1
            y0, y1 = tile_spec.geometry.halo.y0, tile_spec.geometry.halo.y1
            tile_mask = mask[y0:y1, x0:x1] if mask is not None else None
            # skip empty tiles
            if self.optimize and tile_mask is not None and np.all(tile_mask == 0):
                continue
            tile_region = array[y0:y1, x0:x1]

            tile_processed = self._processor(tile_region, tile_spec)
            tiles.append(ProcessedTile(tile_spec=tile_spec, image_data=tile_processed))

        if return_tiles:
            return tiles

        reconstructed = reconstruct(tiles, region_shape)
        return reconstructed[0] if len(reconstructed) == 1 else reconstructed

    def _process_by_chunks(self, streamable: MaskedStreamable) -> None:
        """Process with chunking for large images."""
        shape = streamable.get_shape_hw()
        chunk_grid_spec = GridSpec(size=self.chunk_size, overlap=self.chunk_overlap)
        chunk_specs = list(chunk_grid_spec.build_grid(shape))

        for chunk_spec in chunk_specs:
            x0, x1 = chunk_spec.geometry.halo.x0, chunk_spec.geometry.halo.x1
            y0, y1 = chunk_spec.geometry.halo.y0, chunk_spec.geometry.halo.y1
            chunk_mask = streamable.read_mask_region(self.level, y0, y1, x0, x1)

            # skip empty chunks
            if self.optimize and np.all(chunk_mask == 0):
                continue

            # read region there to optimize disk access, receiving np.ndarray
            chunk_region = streamable.read_region(self.level, self.channels, y0, y1, x0, x1)
            chunk_output = self._process_by_tiles(chunk_region, chunk_mask, return_tiles=False)

            # Apply chunk processor if provided
            if self._chunk_processor:
                self._chunk_processor(chunk_output, chunk_spec)

### -------------------------------------------------------------------------------


class TileFlow:
    """Tile-based image processor.

    Processes large images through tiling with configurable overlap
    and reconstruction. Supports both direct tiling and hierarchical chunking
    for memory-efficient processing of massive datasets.

    Examples
    --------
    >>> processor = TileFlow(tile_size=(128, 128), overlap=(8, 8))
    >>> processor.configure(function=my_function)
    >>> result = processor.run(image)

    >>> # For very large images, use chunking
    >>> processor = TileFlow(tile_size=(128, 128), overlap=(8, 8), chunk_size=(512, 512))
    >>> processor.configure(function=my_function)
    >>> result = processor.run(large_image)
    """

    def __init__(
        self,
        tile_size: tuple[int, int],
        overlap: tuple[int, int] = (0, 0),
        chunk_size: tuple[int, int] | None = None,
        chunk_overlap: tuple[int, int] = (16, 16),
        name: str = "TileFlow",
    ) -> None:
        """Initialize TileFlow processor.

        Parameters
        ----------
        tile_size : tuple[int, int]
            Size of each tile (height, width)
        overlap : tuple[int, int], default=(0, 0)
            Overlap between tiles (height, width)
        chunk_size : tuple[int, int], optional
            If provided, process large images in chunks of this size
        chunk_overlap : tuple[int, int], default=(16, 16)
            Overlap between chunks when using chunked processing
        name : str, default="TileFlow"
            Name of the processor for logging/debugging
        """
        self.tile_size = self._validate_size(tile_size, "tile_size")
        self.overlap = self._validate_size(overlap, "overlap")
        self.chunk_size = self._validate_size(chunk_size, "chunk_size") if chunk_size else None
        self.chunk_overlap = self._validate_size(chunk_overlap, "chunk_overlap")
        self.name = name

        # Validate overlap against tile size
        validate_overlap(self.overlap, self.tile_size)
        if self.chunk_size:
            validate_overlap(self.chunk_overlap, self.chunk_size)

        self._processor: (
            Callable[[np.ndarray], np.ndarray] | Callable[[np.ndarray, TileSpec], np.ndarray] | None
        ) = None
        self._chunk_processor: Callable[[list[np.ndarray], TileSpec], list[np.ndarray]] | None = (
            None
        )
        self._configured = False

    @staticmethod
    def _validate_size(size: tuple[int, int], param_name: str) -> tuple[int, int]:
        """Validate size parameters."""
        if not isinstance(size, (tuple, list)) or len(size) != 2:
            raise ValueError(f"{param_name} must be a tuple of 2 integers")
        h, w = size
        if not isinstance(h, int) or not isinstance(w, int):
            raise ValueError(f"{param_name} must contain integers")
        if h < 0 or w < 0:
            raise ValueError(f"{param_name} must be non-negative, got {size}")
        if "size" in param_name and (h == 0 or w == 0):
            raise ValueError(f"{param_name} must be positive, got {size}")
        return (h, w)

    @staticmethod
    def _get_slice_prefix(ndim: int) -> tuple:
        """Get slice prefix for multi-dimensional arrays."""
        return tuple([slice(None)] * (ndim - 2)) if ndim > 2 else ()

    def _process_single_tile(
        self, streamable: Streamable, tile_spec: TileSpec, slice_prefix: tuple
    ) -> ProcessedTile:
        """Process a single tile."""
        halo_slices = tile_spec.get_halo_slices()
        full_slices = slice_prefix + halo_slices if slice_prefix else halo_slices
        tile_data = streamable[full_slices]

        # Try calling with both arguments first, fall back to single argument for compatibility
        try:
            tile_output = self._processor(tile_data, tile_spec)
        except TypeError:
            # Fall back to single argument for backward compatibility
            tile_output = self._processor(tile_data)

        return ProcessedTile(tile_spec=tile_spec, image_data=tile_output)

    def _process_tiles(
        self,
        streamable: Streamable,
        tile_specs_iter,
        slice_prefix: tuple,
        callback: CompositeCallback,
        stats: ProcessingStats | None = None,
    ) -> list[ProcessedTile]:
        """Process tiles serially."""
        tiles: list[ProcessedTile] = []

        for i, tile_spec in enumerate(tile_specs_iter):
            if callback.has_tile_listeners():
                temp_tile = ProcessedTile(tile_spec=tile_spec, image_data=[])
                callback.on_tile_start(temp_tile, i, stats.total_tiles if stats else 0)

            processed_tile = self._process_single_tile(streamable, tile_spec, slice_prefix)

            if callback.has_tile_listeners():
                callback.on_tile_end(processed_tile, i, stats.total_tiles if stats else 0)
            tiles.append(processed_tile)

            if stats:
                stats.processed_tiles = i + 1

        return tiles

    def configure(
        self,
        function: Callable[[np.ndarray], np.ndarray] | Callable[[np.ndarray, TileSpec], np.ndarray],
        chunk_function: Callable[[list[np.ndarray], TileSpec], list[np.ndarray]] | None = None,
    ) -> None:
        """Configure the processor with processing functions.

        Parameters
        ----------
        function : Callable[[np.ndarray], np.ndarray] | Callable[[np.ndarray, TileSpec], np.ndarray]
            Function to apply to each tile. Can receive just tile data or both tile data and
            tile specification.
        chunk_function : Callable[[list[np.ndarray], TileSpec], list[np.ndarray]], optional
            Function to apply to reconstructed chunks. Receives chunk output and
            chunk specification.
        """
        if not callable(function):
            raise TypeError("function must be callable")
        if chunk_function is not None and not callable(chunk_function):
            raise TypeError("chunk_function must be callable")

        self._processor = function
        self._chunk_processor = chunk_function
        self._configured = True

    def run(
        self,
        data: Any,
        callbacks: list[TileFlowCallback] | None = None,
        return_tiles: bool = False,
    ) -> np.ndarray | list[ProcessedTile] | None:
        """Run processing pipeline on input data.

        Parameters
        ----------
        data : Any
            Input data (np.ndarray or other supported types)
        callbacks : list[TileFlowCallback], optional
            Callbacks for progress tracking
        return_tiles : bool, default=False
            Return individual tiles instead of reconstructed image

        Returns
        -------
        np.ndarray | list[ProcessedTile] | None
            Processed result, individual tiles, or None for chunked processing of massive images
        """
        if not self._configured:
            raise RuntimeError(
                f"Processor '{self.name}' must be configured before use. "
                f"Call processor.configure(function=fn)"
            )

        streamable = as_streamable(data)

        # Validate configuration against actual data (spatial dimensions only)
        spatial_shape = streamable.shape[-2:] if len(streamable.shape) >= 2 else streamable.shape
        validate_tile_size(self.tile_size, spatial_shape)

        # Create unified callback interface
        callback = CompositeCallback(callbacks or [])

        # Initialize statistics for callbacks
        stats = ProcessingStats()
        stats.input_shape = streamable.shape
        stats.tile_size = self.tile_size
        stats.overlap = self.overlap
        stats.chunk_size = self.chunk_size

        try:
            # Start processing
            callback.on_processing_start(stats)

            if self.chunk_size is not None:
                result = self._process_chunked(
                    streamable, spatial_shape, callback, return_tiles, stats
                )
            else:
                result = self._process_direct(
                    streamable, spatial_shape, callback, return_tiles, stats
                )

            # Update final statistics
            if result is not None and hasattr(result, "shape"):
                stats.output_shape = result.shape
            callback.on_processing_end(stats)

            return result

        except Exception as e:
            callback.on_processing_error(e, stats)
            raise

    def _process_direct(
        self,
        streamable: Streamable,
        spatial_shape: tuple[int, int],
        callback: CompositeCallback,
        return_tiles: bool = False,
        stats: ProcessingStats | None = None,
    ) -> np.ndarray | list[ProcessedTile]:
        """Process with direct tiling (no chunking)."""
        grid_spec = GridSpec(size=self.tile_size, overlap=self.overlap)
        tile_specs = list(grid_spec.build_grid(spatial_shape))

        if stats:
            stats.total_tiles = len(tile_specs)

        # Get slice prefix for multi-dimensional arrays
        slice_prefix = TileFlow._get_slice_prefix(len(streamable.shape))

        tiles = self._process_tiles(streamable, tile_specs, slice_prefix, callback, stats)

        if return_tiles:
            return tiles

        reconstructed = reconstruct(tiles)
        return reconstructed[0] if len(reconstructed) == 1 else reconstructed

    def _process_chunked(
        self,
        streamable: Streamable,
        spatial_shape: tuple[int, int],
        callback: CompositeCallback,
        return_tiles: bool = False,
        stats: ProcessingStats | None = None,
    ) -> None:
        """Process with chunking for large images."""
        validate_tile_size(self.chunk_size, spatial_shape)
        chunk_grid_spec = GridSpec(size=self.chunk_size, overlap=self.chunk_overlap)
        chunk_specs = list(chunk_grid_spec.build_grid(spatial_shape))

        if stats:
            stats.total_chunks = len(chunk_specs)

        slice_prefix = TileFlow._get_slice_prefix(len(streamable.shape))

        for i, chunk_spec in enumerate(chunk_specs):
            if callback.has_chunk_listeners():
                chunk_shape = chunk_spec.geometry.halo.shape
                callback.on_chunk_start(i, stats.total_chunks, chunk_shape)

            halo_slices = chunk_spec.get_halo_slices()
            full_slices = slice_prefix + halo_slices if slice_prefix else halo_slices
            chunk_data = streamable[full_slices]

            # Process chunk with tiles
            chunk_streamable = as_streamable(chunk_data)
            chunk_spatial_shape = (
                chunk_data.shape[-2:] if len(chunk_data.shape) >= 2 else chunk_data.shape
            )
            chunk_output = self._process_direct(
                chunk_streamable, chunk_spatial_shape, callback, return_tiles=False, stats=None
            )

            # Apply chunk processor if provided
            if self._chunk_processor:
                chunk_output = self._chunk_processor(chunk_output, chunk_spec)

            if callback.has_chunk_listeners():
                chunk_shape = chunk_spec.geometry.halo.shape
                callback.on_chunk_end(i, stats.total_chunks, chunk_shape)

            if stats:
                stats.processed_chunks = i + 1

        # Return None for chunk processing to save memory

    def summary(self) -> None:
        """Print processor configuration summary."""
        print(f"TileFlow Processor: {self.name}")
        print("=" * 50)
        print(f"Tile size:      {self.tile_size}")
        print(f"Tile overlap:   {self.overlap}")
        if self.chunk_size:
            print(f"Chunk size:     {self.chunk_size}")
            print(f"Chunk overlap:  {self.chunk_overlap}")
            print("Mode:           Hierarchical (chunks → tiles)")
        else:
            print("Mode:           Direct tiling")
        print(f"Configured:     {self._configured}")
        if self._configured:
            print(
                f"Function:       {
                    self._processor.__name__
                    if hasattr(self._processor, '__name__')
                    else 'Custom function'
                }"
            )
        if self._chunk_processor:
            print(
                f"Chunk function: {
                    self._chunk_processor.__name__
                    if hasattr(self._chunk_processor, '__name__')
                    else 'Custom chunk function'
                }"
            )
        print("=" * 50)
