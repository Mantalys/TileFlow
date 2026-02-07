"""Tile-based image processing engine.

This module implements TileFlow's main processing pipeline:
- TileFlow: Main processor class with configure/run interface
- Direct tiling: Process images tile by tile
- Hierarchical chunking: Handle massive images through chunk → tile processing
- Multi-dimensional support: Handle CHW, CHWD, and arbitrary array shapes
"""

from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
from skimage.exposure import rescale_intensity

from tileflow.core import ProcessedTile
from tileflow.reconstruction import reconstruct
from tileflow.tiling import GridSpec


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
    def read_region(
        self, level: int, channels: int | list[int], y0: int, y1: int, x0: int, x1: int
    ) -> np.ndarray:
        pass


class TileFlowMasked:
    def __init__(
        self,
        tile_size: tuple[int, int],
        tile_overlap: tuple[int, int] = (0, 0),
        chunk_size: tuple[int, int] | None = None,
        chunk_overlap: tuple[int, int] = (0, 0),
        optimize=True,
        normalize_range: tuple | None = None,
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
        self.normalize_range = normalize_range

    def configure(
        self,
        level: str,
        channels: list[int],
        function: Callable,
        chunk_function: Callable,
        threshold: int | float | None = None,
    ) -> None:
        if not callable(function):
            raise TypeError("function must be callable")
        if chunk_function is not None and not callable(chunk_function):
            raise TypeError("chunk_function must be callable")

        self._processor = function
        self._chunk_processor = chunk_function
        self._configured = True
        self.level = level
        self.channels = channels
        self.threshold = threshold

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
            array = streamable.read_raster(self.level, self.channels)
            if np.max(array) < self.threshold:
                # consider the chunk empty
                return None
            mask = streamable.read_mask_region(self.level, 0, array.shape[0], 0, array.shape[1])
            result = self._process_by_tiles(array, mask)
        return result

    def normalize_mi_ma(self, x, mi, ma):
        eps = 1e-10
        return (x - mi) / (ma - mi + eps)

    def _process_by_tiles(
        self, array: np.ndarray, mask: np.ndarray | None = None, return_tiles: bool = False
    ) -> np.ndarray | list[ProcessedTile]:
        """Process with direct tiling (no chunking)."""
        if self.normalize_range:
            array = self.normalize_mi_ma(array, *self.normalize_range)
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

            # apply mask if provided
            if self.optimize and tile_mask is not None:
                tile_region = tile_region * tile_mask

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

            if np.max(chunk_region) < self.threshold:
                # consider the chunk empty
                continue

            chunk_output = self._process_by_tiles(chunk_region, chunk_mask, return_tiles=False)

            # apply mask to chunk output
            if self.optimize and chunk_mask is not None:
                chunk_output = chunk_output * chunk_mask

            # Apply chunk processor if provided
            if self._chunk_processor:
                self._chunk_processor(chunk_output, chunk_spec)
