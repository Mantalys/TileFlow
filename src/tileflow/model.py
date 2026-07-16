from abc import abstractmethod, ABC
from collections.abc import Callable
from typing import Any

import numpy as np

from tileflow.core import ProcessedTile, TupleInt2
from tileflow.reconstruction import reconstruct_tiles
from tileflow.tiling import GridSpec


class MaskedStreamable(ABC):
    """Abstract base class for masked streamable image data."""

    @abstractmethod
    def read_raster(self, level: int, channels: int | list[int]) -> np.ndarray:
        pass

    @abstractmethod
    def get_shape_hw(self) -> TupleInt2:
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
        tile_size: TupleInt2,
        tile_overlap: TupleInt2 = (0, 0),
        chunk_size: TupleInt2 | None = None,
        chunk_overlap: TupleInt2 = (0, 0),
        consider_mask=True,
    ):
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.consider_mask = consider_mask
        self._processor = None
        self._chunk_processor = None
        self._configured = False
        self.level = 0
        self.channel_indices = []
        self.thresholds = []
        self.rescale_ranges = []

    def add_channel(
        self,
        channel_index: int,
        threshold: int | float | None = None,
        rescale_range: TupleInt2 | None = None,
    ) -> None:
        if channel_index in self.channel_indices:
            raise ValueError(f"Channel index {channel_index} is already added")
        self.channel_indices.append(channel_index)
        self.thresholds.append(threshold)
        self.rescale_ranges.append(rescale_range)

    def setup(
        self,
        level: int,
        function: Callable,
        chunk_function: Callable | None = None
    ) -> None:
        if self.channel_indices is None or self.thresholds is None or self.rescale_ranges is None:
            raise RuntimeError("Channels must be added before setup")

        if not callable(function):
            raise TypeError("function must be callable")
        if chunk_function is not None and not callable(chunk_function):
            raise TypeError("chunk_function must be callable")

        self.level = level
        self._processor = function
        self._chunk_processor = chunk_function
        self._configured = True

    def run(self, streamable: MaskedStreamable) -> Any:
        if not self._configured:
            raise RuntimeError(
                f"Processor must be configured before use. Call processor.setup(function=fn)"
            )
        # Implementation of run method for masked streamable
        if self.chunk_size is not None:
            result = self._process_by_chunks(streamable)
        else:
            # use only tiles
            if len(self.channel_indices) == 1:
                # case: single channel, array shape is (H, W)
                array = streamable.read_raster(self.level, self.channel_indices)
                if self.thresholds and np.max(array) < self.thresholds[0]:
                    # consider the chunk empty
                    return None
                if array.ndim == 2:
                    array = array[None, :, :]

            else:
                # case: multiple channels, array shape is (C, H, W)
                array = streamable.read_raster(self.level, self.channel_indices)
            if array.ndim != 3:
                raise ValueError(f"Expected array of shape (C, H, W), got {array.shape}")
            mask = (
                streamable.read_mask_region(self.level, 0, array.shape[1], 0, array.shape[2])
                if self.consider_mask
                else None
            )
            result = self._process_by_tiles(array, mask)
        return result

    def normalize_mi_ma(self, x: np.ndarray, mi: float, ma: float) -> np.ndarray:
        eps = 1e-6
        return (x - mi) / (ma - mi + eps)

    def _process_by_tiles(
        self, array: np.ndarray, mask: np.ndarray | None = None, return_tiles: bool = False
    ) -> np.ndarray | list[ProcessedTile]:
        """Process with direct tiling (no chunking)."""
        if len(array.shape) == 2:
            array = array[np.newaxis, :, :]
        # now assume shape is (C, H, W)
        if len(array.shape) != 3:
            raise ValueError(f"Expected array shape (C, H, W), got {array.shape}")
        if mask is not None and len(mask.shape) != 2:
            raise ValueError(f"Expected mask shape (H, W), got {mask.shape}")

        n_channels, region_h, region_w = array.shape

        for i in range(n_channels):
            if self.rescale_ranges[i] is not None:
                vmin, vmax = self.rescale_ranges[i]
                if vmin is not None and vmax is not None:
                    array[i] = self.normalize_mi_ma(array[i], vmin, vmax)

        grid_spec = GridSpec(size=self.tile_size, overlap=self.tile_overlap)

        tiles: list[ProcessedTile] = []
        for tile_spec in grid_spec.build_grid(region_h, region_w):
            x0, x1 = tile_spec.geometry.halo.x0, tile_spec.geometry.halo.x1
            y0, y1 = tile_spec.geometry.halo.y0, tile_spec.geometry.halo.y1
            tile_mask = mask[y0:y1, x0:x1] if mask is not None else None
            # skip empty tiles
            if self.consider_mask and tile_mask is not None and np.all(tile_mask == 0):
                continue
            tile_region = array[:, y0:y1, x0:x1]

            # apply mask if provided
            if self.consider_mask and tile_mask is not None:
                # multiply tile region by mask to apply mask, on each channel
                tile_region = tile_region * tile_mask

            tile_processed = self._processor(tile_region, tile_spec)
            tiles.append(ProcessedTile(tile_spec=tile_spec, image_data=tile_processed))

        if return_tiles:
            return tiles

        reconstructed = reconstruct_tiles(tiles, region_h, region_w)
        return reconstructed[0] if len(reconstructed) == 1 else reconstructed

    def _process_by_chunks(self, streamable: MaskedStreamable) -> None:
        """Process with chunking for large images."""
        shape = streamable.get_shape_hw()
        chunk_grid_spec = GridSpec(size=self.chunk_size, overlap=self.chunk_overlap)

        for chunk_spec in chunk_grid_spec.build_grid(shape[0], shape[1]):
            x0, x1 = chunk_spec.geometry.halo.x0, chunk_spec.geometry.halo.x1
            y0, y1 = chunk_spec.geometry.halo.y0, chunk_spec.geometry.halo.y1
            chunk_mask = streamable.read_mask_region(self.level, y0, y1, x0, x1)

            # skip empty chunks
            if self.consider_mask and np.all(chunk_mask == 0):
                continue

            # read region there to optimize disk access, receiving np.ndarray
            chunk_region = streamable.read_region(self.level, self.channel_indices, y0, y1, x0, x1)

            if len(self.channel_indices) == 1:
                if np.max(chunk_region) < self.thresholds[0]:
                    # consider the chunk empty
                    continue

            chunk_output = self._process_by_tiles(chunk_region, chunk_mask, return_tiles=False)

            # apply mask to chunk output
            if self.consider_mask and chunk_mask is not None:
                chunk_output = chunk_output * chunk_mask

            # Apply chunk processor if provided
            if self._chunk_processor:
                self._chunk_processor(chunk_output, chunk_spec)
