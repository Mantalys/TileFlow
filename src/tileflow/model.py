from abc import abstractmethod, ABC
from collections.abc import Callable
from typing import Any

import numpy as np
from tileflow.core import ProcessedTile, TupleInt2, TileSpec
from tileflow.reconstruction import reconstruct_tiles
from tileflow.tiling import GridSpec


TileProcessor = Callable[
    [np.ndarray, TileSpec],
    np.ndarray | None,
]

ChunkSink = Callable[
    [np.ndarray, TileSpec],
    None,
]


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
        self._tile_processor = None
        self._chunk_sink = None
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
        # check the range is strictly ordered
        if rescale_range is not None and rescale_range[0] >= rescale_range[1]:
            raise ValueError("rescale_range must be ordered (min < max), got {rescale_range}")
        self.rescale_ranges.append(rescale_range)

    def setup(
        self,
        level: int,
        tile_processor: TileProcessor,
        chunk_sink: ChunkSink | None = None,
    ) -> None:
        if not self.channel_indices:
            raise RuntimeError("At least one channel must be added before setup")

        if not callable(tile_processor):
            raise TypeError("function must be callable")
        if chunk_sink is not None and not callable(chunk_sink):
            raise TypeError("chunk_function must be callable")

        self.level = level
        self._tile_processor = tile_processor
        self._chunk_sink = chunk_sink
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
                if self.thresholds[0] is not None and np.max(array) < self.thresholds[0]:
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
        # assume mi < ma and handle division by zero
        return np.clip((x - mi) / (ma - mi), 0, 1)

    def _process_by_tiles(
        self, array: np.ndarray, mask: np.ndarray | None = None, return_tiles: bool = False
    ) -> np.ndarray | list[ProcessedTile]:
        """Process with direct tiling (no chunking)."""
        if array.ndim == 2:
            array = array[np.newaxis, :, :]
        # now assume shape is (C, H, W)
        if array.ndim != 3:
            raise ValueError(f"Expected array shape (C, H, W), got {array.shape}")
        if mask is not None and mask.ndim != 2:
            raise ValueError(f"Expected mask shape (H, W), got {mask.shape}")

        n_channels, region_h, region_w = array.shape

        for i in range(n_channels):
            if self.rescale_ranges[i] is not None:
                vmin, vmax = self.rescale_ranges[i]
                if vmin is not None and vmax is not None:
                    array[i] = self.normalize_mi_ma(array[i], vmin, vmax)

        grid_spec = GridSpec(size=self.tile_size, overlap=self.tile_overlap)

        tiles: list[ProcessedTile] = []
        for tile_spec in grid_spec.iter_tiles(region_h, region_w):
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

            if tile_region.ndim != 3:
                raise ValueError("tile_region must be (C, H, W), got {}".format(tile_region.shape))
            tile_processed = self._tile_processor(tile_region, tile_spec)

            tiles.append(ProcessedTile(tile_spec=tile_spec, image_data=tile_processed))

        if return_tiles:
            return tiles

        reconstructed = reconstruct_tiles(tiles, region_h, region_w)
        return reconstructed[0] if len(reconstructed) == 1 else reconstructed

    def _process_by_chunks(self, streamable: MaskedStreamable) -> None:
        """Process with chunking for large images."""
        shape = streamable.get_shape_hw()
        chunk_grid_spec = GridSpec(size=self.chunk_size, overlap=self.chunk_overlap)

        for chunk_spec in chunk_grid_spec.iter_tiles(shape[0], shape[1]):
            x0, x1 = chunk_spec.geometry.halo.x0, chunk_spec.geometry.halo.x1
            y0, y1 = chunk_spec.geometry.halo.y0, chunk_spec.geometry.halo.y1
            chunk_mask = streamable.read_mask_region(self.level, y0, y1, x0, x1)

            # skip empty chunks
            if self.consider_mask and np.all(chunk_mask == 0):
                continue

            # read region there to optimize disk access, receiving np.ndarray
            chunk_region = streamable.read_region(self.level, self.channel_indices, y0, y1, x0, x1)

            if len(self.channel_indices) == 1:
                if self.thresholds[0] is not None and np.max(chunk_region) < self.thresholds[0]:
                    # consider the chunk empty
                    continue

            chunk_output = self._process_by_tiles(chunk_region, chunk_mask, return_tiles=False)

            # apply mask to chunk output
            if self.consider_mask and chunk_mask is not None:
                chunk_output = chunk_output * chunk_mask

            # Apply chunk processor if provided
            if self._chunk_sink:
                self._chunk_sink(chunk_output, chunk_spec)
