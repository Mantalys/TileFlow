from abc import abstractmethod, ABC
from collections.abc import Callable
from typing import Any

import numpy as np
from tileflow.core import ProcessedTile, TupleInt2, TileSpec
from tileflow.reconstruction import reconstruct_tiles
from tileflow.tiling import GridSpec


TileProcessor = Callable[
    [np.ndarray, TileSpec],
    np.ndarray,
]

ChunkSink = Callable[
    [np.ndarray | None, TileSpec],
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
        if channel_index < 0:
            raise ValueError(f"Channel index must be non-negative, got {channel_index}")
        if channel_index in self.channel_indices:
            raise ValueError(f"Channel index {channel_index} is already added")
        self.channel_indices.append(channel_index)
        self.thresholds.append(threshold)
        # check the range is strictly ordered
        if rescale_range is not None and rescale_range[0] >= rescale_range[1]:
            raise ValueError(
                f"rescale_range must be ordered (min < max), got (min={rescale_range[0]}, max={rescale_range[1]})"
            )
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

        array_f32 = array.astype(np.float32, copy=True)

        for i, rescale_range in enumerate(self.rescale_ranges):
            if rescale_range is None:
                vmin, vmax = array_f32[i].min(), array_f32[i].max()
            else:
                vmin, vmax = rescale_range
            if vmin is None:
                vmin = array_f32[i].min()
            if vmax is None:
                vmax = array_f32[i].max()
            if vmin == vmax:
                array_f32[i].fill(0.0)
            else:
                channel = array_f32[i]
                channel -= vmin
                channel *= 1.0 / (vmax - vmin)
                np.clip(channel, 0.0, 1.0, out=channel)

        # Apply global mask only once instead of once per overlapping tile.
        if self.consider_mask and mask is not None:
            array_f32[:, mask == 0] = 0

        grid_spec = GridSpec(size=self.tile_size, overlap=self.tile_overlap)

        tiles: list[ProcessedTile] = []
        for tile_spec in grid_spec.iter_tiles(region_h, region_w):
            x0, x1 = tile_spec.geometry.halo.x0, tile_spec.geometry.halo.x1
            y0, y1 = tile_spec.geometry.halo.y0, tile_spec.geometry.halo.y1
            if self.consider_mask and mask is not None:
                tile_mask = mask[y0:y1, x0:x1]
                if not np.any(tile_mask):
                    continue
            tile_region = array_f32[:, y0:y1, x0:x1]

            tile_processed = self._tile_processor(tile_region, tile_spec)

            tiles.append(ProcessedTile(tile_spec=tile_spec, data=tile_processed))
        if not tiles:
            raise ValueError("No tiles processed")
        if return_tiles:
            return tiles

        if tiles[0]._data.ndim == 1:
            shape = grid_spec.grid_shape((region_h, region_w))
            return reconstruct_tiles(tiles, shape[0], shape[1])

        return reconstruct_tiles(tiles, region_h, region_w)

    def _process_by_chunks(self, streamable: MaskedStreamable) -> None:
        """Process with chunking for large images."""
        if self.chunk_size is None or self._chunk_sink is None:
            raise RuntimeError("chunk_size and chunk_sink must be set")
        shape = streamable.get_shape_hw()
        chunk_grid_spec = GridSpec(size=self.chunk_size, overlap=self.chunk_overlap)

        for chunk_spec in chunk_grid_spec.iter_tiles(shape[0], shape[1]):
            x0, x1 = chunk_spec.geometry.halo.x0, chunk_spec.geometry.halo.x1
            y0, y1 = chunk_spec.geometry.halo.y0, chunk_spec.geometry.halo.y1
            chunk_mask = streamable.read_mask_region(self.level, y0, y1, x0, x1)

            # skip empty chunks
            if self.consider_mask and np.all(chunk_mask == 0):
                # call chunk sink with None to indicate empty chunk, to allow upper layers to handle it
                self._chunk_sink(None, chunk_spec)
                continue

            # read region there to optimize disk access, receiving np.ndarray
            chunk_region = streamable.read_region(self.level, self.channel_indices, y0, y1, x0, x1)

            if len(self.channel_indices) == 1:
                if self.thresholds[0] is not None and np.max(chunk_region) < self.thresholds[0]:
                    # call chunk sink with None to indicate empty chunk, to allow upper layers to handle it
                    self._chunk_sink(None, chunk_spec)
                    continue

            chunk_output = self._process_by_tiles(chunk_region, chunk_mask, return_tiles=False)

            # apply mask to chunk output
            if self.consider_mask and chunk_mask is not None:
                chunk_output = chunk_output * chunk_mask

            # Apply chunk processor if provided
            if self._chunk_sink:
                self._chunk_sink(chunk_output, chunk_spec)
