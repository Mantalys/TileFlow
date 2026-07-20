import numpy as np
from tileflow.core import Image2D, ProcessedTile


def __reconstruct_1d():
    pass


def __reconstruct_2d(buffer: np.ndarray, tiles: list[ProcessedTile]):
    # Fill in data from each tile
    for tile in tiles:
        core_array = tile.only_core_data()  # (H, W)
        slice_y, slice_x = tile.tile_spec.geometry.core.get_slices()
        buffer[slice_y, slice_x] = core_array
    return buffer


def __reconstruct_3d(buffer: np.ndarray, tiles: list[ProcessedTile]):
    # Fill in data from each tile
    for tile in tiles:
        core_array = tile.only_core_data()  # (C, H, W)
        slice_y, slice_x = tile.tile_spec.geometry.core.get_slices()
        buffer[:, slice_y, slice_x] = core_array
    return buffer


def reconstruct_tiles(tiles: list[ProcessedTile], height, width) -> list[Image2D] | None:
    """Reconstruct full image from processed tiles."""
    if not tiles:
        return None

    dtype = tiles[0]._data.dtype

    match tiles[0]._data.ndim:
        case 1:
            __reconstruct_1d()
        case 2:
            buffer = np.empty((height, width), dtype=dtype, order="C")
            __reconstruct_2d(buffer, tiles)
        case 3:
            buffer = np.empty((tiles[0]._data.shape[0], height, width), dtype=dtype, order="C")
            __reconstruct_3d(buffer, tiles)
    return buffer
