import numpy as np
from tileflow.core import ProcessedTile


def __reconstruct_1d(buffer: np.ndarray, tiles: list[ProcessedTile]):
    # Fill in data from each tile
    for tile in tiles:
        array = tile.only_core_data()  # (x,) vector of features
        x = tile.tile_spec.column
        y = tile.tile_spec.row
        buffer[:, y, x] = array
    return buffer


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


def reconstruct_tiles(tiles: list[ProcessedTile], height, width) -> np.ndarray:
    """Reconstruct full image from processed tiles."""

    dtype = tiles[0]._data.dtype

    match tiles[0]._data.ndim:
        case 1:
            buffer = np.empty((tiles[0]._data.shape[0], height, width), dtype=dtype, order="C")
            __reconstruct_1d(buffer, tiles)
        case 2:
            buffer = np.empty((height, width), dtype=dtype, order="C")
            __reconstruct_2d(buffer, tiles)
        case 3:
            buffer = np.empty((tiles[0]._data.shape[0], height, width), dtype=dtype, order="C")
            __reconstruct_3d(buffer, tiles)
        case _:
            raise ValueError(f"Unsupported ndim: {tiles[0]._data.ndim}")
    return buffer
