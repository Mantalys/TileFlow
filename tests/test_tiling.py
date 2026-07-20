import pytest
from tileflow import GridSpec

SIZE_256 = (256, 256)
OVERLAP_8 = (8, 8)


def test_basics():
    grid = GridSpec(SIZE_256, OVERLAP_8)
    shape_4 = grid.grid_shape((1024, 1024))
    assert shape_4 == (4, 4)
    shape_1 = grid.grid_shape((256, 256))
    assert shape_1 == (1, 1)
    shape_rect = grid.grid_shape((1024, 256))
    assert shape_rect == (4, 1)
    shape_long = grid.grid_shape((64, 256 * 256))
    assert shape_long == (1, 256)
    shape_long = grid.grid_shape((256 * 256, 64))
    assert shape_long == (256, 1)


def test_smaller_shape():
    grid = GridSpec(SIZE_256, OVERLAP_8)
    shape_0 = grid.grid_shape((128, 128))
    assert shape_0 == (1, 1)
    shape_0 = grid.grid_shape((64, 64))
    assert shape_0 == (1, 1)


def test_value_error():
    grid = GridSpec(SIZE_256, OVERLAP_8)
    with pytest.raises(ValueError):
        grid.grid_shape((0, 256))
    with pytest.raises(ValueError):
        grid.grid_shape((256, 0))
    with pytest.raises(ValueError):
        grid.grid_shape((0, 0))
    with pytest.raises(ValueError):
        grid.grid_shape((-16, 256))
    with pytest.raises(ValueError):
        grid.grid_shape((256, -16))
    with pytest.raises(ValueError):
        grid.grid_shape((-16, -16))


def test_build_grid():
    grid = GridSpec(SIZE_256, OVERLAP_8)
    tiles = list(grid.iter_tiles(256, 256))
    assert len(tiles) == 1

    tiles = list(grid.iter_tiles(1032, 1032))
    assert len(tiles) == 16
    assert tiles[5].position.row == 1
    assert tiles[5].position.column == 1

    tiles = list(grid.iter_tiles(128, 128))
    assert len(tiles) == 1
    assert tiles[0].position.row == 0
    assert tiles[0].position.column == 0
