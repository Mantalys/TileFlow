import pytest
from tileflow import TileFlow
import numpy as np

SIZE_256 = (256, 256)


def _tile():
    return 0


def test_add_channel():
    tileflow = TileFlow(SIZE_256)
    tileflow.add_channel(0)
    with pytest.raises(ValueError):
        tileflow.add_channel(0)
    tileflow.add_channel(1, 0)
    tileflow.add_channel(2, 0.5)
    with pytest.raises(ValueError):
        tileflow.add_channel(3, 0, (1, 1))
    with pytest.raises(ValueError):
        tileflow.add_channel(4, 0, (1, 0))
    with pytest.raises(ValueError):
        tileflow.add_channel(-1)


def test_setup():
    tileflow = TileFlow(SIZE_256)
    with pytest.raises(RuntimeError):
        tileflow.setup(0, _tile)
    tileflow.add_channel(0)
    tileflow.setup(0, _tile)


def test_process_by_tiles():
    def _dummy_tile(tile, spec):
        return tile + 1

    tileflow = TileFlow(SIZE_256)
    tileflow.add_channel(0)
    tileflow.setup(0, _dummy_tile)
    dummy_array = np.zeros((1024, 1024))
    tileflow._process_by_tiles(dummy_array)
