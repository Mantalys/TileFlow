import pytest
from tileflow import BBox, TileSpec, TileGeometry, GridIndex, BoundaryEdges, ProcessedTile
import numpy as np


def test_bbox():
    bbox = BBox(0, 0, 512, 512)
    assert bbox.height == 512
    assert bbox.width == 512
    bbox_long = BBox(0, 0, 512, 1024)
    assert bbox_long.height == 1024
    assert bbox_long.width == 512


def test_bbox_methods():
    bbox = BBox(0, 0, 512, 1024)
    assert bbox.shape == (1024, 512)
    assert bbox.contains(256, 512)
    assert not bbox.contains(256, 1024)

    bbox_clamped = bbox.clamp_to(512, 512)
    assert bbox_clamped == BBox(0, 0, 512, 512)

    bbox_2 = BBox.from_size(0, 0, 1024, 320)
    assert bbox_2.shape == (1024, 320)

    bbox_translated = bbox.translate(dy=100, dx=100)
    assert bbox_translated == BBox(100, 100, 612, 1124)
    assert bbox.shape == bbox_translated.shape

    bbox_translated_negative = bbox.translate(dy=-100, dx=-100)
    assert bbox_translated_negative == BBox(0, 0, 512, 1024)

    bbox_expanded = bbox.expand(10, 10, 10, 10)
    assert bbox_expanded == BBox(0, 0, 522, 1034)
    assert not bbox_expanded.contains(-1, -1)

    slices = bbox.get_slices()
    assert slices == (slice(0, 1024), slice(0, 512))


def test_value_error():
    with pytest.raises(ValueError):
        BBox(1024, 0, 512, 1024)


def test_specs():
    core = BBox(0, 0, 512, 1024)
    halo = core.expand(8, 8, 8, 8)
    tile_spec = TileSpec(
        geometry=TileGeometry(core, halo),
        position=GridIndex(0, 0),
        edges=BoundaryEdges(True, True, True, True),
    )
    assert tile_spec.geometry.core == core
    assert tile_spec.geometry.halo == halo
    assert tile_spec.position.row == 0
    assert tile_spec.position.column == 0
    assert tile_spec.edges.top
    assert tile_spec.edges.right
    assert tile_spec.edges.bottom
    assert tile_spec.edges.left
    assert tile_spec.get_slices() == (slice(0, 1024), slice(0, 512))
    assert tile_spec.get_halo_slices() == (slice(0, 1032), slice(0, 520))
    assert tile_spec.geometry.core_in_halo_slices() == (slice(0, 1024), slice(0, 512))


def test_process_tile_corner():
    data_1 = np.random.rand(1)
    data_2 = np.random.rand(1024, 1024)
    data_3 = np.random.rand(3, 1024, 512)
    core = BBox(0, 0, 256, 256)
    halo = core.expand(32, 32, 32, 32)
    tile_spec = TileSpec(
        geometry=TileGeometry(core, halo),
        position=GridIndex(0, 0),
        edges=BoundaryEdges(False, False, False, False),
    )

    tile_1 = ProcessedTile(tile_spec, data_1)
    outcome_1 = tile_1.only_core_data()
    assert outcome_1.shape == (1,)
    assert np.all(outcome_1 == data_1)

    tile_2 = ProcessedTile(tile_spec, data_2)
    outcome_2 = tile_2.only_core_data()
    assert outcome_2.shape == (256, 256)
    assert np.all(outcome_2 == data_2[0:256, 0:256])

    tile_3 = ProcessedTile(tile_spec, data_3)
    outcome_3 = tile_3.only_core_data()
    assert outcome_3.shape == (3, 256, 256)
    assert np.all(outcome_3 == data_3[:, 0:256, 0:256])


def test_process_tile():
    data_1 = np.random.rand(1)
    data_2 = np.random.rand(1024, 1024)

    data_3 = np.random.rand(3, 1024, 512)
    core = BBox(128, 128, 256 + 128, 256 + 128)
    halo = core.expand(32, 32, 32, 32)
    slice_y, slice_x = halo.get_slices()
    print(slice_y, slice_x)

    tile_spec = TileSpec(
        geometry=TileGeometry(core, halo),
        position=GridIndex(0, 0),
        edges=BoundaryEdges(False, False, False, False),
    )

    tile_1 = ProcessedTile(tile_spec, data_1)
    outcome_1 = tile_1.only_core_data()
    assert outcome_1.shape == (1,)
    assert np.all(outcome_1 == data_1)

    data_2_tile = data_2[halo.get_slices()]
    tile_2 = ProcessedTile(tile_spec, data_2_tile)
    outcome_2 = tile_2.only_core_data()
    assert outcome_2.shape == (256, 256)
    assert np.all(outcome_2 == data_2[128:384, 128:384])

    data_3_tile = data_3[:, slice_y, slice_x]
    tile_3 = ProcessedTile(tile_spec, data_3_tile)
    outcome_3 = tile_3.only_core_data()
    assert outcome_3.shape == (3, 256, 256)
    assert np.all(outcome_3 == data_3[:, 128:384, 128:384])
