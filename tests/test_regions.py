"""Tests for region and tile management."""

import numpy as np

from tileflow.core import (
    BBox,
    BoundaryEdges,
    ProcessedTile,
    TileGeometry,
    TilePosition,
    TileSpec,
    new_image,
)


class TestBBox:
    """Test BBox functionality."""

    def test_basic_properties(self):
        """Test basic BBox properties."""
        bbox = BBox(10, 20, 50, 80)
        assert bbox.width == 40
        assert bbox.height == 60
        assert bbox.shape == (60, 40)

    def test_slices(self):
        """Test slice generation."""
        bbox = BBox(10, 20, 50, 80)
        slices = bbox.get_slices()
        assert slices == (slice(20, 80), slice(10, 50))

    def test_from_size(self):
        """Test construction from position and size."""
        bbox = BBox.from_size(y=10, x=20, h=30, w=40)
        assert bbox == BBox(20, 10, 60, 40)

    def test_translate(self):
        """Test translation."""
        bbox = BBox(10, 20, 50, 80)
        translated = bbox.translate(dx=5, dy=10)
        assert translated == BBox(15, 30, 55, 90)

    def test_clamp_to(self):
        """Test clamping to bounds."""
        bbox = BBox(-5, -10, 100, 120)
        clamped = bbox.clamp_to(H=80, W=90)
        assert clamped.x0 >= 0
        assert clamped.y0 >= 0
        assert clamped.x1 <= 90
        assert clamped.y1 <= 80

    def test_contains(self):
        """Test point containment."""
        bbox = BBox(10, 20, 50, 80)
        assert bbox.contains(30, 40)
        assert not bbox.contains(5, 40)
        assert not bbox.contains(30, 15)

    def test_intersects(self):
        """Test bbox intersection."""
        bbox1 = BBox(10, 20, 50, 80)
        bbox2 = BBox(30, 40, 70, 100)
        assert bbox1.intersects(bbox2)

        bbox3 = BBox(60, 90, 80, 110)
        assert not bbox1.intersects(bbox3)

    def test_intersection(self):
        """Test intersection calculation."""
        bbox1 = BBox(10, 20, 50, 80)
        bbox2 = BBox(30, 40, 70, 100)
        intersection = bbox1.intersection(bbox2)
        assert intersection == BBox(30, 40, 50, 80)

        bbox3 = BBox(60, 90, 80, 110)
        assert bbox1.intersection(bbox3) is None

    def test_expand(self):
        """Test expansion."""
        bbox = BBox(10, 20, 50, 80)
        expanded = bbox.expand(left=5, right=10, top=3, bottom=7)
        assert expanded == BBox(5, 17, 60, 87)


class TestBoundaryEdges:
    """Test BoundaryEdges functionality."""

    def test_creation(self):
        """Test edge creation."""
        edges = BoundaryEdges(left=True, right=False, top=True, bottom=False)
        assert edges.left is True
        assert edges.right is False
        assert edges.top is True
        assert edges.bottom is False


class TestTileGeometry:
    """Test TileGeometry functionality."""

    def test_slices(self):
        """Test slice methods."""
        region = BBox(10, 20, 50, 60)
        tile = BBox(5, 15, 55, 65)
        geometry = TileGeometry(core=region, halo=tile)

        assert geometry.get_slices() == region.get_slices()
        assert geometry.get_halo_slices() == tile.get_slices()

    def test_contains(self):
        """Test point containment."""
        region = BBox(10, 20, 50, 60)
        tile = BBox(5, 15, 55, 65)
        geometry = TileGeometry(core=region, halo=tile)

        assert geometry.contains(30, 40)  # In region
        assert not geometry.contains(7, 17)  # In tile but not region


class TestProcessedTile:
    """Test ProcessedTile functionality."""

    def test_single_image(self):
        """Test with single image data."""
        region = BBox(10, 20, 50, 60)
        tile = BBox(5, 15, 55, 65)
        geometry = TileGeometry(core=region, halo=tile)
        position = TilePosition((0, 0), BoundaryEdges(True, False, True, False))
        spec = TileSpec(geometry=geometry, position=position)

        # Create test data matching tile size
        data = np.ones((50, 50), dtype=np.float32)  # tile is 50x50
        processed_tile = ProcessedTile(spec, data)

        assert processed_tile.x_start == 5
        assert processed_tile.y_start == 15
        assert processed_tile.core_bbox == region

    def test_multiple_images(self):
        """Test with multiple image channels."""
        region = BBox(10, 20, 50, 60)
        tile = BBox(5, 15, 55, 65)
        geometry = TileGeometry(core=region, halo=tile)
        position = TilePosition((0, 0), BoundaryEdges(True, False, True, False))
        spec = TileSpec(geometry=geometry, position=position)

        # Create multi-channel data
        data = [np.ones((50, 50)) * 1, np.ones((50, 50)) * 2, np.ones((50, 50)) * 3]
        processed_tile = ProcessedTile(spec, data)

        # Should extract region correctly
        region_data = processed_tile.only_core_image()
        assert len(region_data) == 3
        for i, channel in enumerate(region_data):
            assert channel.shape == (40, 40)  # region size
            assert np.all(channel == i + 1)

    def test_region_extraction(self):
        """Test extracting region of interest from tile data."""
        # Region is (10,20) to (50,60) - size 40x40
        # Tile is (5,15) to (55,65) - size 50x50
        # So region is offset by (5,5) within tile
        region = BBox(10, 20, 50, 60)
        tile = BBox(5, 15, 55, 65)
        geometry = TileGeometry(core=region, halo=tile)
        position = TilePosition((0, 0), BoundaryEdges(True, False, True, False))
        spec = TileSpec(geometry=geometry, position=position)

        # Create tile data with unique values to verify extraction
        tile_data = np.arange(50 * 50).reshape(50, 50).astype(np.float32)
        processed_tile = ProcessedTile(spec, tile_data)

        region_data = processed_tile.only_core_image()
        assert len(region_data) == 1
        assert region_data[0].shape == (40, 40)

        # Verify that extracted data matches expected slice
        expected = tile_data[5:45, 5:45]  # offset by (5,5), size 40x40
        np.testing.assert_array_equal(region_data[0], expected)


class TestUtilities:
    """Test utility functions."""

    def test_new_image_2d(self):
        """Test 2D image creation."""
        img = new_image((100, 200))
        assert img.shape == (100, 200)
        assert img.dtype == np.float32
        assert np.all(img == 0)

    def test_new_image_3d(self):
        """Test 3D image creation."""
        img = new_image((3, 100, 200))
        assert img.shape == (3, 100, 200)
        assert img.dtype == np.float32
        assert np.all(img == 0)

    def test_new_image_custom_dtype(self):
        """Test custom dtype."""
        img = new_image((100, 200), dtype=np.uint8)
        assert img.dtype == np.uint8
