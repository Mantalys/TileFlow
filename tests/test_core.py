"""Tests for core data structures."""

import numpy as np

from tileflow.core import (
    BBox,
    BoundaryEdges,
    ProcessedTile,
    TileGeometry,
    TilePosition,
    TileSpec,
    new_image2d,
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
        bbox3 = BBox(60, 90, 100, 120)

        assert bbox1.intersects(bbox2)
        assert not bbox1.intersects(bbox3)

    def test_intersection(self):
        """Test intersection calculation."""
        bbox1 = BBox(10, 20, 50, 80)
        bbox2 = BBox(30, 40, 70, 100)

        intersection = bbox1.intersection(bbox2)
        assert intersection == BBox(30, 40, 50, 80)

        bbox3 = BBox(60, 90, 100, 120)
        assert bbox1.intersection(bbox3) is None

    def test_expand(self):
        """Test bbox expansion."""
        bbox = BBox(10, 20, 50, 80)
        expanded = bbox.expand(left=5, right=10, top=2, bottom=8)
        assert expanded == BBox(5, 18, 60, 88)


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
        """Test slice generation."""
        core = BBox(10, 20, 50, 80)
        halo = BBox(5, 15, 55, 85)
        geometry = TileGeometry(core=core, halo=halo)

        assert geometry.get_slices() == core.get_slices()
        assert geometry.get_halo_slices() == halo.get_slices()
        assert geometry.contains(30, 40) == core.contains(30, 40)


class TestProcessedTile:
    """Test ProcessedTile functionality."""

    def test_single_image(self):
        """Test with single image."""
        bbox = BBox(0, 0, 10, 10)
        geometry = TileGeometry(core=bbox, halo=bbox)
        position = TilePosition(position=(0, 0), edges=BoundaryEdges(True, True, True, True))
        spec = TileSpec(geometry=geometry, position=position)

        image = np.ones((10, 10))
        region = ProcessedTile(spec, image)

        assert region.x_start == 0
        assert region.y_start == 0
        assert len(region.image_data) == 1

    def test_multiple_images(self):
        """Test with multiple images."""
        bbox = BBox(0, 0, 10, 10)
        geometry = TileGeometry(core=bbox, halo=bbox)
        position = TilePosition(position=(0, 0), edges=BoundaryEdges(True, True, True, True))
        spec = TileSpec(geometry=geometry, position=position)

        images = [np.ones((10, 10)), np.zeros((10, 10))]
        region = ProcessedTile(spec, images)

        assert len(region.image_data) == 2

    def test_core_extraction(self):
        """Test core image extraction."""
        core = BBox(2, 2, 8, 8)
        halo = BBox(0, 0, 10, 10)
        geometry = TileGeometry(core=core, halo=halo)
        position = TilePosition(position=(0, 0), edges=BoundaryEdges(True, True, True, True))
        spec = TileSpec(geometry=geometry, position=position)

        image = np.arange(100).reshape(10, 10)
        region = ProcessedTile(spec, image)

        core_images = region.only_core_image()
        assert len(core_images) == 1
        assert core_images[0].shape == (6, 6)


class TestUtilities:
    """Test utility functions."""

    def test_new_image2d(self):
        """Test image creation."""
        img = new_image2d((100, 200))
        assert img.shape == (100, 200)
        assert img.dtype == np.float32

        img_int = new_image2d((50, 50), dtype=np.int32)
        assert img_int.dtype == np.int32
