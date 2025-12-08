"""Tests for tiling and grid functionality."""

from tileflow.core import BBox, BoundaryEdges
from tileflow.tiling import GridSpec


class TestGridSpec:
    """Test GridSpec functionality."""

    def test_basic_grid_shape(self):
        """Test grid shape calculation."""
        grid = GridSpec(size=(100, 100), overlap=(0, 0))
        shape = grid.grid_shape((500, 600))
        assert shape == (5, 6)

    def test_grid_shape_with_remainder(self):
        """Test grid shape with partial tiles."""
        grid = GridSpec(size=(100, 100), overlap=(0, 0))
        # 550 // 100 = 5, remainder 50 == 50 (100//2), so 5 tiles (not > threshold)
        shape = grid.grid_shape((550, 550))
        assert shape == (5, 5)

        # 530 // 100 = 5, remainder 30 < 50 (100//2), so 5 tiles
        shape = grid.grid_shape((530, 530))
        assert shape == (5, 5)

    def test_edges_from_index(self):
        """Test edge detection."""
        grid = GridSpec(size=(100, 100), overlap=(0, 0))

        # Corner cases
        edges = grid.edges_from_index((0, 0), (3, 3))
        assert edges == BoundaryEdges(left=True, right=False, top=True, bottom=False)

        edges = grid.edges_from_index((2, 2), (3, 3))
        assert edges == BoundaryEdges(left=False, right=True, top=False, bottom=True)

        # Middle case
        edges = grid.edges_from_index((1, 1), (3, 3))
        assert edges == BoundaryEdges(left=False, right=False, top=False, bottom=False)

    def test_build_grid_simple(self):
        """Test simple grid building."""
        grid = GridSpec(size=(50, 50), overlap=(0, 0))
        regions = list(grid.build_grid((100, 100)))

        assert len(regions) == 4  # 2x2 grid

        # Check first region
        first = regions[0]
        assert first.geometry.core == BBox(0, 0, 50, 50)
        assert first.geometry.halo == BBox(0, 0, 50, 50)
        assert first.position.position == (0, 0)
        assert first.position.edges.left is True
        assert first.position.edges.top is True

    def test_build_grid_with_overlap(self):
        """Test grid building with overlap."""
        grid = GridSpec(size=(50, 50), overlap=(10, 10))
        regions = list(grid.build_grid((100, 100)))

        assert len(regions) == 4

        # First region should have expanded halo
        first = regions[0]
        assert first.geometry.core == BBox(0, 0, 50, 50)
        assert first.geometry.halo == BBox(0, 0, 60, 60)  # No left/top expansion for edge

        # Second region (0, 1)
        second = regions[1]
        assert second.geometry.core == BBox(50, 0, 100, 50)
        assert second.geometry.halo == BBox(40, 0, 100, 60)  # Left expansion, no right for edge

    def test_origin_offset(self):
        """Test grid with origin offset."""
        grid = GridSpec(size=(50, 50), overlap=(0, 0), origin=(10, 20))
        regions = list(grid.build_grid((100, 100)))

        first = regions[0]
        # Should start at origin offset
        assert first.geometry.core.x0 == 20
        assert first.geometry.core.y0 == 10

    def test_grid_coverage(self):
        """Test that grid covers entire image."""
        grid = GridSpec(size=(30, 30), overlap=(5, 5))
        regions = list(grid.build_grid((100, 100)))

        # Collect all core regions
        covered_pixels = set()
        for region in regions:
            core = region.geometry.core
            for y in range(core.y0, core.y1):
                for x in range(core.x0, core.x1):
                    covered_pixels.add((y, x))

        # Should cover all pixels
        expected_pixels = set((y, x) for y in range(100) for x in range(100))
        assert covered_pixels == expected_pixels
