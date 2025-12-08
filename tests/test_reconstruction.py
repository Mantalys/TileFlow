"""Tests for image reconstruction."""

import numpy as np
import pytest

from tileflow.core import BBox, BoundaryEdges, ProcessedTile, TileGeometry, TilePosition, TileSpec
from tileflow.reconstruction import reconstruct


class TestReconstruction:
    """Test image reconstruction from tiles."""

    def test_simple_reconstruction(self):
        """Test reconstruction of 2x2 grid."""
        # Create 4 regions for a 2x2 grid of a 100x100 image
        regions = []

        for row in range(2):
            for col in range(2):
                # Define core and halo (same for simplicity)
                x0, y0 = col * 50, row * 50
                x1, y1 = x0 + 50, y0 + 50

                core = BBox(x0, y0, x1, y1)
                halo = core  # No overlap for this test

                edges = BoundaryEdges(
                    left=(col == 0), right=(col == 1), top=(row == 0), bottom=(row == 1)
                )

                geometry = TileGeometry(core=core, halo=halo)
                position = TilePosition(position=(row, col), edges=edges)
                spec = TileSpec(geometry=geometry, position=position)

                # Create test data - fill with unique value
                data = np.full((50, 50), row * 2 + col, dtype=np.float32)
                region = ProcessedTile(spec, data)
                regions.append(region)

        # Reconstruct
        result = reconstruct(regions)

        assert len(result) == 1  # Single channel
        assert result[0].shape == (100, 100)

        # Check each quadrant has correct value
        assert np.all(result[0][0:50, 0:50] == 0)  # top-left
        assert np.all(result[0][0:50, 50:100] == 1)  # top-right
        assert np.all(result[0][50:100, 0:50] == 2)  # bottom-left
        assert np.all(result[0][50:100, 50:100] == 3)  # bottom-right

    def test_reconstruction_with_multiple_channels(self):
        """Test reconstruction with multiple output channels."""
        # Single region with 3 channels
        core = BBox(0, 0, 50, 50)
        halo = core
        edges = BoundaryEdges(True, True, True, True)

        geometry = TileGeometry(core=core, halo=halo)
        position = TilePosition(position=(0, 0), edges=edges)
        spec = TileSpec(geometry=geometry, position=position)

        # Create 3-channel data
        data = [np.ones((50, 50)) * 1, np.ones((50, 50)) * 2, np.ones((50, 50)) * 3]
        region = ProcessedTile(spec, data)

        result = reconstruct([region])

        assert len(result) == 3
        for i, channel in enumerate(result):
            assert channel.shape == (50, 50)
            assert np.all(channel == i + 1)

    def test_reconstruction_validation(self):
        """Test reconstruction validation."""
        # Create region without right edge
        core = BBox(0, 0, 50, 50)
        edges = BoundaryEdges(True, False, True, True)  # No right edge

        geometry = TileGeometry(core=core, halo=core)
        position = TilePosition(position=(0, 0), edges=edges)
        spec = TileSpec(geometry=geometry, position=position)

        data = np.ones((50, 50))
        region = ProcessedTile(spec, data)

        with pytest.raises(ValueError, match="right edge"):
            reconstruct([region])

    def test_reconstruction_with_overlap(self):
        """Test reconstruction handling overlap correctly."""
        # Create 2 overlapping regions
        regions = []

        # First region: 0-60 with core 0-50
        core1 = BBox(0, 0, 50, 50)
        halo1 = BBox(0, 0, 60, 50)  # Extended right
        edges1 = BoundaryEdges(True, False, True, True)

        geometry1 = TileGeometry(core=core1, halo=halo1)
        position1 = TilePosition(position=(0, 0), edges=edges1)
        spec1 = TileSpec(geometry=geometry1, position=position1)

        data1 = np.ones((50, 60))  # Halo size
        region1 = ProcessedTile(spec1, data1)
        regions.append(region1)

        # Second region: 40-100 with core 50-100
        core2 = BBox(50, 0, 100, 50)
        halo2 = BBox(40, 0, 100, 50)  # Extended left
        edges2 = BoundaryEdges(False, True, True, True)

        geometry2 = TileGeometry(core=core2, halo=halo2)
        position2 = TilePosition(position=(0, 1), edges=edges2)
        spec2 = TileSpec(geometry=geometry2, position=position2)

        data2 = np.ones((50, 60)) * 2  # Different value
        region2 = ProcessedTile(spec2, data2)
        regions.append(region2)

        # Reconstruct
        result = reconstruct(regions)

        assert result[0].shape == (50, 100)
        # Core regions should have their respective values
        assert np.all(result[0][0:50, 0:50] == 1)
        assert np.all(result[0][0:50, 50:100] == 2)

    def test_empty_regions_handling(self):
        """Test reconstruction handles empty regions gracefully."""
        # Create one valid region
        core = BBox(0, 0, 50, 50)
        edges = BoundaryEdges(True, True, True, True)

        geometry = TileGeometry(core=core, halo=core)
        position = TilePosition(position=(0, 0), edges=edges)
        spec = TileSpec(geometry=geometry, position=position)

        data = np.ones((50, 50))
        region = ProcessedTile(spec, data)

        # Set image_data to None to simulate empty region
        region.image_data = None

        # Should handle gracefully - though this creates invalid state
        # In practice, this tests the None check in reconstruction
        with pytest.raises(ValueError, match="valid image data"):  # Expected due to None data
            reconstruct([region])
