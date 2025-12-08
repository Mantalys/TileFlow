"""Grid-based tiling system for processing large images efficiently.

This module implements TileFlow's core tiling strategy:
- GridSpec: Defines tile size, overlap, and grid origin
- Automatic edge detection and boundary handling
- Lazy grid generation for memory efficiency
- Support for arbitrary image dimensions

The grid system handles overlap calculations automatically to ensure
seamless reconstruction without processing artifacts.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

from tileflow.core import BBox, BoundaryEdges, TileGeometry, TilePosition, TileSpec


@dataclass(frozen=True)
class GridSpec:
    """Specification for grid-based image tiling.

    Parameters
    ----------
    size : tuple[int, int]
        Size of each tile (height, width)
    overlap : tuple[int, int]
        Overlap/padding around each region (height, width)
    origin : tuple[int, int], default=(0, 0)
        Origin offset for the grid (y, x)
    """

    size: tuple[int, int]  # (height, width) - size of each region
    overlap: tuple[int, int]  # (height, width) - overlap/padding around region
    origin: tuple[int, int] = (0, 0)  # (y, x) origin offset

    @lru_cache(maxsize=128)
    def grid_shape(self, shape: tuple[int, int]) -> tuple[int, int]:
        """Calculate grid dimensions (rows, cols) for the given image shape."""
        H, W = shape[:2]
        n_rows = H // self.size[0] + (1 if H % self.size[0] > self.size[0] // 2 else 0)
        n_cols = W // self.size[1] + (1 if W % self.size[1] > self.size[1] // 2 else 0)
        return (n_rows, n_cols)

    def build_grid(self, image_shape: tuple[int, int]) -> Iterator[TileSpec]:
        """Generate tile specifications for processing the image."""
        grid_shape = self.grid_shape(image_shape)
        rh, rw = self.size
        for row in range(grid_shape[0]):
            for col in range(grid_shape[1]):
                edges = self.edges_from_index((row, col), grid_shape)

                # Calculate base region position
                x_start = col * rw + self.origin[1]
                y_start = row * rh + self.origin[0]
                width = rw
                height = rh

                # Expand to create tile bounds (with overlap)
                tile_x_start = x_start
                tile_y_start = y_start
                tile_width = width
                tile_height = height

                # Add overlap on non-boundary edges
                if not edges.left:
                    tile_x_start -= self.overlap[1]
                    tile_width += self.overlap[1]
                if not edges.right:
                    tile_width += self.overlap[1]
                tile_x_end = tile_x_start + tile_width
                tile_x_end = min(tile_x_end, image_shape[1])
                if edges.right and tile_x_end < image_shape[1]:
                    tile_x_end = image_shape[1]

                if not edges.top:
                    tile_y_start -= self.overlap[0]
                    tile_height += self.overlap[0]
                if not edges.bottom:
                    tile_height += self.overlap[0]
                tile_y_end = tile_y_start + tile_height
                tile_y_end = min(tile_y_end, image_shape[0])
                if edges.bottom and tile_y_end < image_shape[0]:
                    tile_y_end = image_shape[0]

                # Calculate region bounds (area of interest for reconstruction)
                if edges.left:
                    region_x_start = self.origin[1]
                else:
                    region_x_start = tile_x_start + self.overlap[1]
                if edges.right:
                    region_x_end = image_shape[1]
                else:
                    region_x_end = tile_x_end - self.overlap[1]

                if edges.top:
                    region_y_start = self.origin[0]
                else:
                    region_y_start = tile_y_start + self.overlap[0]
                if edges.bottom:
                    region_y_end = image_shape[0]
                else:
                    region_y_end = tile_y_end - self.overlap[0]

                region_bbox = BBox(region_x_start, region_y_start, region_x_end, region_y_end)
                tile_bbox = BBox(tile_x_start, tile_y_start, tile_x_end, tile_y_end)

                geometry = TileGeometry(core=region_bbox, halo=tile_bbox)
                position = TilePosition(position=(row, col), edges=edges)
                yield TileSpec(geometry=geometry, position=position)

    def edges_from_index(
        self, index: tuple[int, int], grid_shape: tuple[int, int]
    ) -> BoundaryEdges:
        """Determine which edges are at the boundary for a given grid index."""
        row, col = index
        n_rows, n_cols = grid_shape
        return BoundaryEdges(
            left=(col == 0),
            right=(col == n_cols - 1),
            top=(row == 0),
            bottom=(row == n_rows - 1),
        )
