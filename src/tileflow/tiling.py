from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from tileflow.core import BBox, BoundaryEdges, TileGeometry, TilePosition, TileSpec, TupleInt2


def _edges_from_index(index: TupleInt2, grid_shape: TupleInt2) -> BoundaryEdges:
    """Determine which edges are at the boundary for a given grid index."""
    row, col = index
    n_rows, n_cols = grid_shape
    return BoundaryEdges(
        left=(col == 0),
        right=(col == n_cols - 1),
        top=(row == 0),
        bottom=(row == n_rows - 1),
    )


@dataclass(frozen=True)
class GridSpec:
    """Specification for grid-based image tiling.

    Parameters
    ----------
    size : TupleInt2
        Size of each tile (height, width)
    overlap : TupleInt2
        Overlap/padding around each region (height, width)
    origin : TupleInt2, default=(0, 0)
        Origin offset for the grid (y, x)
    """

    size: TupleInt2  # (height, width) - size of each region
    overlap: TupleInt2  # (height, width) - overlap/padding around region
    origin: TupleInt2 = (0, 0)  # (y, x) origin offset

    @lru_cache(maxsize=128)
    def grid_shape(self, shape: TupleInt2) -> TupleInt2:
        """Calculate grid dimensions (rows, cols) for the given image shape."""
        H, W = shape[:2]
        n_rows = H // self.size[0] + (1 if H % self.size[0] > self.size[0] // 2 else 0)
        n_cols = W // self.size[1] + (1 if W % self.size[1] > self.size[1] // 2 else 0)
        return (n_rows, n_cols)

    def build_grid(self, region_height: int, region_width: int) -> Iterator[TileSpec]:
        """Generate tile specifications for processing the image."""
        grid_shape = self.grid_shape((region_height, region_width))
        rh, rw = self.size
        for row in range(grid_shape[0]):
            for col in range(grid_shape[1]):
                edges = _edges_from_index((row, col), grid_shape)

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
                tile_x_end = min(tile_x_end, region_width)
                if edges.right and tile_x_end < region_width:
                    tile_x_end = region_width

                if not edges.top:
                    tile_y_start -= self.overlap[0]
                    tile_height += self.overlap[0]
                if not edges.bottom:
                    tile_height += self.overlap[0]
                tile_y_end = tile_y_start + tile_height
                tile_y_end = min(tile_y_end, region_height)
                if edges.bottom and tile_y_end < region_height:
                    tile_y_end = region_height

                # Calculate region bounds (area of interest for reconstruction)
                region_x_start = self.origin[1] if edges.left else tile_x_start + self.overlap[1]
                region_x_end = region_width if edges.right else tile_x_end - self.overlap[1]

                region_y_start = self.origin[0] if edges.top else tile_y_start + self.overlap[0]
                region_y_end = region_height if edges.bottom else tile_y_end - self.overlap[0]

                region_bbox = BBox(region_x_start, region_y_start, region_x_end, region_y_end)
                tile_bbox = BBox(tile_x_start, tile_y_start, tile_x_end, tile_y_end)

                geometry = TileGeometry(core=region_bbox, halo=tile_bbox)
                position = TilePosition(position=(row, col), edges=edges)
                yield TileSpec(geometry=geometry, position=position)
