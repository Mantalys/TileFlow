from typing import NamedTuple, Optional

import numpy as np

# Support both 2D and multi-dimensional images
Image2D = np.ndarray
ImageData = np.ndarray  # More general type for multi-dimensional data
TupleInt2 = tuple[int, int]


class BoundaryEdges(NamedTuple):
    """Immutable representation of tile boundary flags.

    Indicates which edges of a tile are at the boundary of the image grid.
    Using NamedTuple keeps instances compact and fast to create/compare.
    """

    left: bool
    right: bool
    top: bool
    bottom: bool


class BBox(NamedTuple):
    """Immutable bounding box [x0:x1, y0:y1] with geometric operations.

    Using NamedTuple for memory efficiency and fast operations.
    """

    x0: int
    y0: int
    x1: int
    y1: int

    # Convenience
    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def get_slices(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)

    @classmethod
    def from_size(cls, y: int, x: int, h: int, w: int) -> "BBox":
        return cls(x, y, x + w, y + h)

    def translate(self, dy: int = 0, dx: int = 0) -> "BBox":
        return BBox(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)

    def clamp_to(self, H: int, W: int) -> "BBox":
        x0 = max(0, min(self.x0, W))
        y0 = max(0, min(self.y0, H))
        x1 = max(0, min(self.x1, W))
        y1 = max(0, min(self.y1, H))
        x0 = min(x0, x1)
        y0 = min(y0, y1)
        return BBox(x0, y0, x1, y1)

    def contains(self, x: int, y: int) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def intersects(self, other: "BBox") -> bool:
        return not (
            self.x1 <= other.x0 or self.x0 >= other.x1 or self.y1 <= other.y0 or self.y0 >= other.y1
        )

    def intersection(self, other: "BBox") -> Optional["BBox"]:
        if not self.intersects(other):
            return None
        return BBox(
            max(self.x0, other.x0),
            max(self.y0, other.y0),
            min(self.x1, other.x1),
            min(self.y1, other.y1),
        )

    def expand(self, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> "BBox":
        return BBox(self.x0 - left, self.y0 - top, self.x1 + right, self.y1 + bottom)


class TileGeometry(NamedTuple):
    """Tile geometry specification with core and halo regions.

    The core region is the area of interest for reconstruction,
    while the halo includes overlap areas for seamless processing.
    """

    core: BBox
    halo: BBox

    def get_slices(self) -> tuple[slice, slice]:
        return self.core.get_slices()

    def get_halo_slices(self) -> tuple[slice, slice]:
        return self.halo.get_slices()

    def contains(self, x: int, y: int) -> bool:
        return self.core.contains(x, y)


class TilePosition(NamedTuple):
    """Position of a tile in the processing grid.

    Contains both grid coordinates and boundary edge information.
    """

    position: tuple[int, int]  # (row, column) in the grid
    edges: BoundaryEdges


class TileSpec(NamedTuple):
    """Complete specification of a tile in the processing grid.

    Combines geometry (core and halo bounding boxes) with position
    information for comprehensive tile description.
    """

    geometry: TileGeometry
    position: TilePosition

    def get_slices(self) -> tuple[slice, slice]:
        return self.geometry.get_slices()

    def get_halo_slices(self) -> tuple[slice, slice]:
        return self.geometry.get_halo_slices()

    def contains(self, x: int, y: int) -> bool:
        return self.geometry.contains(x, y)


class ProcessedTile:
    """Container for processed image data associated with a specific tile.

    Stores both the tile specification and the processed image data,
    enabling proper reconstruction and spatial referencing.
    """

    def __init__(self, tile_spec: TileSpec, image_data: list[Image2D] | Image2D) -> None:
        """Initialize processed tile.

        Parameters
        ----------
        tile_spec : TileSpec
            Specification of the tile
        image_data : list[Image2D] | Image2D
            Processed image data for this tile
        """
        self.tile_spec = tile_spec
        self.image_data: list[Image2D] = (
            image_data if isinstance(image_data, list) else [image_data]
        )

    @property
    def x_start(self) -> int:
        return self.tile_spec.geometry.halo.x0

    @property
    def y_start(self) -> int:
        return self.tile_spec.geometry.halo.y0

    @property
    def core_bbox(self) -> BBox:
        return self.tile_spec.geometry.core
