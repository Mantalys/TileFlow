from typing import Optional
import numpy as np
from dataclasses import dataclass

# Support both 2D and multi-dimensional images
Image2D = np.ndarray
ImageData = np.ndarray  # More general type for multi-dimensional data
TupleInt2 = tuple[int, int]


@dataclass(frozen=True, slots=True)
class BoundaryEdges:
    """Immutable representation of tile boundary flags.

    Indicates which edges of a tile are at the boundary of the image grid.
    Using NamedTuple keeps instances compact and fast to create/compare.
    """

    left: bool
    right: bool
    top: bool
    bottom: bool


@dataclass(frozen=True, slots=True)
class BBox:
    """Immutable bounding box [x0:x1, y0:y1] with geometric operations.

    Using NamedTuple for memory efficiency and fast operations.
    """

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self):
        if self.x0 < 0 or self.y0 < 0 or self.x1 < 0 or self.y1 < 0:
            raise ValueError(
                "Invalid BBox: expected non-negative coordinates, "
                f"got ({self.x0}, {self.y0}, {self.x1}, {self.y1})"
            )
        if self.x0 > self.x1 or self.y0 > self.y1:
            raise ValueError(
                "Invalid BBox: expected x0 <= x1 and y0 <= y1, "
                f"got ({self.x0}, {self.y0}, {self.x1}, {self.y1})"
            )

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
        if self.x0 + dx < 0:
            dx = -self.x0
        if self.y0 + dy < 0:
            dy = -self.y0
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

    def encloses(self, other: "BBox") -> bool:
        return (
            self.x0 <= other.x0
            and self.y0 <= other.y0
            and other.x1 <= self.x1
            and other.y1 <= self.y1
        )

    def intersects(self, other: "BBox") -> bool:
        return max(self.x0, other.x0) < min(self.x1, other.x1) and max(self.y0, other.y0) < min(
            self.y1, other.y1
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
        if self.x0 - left < 0:
            left = 0
        else:
            left = self.x0 - left
        if self.y0 - top < 0:
            top = 0
        else:
            top = self.y0 - top

        return BBox(left, top, self.x1 + right, self.y1 + bottom)


@dataclass(frozen=True, slots=True)
class TileGeometry:
    """Tile geometry specification with core and halo regions.

    The core region is the area of interest for reconstruction,
    while the halo includes overlap areas for seamless processing.
    """

    core: BBox
    halo: BBox

    def __post_init__(self) -> None:
        if not self.halo.encloses(self.core):
            raise ValueError(f"Core BBox {self.core} must be enclosed by halo BBox {self.halo}")

    @property
    def core_in_halo(self) -> BBox:
        return BBox(
            x0=self.core.x0 - self.halo.x0,
            y0=self.core.y0 - self.halo.y0,
            x1=self.core.x1 - self.halo.x0,
            y1=self.core.y1 - self.halo.y0,
        )

    def get_slices(self) -> tuple[slice, slice]:
        return self.core.get_slices()

    def get_halo_slices(self) -> tuple[slice, slice]:
        return self.halo.get_slices()

    def core_in_halo_slices(self) -> tuple[slice, slice]:
        return self.core_in_halo.get_slices()

    def contains(self, x: int, y: int) -> bool:
        return self.core.contains(x, y)


@dataclass(frozen=True, slots=True)
class GridIndex:
    row: int
    column: int

    def __post_init__(self) -> None:
        if self.row < 0 or self.column < 0:
            raise ValueError("Grid indices must be non-negative")


@dataclass(frozen=True, slots=True)
class TileSpec:
    """Complete specification of a tile in the processing grid.

    Combines geometry (core and halo bounding boxes) with position
    information for comprehensive tile description.
    """

    geometry: TileGeometry
    position: GridIndex  # (row, column) in the grid
    edges: BoundaryEdges

    def get_slices(self) -> tuple[slice, slice]:
        return self.geometry.get_slices()

    def get_halo_slices(self) -> tuple[slice, slice]:
        return self.geometry.get_halo_slices()

    def contains(self, x: int, y: int) -> bool:
        return self.geometry.contains(x, y)

    @property
    def row(self) -> int:
        return self.position.row

    @property
    def column(self) -> int:
        return self.position.column


class ProcessedTile:
    """Container for processed image data associated with a specific tile.

    Stores both the tile specification and the processed image data,
    enabling proper reconstruction and spatial referencing.
    """

    def __init__(self, tile_spec: TileSpec, data: np.ndarray) -> None:
        """Initialize processed tile.

        Parameters
        ----------
        tile_spec : TileSpec
            Specification of the tile
        data : np.ndarray
            Processed data for this tile
        """
        self.tile_spec = tile_spec
        if type(data) is not np.ndarray:
            data = np.asarray(data)
        self._data = data

    def only_core_data(self) -> np.ndarray:
        """Extract the core part of the processed tile data."""

        slice_y, slice_x = self.tile_spec.geometry.core_in_halo_slices()

        match self._data.ndim:
            case 1:
                # Single 1D array
                return self._data
            case 2:
                # Single 2D array
                return self._data[slice_y, slice_x]
            case 3:
                # 3D array (C, H, W) - extract spatial region from all channels
                return self._data[:, slice_y, slice_x]
            case _:
                raise ValueError(f"Unsupported data dimension: {self._data.ndim}")
