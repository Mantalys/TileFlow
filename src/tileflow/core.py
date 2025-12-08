"""Core data structures for image tiling and spatial region management.

This module provides the fundamental building blocks for TileFlow's tiling system:
- BBox: Immutable bounding box operations
- RegionEdges: Boundary detection for grid edges  
- RegionGeometry: Core and halo region specifications
- RegionSpec: Complete tile specification with position and geometry
- RegionImage: Container for processed tile data

All structures use NamedTuple for performance and immutability.
"""

from typing import NamedTuple, Optional

import numpy as np

# Support both 2D and multi-dimensional images
Image2D = np.ndarray
ImageData = np.ndarray  # More general type for multi-dimensional data


def new_image2d(shape: tuple[int, int], dtype: np.dtype = np.float32) -> Image2D:
    """Create a new 2D image with the specified shape and dtype."""
    return np.zeros(shape, dtype=dtype)


def new_image(shape: tuple[int, ...], dtype: np.dtype = np.float32) -> ImageData:
    """Create a new image with the specified shape and dtype.

    Supports both 2D (H, W) and 3D (C, H, W) images.
    """
    return np.zeros(shape, dtype=dtype)


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

    def only_core_image(self) -> list[Image2D]:
        """Extract the core part of the processed tile data."""
        core_bbox = self.tile_spec.geometry.core
        halo_bbox = self.tile_spec.geometry.halo
        # Crop the image data corresponding to the core bbox
        if self.image_data is None:
            return None
        if halo_bbox.x0 >= halo_bbox.x1 or halo_bbox.y0 >= halo_bbox.y1:
            return np.zeros((0, 0), dtype=self.image_data.dtype)
        if core_bbox.x0 >= core_bbox.x1 or core_bbox.y0 >= core_bbox.y1:
            return np.zeros((0, 0), dtype=self.image_data.dtype)
        # Calculate the crop indices
        # Note: We assume the image_data is large enough to accommodate the core bbox
        if (
            core_bbox.x0 < halo_bbox.x0
            or core_bbox.x1 > halo_bbox.x1
            or core_bbox.y0 < halo_bbox.y0
            or core_bbox.y1 > halo_bbox.y1
        ):
            raise ValueError("Core bbox must be within the halo bbox.")
        # Crop the image data to get the core part
        # This assumes the image_data is large enough to accommodate the core bbox

        # Calculate relative slice indices
        rel_y0 = core_bbox.y0 - halo_bbox.y0
        rel_y1 = core_bbox.y1 - halo_bbox.y0
        rel_x0 = core_bbox.x0 - halo_bbox.x0
        rel_x1 = core_bbox.x1 - halo_bbox.x0

        # Handle different data formats
        if isinstance(self.image_data, list):
            # List of arrays (could be 2D or 3D)
            result = []
            for img in self.image_data:
                if img.ndim == 2:
                    result.append(img[rel_y0:rel_y1, rel_x0:rel_x1])
                elif img.ndim == 3:
                    result.append(img[:, rel_y0:rel_y1, rel_x0:rel_x1])
                else:
                    # Higher dimensions - extract spatial region from last 2 dims
                    spatial_slice = tuple(
                        [slice(None)] * (img.ndim - 2)
                        + [slice(rel_y0, rel_y1), slice(rel_x0, rel_x1)]
                    )
                    result.append(img[spatial_slice])
            return result
        # Single array (could be 2D or 3D)
        elif self.image_data.ndim == 2:
            # Single 2D array
            return [self.image_data[rel_y0:rel_y1, rel_x0:rel_x1]]
        elif self.image_data.ndim == 3:
            # 3D array (C, H, W) - extract spatial region from all channels
            extracted = self.image_data[:, rel_y0:rel_y1, rel_x0:rel_x1]
            return [extracted]
        else:
            # Higher dimensions - extract spatial region from last 2 dims
            spatial_slice = tuple(
                [slice(None)] * (self.image_data.ndim - 2)
                + [slice(rel_y0, rel_y1), slice(rel_x0, rel_x1)]
            )
            return [self.image_data[spatial_slice]]
