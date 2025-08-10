# from tileflow.streamer import ImageStreamer, LargeImageStreamer, ZarrStreamer
from typing import NamedTuple, Tuple, Optional, List

import numpy as np

# Type alias for a 2D array, used for image data
Image2D = np.ndarray


def new_image2d(shape: Tuple[int, int], dtype: np.dtype = np.float32) -> Image2D:
    """Create a new 2D image with the specified shape and dtype."""
    return np.zeros(shape, dtype=dtype)


class RegionEdges(NamedTuple):
    """Immutable, ultra-light representation of the edges of a region.

    Using NamedTuple keeps instances as compact as tuples and very fast to
    create/compare. Methods return new RegionEdges objects.
    """

    left: bool
    right: bool
    top: bool
    bottom: bool


class BBox(NamedTuple):
    """Immutable, ultra-light rectangle [y0:y1, x0:x1].

    Using NamedTuple keeps instances as compact as tuples and very fast to
    create/compare. Methods return new BBox objects.
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
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    def get_slices(self) -> Tuple[slice, slice]:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)

    @classmethod
    def from_size(cls, y: int, x: int, h: int, w: int) -> "BBox":
        return cls(y, x, y + h, x + w)

    def translate(self, dy: int = 0, dx: int = 0) -> "BBox":
        return BBox(self.y0 + dy, self.x0 + dx, self.y1 + dy, self.x1 + dx)

    def clamp_to(self, H: int, W: int) -> "BBox":
        x0 = max(0, min(self.x0, W))
        y0 = max(0, min(self.y0, H))
        x1 = max(0, min(self.x1, W))
        y1 = max(0, min(self.y1, H))
        if x1 < x0:
            x0 = x1
        if y1 < y0:
            y0 = y1
        return BBox(x0, y0, x1, y1)

    def contains(self, x: int, y: int) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def intersects(self, other: "BBox") -> bool:
        return not (
            self.x1 <= other.x0
            or self.x0 >= other.x1
            or self.y1 <= other.y0
            or self.y0 >= other.y1
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

    def expand(
        self, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0
    ) -> "BBox":
        return BBox(self.x0 - left, self.y0 - top, self.x1 + right, self.y1 + bottom)


class RegionGeometry(NamedTuple):
    core: BBox
    halo: BBox

    def get_slices(self) -> Tuple[slice, slice]:
        return self.core.get_slices()

    def get_halo_slices(self) -> Tuple[slice, slice]:
        return self.halo.get_slices()

    def contains(self, x: int, y: int) -> bool:
        return self.core.contains(x, y)


class RegionPosition(NamedTuple):
    """Position of a region in the grid, defined by its core and halo BBoxes."""

    position: Tuple[int, int]  # (row, column) in the grid
    edges: RegionEdges


class RegionSpec(NamedTuple):
    """Specification of a region in the grid."""

    geometry: RegionGeometry
    position: RegionPosition

    def get_slices(self) -> Tuple[slice, slice]:
        return self.geometry.get_slices()

    def get_halo_slices(self) -> Tuple[slice, slice]:
        return self.geometry.get_halo_slices()

    def contains(self, x: int, y: int) -> bool:
        return self.geometry.contains(x, y)


class RegionImage:
    def __init__(self, region_spec: RegionSpec, image_data: List[Image2D] | Image2D):
        self.region_spec = region_spec
        self.image_data: List[Image2D] = (
            image_data if isinstance(image_data, list) else [image_data]
        )

    def only_core_image(self) -> List[Image2D]:
        """Returns the core part of the image data."""
        core_bbox = self.region_spec.geometry.core
        halo_bbox = self.region_spec.geometry.halo
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

        return [
            img[
                core_bbox.y0 - halo_bbox.y0 : core_bbox.y1 - halo_bbox.y0,
                core_bbox.x0 - halo_bbox.x0 : core_bbox.x1 - halo_bbox.x0,
            ]
            for img in self.image_data
        ]
