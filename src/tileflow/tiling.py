from tileflow.core import RegionSpec, RegionGeometry, RegionPosition, BBox, RegionEdges
from typing import Tuple, Iterator
from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class GridSpec:
    size: Tuple[int, int]  # (height, width), raw size of the region
    halo: Tuple[
        int, int
    ]  # (height, width), size of the halo around the region, used for overlap
    origin: Tuple[int, int] = (0, 0)  # (y, x) origin of the grid, default is (0, 0)

    def grid_shape(self, shape: Tuple[int, int]) -> Tuple[int, int]:
        H, W = shape[:2]
        n_rows = H // self.size[0] + (1 if H % self.size[0] > self.size[0] // 2 else 0)
        n_cols = W // self.size[1] + (1 if W % self.size[1] > self.size[1] // 2 else 0)
        return (n_rows, n_cols)

    def build_grid(self, image_shape: Tuple[int, int]) -> Iterator[RegionSpec]:
        grid_shape = self.grid_shape(image_shape)
        rh, rw = self.size
        for row in range(grid_shape[0]):
            for col in range(grid_shape[1]):
                edges = self.edges_from_index((row, col), grid_shape)
                x_start = col * rw + self.origin[1]
                y_start = row * rh + self.origin[0]
                width = rw
                height = rh

                # precomputed chunk relative position, manage horizontal first
                if not edges.left:
                    x_start -= self.halo[1]  # shift to the left to create overlap
                    width += self.halo[1]  # increase width to account for overlap
                if not edges.right:
                    width += self.halo[1]  # increase width to account for overlap
                x_end = x_start + width
                if x_end > image_shape[1]:
                    x_end = image_shape[1]
                if edges.right and x_end < image_shape[1]:
                    x_end = image_shape[1]

                if edges.left:
                    core_x_start = 0
                else:
                    core_x_start = x_start + self.halo[1]
                if edges.right:
                    core_x_end = image_shape[1]
                else:
                    core_x_end = x_end - self.halo[1]

                # now we do the same for the vertical position
                if not edges.top:
                    y_start -= self.halo[0]
                    height += self.halo[0]
                if not edges.bottom:
                    height += self.halo[0]
                y_end = y_start + height
                if y_end > image_shape[0]:
                    y_end = image_shape[0]
                if edges.bottom and y_end < image_shape[0]:
                    y_end = image_shape[0]
                if edges.top:
                    core_y_start = 0
                else:
                    core_y_start = y_start + self.halo[0]
                if edges.bottom:
                    core_y_end = image_shape[0]
                else:
                    core_y_end = y_end - self.halo[0]
                core_bbox = BBox(core_x_start, core_y_start, core_x_end, core_y_end)
                halo_bbox = BBox(x_start, y_start, x_end, y_end)

                geometry = RegionGeometry(core=core_bbox, halo=halo_bbox)
                position = RegionPosition(position=(row, col), edges=edges)
                yield RegionSpec(geometry=geometry, position=position)

    def edges_from_index(
        self, yx: Tuple[int, int], grid_shape: Tuple[int, int]
    ) -> RegionEdges:
        nrows, ncols = grid_shape
        y, x = yx
        left = x == 0
        top = y == 0
        right = x == ncols - 1
        bottom = y == nrows - 1
        return RegionEdges(left, right, top, bottom)
