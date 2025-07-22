"""
Stamps per-tile masks into a chunk-level feature map with boundary clamping.
"""

import numpy as np
from typing import Tuple
from Streaming.utils import BBox


def stamp(
    fmap: np.ndarray, local: np.ndarray, core: BBox, offset: Tuple[int, int]
) -> None:
    cx, cy = offset
    x0, y0, x1, y1 = core
    # Destination in fmap
    dst_x0 = max(x0 - cx, 0)
    dst_y0 = max(y0 - cy, 0)
    dst_x1 = min(x1 - cx, fmap.shape[1])
    dst_y1 = min(y1 - cy, fmap.shape[0])
    # Source in local
    src_x0 = dst_x0 - (x0 - cx)
    src_y0 = dst_y0 - (y0 - cy)
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    fmap[dst_y0:dst_y1, dst_x0:dst_x1] = local[src_y0:src_y1, src_x0:src_x1]
