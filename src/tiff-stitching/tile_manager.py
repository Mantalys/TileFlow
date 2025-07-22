"""
Lazy loader and cache manager for image tiles (with optional halo padding).
"""

import numpy as np
from typing import Dict
from tifffile import TiffFile
from Streaming.utils import BBox, Shape
from skimage.exposure import rescale_intensity
import cv2

class TileManager:
    _cache: Dict[BBox, np.ndarray] = {}
    _tiff: TiffFile = None
    _level: int = 0
    _shape: Shape = None
    _array: np.ndarray = None

    @classmethod
    def open(cls, path: str) -> None:
        """
        Initialize file handles for TIFF.
        """
        cls._tiff = TiffFile(path) # TODO replace by imread + zarr
        cls._cache.clear()
        cls._level = 0

    @classmethod
    def read_raw(cls, bbox: BBox) -> np.ndarray:
        """
        Read raw image data for `bbox` from WSI.
        """
        x0, y0, x1, y1 = bbox
        if cls._tiff is not None:
            return cls._array[y0:y1, x0:x1]
        raise RuntimeError("TileManager not initialized with WSI path")

    @classmethod
    def cache(cls, tile: Dict, data: np.ndarray) -> None:
        """
        Store raw or padded tile array in memory.
        """
        cls._cache[tile["bbox"]] = data

    @classmethod
    def loaded(cls, tile: Dict) -> bool:
        return tile["bbox"] in cls._cache

    @classmethod
    def get(cls, tile: Dict) -> np.ndarray:
        """
        Retrieve cached tile (raw + pad applied).
        """
        return cls._cache[tile["bbox"]]

    @classmethod
    def unload(cls, tile: Dict) -> None:
        """
        Remove tile from cache to free memory.
        """
        cls._cache.pop(tile["bbox"], None)


class TileManagerFCYX(TileManager):
    _frames: int = 0
    _frame: int = 0

    @classmethod
    def open(cls, path: str) -> None:
        super().open(path)
        array = cls._tiff.asarray()[:, 3, 400:, 256:]  # CD8 channel
        F, H, W = array.shape
        array = array[cls._frame]  # take the first frame to get shape
        cls._array = rescale_intensity(array, in_range=(210, 500), out_range=(0, 1)).astype(np.float32)
        cls._frames = F
        cls._shape = (1, H, W)
        cv2.imwrite(
            "first_frame.png", (cls._array * 255).astype(np.uint8)
        )

    @classmethod
    def next_frame(cls) -> bool:
        """
        Move to the next frame in the time series.
        Returns True if there are more frames, False otherwise.
        """
        cls._frame += 1
        if cls._frame < cls._frames:
            array = cls._tiff.asarray()[cls._frame, 3, 400:, 256:]
            cls._array = rescale_intensity(array, in_range=(210, 500), out_range=(0, 1)).astype(np.float32)
            cls._cache.clear()
            return True
        return False

   

class TileManagerCYX(TileManager):
    @classmethod
    def open(cls, path: str) -> None:
        super().open(path)

        # TODO: remove hardcoded, connect with app
        cls._array = cls._tiff.asarray()[6]  # DAPI channel
        cls._array = rescale_intensity(cls._array, in_range=(1, 10)).astype(
            np.float32
        )  # user parameter
        # cls._array = cls._array[:720, :720] # crop to process faster
        H, W = cls._array.shape
        cls._shape = (1, H, W)  # (DAPI, H, W)
        # end of TODO
