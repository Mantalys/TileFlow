import numpy as np
from tileflow.tiling import GridSpec
from tileflow.core import RegionImage
from tileflow.reconstruction import reconstruct
from typing import List


class SliderImage:
    def __init__(self, tile_size, overlap):
        self.tile_size = tile_size
        self.overlap = overlap
        self.tile_function = None

    def set_image(self, image):
        """
        Set the image to be processed.
        """
        self.image = image
        self.grid = self.build_grid(image.shape)

    def build_grid(self, image_shape):
        """
        Build a grid of regions based on the image shape.
        """
        chunk_grid_spec = GridSpec(size=self.tile_size, halo=self.overlap)
        n = np.prod(chunk_grid_spec.grid_shape(image_shape))
        return chunk_grid_spec.build_grid(image_shape), n

    def set_tile_function(self, func):
        """
        Set a processing function to be applied to each tile.
        """
        self.tile_function = func

    def process(self, image, return_reconstruct=True, callbacks=None) -> np.ndarray:
        # self.display_grid_over_image()
        regions: List[RegionImage] = []
        grid, n = self.build_grid(image.shape)
        for i, tile in enumerate(grid):
            tile_np = image[tile.get_halo_slices()]
            tile_output = self.tile_function(tile_np)
            region = RegionImage(region_spec=tile, image_data=tile_output)
            if callbacks:
                for callback in callbacks:
                    callback.on_tile_processed(region, i + 1, n)
            # stitch a tile and its content

            regions.append(region)
        if return_reconstruct:
            return reconstruct(regions)
        return regions


class SliderLargeImage:
    """Streamer for processing large images in chunks. Still an image that can be loaded in memory."""

    def __init__(self, image, chunk_size, tile_size, overlap):
        self.image = image
        self.chunk_size = chunk_size
        self.tile_size = tile_size
        self.overlap = overlap
        self.tile_function = None
        self.chunk_function = None

    def set_tile_function(self, func):
        self.tile_function = func

    def set_chunk_function(self, func):
        """
        Set a processing function to be applied to each chunk.
        """
        self.chunk_function = func

    def build_grid(self, image_shape):
        """
        Build a grid of regions based on the image shape.
        """
        chunk_grid_spec = GridSpec(size=self.chunk_size, halo=(16, 16))
        n = np.prod(chunk_grid_spec.grid_shape(image_shape))
        return chunk_grid_spec.build_grid(image_shape), n

    def process(self, image, return_reconstruct=True, callbacks=None) -> np.ndarray:
        regions: List[RegionImage] = []
        tile_streamer = SliderImage(tile_size=self.tile_size, overlap=self.overlap)
        tile_streamer.set_tile_function(self.tile_function)
        grid, n = self.build_grid(image.shape)
        for i, chunk in enumerate(grid):
            chunk_np = image[chunk.get_halo_slices()]
            chunk_output = tile_streamer.process(
                chunk_np, return_reconstruct=True, callbacks=callbacks
            )
            if self.chunk_function:
                chunk_output = self.chunk_function(chunk_output)
            region = RegionImage(region_spec=chunk, image_data=chunk_output)
            if callbacks:
                for callback in callbacks:
                    callback.on_chunk_processed(region, i + 1, n)
            regions.append(region)
        if callbacks:
            for callback in callbacks:
                callback.on_processing_complete(regions)
        if return_reconstruct:
            return reconstruct(regions)
        return regions


class TileFlow:
    def __init__(self, streamer):
        self.streamer = SliderImage(
            image=None, tile_size=self.tile_size, overlap=self.streamer.tile_overlap
        )

    @classmethod
    def for_numpy(cls, tile_size, overlap):
        """
        Create a TileFlow instance from a NumPy array. Assumes the array is a 2D image.
        """
        return SliderImage(tile_size=tile_size, overlap=overlap)

    @classmethod
    def for_large_numpy(
        cls,
        image_np,
        tile_size,
        overlap,
        chunk_size,
    ):
        """
        Create a TileFlow instance from a large NumPy array.
        """
        return SliderLargeImage(
            image=image_np, chunk_size=chunk_size, tile_size=tile_size, overlap=overlap
        )
