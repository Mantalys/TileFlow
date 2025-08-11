from src.tiff_stitching.feature_merger import stamp
import numpy as np
import cv2
from tileflow.tiling import GridSpec
from tileflow.core import RegionImage
from tileflow.reconstruction import reconstruct
from typing import List


class NewImageStreamer:
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

    def display_grid_over_image(self):
        """
        Display the grid of regions over the image.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig, ax = plt.subplots()
        ax.imshow(self.image, cmap="viridis")
        ax.axis("off")
        ax.set_aspect("equal")

        for tile in self.grid:
            x0, y0, x1, y1 = tile.geometry.halo
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor="blue",
                    linewidth=0.5,
                )
            )
            cx0, cy0, cx1, cy1 = tile.geometry.core
            ax.add_patch(
                Rectangle(
                    (cx0, cy0),
                    cx1 - cx0,
                    cy1 - cy0,
                    fill=False,
                    edgecolor="green",
                    linewidth=0.8,
                )
            )
        plt.show()

    def build_grid(self, image_shape):
        """
        Build a grid of regions based on the image shape.
        """
        chunk_grid_spec = GridSpec(size=self.tile_size, halo=self.overlap)
        return chunk_grid_spec.build_grid(image_shape)

    def set_tile_function(self, func):
        """
        Set a processing function to be applied to each tile.
        """
        self.tile_function = func

    def process(self, image) -> np.ndarray:
        self.set_image(image)
        # self.display_grid_over_image()
        regions: List[RegionImage] = []
        for tile in self.grid:
            tile_np = image[tile.get_halo_slices()]
            tile_output = self.tile_function(tile_np)
            # stitch a tile and its content
            region = RegionImage(region_spec=tile, image_data=tile_output)
            regions.append(region)
        return reconstruct(regions)


class LargeImageStreamer:
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
        chunk_grid_spec = GridSpec(size=self.chunk_size, halo=self.tile_size)
        return chunk_grid_spec.build_grid(image_shape)

    def process(self, image):
        regions: List[RegionImage] = []
        tile_streamer = NewImageStreamer(tile_size=self.tile_size, overlap=self.overlap)
        tile_streamer.set_tile_function(self.tile_function)
        chunk_grid = self.build_grid(image.shape)
        for chunk in chunk_grid:
            chunk_np = image[chunk.get_halo_slices()]
            chunk_output = tile_streamer.process(chunk_np)
            if self.chunk_function:
                chunk_output = self.chunk_function(chunk_output)
            region = RegionImage(region_spec=chunk, image_data=chunk_output)
            regions.append(region)
        return reconstruct(regions)


class TileFlow:
    def __init__(self, streamer):
        self.streamer = NewImageStreamer(
            image=None, tile_size=self.tile_size, overlap=self.streamer.tile_overlap
        )

    @classmethod
    def for_numpy(cls, tile_size, overlap):
        """
        Create a TileFlow instance from a NumPy array. Assumes the array is a 2D image.
        """
        return NewImageStreamer(tile_size=tile_size, overlap=overlap)

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
        return LargeImageStreamer(
            image=image_np, chunk_size=chunk_size, tile_size=tile_size, overlap=overlap
        )

    @classmethod
    def for_zarr(cls, zarr_array, tile_size, tile_overlap, chunk_size, chunk_overlap):
        """
        Create a TileFlow instance from a Zarr array.
        """
        streamer = ZarrStreamer(
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        streamer.set_data(zarr_array)
        return cls(streamer)

    @classmethod
    def for_tiff(
        cls, tifffile_path, tile_size, tile_overlap, chunk_size, chunk_overlap
    ):
        """
        Create a TileFlow instance from a TIFF file.
        """
        streamer = LargeImageStreamer(
            tile_size=tile_size,
            overlap=tile_overlap,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        streamer.set_tifffile(tifffile_path)
        return cls(streamer)


class Streamer:
    def __init__(self, tile_size, tile_overlap):
        """
        Initialize the Streamer with the given tile size and overlap.
        :param tile_size: Size of the tiles to be created.
        :param tile_overlap: Overlap between the tiles.
        """
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.stride = tile_size - tile_overlap
        self.half_overlap = tile_overlap // 2
        self._tiles = []
        self._data = None
        self._reconstructed = None
        self._shape = None

    def _build_1_chunks(self, overlap: int, context: int):
        chunks_size = self.config.chunk_size
        self.chunks = []
        h, w = self.config.image_size[-2:]
        x0, y0, x1, y1 = 0, 0, 0, 0
        print(y1)
        for i in range(0, self.config.nb_chunks):
            if i == 0:  # cas particulier car en (0,0) on fait pas l'overlap
                x1 += 512 + context
                y1 += 512 + context
            else:
                x1 += 512 + context
                y1 += 512 + context
                x0 += 512 + context
                y0 += 512 + context

            temp_tiles = self._get_tiles_coord(x0, y0, x1, y1)

            self.chunks.append(
                {
                    "index": (i),
                    "bbox": (x0, y0, x1, y1),
                    "pad": (0, 0, 0, 0),
                    "overlap": (overlap),
                    "core_box": (
                        x0 + context,
                        y0 + context,
                        x1 + context,
                        y1 + context,
                    ),
                    "tiles": (temp_tiles),
                }
            )

    def _get_tiles_coord(self, x_0: int, x_1: int, y_0: int, y_1: int):
        tiles_coord = []
        for tile in self._tiles:
            tile_x0, tile_y0, tile_x1, tile_y1 = tile["core_bbox"]
            # if ((tile_x0 > x_0) and (tile_x0 < x_1) and (tile_x1 >x_0) and (tile_x1<x_1) and (tile_y0 > y_0) and (tile_y0<y_1) and (tile_y1>y_0) and (tile_y1<y_1)):
            if not (
                tile_x1 <= x_0 or tile_x0 >= x_1 or tile_y1 <= y_0 or tile_y0 >= y_1
            ):
                tiles_coord.append(tile)
        return tiles_coord

    def _build_tiles(self):
        self._tiles = []
        h, w = self._data.shape
        xs = list(range(0, w, self.stride))
        ys = list(range(0, h, self.stride))
        for y in ys:
            for x in xs:
                x1 = min(x + self.config.tile_size, w)
                y1 = min(y + self.config.tile_size, h)
                pad_right = (
                    self.config.tile_size - (x1 - x)
                    if (x1 - x) < self.config.tile_size
                    else 0
                )
                pad_bottom = (
                    self.config.tile_size - (y1 - y)
                    if (y1 - y) < self.config.tile_size
                    else 0
                )
                # core coords: half-overlap inside except at image edges
                core_x0 = x + (self.half_overlap if x > 0 else 0)
                core_y0 = y + (self.half_overlap if y > 0 else 0)
                core_x1 = x1 - (self.half_overlap if x1 < w else 0) + pad_right
                core_y1 = y1 - (self.half_overlap if y1 < h else 0) + pad_bottom
                self._tiles.append(
                    {
                        "bbox": (x, y, x1, y1),
                        "pad": (0, 0, pad_right, pad_bottom),
                        "core_bbox": (core_x0, core_y0, core_x1, core_y1),
                    }
                )

    def _build_reconstructed(self):
        self._reconstructed = [
            np.zeros((self._shape[0], self._shape[1]), dtype=np.float32)
            for _ in range(self.config.n_features)
        ]

    def preview(self):
        """
        Preview the tiles generated from the data.
        This method visualizes the tiles by printing their bounding boxes.
        It display the image with tiles overlaid.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        if self._data is None:
            raise ValueError("No data set. Please set data before previewing.")
        fig, ax = plt.subplots()
        ax.imshow(self._data, cmap="gray")
        # remove axis
        ax.axis("off")
        ax.set_aspect("equal")

        for tile in self._tiles:
            x0, y0, x1, y1 = tile["bbox"]
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor="blue",
                    linewidth=0.5,
                )
            )
            cx0, cy0, cx1, cy1 = tile["core_bbox"]
            ax.add_patch(
                Rectangle(
                    (cx0, cy0),
                    cx1 - cx0,
                    cy1 - cy0,
                    fill=False,
                    edgecolor="green",
                    linewidth=0.8,
                )
            )
        plt.show()
        plt.close(fig)

    def preview_one_chunk(self):
        if not self.chunks:
            self.preview()
        else:
            "Faire le traitement pour in self chunks preview pour chaque tiles "
            for chunks in self.chunks:
                "ça marche pas parce qu'on veut le faire sur tile in self.chunks[tiles] et pas self.tiles"


class ImageStreamer(Streamer):
    def __init__(self, config):
        super().__init__(config)

    def set_data(self, data: np.ndarray):
        assert isinstance(data, np.ndarray), "Image must be a numpy array"
        assert data.ndim == 2, "Image must be a 2D array (grayscale)"
        assert data.dtype == np.float32, "Image must be of type float32"
        self._data = data
        self._shape = data.shape
        self._build_tiles()
        self._build_reconstructed()

    def batchify(self, batch_size: int, scale: float = 1.0):
        """
        Get a batch of tiles from the data using generator.
        :param batch_size: Number of tiles to return in the batch.
        :return: List of tiles in the batch.
        """
        if not self._tiles:
            raise ValueError("No tiles available. Please set data first.")

        # create a python generator to yield tiles
        def tile_generator():
            for i in range(0, len(self._tiles), batch_size):
                tiles = self._tiles[i : i + batch_size]
                tiles = [
                    self._data[
                        tile["bbox"][1] : tile["bbox"][3],
                        tile["bbox"][0] : tile["bbox"][2],
                    ]
                    for tile in tiles
                ]
                # Pad tiles to ensure they are all the same size
                for j in range(len(tiles)):
                    tile = tiles[j]
                    if (
                        tile.shape[0] < self.config.tile_size
                        or tile.shape[1] < self.config.tile_size
                    ):
                        pad_height = max(0, self.config.tile_size - tile.shape[0])
                        pad_width = max(0, self.config.tile_size - tile.shape[1])
                        tile = np.pad(
                            tile,
                            ((0, pad_height), (0, pad_width)),
                            mode="constant",
                            constant_values=0,
                        )
                    if scale != 1.0:
                        # Resize the tile if scale is not 1.0
                        tile = cv2.resize(
                            tile,
                            (
                                self.config.tile_size * scale,
                                self.config.tile_size * scale,
                            ),
                            interpolation=cv2.INTER_CUBIC,
                        )
                    tiles[j] = tile
                stacked = np.stack(tiles, axis=0)
                # add channel dimension at the end
                stacked = stacked[
                    ..., np.newaxis
                ]  # shape: (batch_size, tile_size, tile_size, 1)
                yield stacked

        return tile_generator()

    def stamp_features(self, tile_index: int, features: list):
        assert len(features) == self.config.n_features, (
            "Features length must match n_features in config"
        )
        for i in range(len(features[0])):
            tile = self._tiles[tile_index]
            x0, y0, x1, y1 = tile["bbox"]

            pad_top = self.half_overlap if y0 > 0 else 0
            pad_left = self.half_overlap if x0 > 0 else 0
            pad_bot = self.half_overlap if y1 < self._shape[0] else 0
            pad_right = self.half_overlap if x1 < self._shape[1] else 0

            for feature_index in range(self.config.n_features):
                f = features[feature_index][i]
                f = cv2.resize(
                    f,
                    (self.config.tile_size, self.config.tile_size),
                    interpolation=cv2.INTER_CUBIC,
                )
                unpadded = f[
                    pad_top : f.shape[0] - pad_bot,
                    pad_left : f.shape[1] - pad_right,
                ]
                stamp(
                    self._reconstructed[feature_index],
                    unpadded,
                    tile["core_bbox"],
                    (0, 0),
                )
            tile_index += 1
        return tile_index


class ZarrStreamer(Streamer):
    def __init__(
        self, tile_size: int, tile_overlap: int, chunk_size: int, chunk_overlap: int
    ):
        super().__init__(
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def set_data(self, data: np.ndarray):
        assert isinstance(data, np.ndarray), "Image must be a numpy array"
        assert data.ndim == 2, "Image must be a 2D array (grayscale)"
        assert data.dtype == np.float32, "Image must be of type float32"
        self._data = data
        self._shape = data.shape
        self._build_1_chunks(self.config.overlap, self.config.context)
