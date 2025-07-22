from pydantic import BaseModel
from src.tiff_stitching.feature_merger import stamp
import numpy as np
import cv2


class StreamerConfig(BaseModel):
    """
    Configuration for the Streamer.
    This class defines the parameters for streaming data in chunks.
    """

    tile_size: int
    overlap: int
    chunk_size: int
    n_features: int


class Streamer:
    def __init__(self, config: StreamerConfig):
        """
        Initialize the Streamer with the given configuration.
        :param config: StreamerConfig object containing tile size, overlap, and chunk size.
        """
        self.config = config
        self.stride = config.tile_size - config.overlap
        self.half_overlap = config.overlap // 2
        self._tiles = []
        self._data = None
        self._reconstructed = None
        self._shape = None

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


class ImageStreamer(Streamer):
    def __init__(self, config: StreamerConfig):
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


class SlideStreamer(Streamer):
    def __init__(self, config: StreamerConfig):
        pass
