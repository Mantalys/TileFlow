from tifffile import imread

from skimage.exposure import rescale_intensity
from skimage.segmentation import expand_labels
import numpy as np

# from src.tiff_stitching.core.streamer import ImageStreamer, StreamerConfig
from src.tiff_stitching.utils import Edge
from csbdeep.utils import normalize
from stitch import ChunkShape, Chunk2D, stitching_list, extract_polygons
import matplotlib.pyplot as plt
import time
import cv2
from typing import List
from test_model_nn import UnetS4
from tileflow.streamer import TileFlow


class DummySobelModel:
    def _process(self, image):
        # Dummy processing function that applies a Sobel filter
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        return magnitude

    def predict_numpy(self, image_np):
        tileflow = TileFlow.from_numpy(image_np, tile_size=(128, 128), overlap=(4, 4))
        tileflow.set_tile_function(self._process)
        reconstructed = tileflow.process(image_np)
        return reconstructed[0]


if __name__ == "__main__":
    model_path = r"/home/kevin/Workspace/mantalys/pypelines/packages/mantaplex/models/stardist_r4-f24/model.onnx"
    # image = r"/home/kevin/Downloads/LuCa-7color_[17572,60173]_3x3component_data.tif"
    image = r"/home/kevin/Downloads/LuCa-7color_[13860,52919]_1x1component_data.tif"
    image_np = imread(image).astype(np.float32)[6][:880, :880]  # Read the image and convert to float32
    image_np = normalize(
        image_np, pmin=1, pmax=99.8, axis=(0, 1)
    )  # Normalize to [0, 1]

    model = UnetS4(model_path)  # DummySobelModel() # UnetS4(model_path)  #
    time_start = time.time()
    reconstructed = model.predict_large_numpy(image_np)
    print(
        f"Reconstructed image shape: {reconstructed.shape}, dtype: {reconstructed.dtype}"
    )
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")

    plt.figure()
    plt.imshow(reconstructed, cmap="viridis")
    plt.axis("off")
    plt.show()

    exit()

    model_path = (
        r"/home/valentin-poque-irit/Téléchargements/model_onnx+luca_dapi/model.onnx"
    )

    # image_np = rescale_intensity(image_np, in_range=(1, 10), out_range=(0, 1))  # Rescale intensity to [0, 1]

    model = StreamingModel(
        streamer=ImageStreamer(
            config=StreamerConfig(
                tile_size=128,
                overlap=8,
                chunk_size=512,
                n_features=2,
                context=0,
                nb_chunks=0,
                image_size=image_np.shape,
            )
        ),
        backend=StardistS4(model_path),
        postprocessing=SobelMagnitude(),
    )

    output = model.stream(image_np.copy())  # 5979 cells
    nb_cell = len(np.unique(output)) - 1
    print(nb_cell)

    ### START THE STITCH TEST

    tile_size = 128
    overlap_chunk = 1
    overlap = tile_size * overlap_chunk

    chunks: List[Chunk2D] = []
    for chunk_row in range(chunk_grid[0]):
        is_top = chunk_row == 0
        is_bottom = chunk_row == chunk_grid[0] - 1
        for chunk_column in range(chunk_grid[1]):
            is_left = chunk_column == 0
            is_right = chunk_column == chunk_grid[1] - 1
            edges: Edge = (
                is_left,  # left
                is_top,  # top
                is_right,  # right
                is_bottom,  # bottom
            )
            print(f"edges: {edges}")

            # precomputed chunk relative position, manage horizontal first
            width = chunk_size[1]
            x_start = chunk_column * width
            if not is_left:
                x_start -= overlap  # shift to the left to create overlap
                width += overlap  # increase width to account for overlap
            if not is_right:
                width += overlap  # increase width to account for overlap
            x_end = x_start + width
            if x_end > w:
                x_end = w
            if is_right and x_end < w:
                x_end = w

            if is_left:
                core_x_start = 0
            else:
                core_x_start = x_start + overlap
            if is_right:
                core_x_end = w
            else:
                core_x_end = x_end - overlap

            # now we do the same for the vertical position
            height = chunk_size[0]
            y_start = chunk_row * height
            if not is_top:
                y_start -= overlap
                height += overlap  # increase height to account for overlap
            if not is_bottom:
                height += overlap  # increase height to account for overlap
            y_end = y_start + height
            if y_end > h:
                y_end = h
            if is_bottom and y_end < h:
                y_end = h
            if is_top:
                core_y_start = 0
            else:
                core_y_start = y_start + overlap
            if is_bottom:
                core_y_end = h
            else:
                core_y_end = y_end - overlap

            # Create the chunk shape
            context = (x_start, y_start, x_end, y_end)
            core = (core_x_start, core_y_start, core_x_end, core_y_end)

            print(f"Chunk context: {context}, core: {core}, chunk size: {chunk_size}, ")
            # Create the chunk shape

            # on fait la même avec le core pour remplacer get_xmin et on fait une fonction qui dit si on est dans la bbox du core
            chunk_shape = ChunkShape(context=context, core=core)
            chunk = Chunk2D(
                shape=chunk_shape,
                edges=edges,
                position=(
                    chunk_row,
                    chunk_column,
                ),  # Assigning position based on row and column
            )
            chunk_np = image_np[chunk.get_slice()]
            output_chunk = model.stream(chunk_np)
            # output_chunk = expand_labels(output_chunk, distance=5)  # Expand labels to avoid small gaps
            chunk.set_array(output_chunk)
            chunks.append(chunk)

    cv2.imwrite("complete.png", (image_np * 255).astype(np.uint8))
    for i, chunk in enumerate(chunks):
        cv2.imwrite(f"chunk{i + 1}.png", (chunk.array * 255).astype(np.uint8))
    output_viridis = cv2.applyColorMap(output.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    cv2.imwrite("output.png", output_viridis)

    image_full = extract_polygons(chunks)

    cell_found_stitched = len(np.unique(image_full)) - 1

    bin_reconstructed = image_full.copy()
    bin_reconstructed[bin_reconstructed != 0] = 1

    bin_output = output.copy()
    bin_output[output != 0] = 1

    diff = bin_reconstructed - bin_output

    plt.figure()
    plt.imshow(diff, cmap="viridis")
    plt.show()

    assert image_full.shape == image_np.shape, (
        f"Shape missmatch, exepted {image_np.shape}, shape obtenu {image_full.shape}"
    )
    assert image_full.dtype == np.uint16, (
        f"Type missmatch, excepected {np.uint16}, got {image_full.dtype}"
    )
    assert cell_found_stitched == nb_cell, (
        f"Nb cellulles missmatch, excepted {nb_cell}, got {cell_found_stitched}"
    )
