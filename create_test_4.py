from tifffile import imread

from skimage.exposure import rescale_intensity
import numpy as np
from src.tiff_stitching.core.streamer import ImageStreamer, StreamerConfig
from src.tiff_stitching.core.model import StreamingModel, SobelMagnitude, StardistS4
from csbdeep.utils import normalize
from stitch import stitching, Chunk, stitching_list
import matplotlib.pyplot as plt
import time
import cv2


if __name__ == "__main__":
    model_path = (
        f"/home/valentin-poque-irit/Téléchargements/model_onnx+luca_dapi/model.onnx"
    )

    image = (
        r"/home/valentin-poque-irit/Téléchargements/model_onnx+luca_dapi/luca_dapi.tif"
    )
    image_np = imread(image).astype(np.float32)[
        0
    ]  # Read the image and convert to float32
    image_np = normalize(
        image_np, pmin=1, pmax=99.8, axis=(0, 1)
    )  # Normalize to [0, 1]
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

    # Now, we build an automatic splitting of the image into n_chunks, but still horizontal
    # This is a test to see if the stitching works correctly
    h, w = image_np.shape
    print(f"Image shape: {image_np.shape}")
    chunk_size = (1400, 512)  # Size of each chunk
    half_chunk_size = (
        chunk_size[0] // 2,
        chunk_size[1] // 2,
    )  # Half size of each chunk for overlap calculation
    # compute the number of chunks based on the image size and chunk size, as a 2D grid.
    # Eg. if the image is 1024x2048 and chunk size is 512, we will have 2 chunks in height and 4 in width.
    # But if the image size is not a multiple of the chunk size, we will have an additional chunk for the remaining pixels.
    # If the remaining pixels are less than the overlap, we will not create a chunk for them, and will add it to the last chunk.
    chunk_grid = (
        h // chunk_size[0] + (1 if h % chunk_size[0] > half_chunk_size[0] else 0),
        w // chunk_size[1] + (1 if w % chunk_size[1] > half_chunk_size[1] else 0),
    )
    print(f"Number of chunks: {chunk_grid}")
    chunk_list_output = []
    for chunk_row in range(chunk_grid[0]):
        is_top = chunk_row == 0
        is_bottom = chunk_row == chunk_grid[0] - 1
        for chunk_column in range(chunk_grid[1]):
            # precomputed chunk relative position
            is_left = chunk_column == 0
            is_right = chunk_column == chunk_grid[1] - 1
            x_start = chunk_column * w_chunk - (overlap) if chunk_column > 0 else 0
            y_start = chunk_row * h_chunk - (overlap) if chunk_row > 0 else 0
            x_end = x_start + w_chunk + (overlap) if not is_right else w
            y_end = y_start + h_chunk + (overlap) if not is_bottom else h
            # on fait la même avec le core pour remplacer get_xmin et on fait une fonction qui dit si on est dans la bbox du core
            chunk_infos = Chunk(
                x_start=x_start,
                y_start=y_start,
                y_end=y_end,
                x_end=x_end,
                position=(
                    chunk_row,
                    chunk_column,
                ),  # Assigning position based on row and column
            )
            print(
                f"Chunk: {chunk_infos}, height: {chunk_infos.height}, width: {chunk_infos.width}"
            )
            chunk_np = chunk_infos.chunk_image(image_np)
            print(
                f"Chunk {chunk_row + chunk_column * chunk_grid[1]} shape: {chunk_np.shape}"
            )
            output_chunk = model.stream(chunk_np.copy())
            print(
                f"Chunk {chunk_row + chunk_column * chunk_grid[1]} unique labels: {len(np.unique(output_chunk)) - 1}"
            )
            chunk_list_output.append((chunk_infos, output_chunk))

    print(f"TEST {chunk_list_output[1][0].get_valid_xmax(10)}")
    cv2.imwrite("complete.png", (image_np * 255).astype(np.uint8))
    for i, (chunk_infos, output_chunk) in enumerate(chunk_list_output):
        cv2.imwrite(f"chunk{i + 1}.png", (output_chunk * 255).astype(np.uint8))
    output_viridis = cv2.applyColorMap(output.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    cv2.imwrite("output.png", output_viridis)

    image_full = stitching_list(
        chunk_list_output,
        chunk_grid=chunk_grid,
        overlap=overlap,
    )

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
    assert len(np.unique(image_full - 1)) == nb_cell, (
        f"Nb cellulles missmatch, excepted {nb_cell}, got {len(np.unique(image_full - 1))}"
    )
