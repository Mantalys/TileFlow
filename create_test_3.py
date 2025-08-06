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

    # Now, we build an automatic splitting of the image into n_chunks, but still horizontal
    # This is a test to see if the stitching works correctly
    h, w = image_np.shape
    n_chunks = 4  # Number of chunks to split the image into
    chunk_height = h
    w_chunk = w // n_chunks

    chunk_list_output = []
    chunk_grid = (1, n_chunks)  # 1 row, n_chunks columns
    for i in range(n_chunks):
        x_start = i * (w_chunk - tile_size * overlap_chunk) if i > 0 else 0
        x_end = (
            x_start + w_chunk + (tile_size * overlap_chunk) if i < n_chunks - 1 else w
        )
        chunk_infos = Chunk(
            x_start=x_start,
            y_start=0,
            y_end=chunk_height,
            x_end=x_end,
            position=(1, i),  # Assigning position based on the loop index
        )
        print(
            f"Chunk {i}: {chunk_infos}, height: {chunk_infos.height}, width: {chunk_infos.width}"
        )
        chunk_np = chunk_infos.chunk_image(image_np)
        print(f"Chunk {i} shape: {chunk_np.shape}")
        output_chunk = model.stream(chunk_np.copy())
        print(f"Chunk {i} unique labels: {len(np.unique(output_chunk)) - 1}")
        chunk_list_output.append(
            (chunk_infos, output_chunk)
        )  # Store chunk info and output
    #    print(f"TEST {chunk_list_output[1][0].get_valid_xmax(10)}")
    cv2.imwrite("complete.png", (image_np * 255).astype(np.uint8))
    for i, (chunk_infos, output_chunk) in enumerate(chunk_list_output):
        cv2.imwrite(f"chunk{i + 1}.png", (output_chunk * 255).astype(np.uint8))
    output_viridis = cv2.applyColorMap(output.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    cv2.imwrite("output.png", output_viridis)
    # del chunk_list_output[0]
    # del chunk_list_output[1]
    image_full = stitching_list(
        chunk_list_output,
        chunk_grid=chunk_grid,
        overlap=overlap_chunk * tile_size,
        tile_size=tile_size,
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
    assert len(np.unique(image_full)) - 1 == nb_cell, (
        f"Nb cellulles missmatch, excepted {nb_cell}, got {len(np.unique(image_full)) - 1}"
    )
