from tifffile import imread

from skimage.exposure import rescale_intensity
import numpy as np
from src.tiff_stitching.core.streamer import ImageStreamer, StreamerConfig
from src.tiff_stitching.core.model import StreamingModel, SobelMagnitude, StardistS4
from csbdeep.utils import normalize
from stitch import stitching, Chunk
import matplotlib.pyplot as plt
import time
import cv2

# magenta : 237 26 253
# yellow : 255 192 0
# cyan : 124 212 226


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
    # image_np = rescale_intensity(image_np, in_range=(1, 10), out_range=(0, 1))  # Rescale intensity to [0, 1]

    model = StreamingModel(
        streamer=ImageStreamer(
            config=StreamerConfig(
                tile_size=200
                ,
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
    print(image_np.shape)

    h, w = image_np.shape
    h_chunk = h
    w_chunk = w // 2
    tile_size = 128
    overlap_chunk = 1

    chunk_1 = Chunk(
        x_start=0,
        y_start=0,
        y_end=h_chunk,
        x_end=w_chunk + (tile_size * overlap_chunk),
        position=1,
    )
    chunk_2 = Chunk(
        y_start=0,
        x_start=w_chunk - (tile_size * overlap_chunk),
        y_end=h_chunk,
        x_end=w_chunk + w_chunk,
        position=1,
    )

    print(chunk_1, chunk_1.height, chunk_1.width)
    print(chunk_2, chunk_2.height, chunk_2.width)

    image_np = normalize(
        image_np, pmin=1, pmax=99.8, axis=(0, 1)
    )  # Normalize to [0, 1]
    # model.streamer.preview()

    # image_np = rescale_intensity(image_np, out_range=(0, 1))
    time_start = time.time()
    output = model.stream(image_np.copy())  # 5979 cells
    nb_cell = len(np.unique(output)) - 1
    print(nb_cell)

    chunk_1_np = chunk_1.chunk_image(image_np)
    print(chunk_1_np.shape)
    # model.streamer.preview()
    output_chunk_1 = model.stream(chunk_1_np.copy())
    print(len(np.unique(output_chunk_1)) - 1)

    chunk_2_np = chunk_2.chunk_image(image_np)
    print(chunk_2_np.shape)
    # model.streamer.preview()
    output_chunk_2 = model.stream(chunk_2_np.copy())
    print(len(np.unique(output_chunk_2)) - 1)

    cv2.imwrite("complete.png", (image_np * 255).astype(np.uint8))
    cv2.imwrite("chunk1.png", (chunk_1_np * 255).astype(np.uint8))
    cv2.imwrite("chunk2.png", (chunk_2_np * 255).astype(np.uint8))
    output_viridis = cv2.applyColorMap(output.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    cv2.imwrite("output.png", output_viridis)

    image_full = stitching(
        chunk_1=output_chunk_1,
        chunk_2=output_chunk_2,
        coord_chunk_1=chunk_1,
        coord_chunk_2=chunk_2,
        overlap=overlap_chunk,
        chunk_size=0,
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
    assert len(np.unique(image_full - 1)) == nb_cell, (
        f"Nb cellulles missmatch, excepted {nb_cell}, got {len(np.unique(image_full - 1))}"
    )
