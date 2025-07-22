from tifffile import imread

from skimage.exposure import rescale_intensity
import numpy as np
from src.tiff_stitching.core.streamer import ImageStreamer, StreamerConfig
from src.tiff_stitching.core.model import StreamingModel, SobelMagnitude
from Adapters.test_stardist_kartezio import Stardist2
from csbdeep.utils import normalize

import matplotlib.pyplot as plt
import time

# magenta : 237 26 253
# yellow : 255 192 0
# cyan : 124 212 226


if __name__ == "__main__":
    rays = 4
    filters = 32
    model_name = f"stardist_r{rays}_f{filters}"
    model_path = f"Stardist/models/{model_name}/model.onnx"

    model = StreamingModel(
        streamer=ImageStreamer(
            config=StreamerConfig(
                tile_size=256, overlap=8, chunk_size=512, n_features=2
            )
        ),
        backend=Stardist2(model_path, directions=rays),
        postprocessing=SobelMagnitude(),
    )

    image = r"c:\Users\corta\Documents\Mantalys\Mantaplex\Dataset\luca_dapi.tif"
    image_np = imread(image).astype(np.float32)[
        0
    ]  # Read the image and convert to float32
    # image_np = rescale_intensity(image_np, in_range=(1, 10), out_range=(0, 1))  # Rescale intensity to [0, 1]
    image_np = normalize(
        image_np, pmin=1, pmax=99.8, axis=(0, 1)
    )  # Normalize to [0, 1]
    # image_np = rescale_intensity(image_np, out_range=(0, 1))
    time_start = time.time()
    output = model.stream(image_np)
    print(f"Streaming time: {time.time() - time_start:.2f} seconds")
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np, cmap="viridis")
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(output, cmap="viridis")
    plt.title("Labels")
    plt.axis("off")
    plt.show()
