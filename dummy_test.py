from tileflow.examples import generate_test_image, max_filter2d
from tileflow.model import TileFlow
from tileflow import estimate_memory_usage
import numpy as np
import time


if __name__ == "__main__":
    image = generate_test_image(shape=(3200, 3200))
    max_image = max_filter2d(image, k=9)
    model_overlap = TileFlow(tile_size=(128, 128), overlap=(8, 8), n_workers=2)
    model_overlap.configure(function=max_filter2d)
    model_overlap.summary()

    time_start = time.time()
    max_image_overlap = model_overlap.run(image)
    print(f"Processing time with overlap: {time.time() - time_start:.2f} seconds")

    model_no_overlap = TileFlow(tile_size=(128, 128), overlap=(0, 0), chunk_size=(512, 512), n_workers=2)
    model_no_overlap.configure(function=max_filter2d)
    time_start = time.time()
    max_image_no_overlap = model_no_overlap.run(image)
    print(f"Processing time without overlap: {time.time() - time_start:.2f} seconds")

    # print the error between raw and reconstructed images
    error_no_overlap = np.abs(max_image - max_image_no_overlap)
    error_with_overlap = np.abs(max_image - max_image_overlap)
    print("Error without overlap:", np.mean(error_no_overlap))
    print("Error with overlap:", np.mean(error_with_overlap))

    memory_info = estimate_memory_usage(
        image_shape=(2048, 2048), tile_size=(128, 128), overlap=(8, 8)
    )
    print(f"Peak memory: {memory_info['peak_memory_mb']:.1f} MB")
