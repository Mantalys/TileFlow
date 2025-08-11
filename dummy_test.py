from tileflow.dummy import generate_dummy, DummySobelModel
import numpy as np


if __name__ == "__main__":
    model = DummySobelModel(tile_size=(128, 128), overlap=(0, 0))
    image = generate_dummy(shape=(320, 320))

    raw_sobel = model._sobel_filter(image)

    reconstructed_image_no_overlap = model.predict_numpy(image)

    model_with_overlap = DummySobelModel(tile_size=(128, 128), overlap=(4, 4))
    reconstructed_image_with_overlap = model_with_overlap.predict_numpy(image)

    # print the error between raw and reconstructed images
    error_no_overlap = np.abs(raw_sobel - reconstructed_image_no_overlap)
    error_with_overlap = np.abs(raw_sobel - reconstructed_image_with_overlap)
    print("Error without overlap:", np.mean(error_no_overlap))
    print("Error with overlap:", np.mean(error_with_overlap))
