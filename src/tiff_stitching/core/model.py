from src.tiff_stitching.core.streamer import Streamer
import numpy as np
import cv2
from skimage.exposure import rescale_intensity
from skimage.segmentation import watershed
from skimage.morphology import disk


MORPH_DISK_KERNEL = disk(1)


SOBEL_4 = {
    0: np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    1: np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    2: np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
    3: np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
}


class StardistS4:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = rt.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.directions = [0, 1, 2, 3]
        self.n_features = 2 if len(self.directions) > 0 else 1

    def batch(self, batch):
        return self.session.run(output_names=None, input_feed={self.input_name: batch})

    def predict_features(self, image):
        if self.directions == []:
            outputs = self.session.run(
                output_names=self.outputs, input_feed={self.input_name: image}
            )
            magnitude = None
        else:
            outputs = self.batch(image)
            gradients = outputs[1]
            filtered = [
                cv2.filter2D(gradients[0, :, :, i], cv2.CV_32F, SOBEL_4[i])
                for i in self.directions
            ]
            magnitude = np.max(filtered, axis=0)
            magnitude[magnitude < 0] = 0
            cv2.normalize(magnitude, magnitude, 0, 1, cv2.NORM_MINMAX)
            # cv2.imwrite("magnitude.png", (magnitude * 255).astype(np.uint8))

        probability_map = outputs[0][0, :, :, 0]

        return probability_map, magnitude

    def predict(self, image):
        probability, magnitude = self.predict_features(image)

        probability[probability < 0.1] = 0
        probability = rescale_intensity(probability, out_range=(0, 1))
        # magnitude[magnitude < 0] = 0

        markers_mask = ((probability > 0) & (magnitude < 0.1)).astype(np.uint8)
        markers_mask = cv2.morphologyEx(markers_mask, cv2.MORPH_OPEN, MORPH_DISK_KERNEL)
        # magnitude = rescale_intensity(magnitude, in_range=(0, 255))
        # cv2.imwrite("magnitude.png", magnitude.astype(np.uint8) * 255)
        markers = cv2.connectedComponents(markers_mask, connectivity=4)[1]

        labels = watershed(
            magnitude,
            markers=markers,
            mask=probability > 0,
            compactness=1,
            watershed_line=False,
            connectivity=2,
        )

        return labels


class SobelMagnitude:
    def __init__(self):
        self.directions = [0, 1, 2, 3]

    def postprocess(self, features):
        probability = features[0]  # Assuming the first feature is the probability map
        magnitude = features[
            1
        ]  # Assuming the second feature is the magnitude of gradients

        probability[probability < 0.1] = 0
        probability = rescale_intensity(probability, out_range=(0, 1))
        # magnitude[magnitude < 0] = 0

        markers_mask = ((probability > 0) & (magnitude < 0.1)).astype(np.uint8)
        markers_mask = cv2.morphologyEx(markers_mask, cv2.MORPH_OPEN, MORPH_DISK_KERNEL)
        markers = cv2.connectedComponents(markers_mask, connectivity=4)[1]

        labels = watershed(
            magnitude,
            markers=markers,
            mask=probability > 0,
            compactness=0,
            watershed_line=True,
            connectivity=2,
        )

        return labels

    def postprocess_batch(self, batch):
        # keep probabilities as is, but compute the magnitude of the gradients
        batch_probability = batch[0]  # Assuming batch[0] is the probability map
        # batch_probability[batch_probability < 0.005] = 0  # Ensure no negative values
        batch_gradients = batch[1]  # Assuming batch[1] contains the 32 gradients

        batch_features = np.zeros(
            (
                2,
                len(batch_probability),
                batch_probability.shape[1],
                batch_probability.shape[2],
            ),
            dtype=np.float32,
        )
        batch_features[0] = batch_probability[
            :, :, :, 0
        ]  # Probability map, assuming the first channel is the probability
        for tile_index in range(len(batch_probability)):
            filtered = [
                cv2.filter2D(
                    batch_gradients[tile_index, :, :, i], cv2.CV_32F, SOBEL_4[i]
                )
                for i in self.directions
            ]
            magnitude = np.max(filtered, axis=0)
            magnitude[magnitude < 0] = 0
            cv2.normalize(magnitude, magnitude, 0, 1, cv2.NORM_MINMAX)
            batch_features[1][tile_index] = magnitude  # Magnitude of gradients
        return batch_features


class StreamingModel:
    def __init__(self, streamer: Streamer, backend, postprocessing, scale=1.0):
        self.backend = backend
        self.postprocessing = postprocessing
        self.streamer = streamer
        self.scale = scale

    def stream(self, data):
        self.streamer.set_data(data)
        # self.streamer.preview()
        batchs = self.streamer.batchify(batch_size=16, scale=self.scale)

        tile_index = 0
        for batch in batchs:
            p_batch = self.backend.batch(batch)
            batch_features = self.postprocessing.postprocess_batch(p_batch)
            # shape is (2, batch_size, height, width) for features
            # plot the first tile of the batch

            tile_index = self.streamer.stamp_features(tile_index, batch_features)
        output = self.postprocessing.postprocess(self.streamer._reconstructed)
        return output
