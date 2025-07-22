"""
Cleans and labels the stitched feature map.
"""

import numpy as np
import cv2
from typing import Dict
from skimage.exposure import rescale_intensity
from skimage.segmentation import watershed
from skimage.morphology import disk

MORPH_DISK_KERNEL = disk(2)


class PostProcessor:
    @staticmethod
    def from_features_to_labels(fmap: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Morphological cleanup and connected component labeling.
        """
        magnitude = fmap["magnitude"]
        magnitude = cv2.GaussianBlur(magnitude, (3, 3), 0)
        magnitude[magnitude < 20] = 0
        magnitude = magnitude / np.max(magnitude)
        magnitude = rescale_intensity(magnitude, out_range=(0, 255))

        probability_map = fmap["probability"]
        probability_map = cv2.GaussianBlur(probability_map, (3, 3), 0)
        probability_map[probability_map < 0.005] = 0
        probability_map = rescale_intensity(probability_map, out_range=(0, 255))

        markers_mask = ((magnitude <= 0) & (probability_map > 0)).astype(np.uint8)
        markers_mask = cv2.morphologyEx(markers_mask, cv2.MORPH_OPEN, MORPH_DISK_KERNEL)

        markers = cv2.connectedComponents(markers_mask, connectivity=4)[1]

        signal = cv2.subtract(magnitude, probability_map)
        # signal = cv2.GaussianBlur(signal, (3, 3), 0)
        cv2.imwrite("signal.png", signal)

        labels = watershed(
            signal,
            markers=markers,
            mask=probability_map > 0,
            watershed_line=True,
            compactness=0,
        )
        print(f"Found {len(np.unique(labels)) - 1} objects")
        return labels.astype(np.uint16)
