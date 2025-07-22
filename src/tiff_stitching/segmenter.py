"""
Wrapper for the instance-segmentation model.
"""

import numpy as np
from typing import Dict
from tile_manager import TileManager
import cv2
from Adapters.test_stardist_kartezio import Stardist2
from utils import BBox


stardist_model = r"c:\Users\corta\Documents\Mantalys\Mantaplex\Dev\pypelines\stardist-fluo_light.onnx"
model = Stardist2(stardist_model, directions=8)


def predict_with_model(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Run the segmentation model on the image.
    """
    # assert size is (128, 128) and dtype is float32, and values are in [0, 1]
    formatted_image = cv2.resize(
        image,
        (256, 256),
        interpolation=cv2.INTER_CUBIC,
    )
    formatted_image = formatted_image[:, :, np.newaxis]
    formatted_image = formatted_image[np.newaxis, :, :, :]
    features = model.predict_features(formatted_image)
    probability, magnitude = features
    magnitude = cv2.resize(
        magnitude,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    probability = cv2.resize(
        probability,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    return {
        "probability": probability,
        "magnitude": magnitude,
    }


class ModelError(Exception):
    pass


class Segmenter:
    # Map tile bbox -> dict of feature_name -> feature array
    _outputs: Dict[BBox, Dict[str, np.ndarray]] = {}

    @classmethod
    def invoke_model(cls, tile: Dict) -> None:
        """
        Run the segmentation model on a tile and cache result.
        """
        raw = TileManager.get(tile)
        try:
            features = predict_with_model(raw)
        except Exception as e:
            raise ModelError(e)
        cls._outputs[tile["bbox"]] = features

    @classmethod
    def has_output(cls, tile: Dict) -> bool:
        return tile["bbox"] in cls._outputs

    @classmethod
    def get_features(cls, tile: dict) -> Dict[str, np.ndarray]:
        """
        Retrieve all computed feature maps for a tile.
        """
        return cls._outputs[tile["bbox"]]

    @classmethod
    def get_feature(cls, tile: dict, name: str) -> np.ndarray:
        """
        Retrieve a single feature map by name.
        """
        return cls._outputs[tile["bbox"]][name]

    @classmethod
    def clear_output(cls, tile: Dict) -> None:
        cls._outputs.pop(tile["bbox"], None)
