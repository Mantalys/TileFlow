from src.tiff_stitching.core.streamer import ImageStreamer
from src.tiff_stitching.core.streamer import StreamerConfig, Streamer
from pydantic import BaseModel
import numpy as np
import cv2


def main():
    print("Hello from tiff-stitching!")
    path = "/home/valentin-poque-irit/Bureau/Dataset/Neuro_Melanine/PNG_FULL/PATIENS_PNG_FULL/01-001/r_ax_01_001_m00_t1w-nm/slice_061.png"
    image = cv2.imread(path)
    image = image.mean(axis=2)  # Garde uniquement le canal rouge
    image = image.astype(np.float32)
    print(image.shape)
    param = {
        "tile_size": 40,
        "overlap": 1,
        "chunk_size": 10,
        "n_features": 1,
    }
    config = StreamerConfig(**param)
    streamer = Streamer(config=config)
    ImageStreamer.set_data(streamer, data=image)
    ImageStreamer.preview(streamer)


if __name__ == "__main__":
    main()
