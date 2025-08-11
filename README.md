# TileFlow

Fast, memory‑efficient image tiling & reconstruction for deep learning and whole‑slide images.

## Features
- Zero‑copy numpy views for tiles (when possible)
- Overlap/stride control with accurate merge
- Lazy pipelines with pluggable processors
- WSI backends (TIFF, Zarr/Dask)

## Install
pip install tileflow

## Quickstart
```python
from tileflow.dummy import generate_dummy, DummySobelModel
import numpy as np


model = DummySobelModel(tile_size=(128, 128), overlap=(0, 0))
image = generate_dummy(shape=(320, 320))

raw_sobel = model._sobel_filter(image)

reconstructed_image_no_overlap = model.predict_numpy(image)

model_with_overlap = DummySobelModel(tile_size=(128, 128), overlap=(4, 4))
reconstructed_image_with_overlap = model_with_overlap.predict_numpy(image)

# print the error between raw and reconstructed images
error_no_overlap = np.abs(raw_sobel - reconstructed_image_no_overlap)
error_with_overlap = np.abs(raw_sobel - reconstructed_image_with_overlap)
print("Error without overlap:", np.mean(error_no_overlap)) # 0.00026957, correspong to tile glitches
print("Error with overlap:", np.mean(error_with_overlap)) # 0, perfect reconstruction with overlap
```

## Credits

Valentin Poque (Jully-September 2025 Intern)