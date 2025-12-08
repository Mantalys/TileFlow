"""Synthetic image generators and reference processing implementations.

This module provides:
- Synthetic image generation (Perlin noise, random max filters)
- Multi-channel CHW format support 
- SobelEdgeDetector: Reference implementation using TileFlow
- Test data generators for microscopy-like multi-channel images

All generators create realistic test data suitable for validating
image processing pipelines and demonstrating TileFlow capabilities.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from tileflow.core import TileSpec
from tileflow.model import TileFlow

ImageMode = Literal["perlin", "random_max"]


def _perlin_2d(shape: tuple[int, int], scale: int, *, seed: int | None = None) -> np.ndarray:
    """Generate 2D Perlin noise in [0, 1]. Scale is approx cell size in pixels."""
    H, W = shape
    rng = np.random.default_rng(seed)

    # gradient lattice size (one extra to cover tail)
    gy = H // scale + 2
    gx = W // scale + 2
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(gy, gx))
    g = np.stack([np.cos(theta), np.sin(theta)], axis=-1)  # (gy, gx, 2)

    # pixel coords
    ys = np.arange(H)[:, None] / scale
    xs = np.arange(W)[None, :] / scale

    i = np.floor(ys).astype(int)
    j = np.floor(xs).astype(int)
    fy = ys - i
    fx = xs - j

    # corners: (i,j), (i+1,j), (i,j+1), (i+1,j+1)
    def dot(ix, iy, dx, dy):
        vec = g[iy, ix]  # (H, W, 2) via broadcasting
        return vec[..., 0] * dx + vec[..., 1] * dy

    n00 = dot(j, i, fx, fy)
    n10 = dot(j + 1, i, fx - 1, fy)
    n01 = dot(j, i + 1, fx, fy - 1)
    n11 = dot(j + 1, i + 1, fx - 1, fy - 1)

    # fade and lerp
    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    u = fade(fx)
    v = fade(fy)
    nx0 = n00 * (1 - u) + n10 * u
    nx1 = n01 * (1 - u) + n11 * u
    n = nx0 * (1 - v) + nx1 * v

    # normalize to [0,1]
    n = (n - n.min()) / (np.ptp(n) + 1e-12)
    return n.astype(np.float32)


def perlin_fbm(
    shape: tuple[int, int],
    *,
    base_scale: int = 64,
    octaves: int = 3,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate fractal Brownian motion using Perlin noise. Returns float32 in [0, 1]."""
    H, W = shape
    out = np.zeros((H, W), dtype=np.float32)
    amp = 1.0
    total = 0.0
    rng = np.random.default_rng(seed)
    for o in range(octaves):
        sc = max(1, int(round(base_scale / (lacunarity**o))))
        out += amp * _perlin_2d(shape, sc, seed=int(rng.integers(0, 2**31 - 1)))
        total += amp
        amp *= persistence
    out /= total + 1e-12
    return out


def max_filter2d(img: np.ndarray, k: int | tuple[int, int] = (9, 9)) -> np.ndarray:
    """Apply 2D max filter using sliding window and reflection padding."""
    if isinstance(k, int):
        ky = kx = int(k)
    else:
        ky, kx = map(int, k)
    assert ky > 0 and kx > 0 and ky % 2 == 1 and kx % 2 == 1, "k must be odd and >0"

    py = ky // 2
    px = kx // 2
    pad = np.pad(img, ((py, py), (px, px)), mode="reflect")
    win = sliding_window_view(pad, (ky, kx))  # (H, W, ky, kx)
    return win.max(axis=(-2, -1))


def generate_test_image(
    shape: tuple[int, int] = (1024, 1024),
    *,
    mode: ImageMode = "perlin",
    seed: int | None = 0,
    perlin_scale: int = 64,
    perlin_octaves: int = 3,
    max_k: int = 9,
) -> np.ndarray:
    """Generate synthetic test images for demos and testing.

    Parameters
    ----------
    shape : tuple[int, int], default=(1024, 1024)
        Output image dimensions (height, width)
    mode : {"perlin", "random_max"}, default="perlin"
        Image generation mode
    seed : int, optional, default=0
        Random seed for reproducibility
    perlin_scale : int, default=64
        Base scale for Perlin noise
    perlin_octaves : int, default=3
        Number of octaves for fractal noise
    max_k : int, default=9
        Kernel size for max filter mode

    Returns
    -------
    np.ndarray
        Generated test image as float32 in [0, 1]
    """
    H, W = shape
    if mode == "perlin":
        img = perlin_fbm((H, W), base_scale=perlin_scale, octaves=perlin_octaves, seed=seed)
    elif mode == "random_max":
        rng = np.random.default_rng(seed)
        img = rng.random((H, W), dtype=np.float32)
        img = max_filter2d(img, max_k)
        # normalize after max filter
        img = (img - img.min()) / (np.ptp(img) + 1e-12)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return img


def generate_multichannel_image(
    shape: tuple[int, int, int] = (8, 1024, 1024),
    *,
    seed: int | None = 42,
    channel_modes: list[ImageMode] | None = None,
) -> np.ndarray:
    """Generate synthetic multi-channel test images in CHW format.

    Parameters
    ----------
    shape : tuple[int, int, int], default=(8, 1024, 1024)
        Output image dimensions (channels, height, width)
    seed : int, optional, default=42
        Random seed for reproducibility
    channel_modes : list[ImageMode], optional
        Modes for each channel. If None, uses default patterns

    Returns
    -------
    np.ndarray
        Generated CHW image as float32 in [0, 1]
    """
    C, H, W = shape

    if channel_modes is None:
        # Default channel patterns for microscopy-like data
        channel_modes = ["random_max", "perlin"] + ["perlin"] * (C - 2)

    # Ensure we have enough modes
    while len(channel_modes) < C:
        channel_modes.append("perlin")

    image_chw = np.zeros((C, H, W), dtype=np.float32)

    for c in range(C):
        mode = channel_modes[c % len(channel_modes)]

        if c == 0 and mode == "random_max":  # DAPI-like nuclei
            image_chw[c] = generate_test_image((H, W), mode=mode, seed=seed + c, max_k=15)
        elif c == 1 and mode == "perlin":  # Cytoplasm-like
            image_chw[c] = generate_test_image((H, W), mode=mode, seed=seed + c, perlin_scale=32)
        else:  # Other channels
            image_chw[c] = generate_test_image((H, W), mode=mode, seed=seed + c)

    return image_chw


class SobelEdgeDetector:
    """Reference implementation of Sobel edge detection using TileFlow."""

    def __init__(
        self, tile_size: tuple[int, int] = (128, 128), overlap: tuple[int, int] = (8, 8)
    ) -> None:
        """Initialize edge detector with tiling parameters.

        Parameters
        ----------
        tile_size : tuple[int, int], default=(128, 128)
            Size of processing tiles
        overlap : tuple[int, int], default=(8, 8)
            Overlap between tiles for seamless reconstruction
        """
        self.tile_size = tile_size
        self.overlap = overlap

    def _sobel_filter(self, image: np.ndarray, tile_spec: TileSpec) -> np.ndarray:
        """Apply Sobel edge detection to image tile."""
        img = image.astype(np.float32)
        pad = np.pad(img, 1, mode="reflect")
        gx = (
            pad[:-2, :-2]
            + 2 * pad[1:-1, :-2]
            + pad[2:, :-2]
            - (pad[:-2, 2:] + 2 * pad[1:-1, 2:] + pad[2:, 2:])
        )
        gy = (
            pad[:-2, :-2]
            + 2 * pad[:-2, 1:-1]
            + pad[:-2, 2:]
            - (pad[2:, :-2] + 2 * pad[2:, 1:-1] + pad[2:, 2:])
        )
        mag = np.sqrt(gx * gx + gy * gy)
        return mag.astype(np.float32)

    def process(self, image: np.ndarray) -> np.ndarray:
        """Process image using tiled Sobel edge detection.

        Parameters
        ----------
        image : np.ndarray
            Input image to process

        Returns
        -------
        np.ndarray
            Edge-detected image
        """
        processor = TileFlow(tile_size=self.tile_size, overlap=self.overlap)
        processor.configure(function=self._sobel_filter)
        return processor.run(image)
