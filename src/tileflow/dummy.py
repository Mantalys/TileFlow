from __future__ import annotations
from typing import Tuple, Literal, Optional
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tileflow.slider import TileFlow


Mode = Literal["perlin", "random_max"]

# -----------------------------
# Perlin noise (vectorized, NumPy‑only)
# -----------------------------


def _perlin_2d(shape: Tuple[int, int], scale: int, *, seed: Optional[int] = None) -> np.ndarray:
    """2D Perlin noise in [0, 1]. `scale` is approx cell size in pixels."""
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
    shape: Tuple[int, int],
    *,
    base_scale: int = 64,
    octaves: int = 3,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Fractal Brownian motion (sum of octaves of Perlin). Returns float32 in [0, 1]."""
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


# -----------------------------
# Max filter (NumPy‑only, same‑size via reflect padding)
# -----------------------------


def max_filter2d(img: np.ndarray, k: int | Tuple[int, int]) -> np.ndarray:
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


# -----------------------------
# Public helper
# -----------------------------


def generate_dummy(
    shape: Tuple[int, int] = (1024, 1024),
    *,
    mode: Mode = "perlin",
    seed: Optional[int] = 0,
    perlin_scale: int = 64,
    perlin_octaves: int = 3,
    max_k: int = 9,
) -> np.ndarray:
    """Create a demo image with the specified mode."""
    H, W = shape
    if mode == "perlin":
        img = perlin_fbm((H, W), base_scale=perlin_scale, octaves=perlin_octaves, seed=seed)
    elif mode == "random_max":
        rng = np.random.default_rng(seed)
        img = rng.random((H, W), dtype=np.float32)
        img = max_filter2d(img, max_k)
        # normalize after max filter
        img = (img - img.min()) / (img.ptp() + 1e-12)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return img


class DummySobelModel:
    def __init__(self, tile_size: Tuple[int, int] = (128, 128), overlap: Tuple[int, int] = (2, 2)):
        self.tile_size = tile_size
        self.overlap = overlap

    def _sobel_filter(self, image):
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

    def predict_numpy(self, image_np):
        tileflow = TileFlow.for_numpy(tile_size=self.tile_size, overlap=self.overlap)
        tileflow.set_tile_function(self._sobel_filter)
        reconstructed = tileflow.process(image_np)
        return reconstructed[0]
