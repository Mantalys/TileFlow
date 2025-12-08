"""Utility functions and validation helpers."""

import numpy as np

from tileflow.tiling import GridSpec


def validate_tile_size(tile_size: tuple[int, int], image_shape: tuple[int, int]) -> None:
    """Validate tile size against image dimensions.

    Parameters
    ----------
    tile_size : tuple[int, int]
        Tile size (height, width)
    image_shape : tuple[int, int]
        Image shape (height, width)

    Raises
    ------
    ValueError
        If tile size is invalid for the image
    """
    tile_h, tile_w = tile_size
    img_h, img_w = image_shape[:2]

    if tile_h > img_h:
        raise ValueError(f"Tile height {tile_h} cannot exceed image height {img_h}")
    if tile_w > img_w:
        raise ValueError(f"Tile width {tile_w} cannot exceed image width {img_w}")

    # Warn if tiles are very small (potential performance issue)
    if tile_h < 32 or tile_w < 32:
        import warnings

        warnings.warn(
            f"Very small tile size {tile_size} may impact performance. "
            f"Consider using larger tiles (e.g., 128x128 or larger).",
            UserWarning,
        )


def validate_overlap(overlap: tuple[int, int], tile_size: tuple[int, int]) -> None:
    """Validate overlap against tile size.

    Parameters
    ----------
    overlap : tuple[int, int]
        Overlap size (height, width)
    tile_size : tuple[int, int]
        Tile size (height, width)

    Raises
    ------
    ValueError
        If overlap is invalid for the tile size
    """
    overlap_h, overlap_w = overlap
    tile_h, tile_w = tile_size

    if overlap_h >= tile_h // 2:
        raise ValueError(
            f"Overlap height {overlap_h} must be less than half tile height {tile_h // 2}"
        )
    if overlap_w >= tile_w // 2:
        raise ValueError(
            f"Overlap width {overlap_w} must be less than half tile width {tile_w // 2}"
        )

    if overlap_h < 0 or overlap_w < 0:
        raise ValueError(f"Overlap must be non-negative, got {overlap}")


def estimate_memory_usage(
    image_shape: tuple[int, int],
    tile_size: tuple[int, int],
    overlap: tuple[int, int],
    dtype: np.dtype = np.float32,
    chunk_size: tuple[int, int] | None = None,
) -> dict[str, float]:
    """Estimate memory usage for processing configuration.

    Parameters
    ----------
    image_shape : tuple[int, int]
        Shape of input image
    tile_size : tuple[int, int]
        Size of tiles
    overlap : tuple[int, int]
        Overlap between tiles
    dtype : np.dtype, default=np.float32
        Data type for calculations
    chunk_size : tuple[int, int], optional
        Chunk size if using chunked processing

    Returns
    -------
    dict[str, float]
        Memory usage estimates in MB
    """
    bytes_per_element = np.dtype(dtype).itemsize
    img_h, img_w = image_shape

    # Original image memory
    original_mb = (img_h * img_w * bytes_per_element) / (1024**2)

    if chunk_size:
        chunk_h, chunk_w = chunk_size
        chunk_with_overlap_h = min(chunk_h + 2 * overlap[0], img_h)
        chunk_with_overlap_w = min(chunk_w + 2 * overlap[1], img_w)
        peak_mb = (chunk_with_overlap_h * chunk_with_overlap_w * bytes_per_element * 2) / (1024**2)
    else:
        # Worst case: original + reconstructed
        peak_mb = original_mb * 2

    grid_spec = GridSpec(size=tile_size, overlap=overlap)
    if chunk_size:
        chunk_grid_spec = GridSpec(size=chunk_size, overlap=(16, 16))
        num_chunks = len(list(chunk_grid_spec.build_grid(image_shape)))
        avg_tiles_per_chunk = len(list(grid_spec.build_grid(chunk_size))) if chunk_size else 0
        total_tiles = num_chunks * avg_tiles_per_chunk
    else:
        total_tiles = len(list(grid_spec.build_grid(image_shape)))

    return {
        "original_image_mb": original_mb,
        "peak_memory_mb": peak_mb,
        "total_tiles": total_tiles,
        "processing_mode": "chunked" if chunk_size else "direct",
    }
