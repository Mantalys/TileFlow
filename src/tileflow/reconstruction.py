"""Image reconstruction from processed tiles."""

import numpy as np

from tileflow.core import Image2D, ProcessedTile, new_image2d


def reconstruct(tiles: list[ProcessedTile], region_shape: tuple[int, int]) -> list[Image2D]:
    """Reconstruct full image from processed tiles."""
    print(f"Reconstructing image of shape: {region_shape} from {len(tiles)} tiles")
    height_reconstructed, width_reconstructed = region_shape
    last_tile = tiles[-1]
    # Handle both single array and list of arrays
    if isinstance(last_tile.image_data, list):
        sample_data = last_tile.image_data
    else:
        sample_data = [last_tile.image_data]

    # Determine output shape from processed data
    # For multi-dimensional output, preserve the shape structure
    sample_shape = sample_data[0].shape
    if len(sample_shape) >= 3:  # Multi-channel output (C, H, W)
        output_channels = sample_shape[0]
        output_shape = (output_channels, height_reconstructed, width_reconstructed)
    else:  # 2D output (H, W)
        output_shape = (height_reconstructed, width_reconstructed)

    # Create output arrays with optimized allocation
    if len(sample_shape) >= 3:
        # Multi-channel: create single array with optimized memory layout
        reconstructed = [np.empty(output_shape, dtype=sample_data[0].dtype, order="C")]
        # Zero-fill only if needed (some processors might fill completely)
        reconstructed[0].fill(0)
    else:
        # Single channel with pre-allocated contiguous arrays
        reconstructed = [
            np.empty(
                (height_reconstructed, width_reconstructed),
                dtype=data.dtype,
                order="C",  # C-contiguous for better cache performance
            )
            for data in sample_data
        ]
        # Zero-fill efficiently
        for arr in reconstructed:
            arr.fill(0)

    # Fill in data from each tile
    for tile in tiles:
        if tile.image_data is None:
            continue

        region_bbox = tile.tile_spec.geometry.core
        region_images = tile.only_core_image()

        if region_images is None:
            continue

        # Handle different output formats with optimized copying
        if len(sample_shape) >= 3:
            # Multi-channel processor output - use optimized copy
            region_data = region_images[0] if isinstance(region_images, list) else region_images
            target_view = reconstructed[0][
                :, region_bbox.y0 : region_bbox.y1, region_bbox.x0 : region_bbox.x1
            ]
            # Use copyto for better performance when shapes match
            if target_view.shape == region_data.shape:
                np.copyto(target_view, region_data, casting="same_kind")
            else:
                target_view[:] = region_data
        else:
            # Traditional channel-wise reconstruction with optimized copies
            for i, region_img in enumerate(region_images):
                target_view = reconstructed[i][
                    region_bbox.y0 : region_bbox.y1, region_bbox.x0 : region_bbox.x1
                ]
                if target_view.shape == region_img.shape:
                    np.copyto(target_view, region_img, casting="same_kind")
                else:
                    target_view[:] = region_img

    return reconstructed
