#!/usr/bin/env python3
"""Basic usage example for TileFlow with CHW numpy arrays.

This script demonstrates:
- Processing multi-channel images in CHW format [8, 2048, 2048]
- Channel-specific processing (e.g., DAPI channel for nuclei segmentation)
- Basic TileFlow API usage
"""

import numpy as np

from tileflow.examples import SobelEdgeDetector, generate_test_image


def create_multichannel_image(channels=8, height=2048, width=2048, seed=42):
    """Create a synthetic CHW image [C, H, W] with different patterns per channel."""
    np.random.seed(seed)
    image = np.zeros((channels, height, width), dtype=np.float32)
    
    for c in range(channels):
        if c == 0:  # DAPI channel - nuclei-like structures
            image[c] = generate_test_image(
                (height, width), mode="random_max", seed=seed + c, max_k=15
            )
        elif c == 1:  # FITC channel - cytoplasm-like
            image[c] = generate_test_image(
                (height, width), mode="perlin", seed=seed + c, perlin_scale=32
            )
        else:  # Other channels - varied patterns
            image[c] = generate_test_image((height, width), mode="perlin", seed=seed + c)
    
    return image


def simple_nuclei_segmentation(channel_data):
    """Segment nuclei using edge detection and thresholding."""
    # Apply Sobel edge detection
    sobel = SobelEdgeDetector(tile_size=(256, 256), overlap=(16, 16))
    edges = sobel.process(channel_data)
    
    # Simple thresholding
    threshold = np.percentile(edges, 95)
    nuclei_mask = edges > threshold
    
    return nuclei_mask.astype(np.uint8)


def process_specific_channel(image_chw, channel_idx=0):
    """Process a specific channel from CHW image."""
    print(f"Processing channel {channel_idx} (shape: {image_chw[channel_idx].shape})")
    
    # Extract single channel
    channel_data = image_chw[channel_idx]
    
    if channel_idx == 0:  # DAPI channel
        result = simple_nuclei_segmentation(channel_data)
        print(f"  → Nuclei segmentation complete. Found {np.sum(result)} nuclei pixels")
    else:
        # Generic edge detection for other channels
        sobel = SobelEdgeDetector(tile_size=(256, 256), overlap=(16, 16))
        result = sobel.process(channel_data)
        print(f"  → Edge detection complete. Max edge strength: {result.max():.3f}")
    
    return result


def main():
    """Demonstrate basic TileFlow usage with multi-channel images."""
    print("TileFlow Basic Usage Example")
    print("=" * 40)
    
    # Create synthetic multi-channel image
    print("Creating synthetic CHW image [8, 2048, 2048]...")
    image_chw = create_multichannel_image()
    print(f"Image shape: {image_chw.shape}")
    print(f"Data type: {image_chw.dtype}")
    print(f"Memory usage: {image_chw.nbytes / 1024 / 1024:.1f} MB")
    print()
    
    # Process DAPI channel (channel 0)
    print("Processing DAPI channel (0) for nuclei segmentation:")
    process_specific_channel(image_chw, channel_idx=0)
    print()
    
    # Process another channel
    print("Processing FITC channel (1):")
    process_specific_channel(image_chw, channel_idx=1)
    print()
    
    # Demonstrate processing all channels
    print("Processing all channels:")
    results = []
    for i in range(image_chw.shape[0]):
        result = process_specific_channel(image_chw, channel_idx=i)
        results.append(result)
    
    print(f"\nProcessed {len(results)} channels successfully!")
    print("Results can be used for:")
    print("  - Multi-channel analysis")
    print("  - Channel-specific segmentation")
    print("  - Feature extraction pipelines")


if __name__ == "__main__":
    main()