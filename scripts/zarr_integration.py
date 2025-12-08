#!/usr/bin/env python3
"""Zarr integration example for TileFlow with large multi-channel datasets.

This script demonstrates:
- Converting CHW numpy arrays to zarr format
- Processing zarr-backed images with TileFlow
- Memory-efficient handling of large datasets
- Zarr chunking strategies for optimal performance
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import zarr

from tileflow.examples import SobelEdgeDetector, generate_test_image
from tileflow.model import TileFlow


def create_zarr_dataset(shape=(8, 2048, 2048), chunks=(1, 512, 512), dtype=np.float32):
    """Create a zarr dataset with synthetic multi-channel data."""
    # Create temporary zarr store
    temp_dir = Path(tempfile.mkdtemp(prefix="tileflow_zarr_"))
    zarr_path = temp_dir / "multichannel.zarr"
    
    print(f"Creating zarr dataset at: {zarr_path}")
    print(f"Shape: {shape}, Chunks: {chunks}, Dtype: {dtype}")
    
    # Create zarr array
    z_array = zarr.open(
        str(zarr_path), 
        mode="w", 
        shape=shape, 
        chunks=chunks, 
        dtype=dtype
    )
    
    # Fill with synthetic data
    print("Generating synthetic data...")
    for c in range(shape[0]):
        print(f"  Channel {c}... ", end="", flush=True)
        
        if c == 0:  # DAPI - nuclei structures
            channel_data = generate_test_image(shape[1:], mode="random_max", seed=42 + c, max_k=15)
        elif c == 1:  # FITC - cytoplasm
            channel_data = generate_test_image(
                shape[1:], mode="perlin", seed=42 + c, perlin_scale=32
            )
        else:  # Other channels
            channel_data = generate_test_image(shape[1:], mode="perlin", seed=42 + c)
        
        z_array[c] = channel_data
        print("✓")
    
    print(f"Zarr dataset created. Size on disk: {get_zarr_size_mb(zarr_path):.1f} MB")
    return str(zarr_path), temp_dir


def get_zarr_size_mb(zarr_path: str) -> float:
    """Calculate size of zarr dataset on disk."""
    path = Path(zarr_path)
    if path.is_dir():
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    else:
        size = path.stat().st_size
    return size / 1024 / 1024


def process_zarr_channel(zarr_path: str, channel_idx: int, processor_type="sobel"):
    """Process a specific channel from zarr dataset."""
    print(f"Processing zarr channel {channel_idx} with {processor_type}...")
    
    # Open zarr array
    z_array = zarr.open(zarr_path, mode="r")
    
    # Extract channel (this creates a zarr array view, not a copy)
    channel_zarr = z_array[channel_idx]
    print(f"  Channel shape: {channel_zarr.shape}")
    if hasattr(channel_zarr, "chunks"):
        print(f"  Channel chunks: {channel_zarr.chunks}")
    
    # Convert to numpy for processing (loads into memory)
    # For very large datasets, you'd process chunks iteratively
    channel_np = np.array(channel_zarr)
    
    # Process based on type
    if processor_type == "sobel":
        sobel = SobelEdgeDetector(tile_size=(256, 256), overlap=(16, 16))
        result = sobel.process(channel_np)
    elif processor_type == "nuclei" and channel_idx == 0:
        # DAPI nuclei segmentation
        sobel = SobelEdgeDetector(tile_size=(256, 256), overlap=(16, 16))
        edges = sobel.process(channel_np)
        threshold = np.percentile(edges, 95)
        result = (edges > threshold).astype(np.uint8)
    else:
        # Custom processing
        def custom_filter(tile):
            return np.clip(tile * 2.0, 0, 1)
        
        processor = TileFlow(tile_size=(256, 256), overlap=(16, 16))
        processor.configure(function=custom_filter)
        result = processor.run(channel_np)
    
    print(f"  ✓ Processed. Output range: [{result.min():.3f}, {result.max():.3f}]")
    return result


def save_zarr_results(results: dict, output_path: str):
    """Save processing results to zarr format."""
    print(f"\nSaving results to zarr: {output_path}")
    
    # Determine output shape
    first_result = next(iter(results.values()))
    channels = len(results)
    height, width = first_result.shape
    
    # Create output zarr array
    z_output = zarr.open(
        output_path,
        mode="w",
        shape=(channels, height, width),
        chunks=(1, 512, 512),
        dtype=first_result.dtype
    )
    
    # Save results
    for channel_idx, result in results.items():
        z_output[channel_idx] = result
        print(f"  Saved channel {channel_idx} ✓")
    
    print(f"Results saved. Size: {get_zarr_size_mb(output_path):.1f} MB")


def zarr_to_numpy_roundtrip(zarr_path: str):
    """Demonstrate zarr to numpy conversion and back."""
    print("\nZarr ↔ NumPy roundtrip test:")
    
    # Load zarr as numpy
    z_array = zarr.open(zarr_path, mode="r")
    numpy_array = np.array(z_array)
    print(f"  Loaded zarr to numpy: {numpy_array.shape}")
    
    # Process with TileFlow
    sobel = SobelEdgeDetector(tile_size=(128, 128), overlap=(8, 8))
    processed_channel = sobel.process(numpy_array[0])  # Process first channel
    print(f"  Processed channel 0: {processed_channel.shape}")
    
    # Save back to zarr
    temp_dir = Path(tempfile.mkdtemp(prefix="tileflow_roundtrip_"))
    output_zarr = temp_dir / "processed.zarr"
    
    z_output = zarr.open(
        str(output_zarr),
        mode="w",
        shape=processed_channel.shape,
        chunks=(512, 512),
        dtype=processed_channel.dtype
    )
    z_output[:] = processed_channel
    
    print(f"  Saved processed result to zarr: {get_zarr_size_mb(str(output_zarr)):.1f} MB")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    print("  ✓ Roundtrip successful")


def main():
    """Demonstrate zarr integration with TileFlow."""
    print("TileFlow Zarr Integration Example")
    print("=" * 40)
    
    # Create zarr dataset
    zarr_path, temp_dir = create_zarr_dataset()
    
    try:
        # Process individual channels
        print("\n" + "=" * 40)
        print("Channel-specific processing:")
        
        results = {}
        
        # Process DAPI channel (0) for nuclei
        results[0] = process_zarr_channel(zarr_path, 0, "nuclei")
        
        # Process FITC channel (1) with Sobel
        results[1] = process_zarr_channel(zarr_path, 1, "sobel")
        
        # Process other channels generically
        for ch in range(2, min(4, zarr.open(zarr_path).shape[0])):
            results[ch] = process_zarr_channel(zarr_path, ch, "custom")
        
        # Save results to zarr
        output_path = str(temp_dir / "results.zarr")
        save_zarr_results(results, output_path)
        
        # Demonstrate roundtrip
        zarr_to_numpy_roundtrip(zarr_path)
        
        print("\n" + "=" * 40)
        print("Summary:")
        print(f"  ✓ Created zarr dataset: {get_zarr_size_mb(zarr_path):.1f} MB")
        print(f"  ✓ Processed {len(results)} channels")
        print(f"  ✓ Saved results: {get_zarr_size_mb(output_path):.1f} MB")
        print("  ✓ Zarr ↔ NumPy roundtrip successful")
        print("\nZarr benefits with TileFlow:")
        print("  • Memory-efficient loading of large datasets")
        print("  • Chunked storage for optimal I/O")
        print("  • Compression reduces storage requirements")
        print("  • Compatible with existing TileFlow workflows")
        
    finally:
        # Cleanup temporary files
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()