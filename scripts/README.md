# TileFlow Example Scripts

This directory contains comprehensive examples demonstrating TileFlow's capabilities with multi-channel images and zarr integration.

## Scripts Overview

### `basic_usage.py`
Basic interface testing for CHW numpy arrays.
- Creates synthetic 8-channel image [8, 2048, 2048]
- Demonstrates channel-specific processing (DAPI nuclei segmentation)
- Shows independent channel processing workflows

**Run:** `uv run python scripts/basic_usage.py`

### `multichannel_processing.py`
Advanced multi-channel processing with specialized processors.
- Independent processing pipelines for different channels
- DAPI nuclei segmentation vs generic edge detection
- Concurrent processing demonstration
- Channel-specific parameter optimization

**Run:** `uv run python scripts/multichannel_processing.py`

### `zarr_integration.py`
Zarr dataset handling and integration.
- Converting numpy CHW arrays to zarr format
- Processing zarr-backed images with TileFlow
- Memory-efficient handling of large datasets
- Demonstrates zarr ↔ numpy roundtrip workflows

**Dependencies:** Requires `zarr` package (`uv add zarr`)
**Run:** `uv run python scripts/zarr_integration.py`

## Key Demonstrations

✅ **CHW Format Support**: All scripts work with [Channels, Height, Width] numpy arrays
✅ **Channel Independence**: Each channel can use different processing algorithms  
✅ **Memory Efficiency**: TileFlow's tile-based processing keeps memory usage low
✅ **Zarr Compatibility**: Seamless integration with zarr for large dataset handling
✅ **Concurrent Processing**: Multiple channels can be processed simultaneously

## Use Cases Covered

- **DAPI Channel Processing**: Nuclei segmentation with edge detection + thresholding
- **Multi-fluorophore Analysis**: Different algorithms per fluorescent channel
- **Large Dataset Handling**: Zarr-backed processing for datasets larger than RAM
- **Batch Processing**: Concurrent channel processing for throughput optimization

## Performance Notes

- **Memory Usage**: ~128 MB for 8×2048×2048 float32 images in RAM
- **Zarr Compression**: ~98 MB on disk (23% compression) for same dataset
- **Tile Processing**: 256×256 tiles with 16px overlap optimal for most use cases
- **Concurrent Processing**: 4 channels can be processed simultaneously