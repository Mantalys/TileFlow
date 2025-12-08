"""Examples demonstrating the enhanced TileFlow callback system.

This script shows how to use the new callback system for monitoring
energy consumption, memory usage, performance metrics, and progress
during tile-based image processing.
"""

import numpy as np

from tileflow.callback import (
    CodeCarbonTracker,
    CompositeCallback,
    MemoryTracker,
    MetricsCallback,
    ProgressCallback,
)
from tileflow.examples import generate_multichannel_image
from tileflow.model import TileFlow


def example_basic_callbacks():
    """Basic callback usage examples."""
    print("=== Basic Callback Usage ===\n")
    
    # Create test image
    image = generate_multichannel_image(shape=(3, 512, 512))
    print(f"Processing image shape: {image.shape}")
    
    # Simple processing function
    def extract_first_channel(x):
        """Extract and enhance first channel."""
        if x.ndim == 3:
            return x[0:1] * 1.5  # Keep channel dimension
        return x * 1.5
    
    processor = TileFlow(tile_size=(128, 128), overlap=(16, 16))
    processor.configure(function=extract_first_channel)
    
    # Basic progress tracking
    progress = ProgressCallback(verbose=True, show_rate=True)
    result = processor.run(image, callbacks=[progress])
    
    print(f"Result shape: {result.shape}\n")


def example_memory_tracking():
    """Memory usage monitoring example."""
    print("=== Memory Tracking Example ===\n")
    
    # Create larger image for memory tracking
    image = np.random.rand(2048, 2048).astype(np.float32)
    print(f"Processing large image: {image.shape}")
    
    def memory_intensive_function(x):
        """Function that uses extra memory."""
        # Create temporary arrays to increase memory usage
        temp1 = x * 2.0
        temp2 = np.sin(temp1)
        temp3 = np.exp(-temp2)
        return temp3 * 0.5
    
    processor = TileFlow(tile_size=(256, 256), overlap=(32, 32))
    processor.configure(function=memory_intensive_function)
    
    # Track memory usage in detail
    memory_tracker = MemoryTracker(detailed=True)
    result = processor.run(image, callbacks=[memory_tracker])
    
    # Get detailed memory statistics
    memory_stats = memory_tracker.get_memory_stats()
    print(f"\nDetailed Memory Statistics:")
    print(f"Peak memory delta: {memory_stats.get('peak_delta_bytes', 0) / 1024 / 1024:.2f} MB")
    print(f"Average per tile: {memory_stats.get('average_per_tile_bytes', 0) / 1024:.2f} KB")
    print(f"Total tiles tracked: {len(memory_stats.get('memory_per_tile_bytes', []))}")
    print()


def example_energy_tracking():
    """Energy consumption tracking example."""
    print("=== Energy Tracking Example ===\n")
    
    # Create computational workload
    image = np.random.rand(1024, 1024).astype(np.float32)
    
    def compute_intensive_function(x):
        """Computationally expensive function."""
        result = x.copy()
        # Multiple mathematical operations
        for _ in range(3):
            result = np.sin(result) + np.cos(result * 2)
            result = np.exp(-np.abs(result))
        return result
    
    processor = TileFlow(tile_size=(128, 128), overlap=(16, 16))
    processor.configure(function=compute_intensive_function)
    
    # Track energy consumption
    carbon_tracker = CodeCarbonTracker(
        project_name="tileflow-demo",
        output_dir="./carbon_logs",
        detailed=True
    )
    
    result = processor.run(image, callbacks=[carbon_tracker])
    
    # Get emissions data
    emissions_data = carbon_tracker.get_emissions_data()
    print(f"Emissions data captured: {bool(emissions_data)}")
    if emissions_data:
        print(f"Processing time: {emissions_data.get('processing_time_s', 0):.2f}s")
        print(f"Total tiles: {emissions_data.get('total_tiles', 0)}")
    print()


def example_comprehensive_monitoring():
    """Complete monitoring suite example."""
    print("=== Comprehensive Monitoring Suite ===\n")
    
    # Create multi-channel test image
    image = generate_multichannel_image(shape=(8, 1024, 1024))
    print(f"Processing multi-channel image: {image.shape}")
    
    def dapi_nuclei_segmentation(x):
        """Simulate DAPI nuclei segmentation on channel 1."""
        if x.ndim == 3:
            # Extract DAPI channel (typically channel 1)
            dapi = x[1:2] if x.shape[0] > 1 else x[0:1]
            # Simple threshold-based segmentation
            threshold = np.percentile(dapi, 75)
            segmented = (dapi > threshold).astype(np.float32)
            return segmented
        else:
            # 2D case
            threshold = np.percentile(x, 75)
            return (x > threshold).astype(np.float32)
    
    # Create comprehensive monitoring suite
    monitoring_callbacks = [
        ProgressCallback(verbose=True, show_rate=True),
        MemoryTracker(detailed=True),
        MetricsCallback(verbose=True),
        CodeCarbonTracker(project_name="nuclei-segmentation", detailed=True)
    ]
    
    # Alternative: use CompositeCallback for cleaner interface
    composite_monitor = CompositeCallback(monitoring_callbacks)
    
    processor = TileFlow(tile_size=(256, 256), overlap=(32, 32))
    processor.configure(function=dapi_nuclei_segmentation)
    
    print("Starting comprehensive monitoring...")
    result = processor.run(image, callbacks=[composite_monitor])
    
    print(f"\nSegmentation complete!")
    print(f"Output shape: {result.shape}")
    print(f"Nuclei pixels detected: {np.sum(result > 0)}")
    print()


def example_chunked_processing_monitoring():
    """Example of monitoring chunked processing."""
    print("=== Chunked Processing with Monitoring ===\n")
    
    # Create very large image that requires chunking
    large_image = np.random.rand(2048, 2048).astype(np.float32)
    print(f"Processing large image: {large_image.shape}")
    
    def sobel_edge_detection(x):
        """Simple Sobel edge detection."""
        if x.ndim > 2:
            # Process only spatial dimensions
            result = np.zeros_like(x)
            for c in range(x.shape[0]):
                # Sobel kernels
                sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
                
                # Add padding for convolution
                padded = np.pad(x[c], 1, mode='reflect')
                
                # Apply Sobel operators
                grad_x = np.zeros_like(x[c])
                grad_y = np.zeros_like(x[c])
                
                for i in range(x.shape[1]):
                    for j in range(x.shape[2]):
                        patch = padded[i:i+3, j:j+3]
                        grad_x[i, j] = np.sum(patch * sobel_x)
                        grad_y[i, j] = np.sum(patch * sobel_y)
                
                result[c] = np.sqrt(grad_x**2 + grad_y**2)
            return result
        else:
            # 2D processing
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            
            # Simple edge detection (limited implementation for demo)
            return np.abs(np.diff(x, axis=0, prepend=0)) + np.abs(np.diff(x, axis=1, prepend=0))
    
    # Use chunked processing for large image
    processor = TileFlow(
        tile_size=(128, 128), 
        overlap=(16, 16),
        chunk_size=(512, 512),
        chunk_overlap=(64, 64)
    )
    processor.configure(function=sobel_edge_detection)
    
    # Monitor both tile and chunk processing
    callbacks = [
        ProgressCallback(verbose=True),
        MemoryTracker(detailed=False),
        MetricsCallback(verbose=True)
    ]
    
    result = processor.run(large_image, callbacks=callbacks)
    print(f"Edge detection complete! Output shape: {result.shape}\n")


def example_custom_callback():
    """Example of creating a custom callback."""
    print("=== Custom Callback Example ===\n")
    
    from tileflow.callback import TileFlowCallback
    
    class TileAnalysisCallback(TileFlowCallback):
        """Custom callback that analyzes tile statistics."""
        
        def __init__(self):
            self.tile_stats = []
            
        def on_tile_end(self, tile, tile_index, total_tiles):
            """Analyze each processed tile."""
            if hasattr(tile, 'image_data') and tile.image_data:
                data = tile.image_data[0] if isinstance(tile.image_data, list) else tile.image_data
                stats = {
                    'tile_index': tile_index,
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data))
                }
                self.tile_stats.append(stats)
                
        def on_processing_end(self, stats):
            """Print analysis summary."""
            if self.tile_stats:
                means = [s['mean'] for s in self.tile_stats]
                stds = [s['std'] for s in self.tile_stats]
                print(f"📊 Tile Analysis Summary:")
                print(f"   Average tile mean: {np.mean(means):.3f}")
                print(f"   Average tile std: {np.mean(stds):.3f}")
                print(f"   Tile variability: {np.std(means):.3f}")
    
    # Create test image
    image = generate_multichannel_image(shape=(2, 256, 256))
    
    def normalize_channels(x):
        """Normalize each channel independently."""
        if x.ndim == 3:
            result = np.zeros_like(x)
            for c in range(x.shape[0]):
                channel = x[c]
                mean_val = np.mean(channel)
                std_val = np.std(channel)
                if std_val > 0:
                    result[c] = (channel - mean_val) / std_val
                else:
                    result[c] = channel - mean_val
            return result
        else:
            mean_val = np.mean(x)
            std_val = np.std(x)
            return (x - mean_val) / std_val if std_val > 0 else x - mean_val
    
    processor = TileFlow(tile_size=(64, 64), overlap=(8, 8))
    processor.configure(function=normalize_channels)
    
    # Use custom callback
    analysis = TileAnalysisCallback()
    result = processor.run(image, callbacks=[analysis])
    
    print(f"Normalization complete! Processed {len(analysis.tile_stats)} tiles\n")


if __name__ == "__main__":
    """Run all callback examples."""
    print("TileFlow Enhanced Callback System Examples")
    print("=" * 50)
    
    example_basic_callbacks()
    example_memory_tracking()
    example_energy_tracking()
    example_comprehensive_monitoring()
    example_chunked_processing_monitoring()
    example_custom_callback()
    
    print("🎉 All callback examples completed!")