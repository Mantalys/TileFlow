"""Enhanced monitoring demonstration with TileFlow callbacks.

This script demonstrates the complete monitoring suite including:
- Progress tracking with performance metrics
- Memory usage monitoring
- Energy consumption tracking (with CodeCarbon)
- Custom analysis callbacks

Run with: python scripts/enhanced_monitoring.py
"""

import numpy as np

from tileflow.callback import (
    CodeCarbonTracker,
    CompositeCallback,
    MemoryTracker,
    MetricsCallback,
    ProgressCallback,
    TileFlowCallback,
)
from tileflow.examples import generate_multichannel_image
from tileflow.model import TileFlow


def demo_comprehensive_monitoring():
    """Demonstrate comprehensive monitoring suite."""
    print("🔬 TileFlow Enhanced Monitoring Demo")
    print("=" * 50)
    
    # Create test data - 8-channel microscopy image
    print("Creating synthetic 8-channel microscopy image...")
    image = generate_multichannel_image(shape=(8, 1024, 1024))
    print(f"Image shape: {image.shape} ({image.nbytes / 1024 / 1024:.1f} MB)")
    
    # Define processing function - DAPI nuclei segmentation
    def nuclei_segmentation_pipeline(x):
        """Multi-stage nuclei segmentation pipeline."""
        if x.ndim == 3:
            # Extract DAPI channel (channel 1)
            dapi = x[1] if x.shape[0] > 1 else x[0]
            
            # Stage 1: Gaussian-like smoothing (approximated)
            smoothed = dapi.copy()
            for _ in range(2):
                smoothed = (
                    0.25 * np.roll(smoothed, 1, axis=0) +
                    0.25 * np.roll(smoothed, -1, axis=0) +
                    0.25 * np.roll(smoothed, 1, axis=1) +
                    0.25 * np.roll(smoothed, -1, axis=1)
                )
            
            # Stage 2: Adaptive thresholding
            local_mean = smoothed.copy()
            for _ in range(3):
                local_mean = (
                    local_mean +
                    np.roll(local_mean, 1, axis=0) +
                    np.roll(local_mean, -1, axis=0) +
                    np.roll(local_mean, 1, axis=1) +
                    np.roll(local_mean, -1, axis=1)
                ) / 5
            
            threshold = local_mean + 0.1 * np.std(smoothed)
            segmented = (smoothed > threshold).astype(np.float32)
            
            # Stage 3: Morphological operations (erosion/dilation)
            result = segmented.copy()
            # Simple erosion
            eroded = np.minimum.reduce([
                result,
                np.roll(result, 1, axis=0),
                np.roll(result, -1, axis=0),
                np.roll(result, 1, axis=1),
                np.roll(result, -1, axis=1)
            ])
            # Simple dilation
            dilated = np.maximum.reduce([
                eroded,
                np.roll(eroded, 1, axis=0),
                np.roll(eroded, -1, axis=0),
                np.roll(eroded, 1, axis=1),
                np.roll(eroded, -1, axis=1)
            ])
            
            return dilated.reshape(1, *dilated.shape)  # Keep channel dimension
        else:
            # Simple threshold for 2D case
            threshold = np.percentile(x, 85)
            return (x > threshold).astype(np.float32).reshape(1, *x.shape)
    
    # Create monitoring suite
    print("\n🔧 Setting up monitoring suite...")
    monitoring_callbacks = [
        ProgressCallback(verbose=True, show_rate=True),
        MemoryTracker(detailed=False),  # Less verbose for cleaner output
        MetricsCallback(verbose=True),
        CodeCarbonTracker(
            project_name="nuclei-segmentation-demo",
            output_dir="./monitoring_logs",
            detailed=True
        )
    ]
    
    # Create processor with moderate tile size for good monitoring
    processor = TileFlow(
        tile_size=(256, 256), 
        overlap=(32, 32),
        name="NucleiSegmentation"
    )
    processor.configure(function=nuclei_segmentation_pipeline)
    
    print("Starting nuclei segmentation with full monitoring...\n")
    
    # Process with monitoring
    result = processor.run(image, callbacks=monitoring_callbacks)
    
    print(f"\n✨ Segmentation Results:")
    print(f"   Output shape: {result.shape}")
    print(f"   Nuclei pixels detected: {np.sum(result > 0):,}")
    print(f"   Nuclei coverage: {(np.sum(result > 0) / result.size) * 100:.2f}%")
    
    # Extract metrics for analysis
    memory_tracker = monitoring_callbacks[1]
    metrics_callback = monitoring_callbacks[2]
    carbon_tracker = monitoring_callbacks[3]
    
    print(f"\n📈 Performance Analysis:")
    memory_stats = memory_tracker.get_memory_stats()
    detailed_metrics = metrics_callback.get_detailed_metrics()
    emissions_data = carbon_tracker.get_emissions_data()
    
    if memory_stats:
        peak_mb = memory_stats['peak_delta_bytes'] / 1024 / 1024
        avg_mb = memory_stats['average_per_tile_bytes'] / 1024 / 1024
        print(f"   Peak memory increase: {peak_mb:.1f} MB")
        print(f"   Average memory per tile: {avg_mb:.1f} MB")
    
    if detailed_metrics:
        print(f"   Processing efficiency: {detailed_metrics['tiles_per_second']:.1f} tiles/sec")
        if detailed_metrics['tile_times_s']:
            tile_times = detailed_metrics['tile_times_s']
            print(f"   Tile processing consistency: {np.std(tile_times):.3f}s std dev")
    
    if emissions_data:
        print(f"   Energy efficiency: {emissions_data.get('emissions_kg', 0):.6f} kg CO₂")


def demo_custom_scientific_callback():
    """Demonstrate custom callback for scientific analysis."""
    print("\n🧪 Custom Scientific Analysis Callback Demo")
    print("=" * 50)
    
    class ImageQualityCallback(TileFlowCallback):
        """Custom callback for image quality assessment."""
        
        def __init__(self):
            self.quality_metrics = []
            self.processing_artifacts = []
            
        def on_tile_end(self, tile, tile_index, total_tiles):
            """Analyze tile quality and artifacts."""
            if hasattr(tile, 'image_data') and tile.image_data:
                data = tile.image_data[0] if isinstance(tile.image_data, list) else tile.image_data
                
                # Calculate quality metrics
                metrics = {
                    'tile_index': tile_index,
                    'signal_to_noise': self._calculate_snr(data),
                    'contrast_ratio': self._calculate_contrast(data),
                    'edge_density': self._calculate_edge_density(data),
                    'uniformity': self._calculate_uniformity(data)
                }
                self.quality_metrics.append(metrics)
                
                # Detect potential artifacts
                if metrics['signal_to_noise'] < 5.0:
                    self.processing_artifacts.append(f"Low SNR in tile {tile_index}")
                if metrics['uniformity'] > 0.8:
                    self.processing_artifacts.append(f"Potential over-smoothing in tile {tile_index}")
                    
        def on_processing_end(self, stats):
            """Generate quality assessment report."""
            if not self.quality_metrics:
                return
                
            # Calculate aggregate statistics
            snr_values = [m['signal_to_noise'] for m in self.quality_metrics]
            contrast_values = [m['contrast_ratio'] for m in self.quality_metrics]
            edge_values = [m['edge_density'] for m in self.quality_metrics]
            
            print(f"\n🔍 Image Quality Assessment:")
            print(f"   Average SNR: {np.mean(snr_values):.2f} ± {np.std(snr_values):.2f}")
            print(f"   Average contrast: {np.mean(contrast_values):.3f} ± {np.std(contrast_values):.3f}")
            print(f"   Average edge density: {np.mean(edge_values):.3f} ± {np.std(edge_values):.3f}")
            
            if self.processing_artifacts:
                print(f"   ⚠️ Artifacts detected: {len(self.processing_artifacts)}")
                for artifact in self.processing_artifacts[:3]:  # Show first 3
                    print(f"     • {artifact}")
            else:
                print(f"   ✅ No processing artifacts detected")
        
        @staticmethod
        def _calculate_snr(data):
            """Calculate signal-to-noise ratio."""
            signal = np.mean(data)
            noise = np.std(data)
            return signal / noise if noise > 0 else float('inf')
        
        @staticmethod
        def _calculate_contrast(data):
            """Calculate RMS contrast."""
            return np.std(data) / np.mean(data) if np.mean(data) > 0 else 0
        
        @staticmethod
        def _calculate_edge_density(data):
            """Calculate edge density using gradient magnitude."""
            if data.size == 0:
                return 0
            grad_x = np.diff(data, axis=1, prepend=0)
            grad_y = np.diff(data, axis=0, prepend=0)
            edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            return np.mean(edge_magnitude)
        
        @staticmethod
        def _calculate_uniformity(data):
            """Calculate uniformity (inverse of variance)."""
            return 1.0 / (1.0 + np.var(data))
    
    # Create test image with varying characteristics
    image = np.random.rand(512, 512).astype(np.float32)
    # Add some structure
    x, y = np.meshgrid(np.linspace(0, 4*np.pi, 512), np.linspace(0, 4*np.pi, 512))
    image += 0.3 * np.sin(x) * np.cos(y)
    
    # Define enhancement function
    def enhance_and_denoise(x):
        """Enhancement with potential for artifacts."""
        # Gaussian-like smoothing
        smoothed = x.copy()
        for _ in range(2):
            smoothed = (
                0.2 * smoothed +
                0.2 * np.roll(smoothed, 1, axis=0) +
                0.2 * np.roll(smoothed, -1, axis=0) +
                0.2 * np.roll(smoothed, 1, axis=1) +
                0.2 * np.roll(smoothed, -1, axis=1)
            )
        
        # Contrast enhancement
        enhanced = smoothed * 1.2
        enhanced = np.clip(enhanced, 0, 1)
        
        return enhanced
    
    processor = TileFlow(tile_size=(128, 128), overlap=(16, 16))
    processor.configure(function=enhance_and_denoise)
    
    # Use quality assessment callback
    callbacks = [
        ProgressCallback(verbose=False),  # Quiet progress
        ImageQualityCallback()
    ]
    
    result = processor.run(image, callbacks=callbacks)
    print(f"Enhancement complete! Output shape: {result.shape}")


def demo_memory_optimized_processing():
    """Demonstrate memory-efficient processing with detailed tracking."""
    print("\n💾 Memory-Optimized Processing Demo")
    print("=" * 50)
    
    # Create large image that would stress memory
    print("Creating large synthetic image for memory testing...")
    large_image = np.random.rand(4096, 4096).astype(np.float32)
    print(f"Large image: {large_image.shape} ({large_image.nbytes / 1024 / 1024:.1f} MB)")
    
    def memory_efficient_filter(x):
        """Memory-efficient processing function."""
        # Use in-place operations where possible
        result = x.copy()
        
        # Simple sharpening filter
        laplacian = (
            4 * result -
            np.roll(result, 1, axis=0) -
            np.roll(result, -1, axis=0) -
            np.roll(result, 1, axis=1) -
            np.roll(result, -1, axis=1)
        )
        
        # Combine with original (unsharp mask)
        result += 0.2 * laplacian
        result = np.clip(result, 0, 1)
        
        return result
    
    # Use chunked processing for memory efficiency
    processor = TileFlow(
        tile_size=(256, 256), 
        overlap=(32, 32),
        chunk_size=(1024, 1024),
        chunk_overlap=(64, 64),
        name="MemoryOptimizedProcessor"
    )
    processor.configure(function=memory_efficient_filter)
    
    # Focus on memory monitoring
    memory_callbacks = [
        ProgressCallback(verbose=True, show_rate=False),
        MemoryTracker(detailed=True)
    ]
    
    print("\nStarting memory-optimized processing...")
    result = processor.run(large_image, callbacks=memory_callbacks)
    
    print(f"Memory-efficient processing complete!")
    print(f"Output shape: {result.shape}")


if __name__ == "__main__":
    """Run all enhanced monitoring demos."""
    print("TileFlow Enhanced Monitoring System")
    print("=" * 60)
    
    # Check for optional dependencies
    try:
        import codecarbon
        print("✅ CodeCarbon available - energy tracking enabled")
    except ImportError:
        print("⚠️  CodeCarbon not installed - energy tracking disabled")
        print("   Install with: pip install codecarbon")
    
    print()
    
    # Run demonstrations
    demo_comprehensive_monitoring()
    demo_custom_scientific_callback()
    demo_memory_optimized_processing()
    
    print("\n🎯 Key Features Demonstrated:")
    print("   • Enhanced progress tracking with performance metrics")
    print("   • Detailed memory usage monitoring") 
    print("   • Energy consumption tracking (with CodeCarbon)")
    print("   • Custom scientific analysis callbacks")
    print("   • Composite callback management")
    print("   • Error handling and graceful degradation")
    print("   • Memory-optimized chunked processing")
    
    print("\n📚 Next Steps:")
    print("   • Install codecarbon for energy tracking: pip install codecarbon")
    print("   • Check ./monitoring_logs/ for detailed carbon footprint data")
    print("   • Customize callbacks for your specific analysis needs")
    print("   • Use CompositeCallback to combine multiple monitoring tools")
    
    print("\n✨ Enhanced monitoring setup complete!")