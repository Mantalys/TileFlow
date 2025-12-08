#!/usr/bin/env python3
"""Advanced multi-channel processing example with independent channel workflows.

This script demonstrates:
- Independent processing pipelines for different channels
- DAPI channel nuclei segmentation workflow
- Concurrent processing of multiple channels
- Channel-specific parameter optimization
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from tileflow.examples import SobelEdgeDetector, generate_test_image
from tileflow.model import TileFlow


class ChannelProcessor:
    """Base class for channel-specific processors."""
    
    def __init__(self, tile_size=(256, 256), overlap=(16, 16)):
        self.tile_size = tile_size
        self.overlap = overlap
    
    def process(self, channel_data: np.ndarray) -> np.ndarray:
        """Process a single channel. Override in subclasses."""
        raise NotImplementedError


class DAPIProcessor(ChannelProcessor):
    """DAPI channel processor for nuclei segmentation."""
    
    def __init__(self, tile_size=(256, 256), overlap=(16, 16), sensitivity=0.95):
        super().__init__(tile_size, overlap)
        self.sensitivity = sensitivity
        self.sobel = SobelEdgeDetector(tile_size=tile_size, overlap=overlap)
    
    def process(self, channel_data: np.ndarray) -> np.ndarray:
        """Segment nuclei from DAPI channel."""
        # Edge detection using TileFlow
        edges = self.sobel.process(channel_data)
        
        # Adaptive thresholding for nuclei detection
        threshold = np.percentile(edges, self.sensitivity * 100)
        nuclei_mask = edges > threshold
        
        # Simple morphological operations using max filter
        from tileflow.examples import max_filter2d
        
        # Close small gaps
        closed = max_filter2d(nuclei_mask.astype(np.float32), k=3) > 0.5
        
        return closed.astype(np.uint8)


class FITCProcessor(ChannelProcessor):
    """FITC channel processor for cytoplasm detection."""
    
    def __init__(self, tile_size=(256, 256), overlap=(16, 16)):
        super().__init__(tile_size, overlap)
    
    def process(self, channel_data: np.ndarray) -> np.ndarray:
        """Detect cytoplasm structures in FITC channel."""
        # Custom tile function for cytoplasm enhancement
        def enhance_cytoplasm(tile):
            # Simple enhancement: smooth then threshold
            from tileflow.examples import max_filter2d
            smoothed = max_filter2d(tile, k=5)
            enhanced = smoothed - tile  # Difference of filters
            return np.clip(enhanced, 0, 1)
        
        processor = TileFlow(tile_size=self.tile_size, overlap=self.overlap)
        processor.configure(function=enhance_cytoplasm)
        return processor.run(channel_data)


class GenericProcessor(ChannelProcessor):
    """Generic processor for other channels."""
    
    def process(self, channel_data: np.ndarray) -> np.ndarray:
        """Apply standard edge detection."""
        sobel = SobelEdgeDetector(tile_size=self.tile_size, overlap=self.overlap)
        return sobel.process(channel_data)


def create_realistic_multichannel_image(channels=8, height=2048, width=2048, seed=42):
    """Create a more realistic multi-channel microscopy image."""
    np.random.seed(seed)
    image = np.zeros((channels, height, width), dtype=np.float32)
    
    # Channel-specific patterns mimicking real microscopy data
    channel_configs = [
        {"mode": "random_max", "max_k": 15, "name": "DAPI"},      # Nuclei
        {"mode": "perlin", "perlin_scale": 32, "name": "FITC"},   # Cytoplasm
        {"mode": "perlin", "perlin_scale": 64, "name": "TRITC"},  # Protein
        {"mode": "random_max", "max_k": 7, "name": "Cy5"},       # Membrane
        {"mode": "perlin", "perlin_scale": 128, "name": "GFP"},   # Organelles
        {"mode": "perlin", "perlin_scale": 16, "name": "mCherry"},  # Vesicles
        {"mode": "random_max", "max_k": 11, "name": "AF647"},    # Antibody
        {"mode": "perlin", "perlin_scale": 96, "name": "BF"},     # Brightfield
    ]
    
    for c, config in enumerate(channel_configs[:channels]):
        print(f"Generating channel {c}: {config['name']}")
        image[c] = generate_test_image(
            (height, width), 
            mode=config["mode"], 
            seed=seed + c,
            **{k: v for k, v in config.items() if k not in ["mode", "name"]}
        )
    
    return image, [config["name"] for config in channel_configs[:channels]]


def process_channels_independently(image_chw: np.ndarray, channel_names: list[str]):
    """Process each channel with its specialized processor."""
    processors = {
        0: DAPIProcessor(sensitivity=0.95),  # DAPI - nuclei
        1: FITCProcessor(),                  # FITC - cytoplasm  
        2: GenericProcessor(),               # TRITC - generic
        3: GenericProcessor(),               # Cy5 - generic
    }
    
    results = {}
    processing_stats = {}
    
    print("Processing channels independently:")
    
    for channel_idx in range(image_chw.shape[0]):
        channel_name = (
            channel_names[channel_idx] if channel_idx < len(channel_names) else f"Ch{channel_idx}"
        )
        print(f"  Channel {channel_idx} ({channel_name})... ", end="", flush=True)
        
        # Select appropriate processor
        processor = processors.get(channel_idx, GenericProcessor())
        
        # Process the channel
        channel_data = image_chw[channel_idx]
        result = processor.process(channel_data)
        
        # Collect stats
        stats = {
            "input_range": (channel_data.min(), channel_data.max()),
            "output_range": (result.min(), result.max()),
            "output_dtype": str(result.dtype),
            "processor": type(processor).__name__
        }
        
        results[channel_idx] = result
        processing_stats[channel_idx] = stats
        
        print(f"✓ [{stats['processor']}]")
    
    return results, processing_stats


def concurrent_channel_processing(image_chw: np.ndarray, channel_names: list[str]):
    """Process multiple channels concurrently."""
    print("\nConcurrent processing demo:")
    
    def process_single_channel(args):
        channel_idx, channel_data, channel_name = args
        processor = DAPIProcessor() if channel_idx == 0 else GenericProcessor()
        result = processor.process(channel_data)
        return channel_idx, result, type(processor).__name__
    
    # Prepare arguments for concurrent processing
    tasks = [
        (i, image_chw[i], channel_names[i] if i < len(channel_names) else f"Ch{i}")
        for i in range(min(4, image_chw.shape[0]))  # Process first 4 channels
    ]
    
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_single_channel, task): task for task in tasks}
        
        for future in as_completed(futures):
            channel_idx, result, processor_name = future.result()
            results[channel_idx] = result
            print(f"  Channel {channel_idx} complete [{processor_name}]")
    
    return results


def main():
    """Demonstrate multi-channel processing with TileFlow."""
    print("TileFlow Multi-Channel Processing Example")
    print("=" * 50)
    
    # Create synthetic multi-channel image
    print("Creating synthetic CHW image [8, 2048, 2048]...")
    image_chw, channel_names = create_realistic_multichannel_image()
    print(f"Image shape: {image_chw.shape}")
    print(f"Channels: {channel_names}")
    print(f"Memory usage: {image_chw.nbytes / 1024 / 1024:.1f} MB")
    print()
    
    # Sequential processing
    results, stats = process_channels_independently(image_chw, channel_names)
    
    print("\nProcessing Statistics:")
    for ch_idx, stat in stats.items():
        ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"Ch{ch_idx}"
        print(
            f"  {ch_name} (Ch{ch_idx}): {stat['input_range']} → "
            f"{stat['output_range']} [{stat['processor']}]"
        )
    
    # Concurrent processing demo
    concurrent_results = concurrent_channel_processing(image_chw, channel_names)
    
    print(f"\nProcessed {len(results)} channels sequentially")
    print(f"Processed {len(concurrent_results)} channels concurrently")
    print("\nKey takeaways:")
    print("  ✓ TileFlow handles CHW format seamlessly")
    print("  ✓ Each channel can use different processing pipelines")
    print("  ✓ Memory-efficient tile-based processing")
    print("  ✓ Concurrent processing supported")


if __name__ == "__main__":
    main()