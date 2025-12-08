"""TileFlow: Fast, memory-efficient image tiling & reconstruction for deep learning."""

from tileflow.backends import NumpyStreamable, Streamable, ZarrStreamable, as_streamable
from tileflow.callback import (
    CodeCarbonTracker,
    CompositeCallback,
    MemoryTracker,
    MetricsCallback,
    ProcessingStats,
    ProgressCallback,
    TileFlowCallback,
)
from tileflow.core import BBox, BoundaryEdges, ProcessedTile, TileGeometry, TileSpec, new_image
from tileflow.examples import SobelEdgeDetector, generate_multichannel_image, generate_test_image
from tileflow.model import TileFlow
from tileflow.utils import estimate_memory_usage, validate_overlap, validate_tile_size

__all__ = [
    "BBox",
    "BoundaryEdges",
    "CodeCarbonTracker",
    "CompositeCallback",
    "MemoryTracker",
    "MetricsCallback",
    "NumpyStreamable",
    "ProcessedTile",
    "ProcessingStats",
    "ProgressCallback",
    "SobelEdgeDetector",
    "Streamable",
    "TileFlow",
    "TileFlowCallback",
    "TileGeometry",
    "TileSpec",
    "ZarrStreamable",
    "as_streamable",
    "estimate_memory_usage",
    "generate_multichannel_image",
    "generate_test_image",
    "new_image",
    "validate_overlap",
    "validate_tile_size",
]
