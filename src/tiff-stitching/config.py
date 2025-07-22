"""
WSI Streaming Instance Segmentation Package

Modular, type-hinted, with robust tile/chunk pipeline and error handling.
"""

from typing import Final

# Core parameters
TILE_SIZE: Final[int] = 64
OVERLAP: Final[int] = 24
STRIDE: Final[int] = TILE_SIZE - OVERLAP  # 96
CHUNK_SIZE: Final[int] = 512  # multiple of STRIDE
PARALLEL_ENABLED: Final[bool] = False
