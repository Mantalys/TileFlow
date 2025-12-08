"""Backend abstractions for different data sources (NumPy, Zarr, etc.)."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

try:
    import zarr

    _ZARR_AVAILABLE = True
except ImportError:
    _ZARR_AVAILABLE = False


class Streamable(ABC):
    """Abstract base class for streamable, tilable data sources.

    Provides a common interface for different data backends like NumPy arrays,
    Zarr arrays, or other memory-mapped data structures.
    """

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the data source."""
        pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Get the data type of the source."""
        pass

    @abstractmethod
    def __getitem__(self, key: tuple[slice, ...]) -> np.ndarray:
        """Extract a region from the data source."""
        pass

    @abstractmethod
    def create_output(self, shape: tuple[int, ...], dtype: np.dtype | None = None) -> "Streamable":
        """Create a new output container with the specified shape and dtype."""
        pass

    @abstractmethod
    def __setitem__(self, key: tuple[slice, ...], value: np.ndarray) -> None:
        """Set a region in the data source."""
        pass


class NumpyStreamable(Streamable):
    """NumPy array backend for TileFlow processing."""

    def __init__(self, array: np.ndarray) -> None:
        """Initialize with a NumPy array.

        Parameters
        ----------
        array : np.ndarray
            The NumPy array to wrap
        """
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(array)}")
        if array.ndim < 2:
            raise ValueError(f"Array must be at least 2D, got {array.ndim}D")

        self._array = array

    @property
    def shape(self) -> tuple[int, ...]:
        """Get array shape."""
        return self._array.shape

    @property
    def dtype(self) -> np.dtype:
        """Get array dtype."""
        return self._array.dtype

    def __getitem__(self, key: tuple[slice, ...]) -> np.ndarray:
        """Extract region from array."""
        return self._array[key]

    def create_output(
        self, shape: tuple[int, ...], dtype: np.dtype | None = None
    ) -> "NumpyStreamable":
        """Create new NumPy array for output."""
        output_dtype = dtype or self.dtype
        return NumpyStreamable(np.zeros(shape, dtype=output_dtype))

    def __setitem__(self, key: tuple[slice, ...], value: np.ndarray) -> None:
        """Set region in array."""
        self._array[key] = value

    @property
    def array(self) -> np.ndarray:
        """Access the underlying NumPy array."""
        return self._array


class ZarrStreamable(Streamable):
    """Zarr array backend for TileFlow processing.

    Supports out-of-core processing for large datasets that don't fit in memory.
    """

    def __init__(self, array) -> None:
        """Initialize with a Zarr array.

        Parameters
        ----------
        array : zarr.Array
            The Zarr array to wrap
        """
        if not _ZARR_AVAILABLE:
            raise ImportError("zarr package is required for ZarrStreamable")

        if not hasattr(array, "shape") or not hasattr(array, "dtype"):
            raise TypeError("Expected zarr.Array-like object")

        if len(array.shape) < 2:
            raise ValueError(f"Array must be at least 2D, got {len(array.shape)}D")

        self._array = array

    @property
    def shape(self) -> tuple[int, ...]:
        """Get array shape."""
        return self._array.shape

    @property
    def dtype(self) -> np.dtype:
        """Get array dtype."""
        return self._array.dtype

    def __getitem__(self, key: tuple[slice, ...]) -> np.ndarray:
        """Extract region from array, returns NumPy array."""
        return np.array(self._array[key])

    def create_output(
        self, shape: tuple[int, ...], dtype: np.dtype | None = None
    ) -> "ZarrStreamable":
        """Create new Zarr array for output."""
        output_dtype = dtype or self.dtype
        # Create in-memory zarr array for output
        output_array = zarr.zeros(shape, dtype=output_dtype)
        return ZarrStreamable(output_array)

    def __setitem__(self, key: tuple[slice, ...], value: np.ndarray) -> None:
        """Set region in array."""
        self._array[key] = value

    @property
    def array(self):
        """Access the underlying Zarr array."""
        return self._array


def as_streamable(data: Any) -> Streamable:
    """Convert various data types to Streamable interface.

    Parameters
    ----------
    data : Any
        Data to convert (currently supports np.ndarray)

    Returns
    -------
    Streamable
        Streamable interface for the data

    Raises
    ------
    TypeError
        If data type is not supported
    """
    if isinstance(data, np.ndarray):
        return NumpyStreamable(data)
    elif isinstance(data, Streamable):
        return data
    elif _ZARR_AVAILABLE and hasattr(data, "shape") and hasattr(data, "dtype"):
        # Duck typing for zarr arrays
        return ZarrStreamable(data)
    else:
        supported = "np.ndarray"
        if _ZARR_AVAILABLE:
            supported += ", zarr.Array"
        raise TypeError(f"Unsupported data type: {type(data)}. Supported: {supported}")
