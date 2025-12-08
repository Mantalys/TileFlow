"""Tests for streamable backends."""

import numpy as np
import pytest

from tileflow.backends import NumpyStreamable, as_streamable


class TestNumpyStreamable:
    """Test NumPy streamable backend."""

    def test_initialization(self):
        """Test initialization with valid array."""
        array = np.zeros((100, 200))
        streamable = NumpyStreamable(array)

        assert streamable.shape == (100, 200)
        assert streamable.dtype == array.dtype

    def test_initialization_errors(self):
        """Test initialization error cases."""
        with pytest.raises(TypeError, match="Expected np.ndarray"):
            NumpyStreamable([1, 2, 3])

        with pytest.raises(ValueError, match="at least 2D"):
            NumpyStreamable(np.array([1, 2, 3]))

    def test_getitem(self):
        """Test array slicing."""
        array = np.arange(100).reshape(10, 10)
        streamable = NumpyStreamable(array)

        subset = streamable[2:5, 3:7]
        expected = array[2:5, 3:7]

        assert np.array_equal(subset, expected)

    def test_setitem(self):
        """Test array assignment."""
        array = np.zeros((10, 10))
        streamable = NumpyStreamable(array)

        streamable[2:5, 3:7] = 42
        assert np.all(array[2:5, 3:7] == 42)
        assert np.all(array[0:2, :] == 0)  # Other areas unchanged

    def test_create_output(self):
        """Test output creation."""
        array = np.ones((10, 10), dtype=np.float32)
        streamable = NumpyStreamable(array)

        output = streamable.create_output((20, 30))
        assert isinstance(output, NumpyStreamable)
        assert output.shape == (20, 30)
        assert output.dtype == np.float32

        # Test with custom dtype
        output_int = streamable.create_output((5, 5), dtype=np.int32)
        assert output_int.dtype == np.int32

    def test_array_access(self):
        """Test direct array access."""
        array = np.ones((10, 10))
        streamable = NumpyStreamable(array)

        assert np.array_equal(streamable.array, array)
        assert streamable.array is array  # Should be same object


class TestAsStreamable:
    """Test streamable conversion function."""

    def test_numpy_conversion(self):
        """Test NumPy array conversion."""
        array = np.zeros((50, 100))
        streamable = as_streamable(array)

        assert isinstance(streamable, NumpyStreamable)
        assert streamable.shape == (50, 100)

    def test_already_streamable(self):
        """Test with already streamable input."""
        array = np.zeros((50, 100))
        original_streamable = NumpyStreamable(array)
        result = as_streamable(original_streamable)

        assert result is original_streamable

    def test_unsupported_type(self):
        """Test error for unsupported types."""
        with pytest.raises(TypeError, match="Unsupported data type"):
            as_streamable([1, 2, 3])

        with pytest.raises(TypeError, match="Unsupported data type"):
            as_streamable("string")
