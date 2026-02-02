"""Tests for example implementations."""

import numpy as np
import pytest

from tileflow.examples import (
    SobelEdgeDetector,
    generate_multichannel_image,
    generate_test_image,
    max_filter2d,
    perlin_fbm,
)


class TestImageGeneration:
    """Test synthetic image generation."""

    def test_generate_test_image_perlin(self):
        """Test Perlin noise generation."""
        image = generate_test_image(shape=(128, 256), mode="perlin", seed=42)

        assert image.shape == (128, 256)
        assert image.dtype == np.float32
        assert 0 <= image.min() <= image.max() <= 1

    def test_generate_test_image_random_max(self):
        """Test random max filter generation."""
        image = generate_test_image(shape=(64, 64), mode="random_max", seed=42)

        assert image.shape == (64, 64)
        assert image.dtype == np.float32
        assert 0 <= image.min() <= image.max() <= 1

    def test_generate_test_image_invalid_mode(self):
        """Test invalid mode rejection."""
        with pytest.raises(ValueError, match="Unknown mode"):
            generate_test_image(mode="invalid")

    def test_reproducibility(self):
        """Test seed reproducibility."""
        img1 = generate_test_image(shape=(64, 64), mode="perlin", seed=123)
        img2 = generate_test_image(shape=(64, 64), mode="perlin", seed=123)

        assert np.array_equal(img1, img2)

    def test_multichannel_generation(self):
        """Test multichannel image generation."""
        image = generate_multichannel_image(shape=(4, 100, 100))
        assert image.shape == (4, 100, 100)
        assert image.dtype == np.float32


class TestPerlinNoise:
    """Test Perlin noise implementation."""

    def test_perlin_fbm_basic(self):
        """Test basic fractal Brownian motion."""
        noise = perlin_fbm((64, 64), base_scale=32, octaves=2, seed=42)

        assert noise.shape == (64, 64)
        assert noise.dtype == np.float32
        assert 0 <= noise.min() <= noise.max() <= 1

    def test_perlin_fbm_reproducible(self):
        """Test Perlin noise reproducibility."""
        noise1 = perlin_fbm((32, 32), seed=789)
        noise2 = perlin_fbm((32, 32), seed=789)

        assert np.array_equal(noise1, noise2)


class TestMaxFilter:
    """Test max filter implementation."""

    def test_max_filter_square_kernel(self):
        """Test max filter with square kernel."""
        # Create test image with known structure
        image = np.zeros((10, 10))
        image[5, 5] = 1.0

        filtered = max_filter2d(image, k=3)

        assert filtered.shape == (10, 10)
        assert filtered[5, 5] == 1.0  # Center preserved
        assert filtered[4, 5] == 1.0  # Neighbors affected
        assert filtered[6, 5] == 1.0

    def test_max_filter_rectangular_kernel(self):
        """Test max filter with rectangular kernel."""
        image = np.random.rand(20, 20).astype(np.float32)
        filtered = max_filter2d(image, k=(3, 5))

        assert filtered.shape == (20, 20)
        assert np.all(filtered >= image)  # Max filter can only increase values

    def test_max_filter_validation(self):
        """Test max filter parameter validation."""
        image = np.zeros((10, 10))

        with pytest.raises(AssertionError, match="must be odd"):
            max_filter2d(image, k=4)  # Even kernel

        with pytest.raises(AssertionError, match="must be odd"):
            max_filter2d(image, k=(3, 4))  # One even dimension


class TestSobelEdgeDetector:
    """Test Sobel edge detection implementation."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = SobelEdgeDetector(tile_size=(64, 64), overlap=(8, 8))

        assert detector.tile_size == (64, 64)
        assert detector.overlap == (8, 8)

    def test_sobel_filter(self):
        """Test Sobel filter on simple image."""
        detector = SobelEdgeDetector()

        # Create image with vertical edge
        image = np.zeros((10, 10))
        image[:, 5:] = 1.0

        # Create a dummy tile spec for testing
        from tileflow.core import TileSpec, TileGeometry, BBox, TilePosition

        tile_spec = TileSpec(
            geometry=TileGeometry(core=BBox(0, 0, 10, 10), halo=BBox(0, 0, 10, 10)),
            position=TilePosition(0, 0),
        )
        edges = detector._sobel_filter(image, tile_spec)

        assert edges.shape == (10, 10)
        assert edges.dtype == np.float32
        assert np.all(edges >= 0)  # Magnitude should be non-negative

        # Should detect the vertical edge
        assert edges[:, 4].sum() > 0  # Edge area should have response

    def test_process_integration(self):
        """Test full processing pipeline."""
        detector = SobelEdgeDetector(tile_size=(32, 32), overlap=(4, 4))

        # Create test image
        image = generate_test_image(shape=(64, 64), mode="perlin", seed=42)
        result = detector.process(image)

        assert result.shape == image.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0)

    def test_multichannel_edge_detection(self):
        """Test edge detection on multichannel images."""
        detector = SobelEdgeDetector(tile_size=(64, 64), overlap=(4, 4))
        multichannel_image = generate_multichannel_image(shape=(3, 128, 128))

        # Process each channel
        for c in range(3):
            channel = multichannel_image[c]
            result = detector.process(channel)
            assert result.shape == channel.shape
