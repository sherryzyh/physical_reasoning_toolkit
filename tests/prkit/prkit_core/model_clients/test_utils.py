"""
Tests for model client utility functions.
"""

import base64
import tempfile
from pathlib import Path

import pytest

from prkit.prkit_core.model_clients.utils import encode_image_to_base64


class TestEncodeImageToBase64:
    """Test cases for encode_image_to_base64 function."""

    def test_encode_image_file(self, tmp_path):
        """Test encoding an image file to base64."""
        # Create a test image file
        image_file = tmp_path / "test.jpg"
        test_data = b"fake image data for testing"
        image_file.write_bytes(test_data)

        result = encode_image_to_base64(str(image_file))

        # Verify it's valid base64
        decoded = base64.b64decode(result)
        assert decoded == test_data

    def test_encode_different_image_types(self, tmp_path):
        """Test encoding different image file types."""
        test_data = b"image content"
        
        for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
            image_file = tmp_path / f"test{ext}"
            image_file.write_bytes(test_data)
            
            result = encode_image_to_base64(str(image_file))
            decoded = base64.b64decode(result)
            assert decoded == test_data

    def test_encode_nonexistent_file(self):
        """Test encoding non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            encode_image_to_base64("/nonexistent/image.jpg")

    def test_encode_empty_file(self, tmp_path):
        """Test encoding empty file."""
        image_file = tmp_path / "empty.jpg"
        image_file.write_bytes(b"")

        result = encode_image_to_base64(str(image_file))
        assert result == ""  # Base64 of empty string is empty string

    def test_encode_large_file(self, tmp_path):
        """Test encoding large file."""
        image_file = tmp_path / "large.jpg"
        large_data = b"x" * 10000  # 10KB
        image_file.write_bytes(large_data)

        result = encode_image_to_base64(str(image_file))
        decoded = base64.b64decode(result)
        assert decoded == large_data
        assert len(result) > len(large_data)  # Base64 encoding increases size
