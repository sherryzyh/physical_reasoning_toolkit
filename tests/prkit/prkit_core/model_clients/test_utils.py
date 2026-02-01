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

    def test_encode_binary_data(self, tmp_path):
        """Test encoding binary image data."""
        image_file = tmp_path / "binary.jpg"
        # Create binary data that looks like a minimal JPEG header
        binary_data = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 100
        image_file.write_bytes(binary_data)

        result = encode_image_to_base64(str(image_file))
        decoded = base64.b64decode(result)
        assert decoded == binary_data

    def test_encode_unicode_filename(self, tmp_path):
        """Test encoding file with unicode characters in path."""
        # Create a file with unicode in the name
        unicode_dir = tmp_path / "测试"
        unicode_dir.mkdir()
        image_file = unicode_dir / "图片.jpg"
        test_data = b"test image data"
        image_file.write_bytes(test_data)

        result = encode_image_to_base64(str(image_file))
        decoded = base64.b64decode(result)
        assert decoded == test_data

    def test_encode_special_characters_in_path(self, tmp_path):
        """Test encoding file with special characters in path."""
        special_dir = tmp_path / "test-dir_with spaces"
        special_dir.mkdir()
        image_file = special_dir / "image (1).jpg"
        test_data = b"special chars test"
        image_file.write_bytes(test_data)

        result = encode_image_to_base64(str(image_file))
        decoded = base64.b64decode(result)
        assert decoded == test_data

    def test_encode_read_permissions(self, tmp_path):
        """Test that function handles file read permissions correctly."""
        image_file = tmp_path / "readable.jpg"
        test_data = b"readable data"
        image_file.write_bytes(test_data)
        # File should be readable by default
        image_file.chmod(0o444)  # Read-only

        result = encode_image_to_base64(str(image_file))
        decoded = base64.b64decode(result)
        assert decoded == test_data

    def test_encode_returns_string(self, tmp_path):
        """Test that function returns a string (not bytes)."""
        image_file = tmp_path / "test.jpg"
        image_file.write_bytes(b"test")
        result = encode_image_to_base64(str(image_file))
        assert isinstance(result, str)
        assert not isinstance(result, bytes)

    def test_encode_base64_format(self, tmp_path):
        """Test that result is valid base64 format."""
        import re
        image_file = tmp_path / "test.jpg"
        image_file.write_bytes(b"test data")
        result = encode_image_to_base64(str(image_file))
        # Base64 should only contain A-Z, a-z, 0-9, +, /, and = for padding
        base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
        assert base64_pattern.match(result) is not None
