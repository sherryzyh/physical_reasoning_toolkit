"""
Unit tests for PhysReason dataset downloader.
"""

import zipfile
from unittest.mock import Mock, patch

import pytest

from prkit.prkit_datasets.downloaders import PhysReasonDownloader


class TestPhysReasonDownloader:
    """Test cases for PhysReasonDownloader."""

    def test_downloader_initialization(self):
        """Test that PhysReasonDownloader can be instantiated."""
        downloader = PhysReasonDownloader()
        assert downloader is not None
        assert downloader.dataset_name == "physreason"

    def test_dataset_name_property(self):
        """Test dataset_name property."""
        downloader = PhysReasonDownloader()
        assert downloader.dataset_name == "physreason"

    def test_download_info(self):
        """Test download_info property."""
        downloader = PhysReasonDownloader()
        info = downloader.download_info
        assert isinstance(info, dict)
        assert "source" in info
        assert "repository" in info
        assert info["repository"] == "zhibei1204/PhysReason"

    @patch("requests.get")
    def test_do_download_success(self, mock_get, temp_dir):
        """Test successful download."""
        downloader = PhysReasonDownloader()
        download_dir = temp_dir / "physreason"
        
        # Mock HTTP response
        mock_response = Mock()
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content.return_value = [b"fake zip content"]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Accessing protected method for testing purposes
        result = downloader._do_download(download_dir, variant="full")  # pylint: disable=protected-access
        
        assert result == download_dir
        assert download_dir.exists()
        zip_file = download_dir / "PhysReason-full.zip"
        assert zip_file.exists()

    def test_do_download_invalid_variant(self, temp_dir):
        """Test download with invalid variant."""
        downloader = PhysReasonDownloader()
        download_dir = temp_dir / "physreason"
        
        with pytest.raises(ValueError, match="Unknown variant"):
            # Accessing protected method for testing purposes
            downloader._do_download(download_dir, variant="invalid")  # pylint: disable=protected-access

    def test_verify_valid_zip(self, temp_dir):
        """Test verify method with valid zip file."""
        downloader = PhysReasonDownloader()
        download_dir = temp_dir / "physreason"
        download_dir.mkdir(parents=True)
        
        zip_file = download_dir / "PhysReason-full.zip"
        with zipfile.ZipFile(zip_file, "w") as zf:
            zf.writestr("test.txt", "test content")
        
        assert downloader.verify(download_dir) is True

    def test_verify_missing_files(self, temp_dir):
        """Test verify method with missing files."""
        downloader = PhysReasonDownloader()
        download_dir = temp_dir / "physreason"
        download_dir.mkdir(parents=True)
        
        assert downloader.verify(download_dir) is False

    def test_verify_invalid_zip(self, temp_dir):
        """Test verify method with invalid zip file."""
        downloader = PhysReasonDownloader()
        download_dir = temp_dir / "physreason"
        download_dir.mkdir(parents=True)
        
        zip_file = download_dir / "PhysReason-full.zip"
        zip_file.write_text("not a zip file")
        
        assert downloader.verify(download_dir) is False
