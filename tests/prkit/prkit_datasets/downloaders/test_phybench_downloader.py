"""
Unit tests for PHYBench dataset downloader.
"""

import json
from unittest.mock import Mock, patch

import pytest

from prkit.prkit_datasets.downloaders import PHYBenchDownloader


class TestPHYBenchDownloader:
    """Test cases for PHYBenchDownloader."""

    def test_downloader_initialization(self):
        """Test that PHYBenchDownloader can be instantiated."""
        downloader = PHYBenchDownloader()
        assert downloader is not None
        assert downloader.dataset_name == "PHYBench"

    def test_dataset_name_property(self):
        """Test dataset_name property."""
        downloader = PHYBenchDownloader()
        assert downloader.dataset_name == "PHYBench"

    def test_download_info(self):
        """Test download_info property."""
        downloader = PHYBenchDownloader()
        info = downloader.download_info
        assert isinstance(info, dict)
        assert "source" in info
        assert "repository" in info
        assert info["repository"] == "Eureka-Lab/PHYBench"

    def test_resolve_download_dir(self, temp_dir, monkeypatch):
        """Test resolve_download_dir method."""
        downloader = PHYBenchDownloader()
        
        # Test with explicit data_dir
        resolved = downloader.resolve_download_dir(str(temp_dir))
        assert resolved.resolve() == temp_dir.resolve()
        
        # Test with environment variable
        monkeypatch.setenv("DATASET_CACHE_DIR", str(temp_dir))
        resolved = downloader.resolve_download_dir()
        assert resolved.resolve() == (temp_dir / "PHYBench").resolve()
        
        # Test default fallback
        monkeypatch.delenv("DATASET_CACHE_DIR", raising=False)
        resolved = downloader.resolve_download_dir()
        assert "PHYSICAL_REASONING_DATASETS" in str(resolved)
        assert "PHYBench" in str(resolved)

    @patch("requests.get")
    def test_do_download_success(self, mock_get, temp_dir):
        """Test successful download."""
        downloader = PHYBenchDownloader()
        download_dir = temp_dir / "PHYBench"
        
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "num_rows_total": 2,
            "rows": [
                {"row": {"id": "test_001", "content": "Question 1", "answer": "Answer 1"}},
                {"row": {"id": "test_002", "content": "Question 2", "answer": "Answer 2"}},
            ],
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Accessing protected method for testing purposes
        result = downloader._do_download(download_dir, split="train")  # pylint: disable=protected-access
        
        assert result == download_dir
        assert download_dir.exists()
        json_file = download_dir / "PHYBench-questions_v1.json"
        assert json_file.exists()
        
        # Verify JSON content
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 2

    def test_do_download_invalid_split(self, temp_dir):
        """Test download with invalid split."""
        downloader = PHYBenchDownloader()
        download_dir = temp_dir / "PHYBench"
        
        with pytest.raises(ValueError, match="only has 'train' split"):
            # Accessing protected method for testing purposes
            downloader._do_download(download_dir, split="test")  # pylint: disable=protected-access

    def test_do_download_missing_requests(self, temp_dir):
        """Test download when requests library is missing."""
        downloader = PHYBenchDownloader()
        download_dir = temp_dir / "PHYBench"
        
        with patch.dict("sys.modules", {"requests": None}):
            with pytest.raises(ImportError, match="requests"):
                # Accessing protected method for testing purposes
                downloader._do_download(download_dir, split="train")  # pylint: disable=protected-access

    def test_verify_valid_dataset(self, temp_dir):
        """Test verify method with valid dataset."""
        downloader = PHYBenchDownloader()
        download_dir = temp_dir / "PHYBench"
        download_dir.mkdir(parents=True)
        
        json_file = download_dir / "PHYBench-questions_v1.json"
        sample_data = [
            {"id": "test_001", "content": "Question 1", "answer": "Answer 1"}
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        assert downloader.verify(download_dir) is True

    def test_verify_missing_file(self, temp_dir):
        """Test verify method with missing file."""
        downloader = PHYBenchDownloader()
        download_dir = temp_dir / "PHYBench"
        download_dir.mkdir(parents=True)
        
        assert downloader.verify(download_dir) is False

    def test_verify_invalid_json(self, temp_dir):
        """Test verify method with invalid JSON."""
        downloader = PHYBenchDownloader()
        download_dir = temp_dir / "PHYBench"
        download_dir.mkdir(parents=True)
        
        json_file = download_dir / "PHYBench-questions_v1.json"
        json_file.write_text("invalid json")
        
        assert downloader.verify(download_dir) is False

    def test_verify_empty_file(self, temp_dir):
        """Test verify method with empty file."""
        downloader = PHYBenchDownloader()
        download_dir = temp_dir / "PHYBench"
        download_dir.mkdir(parents=True)
        
        json_file = download_dir / "PHYBench-questions_v1.json"
        json_file.touch()
        
        assert downloader.verify(download_dir) is False

    def test_is_downloaded(self, temp_dir):
        """Test is_downloaded method."""
        downloader = PHYBenchDownloader()
        download_dir = temp_dir / "PHYBench"
        download_dir.mkdir(parents=True)
        
        json_file = download_dir / "PHYBench-questions_v1.json"
        sample_data = [{"id": "test_001", "content": "Question", "answer": "Answer"}]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        assert downloader.is_downloaded(download_dir) is True
        assert downloader.is_downloaded(temp_dir / "nonexistent") is False
