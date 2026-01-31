"""
Unit tests for PhyX dataset downloader.
"""

import json
from unittest.mock import Mock, patch

import pytest

from prkit.prkit_datasets.downloaders import PhyXDownloader


class TestPhyXDownloader:
    """Test cases for PhyXDownloader."""

    def test_downloader_initialization(self):
        """Test that PhyXDownloader can be instantiated."""
        downloader = PhyXDownloader()
        assert downloader is not None
        assert downloader.dataset_name == "phyx"

    def test_dataset_name_property(self):
        """Test dataset_name property."""
        downloader = PhyXDownloader()
        assert downloader.dataset_name == "phyx"

    def test_download_info(self):
        """Test download_info property."""
        downloader = PhyXDownloader()
        info = downloader.download_info
        assert isinstance(info, dict)
        assert "source" in info
        assert "repository" in info
        assert info["repository"] == "Cloudriver/PhyX"
        assert "huggingface_url" in info
        assert "paper_url" in info
        assert "homepage" in info
        assert "license" in info
        assert info["license"] == "MIT"

    def test_resolve_download_dir(self, temp_dir, monkeypatch):
        """Test resolve_download_dir method."""
        downloader = PhyXDownloader()
        
        # Test with explicit data_dir
        resolved = downloader.resolve_download_dir(str(temp_dir))
        assert resolved.resolve() == temp_dir.resolve()
        
        # Test with environment variable
        monkeypatch.setenv("DATASET_CACHE_DIR", str(temp_dir))
        resolved = downloader.resolve_download_dir()
        assert resolved.resolve() == (temp_dir / "phyx").resolve()
        
        # Test default fallback
        monkeypatch.delenv("DATASET_CACHE_DIR", raising=False)
        resolved = downloader.resolve_download_dir()
        assert "PHYSICAL_REASONING_DATASETS" in str(resolved)
        assert "phyx" in str(resolved)

    @patch("datasets.load_dataset")
    def test_do_download_success_with_datasets_lib(self, mock_load_dataset, temp_dir):
        """Test successful download using datasets library."""
        downloader = PhyXDownloader()
        download_dir = temp_dir / "phyx"
        
        # Mock dataset from datasets library
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([
            {"id": "test_001", "question": "Question 1?", "answer": "Answer 1"},
            {"id": "test_002", "question": "Question 2?", "answer": "Answer 2"},
        ]))
        mock_load_dataset.return_value = mock_dataset
        
        # Accessing protected method for testing purposes
        result = downloader._do_download(download_dir, split="test_mini")  # pylint: disable=protected-access
        
        assert result == download_dir
        assert download_dir.exists()
        json_file = download_dir / "PhyX-test_mini.json"
        assert json_file.exists()
        
        # Verify JSON content
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["id"] == "test_001"

    @patch("requests.get")
    def test_do_download_success_with_requests_fallback(self, mock_get, temp_dir):
        """Test successful download using requests API fallback."""
        downloader = PhyXDownloader()
        download_dir = temp_dir / "phyx"
        
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "num_rows_total": 2,
            "rows": [
                {"row": {"id": "test_001", "question": "Question 1?", "answer": "Answer 1"}},
                {"row": {"id": "test_002", "question": "Question 2?", "answer": "Answer 2"}},
            ],
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Mock datasets library as unavailable
        with patch.dict("sys.modules", {"datasets": None}):
            # Accessing protected method for testing purposes
            result = downloader._do_download(download_dir, split="test_mini")  # pylint: disable=protected-access
        
        assert result == download_dir
        assert download_dir.exists()
        json_file = download_dir / "PhyX-test_mini.json"
        assert json_file.exists()
        
        # Verify JSON content
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 2

    def test_do_download_missing_requests(self, temp_dir):
        """Test download when requests library is missing (and datasets unavailable)."""
        downloader = PhyXDownloader()
        download_dir = temp_dir / "phyx"
        
        with patch.dict("sys.modules", {"requests": None, "datasets": None}):
            with pytest.raises(ImportError, match="requests"):
                # Accessing protected method for testing purposes
                downloader._do_download(download_dir, split="test_mini")  # pylint: disable=protected-access

    def test_do_download_invalid_split(self, temp_dir):
        """Test download with invalid split."""
        downloader = PhyXDownloader()
        download_dir = temp_dir / "phyx"
        
        # PhyX supports test_mini by default, but we can test with an unsupported split
        # The actual implementation may not validate splits strictly, so this test
        # may need adjustment based on actual behavior
        with patch("datasets.load_dataset") as mock_load:
            mock_load.side_effect = ValueError("Split not found")
            with pytest.raises((ValueError, RuntimeError)):
                # Accessing protected method for testing purposes
                downloader._do_download(download_dir, split="invalid_split")  # pylint: disable=protected-access

    def test_verify_valid_dataset(self, temp_dir):
        """Test verify method with valid dataset."""
        downloader = PhyXDownloader()
        download_dir = temp_dir / "phyx"
        download_dir.mkdir(parents=True)
        
        json_file = download_dir / "PhyX-test_mini.json"
        sample_data = [
            {"id": "test_001", "question": "Question 1?", "answer": "Answer 1"}
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        assert downloader.verify(download_dir) is True

    def test_verify_missing_file(self, temp_dir):
        """Test verify method with missing file."""
        downloader = PhyXDownloader()
        download_dir = temp_dir / "phyx"
        download_dir.mkdir(parents=True)
        
        assert downloader.verify(download_dir) is False

    def test_verify_invalid_json(self, temp_dir):
        """Test verify method with invalid JSON."""
        downloader = PhyXDownloader()
        download_dir = temp_dir / "phyx"
        download_dir.mkdir(parents=True)
        
        json_file = download_dir / "PhyX-test_mini.json"
        json_file.write_text("invalid json")
        
        assert downloader.verify(download_dir) is False

    def test_verify_empty_file(self, temp_dir):
        """Test verify method with empty file."""
        downloader = PhyXDownloader()
        download_dir = temp_dir / "phyx"
        download_dir.mkdir(parents=True)
        
        json_file = download_dir / "PhyX-test_mini.json"
        json_file.touch()
        
        assert downloader.verify(download_dir) is False

    def test_verify_empty_list(self, temp_dir):
        """Test verify method with empty list."""
        downloader = PhyXDownloader()
        download_dir = temp_dir / "phyx"
        download_dir.mkdir(parents=True)
        
        json_file = download_dir / "PhyX-test_mini.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump([], f)
        
        assert downloader.verify(download_dir) is False

    def test_verify_finds_any_phyx_file(self, temp_dir):
        """Test verify method finds any PhyX JSON file."""
        downloader = PhyXDownloader()
        download_dir = temp_dir / "phyx"
        download_dir.mkdir(parents=True)
        
        # Create a file with different split name
        json_file = download_dir / "PhyX-custom.json"
        sample_data = [
            {"id": "test_001", "question": "Question 1?", "answer": "Answer 1"}
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        assert downloader.verify(download_dir) is True

    def test_is_downloaded(self, temp_dir):
        """Test is_downloaded method."""
        downloader = PhyXDownloader()
        download_dir = temp_dir / "phyx"
        download_dir.mkdir(parents=True)
        
        json_file = download_dir / "PhyX-test_mini.json"
        sample_data = [{"id": "test_001", "question": "Question?", "answer": "Answer"}]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        assert downloader.is_downloaded(download_dir) is True
        assert downloader.is_downloaded(temp_dir / "nonexistent") is False

    @patch("PIL.Image")
    def test_process_rows_for_json_with_images(self, mock_pil_module, temp_dir):
        """Test _process_rows_for_json handles images correctly."""
        downloader = PhyXDownloader()
        download_dir = temp_dir / "phyx"
        
        # Mock PIL Image
        mock_pil_image = Mock()
        mock_pil_image.mode = "RGB"
        mock_pil_image.convert.return_value = mock_pil_image
        mock_pil_image.save = Mock()
        
        rows = [
            {
                "id": "test_001",
                "question": "Question 1?",
                "answer": "Answer 1",
                "image": mock_pil_image,
            }
        ]
        
        # Accessing protected method for testing purposes
        processed = downloader._process_rows_for_json(rows, download_dir)  # pylint: disable=protected-access
        
        assert len(processed) == 1
        assert "image_paths" in processed[0]
        assert processed[0]["id"] == "test_001"

    def test_process_rows_for_json_without_images(self, temp_dir):
        """Test _process_rows_for_json handles rows without images."""
        downloader = PhyXDownloader()
        download_dir = temp_dir / "phyx"
        
        rows = [
            {
                "id": "test_001",
                "question": "Question 1?",
                "answer": "Answer 1",
            }
        ]
        
        # Accessing protected method for testing purposes
        processed = downloader._process_rows_for_json(rows, download_dir)  # pylint: disable=protected-access
        
        assert len(processed) == 1
        assert processed[0]["id"] == "test_001"
        assert processed[0]["question"] == "Question 1?"

    def test_make_json_serializable(self):
        """Test _make_json_serializable method."""
        downloader = PhyXDownloader()
        
        # Test with various types
        # Accessing protected method for testing purposes
        assert downloader._make_json_serializable(None) is None  # pylint: disable=protected-access
        assert downloader._make_json_serializable("string") == "string"  # pylint: disable=protected-access
        assert downloader._make_json_serializable(42) == 42  # pylint: disable=protected-access
        assert downloader._make_json_serializable(3.14) == 3.14  # pylint: disable=protected-access
        assert downloader._make_json_serializable(True) is True  # pylint: disable=protected-access
        
        # Test with dict
        result = downloader._make_json_serializable({"key": "value"})  # pylint: disable=protected-access
        assert result == {"key": "value"}
        
        # Test with list
        result = downloader._make_json_serializable([1, 2, 3])  # pylint: disable=protected-access
        assert result == [1, 2, 3]
