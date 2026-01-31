"""
Unit tests for SeePhys dataset downloader.
"""

import json
from unittest.mock import Mock, patch

import pytest

from prkit.prkit_datasets.downloaders import SeePhysDownloader


class TestSeePhysDownloader:
    """Test cases for SeePhysDownloader."""

    def test_downloader_initialization(self):
        """Test that SeePhysDownloader can be instantiated."""
        downloader = SeePhysDownloader()
        assert downloader is not None
        assert downloader.dataset_name == "seephys"

    def test_dataset_name_property(self):
        """Test dataset_name property."""
        downloader = SeePhysDownloader()
        assert downloader.dataset_name == "seephys"

    def test_download_info(self):
        """Test download_info property."""
        downloader = SeePhysDownloader()
        info = downloader.download_info
        assert isinstance(info, dict)
        assert "source" in info
        assert "repository" in info
        assert info["repository"] == "SeePhys/SeePhys"

    def test_do_download_missing_datasets(self, temp_dir):
        """Test download when datasets library is missing."""
        downloader = SeePhysDownloader()
        download_dir = temp_dir / "seephys"
        
        with patch.dict("sys.modules", {"datasets": None}):
            with pytest.raises(ImportError, match="datasets"):
                # Accessing protected method for testing purposes
                downloader._do_download(download_dir, split="train")  # pylint: disable=protected-access

    def test_do_download_invalid_split(self, temp_dir):
        """Test download with invalid split."""
        downloader = SeePhysDownloader()
        download_dir = temp_dir / "seephys"
        
        with pytest.raises(ValueError, match="only has 'train' split"):
            # Accessing protected method for testing purposes
            downloader._do_download(download_dir, split="test")  # pylint: disable=protected-access

    @patch("datasets.load_dataset")
    def test_do_download_success(self, mock_load_dataset, temp_dir):
        """Test successful download."""
        downloader = SeePhysDownloader()
        download_dir = temp_dir / "seephys"
        
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.to_pandas.return_value = Mock()
        mock_dataset.__len__ = Mock(return_value=1)
        mock_load_dataset.return_value = mock_dataset
        
        # Mock pandas DataFrame
        try:
            import pandas as pd
            
            mock_df = pd.DataFrame([
                {
                    "index": "test_001",
                    "question": "Question 1?",
                    "answer": "Answer 1",
                }
            ])
            mock_dataset.to_pandas.return_value = mock_df
            
            # Accessing protected method for testing purposes
            result = downloader._do_download(download_dir, split="train")  # pylint: disable=protected-access
            
            assert result.resolve() == download_dir.resolve()
            assert download_dir.exists()
            parquet_file = download_dir / "train.parquet"
            assert parquet_file.exists()
        except ImportError:
            pytest.skip("pandas/pyarrow not available")

    def test_verify_valid_dataset(self, temp_dir):
        """Test verify method with valid dataset."""
        downloader = SeePhysDownloader()
        download_dir = temp_dir / "seephys"
        train_dir = download_dir / "train"
        train_dir.mkdir(parents=True)
        
        # Create sample JSON file
        json_file = train_dir / "001.json"
        sample_data = {"index": "test_001", "question": "Question?", "answer": "Answer"}
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        
        assert downloader.verify(download_dir) is True

    def test_verify_missing_directory(self, temp_dir):
        """Test verify method with missing directory."""
        downloader = SeePhysDownloader()
        download_dir = temp_dir / "seephys"
        download_dir.mkdir(parents=True)
        
        assert downloader.verify(download_dir) is False

    def test_verify_no_json_files(self, temp_dir):
        """Test verify method with no JSON files."""
        downloader = SeePhysDownloader()
        download_dir = temp_dir / "seephys"
        train_dir = download_dir / "train"
        train_dir.mkdir(parents=True)
        
        assert downloader.verify(download_dir) is False

    def test_verify_invalid_json(self, temp_dir):
        """Test verify method with invalid JSON."""
        downloader = SeePhysDownloader()
        download_dir = temp_dir / "seephys"
        train_dir = download_dir / "train"
        train_dir.mkdir(parents=True)
        
        json_file = train_dir / "001.json"
        json_file.write_text("invalid json")
        
        assert downloader.verify(download_dir) is False
