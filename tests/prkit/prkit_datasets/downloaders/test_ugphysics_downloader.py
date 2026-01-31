"""
Unit tests for UGPhysics dataset downloader.
"""

import json
from unittest.mock import Mock, patch

import pytest

from prkit.prkit_datasets.downloaders import UGPhysicsDownloader


class TestUGPhysicsDownloader:
    """Test cases for UGPhysicsDownloader."""

    def test_downloader_initialization(self):
        """Test that UGPhysicsDownloader can be instantiated."""
        downloader = UGPhysicsDownloader()
        assert downloader is not None
        assert downloader.dataset_name == "ugphysics"

    def test_dataset_name_property(self):
        """Test dataset_name property."""
        downloader = UGPhysicsDownloader()
        assert downloader.dataset_name == "ugphysics"

    def test_download_info(self):
        """Test download_info property."""
        downloader = UGPhysicsDownloader()
        info = downloader.download_info
        assert isinstance(info, dict)
        assert "source" in info
        assert "repository" in info
        assert info["repository"] == "UGPhysics/ugphysics"
        assert "domains" in info
        assert "languages" in info

    def test_domains_property(self):
        """Test DOMAINS class property."""
        assert len(UGPhysicsDownloader.DOMAINS) > 0
        assert "ClassicalMechanics" in UGPhysicsDownloader.DOMAINS

    def test_languages_property(self):
        """Test LANGUAGES class property."""
        assert "en" in UGPhysicsDownloader.LANGUAGES
        assert "zh" in UGPhysicsDownloader.LANGUAGES

    def test_do_download_missing_datasets(self, temp_dir):
        """Test download when datasets library is missing."""
        downloader = UGPhysicsDownloader()
        download_dir = temp_dir / "ugphysics"
        
        with patch.dict("sys.modules", {"datasets": None}):
            with pytest.raises(ImportError, match="datasets"):
                # Accessing protected method for testing purposes
                downloader._do_download(download_dir)  # pylint: disable=protected-access

    def test_do_download_invalid_domain(self, temp_dir):
        """Test download with invalid domain."""
        downloader = UGPhysicsDownloader()
        download_dir = temp_dir / "ugphysics"
        
        with patch("datasets.load_dataset"):
            with pytest.raises(ValueError, match="Invalid domains"):
                # Accessing protected method for testing purposes
                downloader._do_download(download_dir, domains=["InvalidDomain"])  # pylint: disable=protected-access

    def test_do_download_invalid_language(self, temp_dir):
        """Test download with invalid language."""
        downloader = UGPhysicsDownloader()
        download_dir = temp_dir / "ugphysics"
        
        with patch("datasets.load_dataset"):
            with pytest.raises(ValueError, match="Invalid languages"):
                # Accessing protected method for testing purposes
                downloader._do_download(download_dir, languages=["invalid"])  # pylint: disable=protected-access

    @patch("datasets.load_dataset")
    def test_do_download_success(self, mock_load_dataset, temp_dir):
        """Test successful download."""
        downloader = UGPhysicsDownloader()
        download_dir = temp_dir / "ugphysics"
        
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([
            {"index": "test_001", "problem": "Question 1?", "answers": "Answer 1"}
        ]))
        mock_dataset.__len__ = Mock(return_value=1)
        mock_load_dataset.return_value = mock_dataset
        
        # Accessing protected method for testing purposes
        result = downloader._do_download(  # pylint: disable=protected-access
            download_dir, domains=["ClassicalMechanics"], languages=["en"]
        )
        
        assert result == download_dir
        assert download_dir.exists()
        domain_dir = download_dir / "ClassicalMechanics"
        assert domain_dir.exists()
        jsonl_file = domain_dir / "en.jsonl"
        assert jsonl_file.exists()

    def test_verify_valid_dataset(self, temp_dir):
        """Test verify method with valid dataset."""
        downloader = UGPhysicsDownloader()
        download_dir = temp_dir / "ugphysics"
        domain_dir = download_dir / "ClassicalMechanics"
        domain_dir.mkdir(parents=True)
        
        jsonl_file = domain_dir / "en.jsonl"
        sample_data = {"index": "test_001", "problem": "Question?", "answers": "Answer"}
        with open(jsonl_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(sample_data) + "\n")
        
        assert downloader.verify(download_dir) is True

    def test_verify_missing_directory(self, temp_dir):
        """Test verify method with missing directory."""
        downloader = UGPhysicsDownloader()
        download_dir = temp_dir / "ugphysics"
        download_dir.mkdir(parents=True)
        
        assert downloader.verify(download_dir) is False

    def test_verify_no_domain_files(self, temp_dir):
        """Test verify method with no domain files."""
        downloader = UGPhysicsDownloader()
        download_dir = temp_dir / "ugphysics"
        download_dir.mkdir(parents=True)
        
        assert downloader.verify(download_dir) is False

    def test_verify_invalid_jsonl(self, temp_dir):
        """Test verify method with invalid JSONL."""
        downloader = UGPhysicsDownloader()
        download_dir = temp_dir / "ugphysics"
        domain_dir = download_dir / "ClassicalMechanics"
        domain_dir.mkdir(parents=True)
        
        jsonl_file = domain_dir / "en.jsonl"
        jsonl_file.write_text("invalid jsonl")
        
        assert downloader.verify(download_dir) is False

    def test_verify_empty_file(self, temp_dir):
        """Test verify method with empty file."""
        downloader = UGPhysicsDownloader()
        download_dir = temp_dir / "ugphysics"
        domain_dir = download_dir / "ClassicalMechanics"
        domain_dir.mkdir(parents=True)
        
        jsonl_file = domain_dir / "en.jsonl"
        jsonl_file.touch()
        
        assert downloader.verify(download_dir) is False

    def test_resolve_download_dir(self, temp_dir, monkeypatch):
        """Test resolve_download_dir method."""
        downloader = UGPhysicsDownloader()
        
        # Test with explicit data_dir
        resolved = downloader.resolve_download_dir(str(temp_dir))
        assert resolved.resolve() == temp_dir.resolve()
        
        # Test with environment variable
        monkeypatch.setenv("DATASET_CACHE_DIR", str(temp_dir))
        resolved = downloader.resolve_download_dir()
        assert resolved.resolve() == (temp_dir / "ugphysics").resolve()
        
        # Test default fallback
        monkeypatch.delenv("DATASET_CACHE_DIR", raising=False)
        resolved = downloader.resolve_download_dir()
        assert "PHYSICAL_REASONING_DATASETS" in str(resolved)
        assert "ugphysics" in str(resolved)
