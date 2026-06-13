"""
Unit tests for the PhysBench dataset downloader.
"""

import json
from unittest.mock import Mock, patch

import pytest

from prkit.datasets.downloaders import PhysBenchDownloader


class TestPhysBenchDownloader:
    """Test cases for PhysBenchDownloader."""

    def test_downloader_initialization(self):
        downloader = PhysBenchDownloader()
        assert downloader.dataset_name == "PhysBench"

    def test_download_info(self):
        downloader = PhysBenchDownloader()
        info = downloader.download_info
        assert info["repository"] == "USC-PSI-Lab/PhysBench"
        assert info["size_bytes"] > 0
        assert "image_zip" in info["optional_media_size_bytes"]

    @patch("requests.get")
    def test_do_download_success(self, mock_get, temp_dir):
        downloader = PhysBenchDownloader()
        download_dir = temp_dir / "PhysBench"

        def _response(payload: bytes):
            response = Mock()
            response.raise_for_status = Mock()
            response.iter_content = Mock(return_value=[payload])
            return response

        mock_get.side_effect = [
            _response(b"# readme"),
            _response(
                json.dumps(
                    [
                        {
                            "idx": 0,
                            "split": "val",
                            "mode": "general",
                            "file_name": ["frame.png"],
                            "question": "A. one\nB. two\nC. three\nD. four",
                        }
                    ]
                ).encode("utf-8")
            ),
        ]

        result = downloader._do_download(
            download_dir
        )  # pylint: disable=protected-access

        assert result == download_dir
        assert (download_dir / "README.md").exists()
        assert (download_dir / "test.json").exists()
        assert (download_dir / "image").exists()
        assert (download_dir / "video").exists()

    def test_do_download_rejects_extract_without_media(self, temp_dir):
        downloader = PhysBenchDownloader()
        download_dir = temp_dir / "PhysBench"

        with pytest.raises(ValueError, match="download_media=True"):
            downloader._do_download(  # pylint: disable=protected-access
                download_dir,
                download_media=False,
                extract_media=True,
            )

    def test_verify_valid_dataset(self, temp_dir):
        downloader = PhysBenchDownloader()
        download_dir = temp_dir / "PhysBench"
        download_dir.mkdir(parents=True)

        with open(download_dir / "test.json", "w", encoding="utf-8") as handle:
            json.dump([{"idx": 0, "question": "test"}], handle)

        assert downloader.verify(download_dir) is True

    def test_verify_invalid_dataset(self, temp_dir):
        downloader = PhysBenchDownloader()
        download_dir = temp_dir / "PhysBench"
        download_dir.mkdir(parents=True)
        (download_dir / "test.json").write_text("not json", encoding="utf-8")

        assert downloader.verify(download_dir) is False
