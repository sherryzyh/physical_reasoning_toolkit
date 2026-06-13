"""
Unit tests for the PHYSICS dataset downloader.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from prkit.datasets.downloaders import PhysicsDownloader
from prkit.datasets.loaders import PhysicsLoader


def _sample_jsonl_record(problem_id: str) -> bytes:
    record = {
        "id": problem_id,
        "questions": f"Question for {problem_id}?",
        "solutions": f"Solution for {problem_id}",
        "final_answers": ["42"],
        "graphs": [],
    }
    return (json.dumps(record) + "\n").encode("utf-8")


def _write_complete_physics_tree(root: Path) -> None:
    downloader = PhysicsDownloader()
    for relative_path in downloader._relative_file_paths():  # pylint: disable=protected-access
        full_path = root / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        problem_id = full_path.stem.replace("_dataset", "")
        full_path.write_bytes(_sample_jsonl_record(problem_id))


class TestPhysicsDownloader:
    """Test cases for PhysicsDownloader."""

    def test_downloader_initialization(self):
        """Test that PhysicsDownloader can be instantiated."""
        downloader = PhysicsDownloader()
        assert downloader is not None
        assert downloader.dataset_name == "PHYSICS"

    def test_download_info(self):
        """Test download_info property."""
        downloader = PhysicsDownloader()
        info = downloader.download_info
        assert isinstance(info, dict)
        assert info["repository"] == "yale-nlp/Physics"
        assert "variants" in info
        assert "splits" in info

    def test_resolve_download_dir(self, temp_dir, monkeypatch):
        """Test resolve_download_dir behavior."""
        downloader = PhysicsDownloader()

        resolved = downloader.resolve_download_dir(str(temp_dir))
        assert resolved.resolve() == temp_dir.resolve()

        monkeypatch.setenv("DATASET_CACHE_DIR", str(temp_dir))
        resolved = downloader.resolve_download_dir()
        assert resolved.resolve() == (temp_dir / "PHYSICS").resolve()

    @patch("requests.get")
    def test_do_download_success(self, mock_get, temp_dir):
        """Test successful PHYSICS download and loader compatibility."""
        downloader = PhysicsDownloader()
        download_dir = temp_dir / "PHYSICS"

        def side_effect(url, timeout):
            del timeout
            file_name = Path(url).name
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.content = _sample_jsonl_record(file_name)
            return mock_response

        mock_get.side_effect = side_effect

        result = downloader._do_download(  # pylint: disable=protected-access
            download_dir,
            variant="full",
            split="eval",
        )

        assert result == download_dir
        assert downloader.verify(download_dir) is True

        dataset = PhysicsLoader().load(
            data_dir=download_dir,
            variant="full",
            split="eval",
            decode_images=False,
        )
        assert len(dataset) == len(PhysicsDownloader.DOMAINS)

    def test_download_rejects_invalid_variant_split_combo(self, temp_dir):
        """Test invalid PHYSICS variant/split combinations are rejected."""
        downloader = PhysicsDownloader()

        with pytest.raises(ValueError, match="Unsupported PHYSICS variant/split combination"):
            downloader.download(
                data_dir=temp_dir / "PHYSICS",
                variant="hard",
                split="test",
            )

    def test_do_download_missing_requests(self, temp_dir):
        """Test download when requests is unavailable."""
        downloader = PhysicsDownloader()
        download_dir = temp_dir / "PHYSICS"

        with patch.dict("sys.modules", {"requests": None}):
            with pytest.raises(ImportError, match="requests"):
                downloader._do_download(download_dir)  # pylint: disable=protected-access

    def test_verify_valid_dataset(self, temp_dir):
        """Test verify succeeds on a complete PHYSICS tree."""
        downloader = PhysicsDownloader()
        download_dir = temp_dir / "PHYSICS"
        _write_complete_physics_tree(download_dir)

        assert downloader.verify(download_dir) is True

    def test_verify_missing_file(self, temp_dir):
        """Test verify fails when a required file is missing."""
        downloader = PhysicsDownloader()
        download_dir = temp_dir / "PHYSICS"
        _write_complete_physics_tree(download_dir)

        missing_file = download_dir / "PHYSICS-test" / "atomic_dataset_test.jsonl"
        missing_file.unlink()

        assert downloader.verify(download_dir) is False

    def test_verify_invalid_json(self, temp_dir):
        """Test verify fails on malformed JSONL."""
        downloader = PhysicsDownloader()
        download_dir = temp_dir / "PHYSICS"
        _write_complete_physics_tree(download_dir)

        invalid_file = download_dir / "atomic_dataset.jsonl"
        invalid_file.write_text("not json\n", encoding="utf-8")

        assert downloader.verify(download_dir) is False
