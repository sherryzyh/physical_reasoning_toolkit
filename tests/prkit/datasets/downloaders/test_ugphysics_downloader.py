"""
Unit tests for UGPhysics dataset downloader.
"""

import json
from unittest.mock import Mock, patch

import pytest

from prkit.datasets.downloaders import UGPhysicsDownloader


class TestUGPhysicsDownloader:
    """Test cases for UGPhysicsDownloader."""

    def test_download_info_reflects_real_public_contract(self):
        """Downloader metadata should expose real splits and variants."""
        downloader = UGPhysicsDownloader()
        info = downloader.download_info

        assert info["repository"] == "UGPhysics/ugphysics"
        assert info["splits"] == ["en", "zh"]
        assert "full" in info["variants"]
        assert "classical_mechanics" in info["variants"]
        assert "mini" not in info["variants"]
        assert info["total_problems"]["en"] == 5520

    def test_resolve_requested_artifacts_defaults_to_full_en(self):
        """Direct downloader usage should resolve to the canonical full/en request."""
        downloader = UGPhysicsDownloader()

        variant, split, domains, languages = (
            downloader._resolve_requested_artifacts(  # pylint: disable=protected-access
                variant=None,
                split=None,
            )
        )

        assert variant == "full"
        assert split is None
        assert "ClassicalMechanics" in domains
        assert languages == ["en"]

    def test_resolve_requested_artifacts_accepts_legacy_aliases(self):
        """Legacy mini/test requests should normalize without breaking callers."""
        downloader = UGPhysicsDownloader()

        with patch.object(downloader.logger, "warning") as mock_warning:
            variant, split, domains, languages = (
                downloader._resolve_requested_artifacts(  # pylint: disable=protected-access
                    variant="mini",
                    split="test",
                    language="zh",
                )
            )

        assert variant == "full"
        assert split == "zh"
        assert "AtomicPhysics" in domains
        assert languages == ["zh"]
        mock_warning.assert_called()

    def test_resolve_requested_artifacts_rejects_conflicting_domains(self):
        """Domain overrides should not contradict a narrowed variant."""
        downloader = UGPhysicsDownloader()

        with pytest.raises(ValueError, match="Conflicting UGPhysics domain request"):
            downloader._resolve_requested_artifacts(  # pylint: disable=protected-access
                variant="atomic_physics",
                domains=["ClassicalMechanics"],
            )

    def test_do_download_missing_datasets(self, temp_dir):
        """Downloading should fail clearly when datasets is unavailable."""
        downloader = UGPhysicsDownloader()
        download_dir = temp_dir / "ugphysics"

        with patch.dict("sys.modules", {"datasets": None}):
            with pytest.raises(ImportError, match="datasets"):
                downloader._do_download(
                    download_dir
                )  # pylint: disable=protected-access

    @patch("datasets.load_dataset")
    def test_do_download_success_writes_manifest(self, mock_load_dataset, temp_dir):
        """Downloader should fetch the requested shard and persist a manifest."""
        downloader = UGPhysicsDownloader()
        download_dir = temp_dir / "ugphysics"

        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(
            return_value=iter(
                [
                    {
                        "index": "test_001",
                        "problem": "Question 1?",
                        "answers": "Answer 1",
                    }
                ]
            )
        )
        mock_dataset.__len__ = Mock(return_value=1)
        mock_load_dataset.return_value = mock_dataset

        result = downloader._do_download(  # pylint: disable=protected-access
            download_dir,
            variant="classical_mechanics",
            split="en",
        )

        assert result.resolve() == download_dir.resolve()
        jsonl_file = download_dir / "ClassicalMechanics" / "en.jsonl"
        manifest_path = download_dir / "download_manifest.json"
        assert jsonl_file.exists()
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["last_request"] == {
            "variant": "classical_mechanics",
            "split": "en",
        }
        assert manifest["files"] == [
            {
                "domain": "ClassicalMechanics",
                "language": "en",
                "path": "ClassicalMechanics/en.jsonl",
                "rows": 1,
            }
        ]
        mock_load_dataset.assert_called_once_with(
            "UGPhysics/ugphysics",
            name="ClassicalMechanics",
            split="en",
        )

    @patch("datasets.load_dataset")
    def test_do_download_raises_on_partial_failure(self, mock_load_dataset, temp_dir):
        """A partial UGPhysics download should raise instead of silently succeeding."""
        downloader = UGPhysicsDownloader()
        download_dir = temp_dir / "ugphysics"

        def fake_load_dataset(dataset_name, name, split):
            if name == "AtomicPhysics":
                raise RuntimeError("network issue")
            dataset = Mock()
            dataset.__iter__ = Mock(
                return_value=iter(
                    [
                        {
                            "index": "test_001",
                            "problem": "Question 1?",
                            "answers": "Answer 1",
                        }
                    ]
                )
            )
            dataset.__len__ = Mock(return_value=1)
            return dataset

        mock_load_dataset.side_effect = fake_load_dataset

        with pytest.raises(RuntimeError, match="download incomplete"):
            downloader._do_download(  # pylint: disable=protected-access
                download_dir,
                domains=["AtomicPhysics", "ClassicalMechanics"],
                languages=["en"],
            )

    def test_download_skips_when_exact_request_exists(self, temp_dir):
        """download() should skip only when the requested shard already exists."""
        downloader = UGPhysicsDownloader()
        download_dir = temp_dir / "ugphysics"
        domain_dir = download_dir / "ClassicalMechanics"
        domain_dir.mkdir(parents=True)
        (domain_dir / "en.jsonl").write_text(
            json.dumps({"index": "1", "problem": "Q", "answers": "A"}) + "\n",
            encoding="utf-8",
        )

        with patch.object(downloader, "_do_download") as mock_do_download:
            result = downloader.download(
                data_dir=download_dir,
                variant="classical_mechanics",
                split="en",
            )

        assert result.resolve() == download_dir.resolve()
        mock_do_download.assert_not_called()

    def test_download_retries_when_requested_split_is_missing(self, temp_dir):
        """A cache with en only should not satisfy a zh request."""
        downloader = UGPhysicsDownloader()
        download_dir = temp_dir / "ugphysics"
        domain_dir = download_dir / "ClassicalMechanics"
        domain_dir.mkdir(parents=True)
        (domain_dir / "en.jsonl").write_text(
            json.dumps({"index": "1", "problem": "Q", "answers": "A"}) + "\n",
            encoding="utf-8",
        )

        with patch.object(
            downloader, "_do_download", return_value=download_dir
        ) as mock_do_download:
            result = downloader.download(
                data_dir=download_dir,
                variant="classical_mechanics",
                split="zh",
            )

        assert result == download_dir
        mock_do_download.assert_called_once()
        assert mock_do_download.call_args.kwargs["languages"] == ["zh"]

    def test_verify_uses_manifest(self, temp_dir):
        """verify() should validate every shard listed in the manifest."""
        downloader = UGPhysicsDownloader()
        download_dir = temp_dir / "ugphysics"
        domain_dir = download_dir / "ClassicalMechanics"
        domain_dir.mkdir(parents=True)
        jsonl_file = domain_dir / "en.jsonl"
        jsonl_file.write_text(
            json.dumps({"index": "1", "problem": "Q", "answers": "A"}) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "dataset": "ugphysics",
            "repository": "UGPhysics/ugphysics",
            "last_request": {"variant": "classical_mechanics", "split": "en"},
            "files": [
                {
                    "domain": "ClassicalMechanics",
                    "language": "en",
                    "path": "ClassicalMechanics/en.jsonl",
                    "rows": 1,
                }
            ],
        }
        (download_dir / "download_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        assert downloader.verify(download_dir) is True

        jsonl_file.unlink()
        assert downloader.verify(download_dir) is False

    def test_verify_fallback_without_manifest(self, temp_dir):
        """verify() should still work against raw shard files without a manifest."""
        downloader = UGPhysicsDownloader()
        download_dir = temp_dir / "ugphysics"
        domain_dir = download_dir / "AtomicPhysics"
        domain_dir.mkdir(parents=True)
        (domain_dir / "zh.jsonl").write_text(
            json.dumps({"index": "1", "problem": "Q", "answers": "A"}) + "\n",
            encoding="utf-8",
        )

        assert downloader.verify(download_dir) is True

    def test_resolve_download_dir(self, temp_dir, monkeypatch):
        """resolve_download_dir should still honor explicit and env-based roots."""
        downloader = UGPhysicsDownloader()

        resolved = downloader.resolve_download_dir(str(temp_dir))
        assert resolved.resolve() == temp_dir.resolve()

        monkeypatch.setenv("DATASET_CACHE_DIR", str(temp_dir))
        resolved = downloader.resolve_download_dir()
        assert resolved.resolve() == (temp_dir / "ugphysics").resolve()
