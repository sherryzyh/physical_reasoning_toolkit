from pathlib import Path

import pytest

from prkit.datasets.downloaders.base_downloader import BaseDownloader


class DummyDownloader(BaseDownloader):
    @property
    def dataset_name(self) -> str:
        return "dummy"

    @property
    def download_info(self):
        return {
            "variants": ["mini", "full"],
            "splits": ["train", "full"],
            "size_bytes": 123,
            "source": "unit-test",
        }

    def _do_download(self, download_dir: Path, **kwargs) -> Path:
        download_dir.mkdir(parents=True, exist_ok=True)
        (download_dir / "payload.txt").write_text(kwargs.get("marker", "ok"))
        return download_dir

    def verify(self, data_dir) -> bool:
        return (Path(data_dir) / "payload.txt").exists()


def test_base_downloader_defaults_and_validation():
    downloader = DummyDownloader()

    assert downloader.get_default_variant() == "full"
    assert downloader.get_default_split() == "full"
    assert downloader.get_available_variants() == ["mini", "full"]
    assert downloader.get_available_splits() == ["train", "full"]
    assert downloader.get_download_size() == 123
    assert downloader.get_download_source() == "unit-test"

    downloader.validate_variant("mini")
    downloader.validate_split("train")

    with pytest.raises(ValueError, match="Unknown variant"):
        downloader.validate_variant("bad")
    with pytest.raises(ValueError, match="Unknown split"):
        downloader.validate_split("bad")


def test_base_downloader_download_skip_and_force(tmp_path):
    downloader = DummyDownloader()
    target_dir = tmp_path / "dummy"

    result = downloader.download(target_dir, marker="first")
    assert result == target_dir
    assert (target_dir / "payload.txt").read_text() == "first"

    skipped = downloader.download(target_dir, marker="second")
    assert skipped == target_dir
    assert (target_dir / "payload.txt").read_text() == "first"

    forced = downloader.download(target_dir, force=True, marker="second")
    assert forced == target_dir
    assert (target_dir / "payload.txt").read_text() == "second"


def test_base_downloader_resolve_and_is_downloaded(tmp_path, monkeypatch):
    downloader = DummyDownloader()

    explicit = downloader.resolve_download_dir(tmp_path / "explicit")
    assert explicit == (tmp_path / "explicit").resolve()

    monkeypatch.setenv("DATASET_CACHE_DIR", str(tmp_path))
    assert downloader.resolve_download_dir() == (tmp_path / "dummy").resolve()

    target_dir = tmp_path / "dummy"
    assert downloader.is_downloaded(target_dir) is False
    downloader.download(target_dir)
    assert downloader.is_downloaded(target_dir) is True


def test_base_downloader_is_downloaded_handles_verify_errors(tmp_path):
    class FailingVerifyDownloader(DummyDownloader):
        def verify(self, data_dir) -> bool:
            raise RuntimeError("verify failed")

    downloader = FailingVerifyDownloader()
    target_dir = tmp_path / "dummy"
    target_dir.mkdir()
    assert downloader.is_downloaded(target_dir) is False


def test_base_downloader_clean_directory(tmp_path):
    downloader = DummyDownloader()
    target_dir = tmp_path / "dummy"
    target_dir.mkdir()
    (target_dir / "payload.txt").write_text("x")

    downloader.clean_directory(target_dir)
    assert not target_dir.exists()
