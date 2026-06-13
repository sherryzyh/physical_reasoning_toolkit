"""
PhysBench dataset downloader.

This downloader fetches the public PhysBench metadata and optionally the large
media archives from Hugging Face.
"""

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from .base_downloader import BaseDownloader


class PhysBenchDownloader(BaseDownloader):
    """Downloader for the PhysBench benchmark."""

    METADATA_URLS = {
        "README.md": "https://huggingface.co/datasets/USC-PSI-Lab/PhysBench/resolve/main/README.md?download=true",
        "test.json": "https://huggingface.co/datasets/USC-PSI-Lab/PhysBench/resolve/main/test.json?download=true",
    }
    MEDIA_URLS = {
        "image.zip": "https://huggingface.co/datasets/USC-PSI-Lab/PhysBench/resolve/main/image.zip?download=true",
        "video.zip": "https://huggingface.co/datasets/USC-PSI-Lab/PhysBench/resolve/main/video.zip?download=true",
    }

    @property
    def dataset_name(self) -> str:
        return "PhysBench"

    @property
    def download_info(self) -> dict[str, Any]:
        return {
            "source": "Hugging Face dataset files",
            "repository": "USC-PSI-Lab/PhysBench",
            "paper_url": "https://arxiv.org/pdf/2501.16411",
            "huggingface_url": "https://huggingface.co/datasets/USC-PSI-Lab/PhysBench",
            "homepage": "https://physbench.github.io/",
            "license": "apache-2.0",
            "format": "JSON + optional ZIP media archives",
            "variants": ["full", "general", "image_only", "image_video"],
            "splits": ["full", "val", "test"],
            "size_bytes": 8132356,
            "optional_media_size_bytes": {
                "metadata": 8132356,
                "image_zip": 3701071392,
                "video_zip": 3721393856,
            },
            "download_method": "direct file download",
            "media_note": (
                "By default only README.md and test.json are downloaded. "
                "Pass download_media=True to fetch image.zip and video.zip."
            ),
        }

    def _do_download(
        self,
        download_dir: Path,
        download_media: bool = False,
        extract_media: bool = False,
        **kwargs,
    ) -> Path:
        del kwargs  # Unused, kept for downloader API compatibility

        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "The 'requests' library is required to download PhysBench. "
                "Install it with: pip install requests"
            ) from exc

        if extract_media and not download_media:
            raise ValueError("extract_media=True requires download_media=True")

        self.logger.info("Downloading PhysBench metadata to %s", download_dir)

        try:
            download_dir.mkdir(parents=True, exist_ok=True)

            for filename, url in self.METADATA_URLS.items():
                self._download_file(
                    requests=requests,
                    url=url,
                    destination=download_dir / filename,
                )

            # Create the expected media directories even for metadata-only installs.
            (download_dir / "image").mkdir(exist_ok=True)
            (download_dir / "video").mkdir(exist_ok=True)

            if download_media:
                for filename, url in self.MEDIA_URLS.items():
                    archive_path = download_dir / filename
                    self._download_file(
                        requests=requests,
                        url=url,
                        destination=archive_path,
                    )
                    if extract_media:
                        target_dir = download_dir / archive_path.stem
                        target_dir.mkdir(exist_ok=True)
                        with zipfile.ZipFile(archive_path, "r") as archive:
                            archive.extractall(target_dir)

            self.logger.info("Successfully downloaded PhysBench to %s", download_dir)
            return download_dir

        except (ImportError, ValueError):
            raise
        except Exception as exc:  # pylint: disable=broad-except
            if download_dir.exists():
                try:
                    shutil.rmtree(download_dir)
                except OSError:
                    pass
            self.logger.error("Failed to download PhysBench: %s", exc)
            raise RuntimeError(f"Download failed: {exc}") from exc

    def verify(self, data_dir: str | Path) -> bool:
        data_dir = Path(data_dir)
        json_file = data_dir / "test.json"
        if not json_file.exists():
            self.logger.warning("PhysBench JSON file not found: %s", json_file)
            return False

        try:
            with open(json_file, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.warning("Failed to validate PhysBench metadata: %s", exc)
            return False

        if not isinstance(data, list) or not data:
            self.logger.warning("PhysBench metadata is empty or not a JSON list")
            return False

        return True

    def _download_file(self, requests, url: str, destination: Path) -> None:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        with open(destination, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
