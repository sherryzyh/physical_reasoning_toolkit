"""
PHYSICS dataset downloader.

This module provides a downloader for the PHYSICS benchmark from the
yale-nlp/Physics GitHub repository.
"""

import json
import shutil
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .base_downloader import BaseDownloader


class PhysicsDownloader(BaseDownloader):
    """Downloader for the PHYSICS benchmark files hosted on GitHub."""

    DOMAINS: tuple[str, ...] = (
        "atomic",
        "electro",
        "mechanics",
        "optics",
        "quantum",
        "statistics",
    )

    FILE_PATTERNS: dict[tuple[str, str], tuple[str, str]] = {
        ("full", "full"): ("", "{}_dataset.jsonl"),
        ("full", "test"): ("PHYSICS-test", "{}_dataset_test.jsonl"),
        ("full", "eval"): ("PHYSICS-eval", "{}_dataset_eval.jsonl"),
        ("hard", "full"): ("PHYSICS-hard", "{}_dataset_hard.jsonl"),
        ("textonly", "full"): ("PHYSICS-textonly", "{}_dataset_textonly.jsonl"),
    }

    RAW_BASE_URL = "https://raw.githubusercontent.com/yale-nlp/Physics/main/PHYSICS"

    @property
    def dataset_name(self) -> str:
        """Return the on-disk dataset directory name."""
        return "PHYSICS"

    @property
    def download_info(self) -> dict[str, Any]:
        """Return download information."""
        return {
            "source": "GitHub raw files",
            "repository": "yale-nlp/Physics",
            "homepage": "https://github.com/yale-nlp/Physics",
            "paper_url": "https://aclanthology.org/2025.findings-acl.610.pdf",
            "repository_url": "https://github.com/yale-nlp/Physics",
            "format": "JSONL",
            "variants": ["full", "hard", "textonly"],
            "splits": ["full", "test", "eval"],
            "license": "MIT",
            "download_method": "raw GitHub file download",
        }

    def download(
        self,
        data_dir: str | Path | None = None,
        force: bool = False,
        variant: str | None = None,
        split: str | None = None,
        **kwargs: Any,
    ) -> Path:
        """
        Download the PHYSICS dataset.

        The downloader fetches the full benchmark layout in one pass so all
        supported PHYSICS variant/split combinations are available locally after
        a single download.
        """
        if variant is None:
            variant = self.get_default_variant() or "full"
        if split is None:
            split = self.get_default_split() or "full"

        self.validate_variant(variant)
        self.validate_split(split)
        self._validate_variant_split_combo(variant, split)

        return super().download(
            data_dir=data_dir,
            force=force,
            variant=variant,
            split=split,
            **kwargs,
        )

    def _do_download(
        self,
        download_dir: Path,
        variant: str = "full",
        split: str = "full",
        **kwargs: Any,
    ) -> Path:
        """
        Download the PHYSICS benchmark files into the PRKit cache layout.

        Args:
            download_dir: Resolved dataset root
            variant: Requested variant for validation/logging
            split: Requested split for validation/logging
            **kwargs: Additional download parameters (unused)
        """
        del kwargs  # Unused, kept for downloader API compatibility

        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "The 'requests' library is required to download PHYSICS. "
                "Install it with: pip install requests"
            ) from exc

        self.logger.info(
            "Downloading PHYSICS benchmark files (requested variant=%s, split=%s)...",
            variant,
            split,
        )
        self.logger.info("Target directory: %s", download_dir)

        try:
            download_dir.mkdir(parents=True, exist_ok=True)

            for relative_path, url in self._iter_file_urls():
                target_path = download_dir / relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                self._download_file(
                    requests=requests,
                    url=url,
                    target_path=target_path,
                )

            if not self.verify(download_dir):
                raise RuntimeError("Downloaded PHYSICS dataset failed verification")

            self.logger.info(
                "Successfully downloaded %d PHYSICS JSONL files to %s",
                len(list(self._relative_file_paths())),
                download_dir,
            )
            return download_dir

        except (ImportError, ValueError):
            raise
        except Exception as exc:
            if download_dir.exists():
                try:
                    shutil.rmtree(download_dir)
                except OSError:
                    pass

            self.logger.error("Failed to download PHYSICS dataset: %s", exc)
            raise RuntimeError(f"Download failed: {exc}") from exc

    def verify(self, data_dir: str | Path) -> bool:
        """
        Verify that the downloaded dataset contains all expected JSONL files.

        Args:
            data_dir: Directory containing the dataset

        Returns:
            True if the dataset looks complete and valid, False otherwise
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            return False

        for relative_path in self._relative_file_paths():
            file_path = data_dir / relative_path
            if not file_path.exists() or not file_path.is_file():
                self.logger.warning("Missing PHYSICS file: %s", file_path)
                return False
            if file_path.stat().st_size == 0:
                self.logger.warning("Empty PHYSICS file: %s", file_path)
                return False
            if not self._is_valid_jsonl(file_path):
                return False

        return True

    def _validate_variant_split_combo(self, variant: str, split: str) -> None:
        if (variant, split) not in self.FILE_PATTERNS:
            raise ValueError(
                f"Unsupported PHYSICS variant/split combination: variant='{variant}', split='{split}'. "
                "Supported combinations: "
                "('full', 'full'|'test'|'eval'), ('hard', 'full'), ('textonly', 'full')"
            )

    def _iter_file_urls(self) -> Iterable[tuple[Path, str]]:
        for relative_path in self._relative_file_paths():
            url = f"{self.RAW_BASE_URL}/{relative_path.as_posix()}"
            yield relative_path, url

    def _relative_file_paths(self) -> list[Path]:
        relative_paths: list[Path] = []
        for (variant, split), (subdir, filename_pattern) in self.FILE_PATTERNS.items():
            del variant, split
            for domain in self.DOMAINS:
                filename = filename_pattern.format(domain)
                relative_paths.append(
                    Path(subdir) / filename if subdir else Path(filename)
                )
        return relative_paths

    def _download_file(self, requests: Any, url: str, target_path: Path) -> None:
        max_retries = 3
        retry_delay_seconds = 3

        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=120)
                response.raise_for_status()

                if not response.content:
                    raise RuntimeError(f"Downloaded empty response from {url}")

                target_path.write_bytes(response.content)
                return
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise

                self.logger.warning(
                    "Failed to download %s (attempt %d/%d): %s. Retrying in %d seconds...",
                    url,
                    attempt + 1,
                    max_retries,
                    exc,
                    retry_delay_seconds,
                )
                time.sleep(retry_delay_seconds)

    def _is_valid_jsonl(self, file_path: Path) -> bool:
        try:
            with open(file_path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    json.loads(line)
                    return True
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.warning("Invalid PHYSICS JSONL file %s: %s", file_path, exc)
            return False

        self.logger.warning("PHYSICS JSONL file has no records: %s", file_path)
        return False
