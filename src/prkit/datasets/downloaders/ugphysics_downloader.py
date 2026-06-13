"""
UGPhysics Dataset Downloader

This module provides a downloader for the UGPhysics dataset from HuggingFace.

For citation information, see prkit.datasets.citations.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from prkit.datasets.ugphysics_common import (
    UGPHYSICS_DOMAIN_COUNTS,
    UGPHYSICS_DOMAIN_VARIANTS,
    UGPHYSICS_PUBLIC_VARIANTS,
    UGPHYSICS_SPLIT_TOTALS,
    UGPHYSICS_SUPPORTED_SPLITS,
    get_requested_domain_dirs,
    normalize_domain_dir_name,
    normalize_language_code,
    normalize_split,
    normalize_variant,
)

from .base_downloader import BaseDownloader


class UGPhysicsDownloader(BaseDownloader):
    """
    Downloader for UGPhysics dataset from HuggingFace.

    The UGPhysics dataset is available at:
    - HuggingFace: https://huggingface.co/datasets/UGPhysics/ugphysics
    - Homepage: https://github.com/YangLabHKUST/UGPhysics
    - Paper: https://openreview.net/pdf?id=EmLiyZGvrR
    """

    DOMAINS = list(UGPHYSICS_DOMAIN_VARIANTS.values())
    LANGUAGES = UGPHYSICS_SUPPORTED_SPLITS

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "ugphysics"

    @property
    def download_info(self) -> Dict[str, Any]:
        """Return download information."""
        return {
            "source": "HuggingFace Datasets Server",
            "repository": "UGPhysics/ugphysics",
            "homepage": "https://github.com/YangLabHKUST/UGPhysics",
            "paper_url": "https://openreview.net/pdf?id=EmLiyZGvrR",
            "huggingface_url": "https://huggingface.co/datasets/UGPhysics/ugphysics",
            "format": "JSONL",
            "variants": UGPHYSICS_PUBLIC_VARIANTS,
            "splits": UGPHYSICS_SUPPORTED_SPLITS,
            "domains": self.DOMAINS,
            "languages": self.LANGUAGES,
            "size_bytes": None,
            "license": "cc-by-nc-sa-4.0",
            "download_method": "datasets library",
            "total_problems": {
                "en": UGPHYSICS_SPLIT_TOTALS["en"],
                "zh": UGPHYSICS_SPLIT_TOTALS["zh"],
                "all": sum(UGPHYSICS_SPLIT_TOTALS.values()),
            },
            "problems_by_domain": UGPHYSICS_DOMAIN_COUNTS["en"],
        }

    def download(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        force: bool = False,
        variant: Optional[str] = None,
        split: Optional[str] = None,
        domains: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> Path:
        """
        Download the UGPhysics dataset from HuggingFace.

        Args:
            data_dir: Target directory for download (None = auto-detect)
            force: If True, clean existing dataset directory and re-download.
            variant: Public UGPhysics variant ("full" or one domain variant)
            split: Public UGPhysics split ("en" or "zh")
            domains: Optional explicit list of domains to download
            languages: Optional explicit list of languages to download
            language: Backward-compatible alias used with legacy split="test"
            **kwargs: Additional download parameters

        Returns:
            Path to the downloaded dataset directory
        """
        (
            normalized_variant,
            normalized_split,
            requested_domains,
            requested_languages,
        ) = self._resolve_requested_artifacts(
            variant=variant,
            split=split,
            domains=domains,
            languages=languages,
            language=language,
        )

        download_dir = self.resolve_download_dir(data_dir)

        if force and download_dir.exists():
            self.clean_directory(download_dir)

        if not force and self._is_request_downloaded(
            download_dir,
            domains=requested_domains,
            languages=requested_languages,
        ):
            self.logger.info(
                "Requested UGPhysics artifacts already exist at %s "
                "(variant=%s, split=%s, domains=%s, languages=%s)",
                download_dir,
                normalized_variant,
                normalized_split,
                requested_domains,
                requested_languages,
            )
            return download_dir

        return self._do_download(
            download_dir=download_dir,
            variant=normalized_variant,
            split=normalized_split,
            domains=requested_domains,
            languages=requested_languages,
            **kwargs,
        )

    def _do_download(
        self,
        download_dir: Path,
        variant: str = "full",
        split: Optional[str] = None,
        domains: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        **kwargs,
    ) -> Path:
        """
        Perform the actual UGPhysics dataset download.

        Downloads data from HuggingFace using the datasets library, which downloads
        entire configs/splits at once.
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' library is required to download UGPhysics. "
                "Install it with: pip install datasets"
            ) from exc

        (
            normalized_variant,
            normalized_split,
            domains,
            languages,
        ) = self._resolve_requested_artifacts(
            variant=variant,
            split=split,
            domains=domains,
            languages=languages,
        )

        self.logger.info("Downloading UGPhysics dataset...")
        self.logger.info("Target directory: %s", download_dir)
        self.logger.info("Variant: %s", normalized_variant)
        if normalized_split is not None:
            self.logger.info("Split: %s", normalized_split)
        self.logger.info("Domains: %s", domains)
        self.logger.info("Languages: %s", languages)
        self.logger.info(
            "Using HuggingFace datasets library to download entire configs/splits at once"
        )

        try:
            download_dir.mkdir(parents=True, exist_ok=True)

            dataset_name = "UGPhysics/ugphysics"
            total_problems = 0
            file_records = []
            failures = []

            for domain in domains:
                domain_dir = download_dir / domain
                domain_dir.mkdir(parents=True, exist_ok=True)

                for language_name in languages:
                    output_file = domain_dir / f"{language_name}.jsonl"
                    self.logger.info(
                        "Downloading %s (%s) using datasets library...",
                        domain,
                        language_name,
                    )

                    try:
                        dataset = load_dataset(
                            dataset_name,
                            name=domain,
                            split=language_name,
                        )

                        self.logger.info(
                            "Loaded %d examples for %s/%s",
                            len(dataset),
                            domain,
                            language_name,
                        )

                        if len(dataset) == 0:
                            failures.append(f"{domain}/{language_name}: empty dataset")
                            continue

                        with open(output_file, "w", encoding="utf-8") as file_obj:
                            for example in dataset:
                                json.dump(dict(example), file_obj, ensure_ascii=False)
                                file_obj.write("\n")

                        total_problems += len(dataset)
                        file_records.append(
                            {
                                "domain": domain,
                                "language": language_name,
                                "path": str(Path(domain) / f"{language_name}.jsonl"),
                                "rows": len(dataset),
                            }
                        )

                        self.logger.info(
                            "Successfully saved %s (%d problems)",
                            output_file,
                            len(dataset),
                        )
                    except Exception as error:
                        self.logger.error(
                            "Failed to download %s/%s: %s",
                            domain,
                            language_name,
                            error,
                        )
                        failures.append(f"{domain}/{language_name}: {error}")

            if failures:
                raise RuntimeError(
                    "UGPhysics download incomplete. Failures: " + "; ".join(failures)
                )

            self._write_manifest(
                download_dir=download_dir,
                variant=normalized_variant,
                split=normalized_split,
                file_records=file_records,
            )

            self.logger.info(
                "Successfully downloaded UGPhysics dataset to %s",
                download_dir,
            )
            self.logger.info("Total problems: %d", total_problems)
            return download_dir

        except (ImportError, ValueError):
            raise
        except (OSError, RuntimeError) as error:
            self.logger.error("Failed to download UGPhysics dataset: %s", error)
            raise RuntimeError(f"Download failed: {error}") from error

    def _resolve_requested_artifacts(
        self,
        variant: Optional[str] = None,
        split: Optional[str] = None,
        domains: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        language: Optional[str] = None,
    ) -> tuple[str, Optional[str], List[str], List[str]]:
        """Normalize downloader requests to concrete domains and languages."""
        normalized_variant = normalize_variant(variant, logger=self.logger)
        normalized_split = (
            normalize_split(split, language=language, logger=self.logger)
            if split is not None or language is not None
            else None
        )

        if domains is None:
            requested_domains = get_requested_domain_dirs(normalized_variant)
        else:
            requested_domains = [normalize_domain_dir_name(domain) for domain in domains]
            if normalized_variant != "full":
                expected_domains = set(get_requested_domain_dirs(normalized_variant))
                if set(requested_domains) != expected_domains:
                    raise ValueError(
                        "Conflicting UGPhysics domain request: "
                        f"variant='{variant}' does not match domains={domains}"
                    )

        if languages is None:
            requested_languages = [normalized_split or "en"]
        else:
            requested_languages = []
            for raw_language in languages:
                normalized_language = normalize_language_code(raw_language)
                if normalized_language not in UGPHYSICS_SUPPORTED_SPLITS:
                    raise ValueError(
                        f"Invalid languages: {languages}. "
                        f"Valid languages: {UGPHYSICS_SUPPORTED_SPLITS}"
                    )
                requested_languages.append(normalized_language)
            requested_languages = sorted(set(requested_languages))
            if normalized_split is not None and requested_languages != [normalized_split]:
                raise ValueError(
                    "Conflicting UGPhysics language request: "
                    f"split='{split}' does not match languages={languages}"
                )

        return (
            normalized_variant,
            normalized_split,
            requested_domains,
            requested_languages,
        )

    def _is_request_downloaded(
        self,
        download_dir: Path,
        domains: List[str],
        languages: List[str],
    ) -> bool:
        """Check whether the exact requested UGPhysics artifacts already exist."""
        if not download_dir.exists():
            return False

        for domain in domains:
            for language_name in languages:
                jsonl_file = download_dir / domain / f"{language_name}.jsonl"
                if not self._verify_jsonl_file(jsonl_file):
                    return False

        return True

    def _verify_jsonl_file(self, jsonl_file: Path) -> bool:
        """Verify that a single UGPhysics JSONL shard exists and is readable."""
        if not jsonl_file.exists() or not jsonl_file.is_file():
            return False

        if jsonl_file.stat().st_size == 0:
            return False

        checked_rows = 0
        try:
            with open(jsonl_file, "r", encoding="utf-8") as file_obj:
                for line in file_obj:
                    if not line.strip():
                        continue
                    json.loads(line)
                    checked_rows += 1
                    if checked_rows >= 3:
                        break
        except (OSError, json.JSONDecodeError):
            return False

        return checked_rows > 0

    def _write_manifest(
        self,
        download_dir: Path,
        variant: str,
        split: Optional[str],
        file_records: List[Dict[str, Any]],
    ) -> None:
        """Persist a lightweight manifest for deterministic verification."""
        manifest_path = download_dir / "download_manifest.json"
        existing_files = {}

        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as file_obj:
                    manifest = json.load(file_obj)
                for record in manifest.get("files", []):
                    existing_files[record["path"]] = record
            except (OSError, json.JSONDecodeError, KeyError):
                existing_files = {}

        for record in file_records:
            existing_files[record["path"]] = record

        manifest = {
            "dataset": self.dataset_name,
            "repository": self.download_info["repository"],
            "last_request": {
                "variant": variant,
                "split": split,
            },
            "files": [existing_files[path] for path in sorted(existing_files.keys())],
        }

        with open(manifest_path, "w", encoding="utf-8") as file_obj:
            json.dump(manifest, file_obj, indent=2, sort_keys=True)

    def verify(self, data_dir: Union[str, Path]) -> bool:
        """
        Verify that the downloaded dataset is complete and valid.

        Args:
            data_dir: Directory containing the dataset

        Returns:
            True if dataset is valid, False otherwise
        """
        data_dir = Path(data_dir)

        if not data_dir.exists():
            return False

        manifest_path = data_dir / "download_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as file_obj:
                    manifest = json.load(file_obj)
                files = manifest.get("files", [])
                if not files:
                    self.logger.warning(
                        "UGPhysics manifest has no file records: %s",
                        manifest_path,
                    )
                    return False

                for record in files:
                    jsonl_file = data_dir / record["path"]
                    if not self._verify_jsonl_file(jsonl_file):
                        self.logger.warning("Invalid or missing shard: %s", jsonl_file)
                        return False

                self.logger.info(
                    "UGPhysics dataset is valid per manifest (%d files) in %s",
                    len(files),
                    data_dir,
                )
                return True
            except (OSError, json.JSONDecodeError, KeyError) as error:
                self.logger.warning("Failed to read UGPhysics manifest: %s", error)

        found_valid_files = 0
        for domain in self.DOMAINS:
            for language_name in self.LANGUAGES:
                jsonl_file = data_dir / domain / f"{language_name}.jsonl"
                if self._verify_jsonl_file(jsonl_file):
                    found_valid_files += 1

        if found_valid_files == 0:
            self.logger.warning("No valid UGPhysics shards found in %s", data_dir)
            return False

        self.logger.info(
            "UGPhysics dataset is valid: %d shard files with data in %s",
            found_valid_files,
            data_dir,
        )
        return True
