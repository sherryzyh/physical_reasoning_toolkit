"""
UGPhysics Dataset Downloader

This module provides a downloader for the UGPhysics dataset from HuggingFace.

For citation information, see prkit.prkit_datasets.citations.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base_downloader import BaseDownloader


class UGPhysicsDownloader(BaseDownloader):
    """
    Downloader for UGPhysics dataset from HuggingFace.

    The UGPhysics dataset is available at:
    - HuggingFace: https://huggingface.co/datasets/UGPhysics/ugphysics
    - Homepage: https://github.com/YangLabHKUST/UGPhysics
    - Paper: https://openreview.net/pdf?id=EmLiyZGvrR
    """

    # List of all domains/configs in UGPhysics
    DOMAINS = [
        "AtomicPhysics",
        "ClassicalElectromagnetism",
        "ClassicalMechanics",
        "Electrodynamics",
        "GeometricalOptics",
        "QuantumMechanics",
        "Relativity",
        "SemiconductorPhysics",
        "Solid-StatePhysics",
        "StatisticalMechanics",
        "TheoreticalMechanics",
        "Thermodynamics",
        "WaveOptics",
    ]

    # Supported languages
    LANGUAGES = ["en", "zh"]

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
            "domains": self.DOMAINS,
            "languages": self.LANGUAGES,
            "size_bytes": None,  # Size varies
            "license": "Research use",
            "download_method": "datasets library",
        }

    def download(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        force: bool = False,
        domains: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        **kwargs,
    ) -> Path:
        """
        Download the UGPhysics dataset from HuggingFace.

        Args:
            data_dir: Target directory for download (None = auto-detect)
            force: If True, clean existing dataset directory and re-download.
                  If False, skip download if dataset already exists.
            domains: List of domains to download (None = all domains)
            languages: List of languages to download (None = all languages)
            **kwargs: Additional download parameters

        Returns:
            Path to the downloaded dataset directory

        Raises:
            FileExistsError: If dataset already exists and force=False
            RuntimeError: If download fails
        """
        # Call parent download method which handles force logic
        return super().download(
            data_dir=data_dir,
            force=force,
            domains=domains,
            languages=languages,
            **kwargs,
        )

    def _do_download(
        self,
        download_dir: Path,
        domains: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        **kwargs,
    ) -> Path:
        """
        Perform the actual UGPhysics dataset download.

        Downloads data from HuggingFace using the datasets library, which downloads
        entire configs/splits at once, avoiding pagination limits and rate limiting issues.

        Args:
            download_dir: Resolved download directory path
            domains: List of domains to download (None = all domains)
            languages: List of languages to download (None = all languages)
            **kwargs: Additional download parameters

        Returns:
            Path to the downloaded dataset directory

        Raises:
            ImportError: If datasets library is not installed
            ValueError: If invalid domain or language is specified
            RuntimeError: If download fails
        """
        # Check if datasets library is available
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' library is required to download UGPhysics. "
                "Install it with: pip install datasets"
            ) from exc

        # Validate domains and languages
        if domains is None:
            domains = self.DOMAINS
        else:
            invalid_domains = [d for d in domains if d not in self.DOMAINS]
            if invalid_domains:
                raise ValueError(
                    f"Invalid domains: {invalid_domains}. "
                    f"Valid domains: {self.DOMAINS}"
                )

        if languages is None:
            languages = self.LANGUAGES
        else:
            invalid_languages = [l for l in languages if l not in self.LANGUAGES]
            if invalid_languages:
                raise ValueError(
                    f"Invalid languages: {invalid_languages}. "
                    f"Valid languages: {self.LANGUAGES}"
                )

        self.logger.info("Downloading UGPhysics dataset...")
        self.logger.info("Target directory: %s", download_dir)
        self.logger.info("Domains: %s", domains)
        self.logger.info("Languages: %s", languages)
        self.logger.info(
            "Using HuggingFace datasets library to download entire configs/splits at once"
        )

        try:
            # Create download directory
            download_dir.mkdir(parents=True, exist_ok=True)

            dataset_name = "UGPhysics/ugphysics"
            total_problems = 0

            # Download each domain and language combination
            for domain in domains:
                domain_dir = download_dir / domain
                domain_dir.mkdir(parents=True, exist_ok=True)

                for language in languages:
                    output_file = domain_dir / f"{language}.jsonl"
                    self.logger.info(
                        "Downloading %s (%s) using datasets library...", domain, language
                    )

                    try:
                        # Load entire dataset config/split at once using datasets library
                        # This downloads the entire dataset in one go, avoiding pagination limits
                        dataset = load_dataset(
                            dataset_name,
                            name=domain,  # config name
                            split=language,  # split name (en or zh)
                        )

                        self.logger.info(
                            "Loaded %d examples for %s/%s", len(dataset), domain, language
                        )

                        if len(dataset) == 0:
                            self.logger.warning(
                                "No data found for %s/%s, skipping...",
                                domain,
                                language,
                            )
                            continue

                        # Convert to list of dictionaries and save as JSONL
                        self.logger.info(
                            "Saving %d problems to %s...", len(dataset), output_file
                        )
                        with open(output_file, "w", encoding="utf-8") as f:
                            for example in dataset:
                                # Convert example to dict (handles Arrow format)
                                row_dict = dict(example)
                                json.dump(row_dict, f, ensure_ascii=False)
                                f.write("\n")

                        self.logger.info(
                            "Successfully saved %s (%d problems)",
                            output_file,
                            len(dataset),
                        )
                        total_problems += len(dataset)

                    except Exception as e:
                        self.logger.error(
                            "Failed to download %s/%s: %s", domain, language, e
                        )
                        # Continue with other domains/languages instead of failing completely
                        continue

            self.logger.info(
                "Successfully downloaded UGPhysics dataset to %s",
                download_dir,
            )
            self.logger.info("Total problems: %d", total_problems)

            return download_dir

        except ImportError:
            # Re-raise ImportError as-is
            raise
        except (OSError, ValueError, RuntimeError) as e:
            # Clean up on error
            if download_dir.exists():
                try:
                    shutil.rmtree(download_dir)
                except OSError:
                    pass

            self.logger.error("Failed to download UGPhysics dataset: %s", e)
            raise RuntimeError(f"Download failed: {e}") from e

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

        # Check that at least one domain directory exists
        found_domains = []
        for domain in self.DOMAINS:
            domain_dir = data_dir / domain
            if domain_dir.exists() and domain_dir.is_dir():
                found_domains.append(domain)

        if not found_domains:
            self.logger.warning(
                "No domain directories found in %s", data_dir
            )
            return False

        # Check that each domain has at least one language file
        valid_domains = 0
        for domain in found_domains:
            domain_dir = data_dir / domain
            has_language_file = False

            for language in self.LANGUAGES:
                jsonl_file = domain_dir / f"{language}.jsonl"
                if jsonl_file.exists() and jsonl_file.is_file():
                    # Verify it's not empty
                    if jsonl_file.stat().st_size > 0:
                        # Verify it's valid JSONL
                        try:
                            with open(jsonl_file, "r", encoding="utf-8") as f:
                                # Check first few lines
                                for i, line in enumerate(f):
                                    if i >= 3:  # Check first 3 lines
                                        break
                                    if line.strip():
                                        json.loads(line)
                            has_language_file = True
                            break
                        except json.JSONDecodeError:
                            self.logger.warning(
                                "Invalid JSONL in %s", jsonl_file
                            )
                            continue

            if has_language_file:
                valid_domains += 1

        if valid_domains == 0:
            self.logger.warning(
                "No valid domain files found in %s", data_dir
            )
            return False

        self.logger.info(
            "UGPhysics dataset is valid: %d domains with data in %s",
            valid_domains,
            data_dir,
        )
        return True
