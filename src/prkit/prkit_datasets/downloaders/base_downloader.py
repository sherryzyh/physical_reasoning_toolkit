"""
Base Dataset Downloader

This module provides a base class for all dataset downloaders in PRKit,
ensuring consistent download handling and standardization across different datasets.
"""

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

from prkit.prkit_core import PRKitLogger


class BaseDownloader(ABC):
    """
    Base class for all dataset downloaders in PRKit.

    This class provides common functionality for:
    - Download location resolution
    - Download progress tracking
    - Dataset validation after download
    - Error handling
    """

    def __init__(self):
        """Initialize the downloader with a logger."""
        self.logger = PRKitLogger.get_logger(self.__class__.__module__)

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """
        Return the name of the dataset (e.g., 'ugphysics', 'phybench').

        Returns:
            Dataset name string
        """
        pass

    @property
    @abstractmethod
    def download_info(self) -> Dict[str, any]:
        """
        Return download information including source URLs, formats, etc.

        Returns:
            Dictionary containing download metadata
        """
        pass

    def get_default_variant(self) -> Optional[str]:
        """
        Get the default variant for this dataset downloader.
        
        Returns "full" if available, otherwise returns the first available variant.
        Returns None if no variants are available.
        
        Returns:
            Default variant string or None
        """
        info = self.download_info
        variants = info.get("variants", [])
        if not variants:
            return None
        
        # Prefer "full" if available
        if "full" in variants:
            return "full"
        
        # Otherwise return the first variant
        return variants[0] if variants else None

    def get_default_split(self) -> Optional[str]:
        """
        Get the default split for this dataset downloader.
        
        Returns "full" if available, otherwise returns the first available split.
        Returns None if no splits are available.
        
        Returns:
            Default split string or None
        """
        info = self.download_info
        splits = info.get("splits", [])
        if not splits:
            return None
        
        # Prefer "full" if available
        if "full" in splits:
            return "full"
        
        # Otherwise return the first split
        return splits[0] if splits else None

    def get_available_variants(self) -> List[str]:
        """
        Get list of available variants for this dataset downloader.
        
        Returns:
            List of variant strings
        """
        info = self.download_info
        return info.get("variants", [])

    def get_available_splits(self) -> List[str]:
        """
        Get list of available splits for this dataset downloader.
        
        Returns:
            List of split strings
        """
        info = self.download_info
        return info.get("splits", [])

    def validate_variant(self, variant: str) -> None:
        """
        Validate that a variant is available for this dataset downloader.
        
        Args:
            variant: Variant to validate
            
        Raises:
            ValueError: If variant is not available
        """
        available = self.get_available_variants()
        if variant not in available:
            raise ValueError(
                f"Unknown variant '{variant}' for dataset '{self.dataset_name}'. "
                f"Available variants: {available}"
            )

    def validate_split(self, split: str) -> None:
        """
        Validate that a split is available for this dataset downloader.
        
        Args:
            split: Split to validate
            
        Raises:
            ValueError: If split is not available
        """
        available = self.get_available_splits()
        if split not in available:
            raise ValueError(
                f"Unknown split '{split}' for dataset '{self.dataset_name}'. "
                f"Available splits: {available}"
            )

    def download(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        force: bool = False,
        **kwargs,
    ) -> Path:
        """
        Download the dataset to the specified directory.

        This method handles common download logic including:
        - Resolving download directory
        - Checking if dataset already exists
        - Cleaning directory if force=True
        - Delegating actual download to _do_download()

        Args:
            data_dir: Target directory for download (None = auto-detect)
            force: If True, clean existing dataset directory and re-download.
                  If False, skip download if dataset already exists.
            **kwargs: Additional download parameters (e.g., variant, split)

        Returns:
            Path to the downloaded dataset directory

        Raises:
            FileExistsError: If dataset already exists and force=False
            DownloadError: If download fails
        """
        download_dir = self.resolve_download_dir(data_dir)

        # Check if already downloaded
        if not force and self.is_downloaded(download_dir):
            self.logger.info(
                "Dataset already exists at %s. Use force=True to re-download.",
                download_dir,
            )
            return download_dir

        # Clean directory if force=True and dataset exists
        if force and download_dir.exists():
            self.clean_directory(download_dir)

        # Delegate actual download to child class implementation
        return self._do_download(download_dir, **kwargs)

    @abstractmethod
    def _do_download(
        self,
        download_dir: Path,
        **kwargs,
    ) -> Path:
        """
        Perform the actual dataset download.

        This method is called by download() after handling force logic.
        Child classes should implement the specific download mechanism here.

        Args:
            download_dir: Resolved download directory path
            **kwargs: Additional download parameters (e.g., variant, split)

        Returns:
            Path to the downloaded dataset directory

        Raises:
            DownloadError: If download fails
        """
        pass

    @abstractmethod
    def verify(self, data_dir: Union[str, Path]) -> bool:
        """
        Verify that the downloaded dataset is complete and valid.

        Args:
            data_dir: Directory containing the dataset

        Returns:
            True if dataset is valid, False otherwise
        """
        pass

    def resolve_download_dir(
        self, data_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Resolve download directory with support for DATASET_CACHE_DIR environment variable.

        Args:
            data_dir: Explicitly provided download directory path

        Returns:
            Resolved Path object pointing to the download directory

        Priority order:
        1. Explicitly provided data_dir (highest priority)
        2. DATASET_CACHE_DIR environment variable + dataset_name
        3. ~/PHYSICAL_REASONING_DATASETS + dataset_name (default fallback)
        """
        # If data_dir is explicitly provided, use it
        if data_dir is not None:
            resolved_dir = Path(data_dir).resolve()
            return resolved_dir

        # Check for DATASET_CACHE_DIR environment variable
        dataset_cache_dir = os.getenv("DATASET_CACHE_DIR")
        if dataset_cache_dir:
            base_dir = Path(dataset_cache_dir).resolve()
            return base_dir / self.dataset_name

        # Fallback to ~/PHYSICAL_REASONING_DATASETS
        # Path.home() already returns an absolute path
        home_data_dir = Path.home() / "PHYSICAL_REASONING_DATASETS"
        return home_data_dir.resolve() / self.dataset_name

    def is_downloaded(self, data_dir: Optional[Union[str, Path]] = None) -> bool:
        """
        Check if the dataset is already downloaded.

        Args:
            data_dir: Directory to check (None = auto-detect)

        Returns:
            True if dataset exists and is valid, False otherwise
        """
        download_dir = self.resolve_download_dir(data_dir)
        if not download_dir.exists():
            return False

        # Verify the dataset
        try:
            return self.verify(download_dir)
        except Exception as e:
            self.logger.warning(f"Verification failed: {e}")
            return False

    def get_download_size(self) -> Optional[int]:
        """
        Get the estimated download size in bytes.

        Returns:
            Size in bytes, or None if size cannot be determined
        """
        info = self.download_info
        return info.get("size_bytes")

    def get_download_source(self) -> str:
        """
        Get the download source description.

        Returns:
            Source description string
        """
        info = self.download_info
        return info.get("source", "Unknown")

    def clean_directory(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Clean (delete) the dataset directory if it exists.

        This method removes the entire dataset directory to ensure a clean
        re-download when force=True.

        Args:
            data_dir: Directory to clean (None = auto-detect)

        Raises:
            OSError: If directory deletion fails
        """
        download_dir = self.resolve_download_dir(data_dir)
        
        if download_dir.exists():
            self.logger.info(
                "Cleaning existing dataset directory: %s", download_dir
            )
            try:
                shutil.rmtree(download_dir)
                self.logger.info("Successfully cleaned directory: %s", download_dir)
            except OSError as e:
                self.logger.error(
                    "Failed to clean directory %s: %s", download_dir, e
                )
                raise
