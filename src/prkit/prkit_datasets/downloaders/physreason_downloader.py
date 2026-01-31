"""
PhysReason Dataset Downloader

This module provides a downloader for the PhysReason dataset from HuggingFace.

For citation information, see prkit.prkit_datasets.citations.
"""

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base_downloader import BaseDownloader


class PhysReasonDownloader(BaseDownloader):
    """
    Downloader for PhysReason dataset from HuggingFace.

    The PhysReason dataset is available at:
    - HuggingFace: https://huggingface.co/datasets/zhibei1204/PhysReason
    - Homepage: https://dxzxy12138.github.io/PhysReason/
    - Paper: https://aclanthology.org/2025.acl-long.811.pdf
    """

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "physreason"

    @property
    def download_info(self) -> Dict[str, Any]:
        """Return download information."""
        return {
            "source": "HuggingFace",
            "repository": "zhibei1204/PhysReason",
            "homepage": "https://dxzxy12138.github.io/PhysReason/",
            "paper_url": "https://aclanthology.org/2025.acl-long.811.pdf",
            "huggingface_url": "https://huggingface.co/datasets/zhibei1204/PhysReason",
            "format": "JSON",
            "variants": ["full", "mini"],
            "splits": ["train"],
            "size_bytes": None,  # Size varies by variant
            "license": "CC BY-NC-SA / MIT",
            "download_method": "HuggingFace direct download",
        }

    def download(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        force: bool = False,
        variant: str = "full",
        **kwargs,
    ) -> Path:
        """
        Download the PhysReason dataset from HuggingFace.

        Args:
            data_dir: Target directory for download (None = auto-detect)
            force: If True, clean existing dataset directory and re-download.
                  If False, skip download if dataset already exists.
            variant: Dataset variant ("full" or "mini")
            **kwargs: Additional download parameters

        Returns:
            Path to the downloaded dataset directory

        Raises:
            ImportError: If requests library is not installed
            FileExistsError: If dataset already exists and force=False
            RuntimeError: If download fails
        """
        # Call parent download method which handles force logic
        return super().download(data_dir=data_dir, force=force, variant=variant, **kwargs)

    def _do_download(
        self,
        download_dir: Path,
        variant: str = "full",
        **kwargs,
    ) -> Path:
        """
        Perform the actual PhysReason dataset download.

        Downloads zip files directly from HuggingFace repository without post-processing.

        Args:
            download_dir: Resolved download directory path
            variant: Dataset variant ("full" or "mini")
            **kwargs: Additional download parameters

        Returns:
            Path to the downloaded dataset directory

        Raises:
            ImportError: If requests library is not installed
            RuntimeError: If download fails
        """
        # Check if requests library is available
        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "The 'requests' library is required to download PhysReason. "
                "Install it with: pip install requests"
            ) from exc

        self.logger.info(
            "Downloading PhysReason dataset (%s variant)...", variant
        )
        self.logger.info("Target directory: %s", download_dir)

        try:
            # Validate variant
            if variant not in ["full", "mini"]:
                raise ValueError(f"Unknown variant: {variant}. Choose 'full' or 'mini'")

            # Create download directory
            download_dir.mkdir(parents=True, exist_ok=True)

            # Build the HuggingFace file download URL
            # Files are available at: https://huggingface.co/datasets/zhibei1204/PhysReason/resolve/main/{filename}
            dataset_repo = "zhibei1204/PhysReason"
            
            # Map variant to filename
            filename_map = {
                "full": "PhysReason-full.zip",
                "mini": "PhysReason-mini.zip",
            }
            filename = filename_map[variant]
            
            # Construct the download URL using HuggingFace's resolve endpoint
            # Note: The ?download=true parameter is required for proper file download
            download_url = (
                f"https://huggingface.co/datasets/{dataset_repo}/resolve/main/{filename}?download=true"
            )
            
            self.logger.info("Downloading from HuggingFace repository...")
            self.logger.info("URL: %s", download_url)
            self.logger.info("File: %s", filename)

            # Download the zip file with progress tracking
            max_retries = 3
            retry_delay = 5  # seconds
            
            for attempt in range(max_retries):
                try:
                    # Stream the download to handle large files
                    response = requests.get(download_url, stream=True, timeout=600)  # 10 minute timeout
                    response.raise_for_status()
                    
                    # Determine output file path
                    output_file = download_dir / filename
                    
                    # Download and save the file
                    self.logger.info("Saving to: %s", output_file)
                    total_size = int(response.headers.get('content-length', 0))
                    
                    with open(output_file, 'wb') as f:
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                
                                # Log progress for large files
                                if total_size > 0 and downloaded % (10 * 1024 * 1024) == 0:  # Every 10MB
                                    progress = (downloaded / total_size) * 100
                                    self.logger.info(
                                        "Download progress: %.1f%% (%d / %d bytes)",
                                        progress, downloaded, total_size
                                    )
                    
                    # Verify file was downloaded
                    if output_file.exists() and output_file.stat().st_size > 0:
                        file_size_mb = output_file.stat().st_size / (1024 * 1024)
                        self.logger.info(
                            "Successfully downloaded PhysReason (%s) to %s (%.2f MB)",
                            variant,
                            output_file,
                            file_size_mb,
                        )
                        break
                    else:
                        raise RuntimeError("Downloaded file is empty or does not exist")
                        
                except requests.exceptions.RequestException as req_err:
                    if attempt < max_retries - 1:
                        self.logger.warning(
                            "Download failed (attempt %d/%d): %s. Retrying in %d seconds...",
                            attempt + 1, max_retries, req_err, retry_delay
                        )
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise RuntimeError(
                            f"Failed to download {filename} after {max_retries} attempts: {req_err}"
                        ) from req_err

            return download_dir

        except (ImportError, ValueError) as e:
            # Re-raise ImportError and ValueError as-is (don't wrap)
            raise
        except (RuntimeError, OSError, Exception) as e:
            # Clean up on error
            if download_dir.exists():
                try:
                    shutil.rmtree(download_dir)
                except OSError:
                    pass

            self.logger.error("Failed to download PhysReason dataset: %s", e)
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

        # Check for downloaded zip files
        full_file = data_dir / "PhysReason-full.zip"
        mini_file = data_dir / "PhysReason-mini.zip"

        # At least one variant file should exist
        if not full_file.exists() and not mini_file.exists():
            self.logger.warning("No variant files found")
            return False

        # Verify that existing files are valid zip files
        try:
            import zipfile
        except ImportError:
            # If zipfile is not available, just check file existence and size
            self.logger.warning("zipfile module not available, skipping zip validation")
            for zip_file in [full_file, mini_file]:
                if zip_file.exists():
                    if not zip_file.is_file():
                        self.logger.warning("%s is not a file", zip_file)
                        return False
                    if zip_file.stat().st_size == 0:
                        self.logger.warning("%s is empty", zip_file)
                        return False
            return True

        # Verify zip files are valid
        for zip_file in [full_file, mini_file]:
            if zip_file.exists():
                # Check that it's a file
                if not zip_file.is_file():
                    self.logger.warning("%s is not a file", zip_file)
                    return False

                # Check that it's not empty
                if zip_file.stat().st_size == 0:
                    self.logger.warning("%s is empty", zip_file)
                    return False

                # Check that it's a valid zip file
                try:
                    with zipfile.ZipFile(zip_file, 'r') as zf:
                        file_list = zf.namelist()
                        self.logger.info(
                            "%s is a valid zip file with %d entries",
                            zip_file.name,
                            len(file_list)
                        )
                except zipfile.BadZipFile as e:
                    self.logger.warning("%s is not a valid zip file: %s", zip_file, e)
                    return False

        return True
