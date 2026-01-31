"""
PHYBench Dataset Downloader

This module provides a downloader for the PHYBench dataset from HuggingFace.

For citation information, see prkit.prkit_datasets.citations.
"""

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base_downloader import BaseDownloader


class PHYBenchDownloader(BaseDownloader):
    """
    Downloader for PHYBench dataset from HuggingFace.

    The PHYBench dataset is available at:
    - HuggingFace: https://huggingface.co/datasets/Eureka-Lab/PHYBench
    - Homepage: https://www.phybench.cn/
    """

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "PHYBench"

    @property
    def download_info(self) -> Dict[str, Any]:
        """Return download information."""
        return {
            "source": "HuggingFace Datasets Server",
            "repository": "Eureka-Lab/PHYBench",
            "paper_url": "https://arxiv.org/pdf/2504.16074",
            "huggingface_url": "https://huggingface.co/datasets/Eureka-Lab/PHYBench",
            "format": "JSON",
            "splits": ["train"],
            "size_bytes": None,  # Size varies
            "license": "Research use",
            "download_method": "datasets-server API",
        }

    def download(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        force: bool = False,
        split: Optional[str] = None,
        **kwargs,
    ) -> Path:
        """
        Download the PHYBench dataset from HuggingFace.

        Args:
            data_dir: Target directory for download (None = auto-detect)
            force: If True, clean existing dataset directory and re-download.
                  If False, skip download if dataset already exists.
            split: Dataset split to download. Defaults to "train" if available.
            **kwargs: Additional download parameters

        Returns:
            Path to the downloaded dataset directory

        Raises:
            ValueError: If split is invalid
            FileExistsError: If dataset already exists and force=False
            RuntimeError: If download fails
        """
        # Use default if not provided
        if split is None:
            split = self.get_default_split() or "train"
        
        # Validate split
        self.validate_split(split)
        
        # Call parent download method which handles force logic
        return super().download(data_dir=data_dir, force=force, split=split, **kwargs)

    def _do_download(
        self,
        download_dir: Path,
        split: str = "train",
        **kwargs,
    ) -> Path:
        """
        Perform the actual PHYBench dataset download.

        Downloads data from HuggingFace datasets server API by paginating through
        all rows and saving as JSON files.

        Args:
            download_dir: Resolved download directory path
            split: Dataset split to download ("train" is the only available split)
            **kwargs: Additional download parameters

        Returns:
            Path to the downloaded dataset directory

        Raises:
            ImportError: If requests library is not installed
            ValueError: If split is not "train"
            RuntimeError: If download fails
        """
        if split != "train":
            raise ValueError(
                f"PHYBench dataset only has 'train' split. Got: {split}"
            )

        # Check if requests library is available
        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "The 'requests' library is required to download PHYBench. "
                "Install it with: pip install requests"
            ) from exc

        self.logger.info("Downloading PHYBench dataset...")
        self.logger.info("Target directory: %s", download_dir)

        try:
            # Create download directory
            download_dir.mkdir(parents=True, exist_ok=True)

            # Base URL for HuggingFace datasets server API
            dataset_name = "Eureka-Lab/PHYBench"
            base_url = "https://datasets-server.huggingface.co/rows"
            
            # Parameters for the API request
            params = {
                "dataset": dataset_name,
                "config": "default",
                "split": split,
            }

            # First, get the total number of rows
            self.logger.info("Fetching dataset info...")
            info_params = params.copy()
            info_params.update({"offset": 0, "length": 1})
            
            response = requests.get(base_url, params=info_params, timeout=60)
            response.raise_for_status()
            info_data = response.json()
            
            total_rows = info_data.get("num_rows_total", 0)
            if total_rows == 0:
                raise RuntimeError("Dataset appears to be empty or inaccessible")
            
            self.logger.info("Total rows in dataset: %d", total_rows)

            # Paginate through all rows
            # HuggingFace API allows up to 100 rows per request
            batch_size = 100
            all_rows = []
            max_retries = 3
            retry_delay = 5  # seconds

            for offset in range(0, total_rows, batch_size):
                current_length = min(batch_size, total_rows - offset)
                self.logger.info(
                    "Fetching rows %d-%d of %d...",
                    offset,
                    offset + current_length - 1,
                    total_rows,
                )

                request_params = params.copy()
                request_params.update({"offset": offset, "length": current_length})

                for attempt in range(max_retries):
                    try:
                        response = requests.get(
                            base_url, params=request_params, timeout=300
                        )
                        response.raise_for_status()
                        data = response.json()

                        # Extract row data from the response
                        rows = data.get("rows", [])
                        if not rows:
                            self.logger.warning(
                                "No rows returned for offset %d", offset
                            )
                            break

                        # Extract the actual row data (the "row" field contains the data)
                        for row_obj in rows:
                            row_data = row_obj.get("row", {})
                            if row_data:
                                all_rows.append(row_data)

                        self.logger.info(
                            "Fetched %d rows (total: %d/%d)",
                            len(rows),
                            len(all_rows),
                            total_rows,
                        )
                        break

                    except requests.exceptions.RequestException as req_err:
                        if attempt < max_retries - 1:
                            self.logger.warning(
                                "Request failed (attempt %d/%d): %s. Retrying in %d seconds...",
                                attempt + 1,
                                max_retries,
                                req_err,
                                retry_delay,
                            )
                            time.sleep(retry_delay)
                            continue
                        else:
                            raise RuntimeError(
                                f"Failed to fetch rows at offset {offset} after {max_retries} attempts: {req_err}"
                            ) from req_err

            if not all_rows:
                raise RuntimeError("No data was downloaded from the dataset")

            self.logger.info("Successfully fetched %d rows", len(all_rows))

            # Save as JSON file
            # The loader expects files like PHYBench-questions_v1.json
            # We'll save the main file as PHYBench-questions_v1.json (for "full" variant)
            output_file = download_dir / "PHYBench-questions_v1.json"
            
            self.logger.info("Saving data to %s...", output_file)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_rows, f, indent=2, ensure_ascii=False)

            self.logger.info(
                "Successfully downloaded PHYBench dataset to %s",
                download_dir,
            )
            self.logger.info("Total problems: %d", len(all_rows))
            self.logger.info("Output file: %s", output_file)

            return download_dir

        except (ImportError, ValueError) as e:
            # Re-raise ImportError and ValueError as-is (don't wrap)
            raise
        except (OSError, RuntimeError) as e:
            # Clean up on error
            if download_dir.exists():
                try:
                    shutil.rmtree(download_dir)
                except OSError:
                    pass

            self.logger.error("Failed to download PHYBench dataset: %s", e)
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

        # Check for the main JSON file
        json_file = data_dir / "PHYBench-questions_v1.json"

        if not json_file.exists():
            self.logger.warning("PHYBench JSON file not found: %s", json_file)
            return False

        if not json_file.is_file():
            self.logger.warning("%s is not a file", json_file)
            return False

        if json_file.stat().st_size == 0:
            self.logger.warning("%s is empty", json_file)
            return False

        # Verify that it's a valid JSON file containing a list
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                self.logger.warning(
                    "JSON file does not contain a list: %s", json_file
                )
                return False

            if len(data) == 0:
                self.logger.warning("JSON file is empty: %s", json_file)
                return False

            # Check that at least one entry has expected fields
            if len(data) > 0:
                first_entry = data[0]
                if not isinstance(first_entry, dict):
                    self.logger.warning(
                        "JSON entries are not dictionaries: %s", json_file
                    )
                    return False

            self.logger.info(
                "PHYBench dataset is valid: %d problems in %s",
                len(data),
                json_file,
            )
            return True

        except json.JSONDecodeError as e:
            self.logger.warning(
                "JSON file is not valid JSON: %s. Error: %s", json_file, e
            )
            return False
        except IOError as e:
            self.logger.warning("Failed to read JSON file: %s. Error: %s", json_file, e)
            return False
