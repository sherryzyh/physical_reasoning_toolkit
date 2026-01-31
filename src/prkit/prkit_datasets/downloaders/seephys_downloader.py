"""
SeePhys Dataset Downloader

This module provides a downloader for the SeePhys dataset from HuggingFace.

For citation information, see prkit.prkit_datasets.citations.
"""

import base64
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from .base_downloader import BaseDownloader

try:
    import pyarrow.parquet as pq
    USE_PYARROW = True
except ImportError:
    USE_PYARROW = False


class SeePhysDownloader(BaseDownloader):
    """
    Downloader for SeePhys dataset from HuggingFace.

    The SeePhys dataset is available at:
    - HuggingFace: https://huggingface.co/datasets/SeePhys/SeePhys
    - Homepage: https://seephys.github.io/
    - Paper: https://openreview.net/pdf?id=APNWmytTCS
    """

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "seephys"

    @property
    def download_info(self) -> Dict[str, Any]:
        """Return download information."""
        return {
            "source": "HuggingFace",
            "repository": "SeePhys/SeePhys",
            "homepage": "https://seephys.github.io/",
            "paper_url": "https://openreview.net/pdf?id=APNWmytTCS",
            "huggingface_url": "https://huggingface.co/datasets/SeePhys/SeePhys",
            "format": "Parquet/JSON",
            "splits": ["train"],
            "size_bytes": None,  # Size varies
            "license": "Research use",
            "download_method": "datasets library",
        }

    def download(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        force: bool = False,
        split: Optional[str] = None,
        **kwargs,
    ) -> Path:
        """
        Download the SeePhys dataset from HuggingFace.

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
        Perform the actual SeePhys dataset download.

        Args:
            download_dir: Resolved download directory path
            split: Dataset split to download ("train" is the only available split)
            **kwargs: Additional download parameters

        Returns:
            Path to the downloaded dataset directory

        Raises:
            ImportError: If datasets library is not installed
            ValueError: If split is not "train"
            RuntimeError: If download fails
        """
        if split != "train":
            raise ValueError(
                f"SeePhys dataset only has 'train' split. Got: {split}"
            )

        # Check if datasets library is available
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' library is required to download SeePhys. "
                "Install it with: pip install datasets"
            ) from exc

        self.logger.info("Downloading SeePhys dataset...")
        self.logger.info("Target directory: %s", download_dir)
        self.logger.info("Resolved absolute path: %s", download_dir.resolve())

        try:
            # Resolve to absolute path (Path.home() already returns absolute, but resolve() ensures canonical form)
            download_dir = download_dir.resolve()
            
            # Create download directory and all parent directories
            # This will create ~/PHYSICAL_REASONING_DATASETS/seephys if ~/PHYSICAL_REASONING_DATASETS doesn't exist
            # parents=True ensures all parent directories are created
            download_dir.mkdir(parents=True, exist_ok=True)
            
            # Verify the directory was created
            if not download_dir.exists():
                raise RuntimeError(f"Failed to create directory: {download_dir}")
            
            self.logger.info("Created directory: %s", download_dir)
            self.logger.info("Directory exists: %s", download_dir.exists())
            self.logger.info("Parent directory exists: %s", download_dir.parent.exists())

            # Create split directory
            split_dir = download_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            # Load dataset from HuggingFace using datasets library
            dataset_name = "SeePhys/SeePhys"
            self.logger.info(
                "Loading dataset from HuggingFace: %s (split: %s)",
                dataset_name,
                split,
            )
            
            # Load the dataset
            dataset = load_dataset(dataset_name, split=split)
            self.logger.info("Loaded %d examples from HuggingFace", len(dataset))
            
            # Convert to pandas DataFrame for easier processing
            self.logger.info("Converting to pandas DataFrame...")
            df = dataset.to_pandas()
            self.logger.info("Converted to DataFrame with %d rows", len(df))
            
            # Step 1: Save the parquet file first
            parquet_file = download_dir / f"{split}.parquet"
            self.logger.info("Saving parquet file to: %s", parquet_file)
            df.to_parquet(parquet_file, engine="pyarrow", index=False)
            self.logger.info("Successfully saved parquet file to: %s", parquet_file)
            
            # Step 2: Create images directory
            images_dir = download_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info("Created images directory: %s", images_dir)
            
            # Step 3: Convert parquet to JSON files (post-processing)
            self.logger.info("Converting parquet file to JSON format...")
            self._convert_parquet_to_json(parquet_file, split_dir, images_dir)
            
            self.logger.info(
                "Successfully downloaded SeePhys dataset to %s",
                download_dir,
            )
            self.logger.info("Total problems: %d", len(df))

            return download_dir

        except (ImportError, ValueError) as e:
            # Re-raise ImportError and ValueError as-is (don't wrap)
            raise
        except (OSError, RuntimeError) as e:
            # Clean up on error
            if download_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(download_dir)
                except OSError:
                    pass

            self.logger.error("Failed to download SeePhys dataset: %s", e)
            raise RuntimeError(f"Download failed: {e}") from e

    def _convert_bytes_in_structure(self, obj: Any) -> Any:
        """
        Recursively convert bytes objects in nested structures to base64 strings.
        
        Args:
            obj: The object to process (can be dict, list, bytes, or other types)
            
        Returns:
            Object with bytes converted to base64 strings
        """
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode('utf-8')
        elif isinstance(obj, dict):
            return {k: self._convert_bytes_in_structure(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_bytes_in_structure(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            # Handle numpy arrays
            try:
                result = obj.tolist()
                return self._convert_bytes_in_structure(result)
            except (ValueError, TypeError):
                # Object array - convert each element
                return [self._convert_bytes_in_structure(item) for item in obj]
        else:
            return obj

    def _convert_parquet_to_json(
        self, parquet_file: Path, output_dir: Path, images_dir: Path
    ) -> None:
        """
        Convert a parquet file to individual JSON files.
        
        This method reads the parquet file using pyarrow and converts each row
        to a separate JSON file. Images from the "images" column are saved to
        the images directory and their paths are added to the JSON files.
        
        Args:
            parquet_file: Path to the parquet file to convert
            output_dir: Directory to save JSON files
            images_dir: Directory to save image files
        """
        if not USE_PYARROW:
            raise ImportError(
                "pyarrow is required for parquet to JSON conversion. "
                "Install it with: pip install pyarrow"
            )
        
        if not parquet_file.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_file}")
        
        # Read parquet file with pyarrow
        self.logger.info("Reading parquet file: %s", parquet_file)
        table = pq.read_table(parquet_file)
        df_dict = table.to_pydict()
        
        # Get number of rows
        if not df_dict:
            self.logger.warning("Parquet file is empty")
            return
        
        num_rows = len(df_dict[list(df_dict.keys())[0]])
        self.logger.info("Found %d rows in parquet file", num_rows)
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure images directory exists
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if "images" column exists and log its type
        if "images" in df_dict and num_rows > 0:
            first_images_value = df_dict["images"][0]
            self.logger.info(
                "Found 'images' column. Type of value in 'images' column (row 0): %s",
                type(first_images_value)
            )
            if first_images_value is not None:
                self.logger.info(
                    "Value in 'images' column (row 0) is not None. Additional info: %s",
                    str(first_images_value)[:200] if hasattr(first_images_value, '__str__') else repr(first_images_value)[:200]
                )
                if isinstance(first_images_value, (list, tuple)) and len(first_images_value) > 0:
                    self.logger.info(
                        "First element in 'images' column (row 0): type=%s",
                        type(first_images_value[0])
                    )
                elif isinstance(first_images_value, dict):
                    self.logger.info(
                        "Keys in 'images' column dict (row 0): %s",
                        list(first_images_value.keys())
                    )
        
        # Convert each row to JSON
        for i in range(num_rows):
            # Create sample dictionary, excluding "images" column initially
            sample_dict = {}
            
            # Process images column first to get problem_index
            image_paths = []
            if "images" in df_dict:
                images_value = df_dict["images"][i]
                
                # Get problem_index for naming images
                # We'll use the index from the row or fallback to i
                problem_index = i
                if "index" in df_dict:
                    problem_index = df_dict["index"][i]
                    if problem_index is None:
                        problem_index = i
                    if isinstance(problem_index, (np.integer, np.floating)):
                        problem_index = int(problem_index.item())
                    else:
                        try:
                            problem_index = int(problem_index)
                        except (ValueError, TypeError):
                            problem_index = i
                else:
                    problem_index = i
                
                # Process images if they exist
                if images_value is not None:
                    # Handle numpy arrays
                    if isinstance(images_value, np.ndarray):
                        try:
                            images_value = images_value.tolist()
                        except (ValueError, TypeError):
                            # If tolist() fails, try to iterate directly
                            images_value = list(images_value)
                    
                    if isinstance(images_value, (list, tuple)):
                        for img_idx, img_data in enumerate(images_value):
                            # Handle numpy array elements
                            if isinstance(img_data, np.ndarray):
                                try:
                                    img_data = img_data.tolist()
                                except (ValueError, TypeError):
                                    img_data = dict(img_data) if hasattr(img_data, '__iter__') else img_data
                            
                            if isinstance(img_data, dict) and "bytes" in img_data:
                                img_bytes = img_data["bytes"]
                                
                                # Handle numpy bytes
                                if isinstance(img_bytes, np.ndarray):
                                    img_bytes = bytes(img_bytes.tobytes())
                                
                                if isinstance(img_bytes, bytes):
                                    # Determine image extension (try to detect from bytes)
                                    # Default to .png, but could be .jpg, .jpeg, etc.
                                    img_ext = ".png"  # Default extension
                                    if img_bytes.startswith(b'\xff\xd8\xff'):
                                        img_ext = ".jpg"
                                    elif img_bytes.startswith(b'\x89PNG'):
                                        img_ext = ".png"
                                    
                                    # Create image filename: <problem_index>_<image_index>
                                    img_filename = f"{problem_index}_{img_idx}{img_ext}"
                                    img_path = images_dir / img_filename
                                    
                                    # Save image bytes to file
                                    with open(img_path, 'wb') as img_file:
                                        img_file.write(img_bytes)
                                    
                                    # Store relative path from output_dir (or absolute path)
                                    # Using relative path: images/<filename>
                                    relative_img_path = f"images/{img_filename}"
                                    image_paths.append(relative_img_path)
                                    self.logger.debug(
                                        "Saved image %d for problem %d to %s",
                                        img_idx, problem_index, img_path
                                    )
            
            # Process all other columns
            for key in df_dict.keys():
                # Skip "images" column as we've already processed it
                if key == "images":
                    continue
                
                # Regular columns
                value = df_dict[key][i]
                
                # Handle numpy types
                if isinstance(value, np.ndarray):
                    # Convert numpy array to list, handling nested bytes
                    try:
                        sample_dict[key] = value.tolist()
                        # Check if the list contains bytes and convert them
                        if isinstance(sample_dict[key], list):
                            sample_dict[key] = self._convert_bytes_in_structure(sample_dict[key])
                    except (ValueError, TypeError):
                        # If tolist() fails (e.g., object array with bytes), try alternative
                        sample_dict[key] = [self._convert_bytes_in_structure(item) for item in value]
                elif isinstance(value, (np.integer, np.floating)):
                    sample_dict[key] = value.item()
                elif isinstance(value, np.bool_):
                    sample_dict[key] = bool(value)
                elif isinstance(value, bytes):
                    # Convert bytes to base64 string
                    sample_dict[key] = base64.b64encode(value).decode('utf-8')
                elif isinstance(value, (dict, list)):
                    # Handle nested structures
                    sample_dict[key] = self._convert_bytes_in_structure(value)
                elif value is None:
                    sample_dict[key] = None
                else:
                    # Check for NaN values (only for scalar types, not arrays)
                    # pd.isna() on arrays returns an array, which causes issues
                    if not isinstance(value, np.ndarray):
                        try:
                            if pd.isna(value):
                                sample_dict[key] = None
                                continue
                        except (ValueError, TypeError):
                            # pd.isna() failed, continue with normal handling
                            pass
                    
                    # Try to convert, but check JSON serializability first
                    try:
                        # Test if it's JSON serializable
                        json.dumps(value)
                        sample_dict[key] = value
                    except (TypeError, ValueError):
                        # Not JSON serializable, try converting bytes in structure
                        sample_dict[key] = self._convert_bytes_in_structure(value)
            
            # Add image_paths to the sample_dict
            sample_dict["image_paths"] = image_paths
            
            # Get file ID for naming the JSON file
            file_id = sample_dict.get('index', i)
            if file_id is None:
                file_id = i
            if isinstance(file_id, (np.integer, np.floating)):
                file_id = str(file_id.item())
            else:
                file_id = str(file_id)
            
            # Final pass: ensure all bytes are converted (safety check)
            sample_dict = self._convert_bytes_in_structure(sample_dict)
            
            # Save as JSON file
            output_file = output_dir / f"{file_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(sample_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(
            "Successfully converted %d samples to JSON files in %s",
            num_rows,
            output_dir,
        )

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

        # Check for train split directory
        train_dir = data_dir / "train"
        
        if not train_dir.exists():
            self.logger.warning("Train split directory not found")
            return False

        if not train_dir.is_dir():
            self.logger.warning("%s is not a directory", train_dir)
            return False

        # Check for JSON files
        json_files = list(train_dir.glob("*.json"))
        if len(json_files) == 0:
            self.logger.warning("No JSON files found in %s", train_dir)
            return False

        # Check that at least one JSON file is valid
        found_valid_json = False
        for json_file in json_files[:5]:  # Check first 5
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    json.load(f)
                found_valid_json = True
                break
            except (json.JSONDecodeError, IOError):
                continue

        if not found_valid_json:
            self.logger.warning(
                "No valid JSON files found in %s", train_dir
            )
            return False

        # Also check for parquet file (optional)
        parquet_file = data_dir / "train.parquet"
        if parquet_file.exists():
            try:
                pd.read_parquet(parquet_file, engine="pyarrow")
            except (ValueError, OSError, ImportError) as e:
                self.logger.warning(
                    "Parquet file exists but is invalid: %s", e
                )
                # Don't fail verification if parquet is invalid, JSON is primary format

        return True
