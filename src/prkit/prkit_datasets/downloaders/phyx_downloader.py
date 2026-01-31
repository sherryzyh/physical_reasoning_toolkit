"""
PhyX Dataset Downloader

This module provides a downloader for the PhyX dataset from HuggingFace.

For citation information, see prkit.prkit_datasets.citations.
"""

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base_downloader import BaseDownloader

# Try to import PIL for image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None


class PhyXDownloader(BaseDownloader):
    """
    Downloader for PhyX dataset from HuggingFace.

    The PhyX dataset is available at:
    - HuggingFace: https://huggingface.co/datasets/Cloudriver/PhyX
    - Homepage: https://phyx-bench.github.io/
    - Paper: https://arxiv.org/pdf/2505.15929v2
    """

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "phyx"

    @property
    def download_info(self) -> Dict[str, Any]:
        """Return download information."""
        return {
            "source": "HuggingFace Datasets Server",
            "repository": "Cloudriver/PhyX",
            "paper_url": "https://arxiv.org/pdf/2505.15929v2",
            "homepage": "https://phyx-bench.github.io/",
            "huggingface_url": "https://huggingface.co/datasets/Cloudriver/PhyX",
            "format": "JSON",
            "splits": ["test_mini"],
            "size_bytes": None,  # Size varies
            "license": "MIT",
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
        Download the PhyX dataset from HuggingFace.

        Args:
            data_dir: Target directory for download (None = auto-detect)
            force: If True, clean existing dataset directory and re-download.
                  If False, skip download if dataset already exists.
            split: Dataset split to download. Defaults to "test_mini" if available.
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
            split = self.get_default_split() or "test_mini"
        
        # Validate split
        self.validate_split(split)
        
        # Call parent download method which handles force logic
        return super().download(data_dir=data_dir, force=force, split=split, **kwargs)

    def _do_download(
        self,
        download_dir: Path,
        split: str = "test_mini",
        **kwargs,
    ) -> Path:
        """
        Perform the actual PhyX dataset download.

        Downloads data from HuggingFace datasets server API by paginating through
        all rows and saving as JSON files.

        Args:
            download_dir: Resolved download directory path
            split: Dataset split to download ("test_mini" is the default)
            **kwargs: Additional download parameters

        Returns:
            Path to the downloaded dataset directory

        Raises:
            ImportError: If requests library is not installed
            ValueError: If split is not supported
            RuntimeError: If download fails
        """
        # Check if requests library is available
        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "The 'requests' library is required to download PhyX. "
                "Install it with: pip install requests"
            ) from exc

        self.logger.info("Downloading PhyX dataset...")
        self.logger.info("Target directory: %s", download_dir)
        self.logger.info("Split: %s", split)

        try:
            # Create download directory
            download_dir.mkdir(parents=True, exist_ok=True)

            # Base URL for HuggingFace datasets server API
            dataset_name = "Cloudriver/PhyX"
            base_url = "https://datasets-server.huggingface.co/rows"

            # Parameters for the API request
            params = {
                "dataset": dataset_name,
                "config": "default",
                "split": split,
            }

            # Try to use HuggingFace datasets library for direct download
            # This allows downloading the entire split in one go
            try:
                import datasets
                self.logger.info("Using HuggingFace datasets library to download entire split...")
                
                # Download the entire split using datasets library
                dataset = datasets.load_dataset(
                    "Cloudriver/PhyX",
                    split=split,
                    trust_remote_code=False
                )
                
                # Convert to list of dictionaries
                all_rows = [row for row in dataset]
                self.logger.info("Successfully downloaded %d rows using datasets library", len(all_rows))
                
            except ImportError:
                # Fallback to using the rows API - but note that it has limits
                self.logger.warning(
                    "HuggingFace datasets library not available. "
                    "For best results, install it with: pip install datasets"
                )
                self.logger.info("Attempting to use rows API (may have limitations)...")
                
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
                
                # Fetch all rows without specifying offset or length parameters
                # The API should return all available rows starting from the beginning
                request_params = params.copy()
                # Don't include offset or length parameters - let API return all rows

                max_retries = 3
                retry_delay = 5  # seconds
                all_rows = []

                for attempt in range(max_retries):
                    try:
                        self.logger.info("Fetching all rows in a single request (no length limit)...")
                        response = requests.get(
                            base_url, params=request_params, timeout=600  # 10 minute timeout for large datasets
                        )
                        response.raise_for_status()
                        data = response.json()

                        # Extract row data from the response
                        rows = data.get("rows", [])
                        if not rows:
                            raise RuntimeError("No rows returned from the API")

                        # Extract the actual row data (the "row" field contains the data)
                        for row_obj in rows:
                            row_data = row_obj.get("row", {})
                            if row_data:
                                all_rows.append(row_data)

                        self.logger.info("Successfully fetched %d rows", len(all_rows))
                        
                        # Verify we got all rows
                        if len(all_rows) < total_rows:
                            self.logger.warning(
                                "Only fetched %d out of %d rows. API may have limits.",
                                len(all_rows), total_rows
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
                                f"Failed to fetch dataset after {max_retries} attempts: {req_err}"
                            ) from req_err

                if not all_rows:
                    raise RuntimeError("No data was downloaded from the dataset")

            # Post-process to handle images and convert to JSON-serializable format
            self.logger.info("Post-processing data to handle images...")
            processed_rows = self._process_rows_for_json(all_rows, download_dir)

            # Validate JSON serializability before saving
            self.logger.info("Validating JSON serializability...")
            try:
                json_str = json.dumps(processed_rows, indent=2, ensure_ascii=False)
                # Test that we can parse it back
                json.loads(json_str)
                self.logger.info("JSON validation passed")
            except (TypeError, ValueError) as e:
                self.logger.warning(
                    "Initial JSON serialization check failed: %s. "
                    "Attempting to fix non-serializable values...",
                    e,
                )
                # Try to fix any remaining issues
                processed_rows = self._fix_json_serialization(processed_rows)
                # Validate again
                try:
                    json_str = json.dumps(processed_rows, indent=2, ensure_ascii=False)
                    json.loads(json_str)
                    self.logger.info("JSON validation passed after fixing")
                except (TypeError, ValueError) as e2:
                    self.logger.error(
                        "JSON serialization still failing after fix attempt: %s", e2
                    )
                    # Log first problematic row for debugging
                    for i, row in enumerate(processed_rows[:5]):
                        try:
                            json.dumps(row)
                        except (TypeError, ValueError) as row_err:
                            self.logger.error(
                                "Row %d is not serializable: %s. Row keys: %s",
                                i,
                                row_err,
                                list(row.keys()) if isinstance(row, dict) else type(row),
                            )
                    raise

            # Save as JSON file
            output_file = download_dir / f"PhyX-{split}.json"

            self.logger.info("Saving data to %s...", output_file)
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(processed_rows, f, indent=2, ensure_ascii=False)
            except (TypeError, ValueError) as e:
                self.logger.error(
                    "Failed to save JSON file: %s. Attempting to fix and retry...", e
                )
                # Try one more time with more aggressive cleaning
                processed_rows = self._fix_json_serialization(processed_rows)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(processed_rows, f, indent=2, ensure_ascii=False)

            self.logger.info(
                "Successfully downloaded PhyX dataset to %s",
                download_dir,
            )
            self.logger.info("Total problems: %d", len(processed_rows))
            self.logger.info("Output file: %s", output_file)

            return download_dir

        except (ImportError, ValueError):
            # Re-raise ImportError and ValueError as-is (don't wrap)
            raise
        except (OSError, RuntimeError) as e:
            # Clean up on error
            if download_dir.exists():
                try:
                    shutil.rmtree(download_dir)
                except OSError:
                    pass

            self.logger.error("Failed to download PhyX dataset: %s", e)
            raise RuntimeError(f"Download failed: {e}") from e

    def _process_rows_for_json(
        self, rows: List[Dict[str, Any]], download_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Process rows to handle images and convert to JSON-serializable format.
        
        This method:
        1. Creates an images directory
        2. Extracts PIL Image objects and saves them as files
        3. Replaces image objects with file paths
        4. Handles other non-serializable objects
        
        Args:
            rows: List of row dictionaries from the dataset
            download_dir: Directory where images will be saved
            
        Returns:
            List of processed row dictionaries that are JSON-serializable
        """
        # Create images directory
        images_dir = download_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Created images directory: %s", images_dir)
        
        processed_rows = []
        
        for idx, row in enumerate(rows):
            try:
                # Ensure row is a dictionary
                if not isinstance(row, dict):
                    self.logger.warning(
                        "Row %d is not a dictionary (type: %s), skipping", idx, type(row)
                    )
                    continue
                
                processed_row = {}
                
                # Get problem ID for naming images
                problem_id = row.get("id") or row.get("index") or idx
                
                # Process each field in the row
                for key, value in row.items():
                    try:
                        # Ensure key is a string
                        key_str = str(key) if not isinstance(key, str) else key
                        
                        # Handle image fields - check common image field names
                        if key_str in ["image", "images", "image_path", "image_paths"]:
                            image_paths = self._process_image_field(
                                value, problem_id, idx, images_dir
                            )
                            # Store as image_paths (standardized field name)
                            # Only set if we have image paths or if it's the first image field
                            if image_paths or "image_paths" not in processed_row:
                                processed_row["image_paths"] = image_paths
                        else:
                            # Process other fields, handling non-serializable objects
                            processed_row[key_str] = self._make_json_serializable(value)
                    except Exception as e:
                        self.logger.warning(
                            "Error processing field '%s' in row %d: %s. Skipping field.",
                            key,
                            idx,
                            e,
                        )
                        # Skip this field but continue processing
                        continue
                
                processed_rows.append(processed_row)
            except Exception as e:
                self.logger.warning(
                    "Error processing row %d: %s. Skipping row.", idx, e
                )
                continue
        
        self.logger.info(
            "Processed %d rows, saved images to %s", len(processed_rows), images_dir
        )
        return processed_rows
    
    def _process_image_field(
        self,
        image_value: Any,
        problem_id: Any,
        fallback_idx: int,
        images_dir: Path,
    ) -> List[str]:
        """
        Process an image field value and save images to files.
        
        Args:
            image_value: The image value (can be PIL Image, list of images, dict, etc.)
            problem_id: Problem ID for naming images
            fallback_idx: Fallback index if problem_id is not usable
            images_dir: Directory to save images
            
        Returns:
            List of relative image paths
        """
        image_paths = []
        
        if image_value is None:
            return image_paths
        
        # Convert problem_id to string for filename
        try:
            if isinstance(problem_id, (int, float)):
                problem_id_str = str(int(problem_id))
            else:
                problem_id_str = str(problem_id)
        except (ValueError, TypeError):
            problem_id_str = str(fallback_idx)
        
        # Handle PIL Image objects
        if PIL_AVAILABLE and isinstance(image_value, Image.Image):
            img_path = self._save_pil_image(
                image_value, problem_id_str, 0, images_dir
            )
            if img_path:
                image_paths.append(img_path)
        
        # Handle list/tuple of images
        elif isinstance(image_value, (list, tuple)):
            for img_idx, img_item in enumerate(image_value):
                if PIL_AVAILABLE and isinstance(img_item, Image.Image):
                    img_path = self._save_pil_image(
                        img_item, problem_id_str, img_idx, images_dir
                    )
                    if img_path:
                        image_paths.append(img_path)
                elif isinstance(img_item, dict):
                    # Handle dict with image data
                    if "bytes" in img_item:
                        img_path = self._save_bytes_image(
                            img_item["bytes"], problem_id_str, img_idx, images_dir
                        )
                        if img_path:
                            image_paths.append(img_path)
        
        # Handle dict with image data
        elif isinstance(image_value, dict):
            if "bytes" in image_value:
                img_path = self._save_bytes_image(
                    image_value["bytes"], problem_id_str, 0, images_dir
                )
                if img_path:
                    image_paths.append(img_path)
            elif PIL_AVAILABLE and "image" in image_value:
                img = image_value["image"]
                if isinstance(img, Image.Image):
                    img_path = self._save_pil_image(
                        img, problem_id_str, 0, images_dir
                    )
                    if img_path:
                        image_paths.append(img_path)
        
        # Handle bytes directly
        elif isinstance(image_value, bytes):
            img_path = self._save_bytes_image(
                image_value, problem_id_str, 0, images_dir
            )
            if img_path:
                image_paths.append(img_path)
        
        return image_paths
    
    def _save_pil_image(
        self, pil_image: Any, problem_id: str, img_idx: int, images_dir: Path
    ) -> Optional[str]:
        """
        Save a PIL Image object to a file.
        
        Args:
            pil_image: PIL Image object
            problem_id: Problem ID string
            img_idx: Image index within the problem
            images_dir: Directory to save the image
            
        Returns:
            Relative path to the saved image, or None if saving failed
        """
        if not PIL_AVAILABLE:
            return None
        
        try:
            # Determine filename
            if img_idx == 0:
                img_filename = f"{problem_id}.png"
            else:
                img_filename = f"{problem_id}_{img_idx}.png"
            
            img_path = images_dir / img_filename
            
            # Convert to RGB if necessary (handles RGBA, P, etc.)
            if pil_image.mode not in ("RGB", "L"):
                pil_image = pil_image.convert("RGB")
            
            # Save the image
            pil_image.save(img_path, "PNG")
            
            # Return relative path
            return f"images/{img_filename}"
        except Exception as e:
            self.logger.warning(
                "Failed to save PIL image for problem %s, image %d: %s",
                problem_id,
                img_idx,
                e,
            )
            return None
    
    def _save_bytes_image(
        self, img_bytes: bytes, problem_id: str, img_idx: int, images_dir: Path
    ) -> Optional[str]:
        """
        Save image bytes to a file.
        
        Args:
            img_bytes: Image bytes
            problem_id: Problem ID string
            img_idx: Image index within the problem
            images_dir: Directory to save the image
            
        Returns:
            Relative path to the saved image, or None if saving failed
        """
        try:
            # Determine file extension from bytes
            if img_bytes.startswith(b"\xff\xd8\xff"):
                ext = ".jpg"
            elif img_bytes.startswith(b"\x89PNG"):
                ext = ".png"
            else:
                ext = ".png"  # Default to PNG
            
            # Determine filename
            if img_idx == 0:
                img_filename = f"{problem_id}{ext}"
            else:
                img_filename = f"{problem_id}_{img_idx}{ext}"
            
            img_path = images_dir / img_filename
            
            # Save the bytes
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            
            # Return relative path
            return f"images/{img_filename}"
        except Exception as e:
            self.logger.warning(
                "Failed to save bytes image for problem %s, image %d: %s",
                problem_id,
                img_idx,
                e,
            )
            return None
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Recursively convert an object to JSON-serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable version of the object
        """
        # Handle None
        if obj is None:
            return None
        
        # Handle PIL Images (should be caught earlier, but safety check)
        if PIL_AVAILABLE and isinstance(obj, Image.Image):
            # Return None - images should be handled separately
            return None
        
        # Handle bytes
        if isinstance(obj, bytes):
            import base64
            try:
                return base64.b64encode(obj).decode("utf-8")
            except Exception:
                return ""
        
        # Handle dict
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                # Ensure key is string
                key_str = str(k) if not isinstance(k, str) else k
                result[key_str] = self._make_json_serializable(v)
            return result
        
        # Handle list/tuple
        if isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        
        # Handle set
        if isinstance(obj, set):
            return [self._make_json_serializable(item) for item in obj]
        
        # Handle numpy types
        try:
            import numpy as np
            
            if isinstance(obj, np.ndarray):
                try:
                    return obj.tolist()
                except (ValueError, TypeError):
                    return [self._make_json_serializable(item) for item in obj]
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.str_):
                return str(obj)
        except ImportError:
            pass
        
        # Handle basic types that are already JSON serializable
        if isinstance(obj, (str, int, float, bool)):
            return obj
        
        # Try to serialize directly
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            # Not serializable, try to convert
            try:
                # Try to get string representation
                obj_str = str(obj)
                # Check if it's a valid JSON string
                try:
                    json.loads(obj_str)
                    return obj_str
                except (json.JSONDecodeError, ValueError):
                    # Not a JSON string, return as regular string
                    return obj_str
            except Exception:
                # Last resort: return empty string
                self.logger.debug(
                    "Could not serialize object of type %s, returning empty string",
                    type(obj),
                )
                return ""
    
    def _fix_json_serialization(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        More aggressively fix JSON serialization issues.
        
        Args:
            data: List of dictionaries to fix
            
        Returns:
            Fixed list of dictionaries
        """
        fixed_data = []
        for idx, row in enumerate(data):
            try:
                if not isinstance(row, dict):
                    self.logger.warning(
                        "Row %d is not a dict (type: %s), skipping", idx, type(row)
                    )
                    continue
                
                fixed_row = {}
                for key, value in row.items():
                    try:
                        # Ensure key is a string
                        key_str = str(key) if not isinstance(key, str) else key
                        
                        # Convert value to JSON-serializable format
                        serialized_value = self._make_json_serializable(value)
                        
                        # Validate that the value can be serialized
                        json.dumps(serialized_value)
                        fixed_row[key_str] = serialized_value
                    except (TypeError, ValueError) as e:
                        # Value is not serializable even after conversion
                        self.logger.debug(
                            "Field '%s' in row %d still not serializable after conversion: %s. "
                            "Setting to null.",
                            key,
                            idx,
                            e,
                        )
                        fixed_row[str(key)] = None
                    except Exception as e:
                        self.logger.warning(
                            "Unexpected error processing field '%s' in row %d: %s. Setting to null.",
                            key,
                            idx,
                            e,
                        )
                        fixed_row[str(key)] = None
                
                # Validate the entire row can be serialized
                try:
                    json.dumps(fixed_row)
                    fixed_data.append(fixed_row)
                except (TypeError, ValueError) as e:
                    self.logger.warning(
                        "Row %d still not serializable after fixing: %s. Skipping row.", idx, e
                    )
                    continue
            except Exception as e:
                self.logger.warning(
                    "Failed to fix row %d, skipping: %s", idx, e
                )
                continue
        return fixed_data

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

        # Check for the main JSON file (test_mini is the default split)
        json_file = data_dir / "PhyX-test_mini.json"

        if not json_file.exists():
            # Try to find any PhyX JSON file
            json_files = list(data_dir.glob("PhyX-*.json"))
            if not json_files:
                self.logger.warning("PhyX JSON file not found in %s", data_dir)
                return False
            json_file = json_files[0]

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
                "PhyX dataset is valid: %d problems in %s",
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
