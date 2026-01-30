"""
PhysReason Dataset Downloader

This module provides a downloader for the PhysReason dataset from HuggingFace.

For citation information, see prkit.prkit_datasets.citations.
"""

import json
import shutil
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
            ImportError: If datasets library is not installed
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

        Args:
            download_dir: Resolved download directory path
            variant: Dataset variant ("full" or "mini")
            **kwargs: Additional download parameters

        Returns:
            Path to the downloaded dataset directory

        Raises:
            ImportError: If datasets library is not installed
            RuntimeError: If download fails
        """
        # Check if datasets library is available
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' library is required to download PhysReason. "
                "Install it with: pip install datasets"
            ) from exc

        self.logger.info(
            "Downloading PhysReason dataset (%s variant)...", variant
        )
        self.logger.info("Target directory: %s", download_dir)

        try:
            # Create download directory
            download_dir.mkdir(parents=True, exist_ok=True)

            # HuggingFace dataset only has "default" config, so we always use that
            # For "mini" variant, we'll sample a subset after downloading
            dataset_name = "zhibei1204/PhysReason"
            config_name = "default"

            # Load dataset from HuggingFace using streaming mode to bypass schema validation
            # The dataset has a schema mismatch (step_1-step_5 vs schema only defining step_1)
            # Streaming mode avoids this issue by not validating against the schema
            self.logger.info(
                "Loading dataset from HuggingFace: %s (config: %s) using streaming mode",
                dataset_name,
                config_name,
            )
            
            # Try loading with streaming first to bypass schema validation
            try:
                streaming_dataset = load_dataset(
                    dataset_name, 
                    config_name, 
                    split="train",
                    streaming=True
                )
                self.logger.info("Loaded dataset in streaming mode")
                
                # Convert streaming dataset to list for processing
                # We need to iterate through it to get examples
                dataset_examples = []
                total_count = 0
                max_to_collect = 200 if variant == "mini" else None  # None means collect all
                
                for example in streaming_dataset:
                    dataset_examples.append(example)
                    total_count += 1
                    
                    # Log progress every 100 examples for full variant
                    if variant == "full" and total_count % 100 == 0:
                        self.logger.info("Collected %d examples so far...", total_count)
                    
                    # For mini variant, stop at 200 examples
                    if max_to_collect is not None and total_count >= max_to_collect:
                        break
                
                self.logger.info("Collected %d examples from streaming dataset", len(dataset_examples))
                max_problems = len(dataset_examples)
                
                if variant == "mini":
                    self.logger.info(
                        "Creating mini variant: using first %d problems",
                        max_problems,
                    )
                elif variant == "full":
                    self.logger.info(
                        "Creating full variant: using all %d problems",
                        max_problems,
                    )
                else:
                    raise ValueError(f"Unknown variant: {variant}. Choose 'full' or 'mini'")
                    
            except Exception as stream_error:
                # Fallback: try without streaming if streaming fails
                self.logger.warning(
                    "Streaming mode failed: %s. Trying non-streaming mode with download_mode...",
                    stream_error
                )
                try:
                    # Try with force_redownload to regenerate cache and bypass schema issues
                    dataset = load_dataset(
                        dataset_name, 
                        config_name, 
                        split="train",
                        download_mode="force_redownload"
                    )
                    self.logger.info("Loaded dataset in non-streaming mode with force_redownload")
                    
                    # Determine how many problems to save based on variant
                    if variant == "mini":
                        # Mini variant: use first 200 problems
                        max_problems = 200
                        self.logger.info(
                            "Creating mini variant: sampling first %d problems from %d total",
                            max_problems,
                            len(dataset),
                        )
                    elif variant == "full":
                        # Full variant: use all problems
                        max_problems = len(dataset)
                        self.logger.info(
                            "Creating full variant: using all %d problems",
                            max_problems,
                        )
                    else:
                        raise ValueError(f"Unknown variant: {variant}. Choose 'full' or 'mini'")
                    
                    # Convert to list and limit to max_problems for consistent processing
                    dataset_examples = list(dataset[:max_problems])
                    self.logger.info("Collected %d examples from dataset", len(dataset_examples))
                except Exception as fallback_error:
                    self.logger.error("Both streaming and non-streaming modes failed")
                    raise RuntimeError(
                        f"Failed to load dataset. Streaming error: {stream_error}. "
                        f"Fallback error: {fallback_error}"
                    ) from fallback_error

            # Create variant directory
            if variant == "full":
                variant_dir = download_dir / "PhysReason_full"
            else:
                variant_dir = download_dir / "PhysReason-mini"

            variant_dir.mkdir(parents=True, exist_ok=True)

            # Convert dataset to the expected directory structure
            # Each problem gets its own directory with problem.json
            self.logger.info("Converting dataset to directory structure...")
            saved_count = 0
            
            # Process collected examples
            for idx, example in enumerate(dataset_examples):
                # Convert example to dict if it's not already
                if hasattr(example, '__dict__'):
                    example_dict = dict(example)
                elif hasattr(example, 'keys'):
                    example_dict = dict(example)
                else:
                    example_dict = example
                
                # Create problem directory
                problem_id = example_dict.get("problem_id", f"problem_{idx:05d}")
                problem_dir = variant_dir / problem_id
                problem_dir.mkdir(exist_ok=True)

                # Save problem.json
                problem_file = problem_dir / "problem.json"
                with open(problem_file, "w", encoding="utf-8") as f:
                    json.dump(example_dict, f, indent=2, ensure_ascii=False)

                # Handle images if present
                if "image" in example_dict and example_dict["image"] is not None:
                    images_dir = problem_dir / "images"
                    images_dir.mkdir(exist_ok=True)

                    # Save image
                    image = example_dict["image"]
                    if hasattr(image, "save"):
                        # PIL Image
                        image_path = images_dir / f"image_{problem_id}.png"
                        image.save(image_path)
                    elif isinstance(image, (str, Path)):
                        # Path to image file
                        image_path = images_dir / Path(image).name
                        shutil.copy2(image, image_path)
                
                saved_count += 1

            self.logger.info(
                "Successfully downloaded PhysReason (%s) to %s",
                variant,
                variant_dir,
            )
            self.logger.info("Saved problems: %d (from %d collected)", saved_count, len(dataset_examples))

            return download_dir

        except (ImportError, ValueError, RuntimeError, OSError) as e:
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

        # Check for both variants
        full_dir = data_dir / "PhysReason_full"
        mini_dir = data_dir / "PhysReason-mini"

        # At least one variant should exist
        if not full_dir.exists() and not mini_dir.exists():
            self.logger.warning("No variant directories found")
            return False

        # Verify structure for existing variants
        for variant_dir in [full_dir, mini_dir]:
            if variant_dir.exists():
                # Check that it's a directory
                if not variant_dir.is_dir():
                    self.logger.warning("%s is not a directory", variant_dir)
                    return False

                # Check for at least one problem directory
                problem_dirs = [d for d in variant_dir.iterdir() if d.is_dir()]
                if len(problem_dirs) == 0:
                    self.logger.warning(
                        "No problem directories found in %s", variant_dir
                    )
                    return False

                # Check that at least one problem has a problem.json file
                found_valid_problem = False
                for problem_dir in problem_dirs[:5]:  # Check first 5
                    problem_file = problem_dir / "problem.json"
                    if problem_file.exists():
                        found_valid_problem = True
                        break

                if not found_valid_problem:
                    self.logger.warning(
                        "No valid problem.json files found in %s", variant_dir
                    )
                    return False

        return True
