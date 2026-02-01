"""
SeePhys dataset loader.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from prkit.prkit_core import PRKitLogger
from prkit.prkit_core.domain import PhysicalDataset

from .base_loader import BaseDatasetLoader


class SeePhysLoader(BaseDatasetLoader):
    """Loader for SeePhys dataset."""

    def __init__(self):
        """Initialize the SeePhys loader with a logger."""
        super().__init__()
        self.logger = PRKitLogger.get_logger(__name__)

    @property
    def name(self) -> str:
        return "seephys"

    @property
    def description(self) -> str:
        return "SeePhys: A visual physics reasoning dataset with questions, images, and captions"

    @property
    def modalities(self) -> List[str]:
        """SeePhys supports both text and image modalities."""
        return ["text", "image"]

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "repository_url": "https://huggingface.co/datasets/SeePhys/SeePhys",
            "license": "Research use",
            "homepage": "https://seephys.github.io/",
            "paper_url": "https://openreview.net/pdf?id=APNWmytTCS",
            "languages": ["en", "zh"],
            "variants": ["full"],  # Variant is kept as a property but not used for loading
            "splits": ["train"],
            "problem_types": ["OE"],
            "total_problems": "2000",
            "source": "SeePhys dataset from HuggingFace",
            "modalities": self.modalities,
        }

    def get_default_variant(self) -> Optional[str]:
        """Return default variant 'full' for SeePhys dataset."""
        return "full"

    def get_default_split(self) -> Optional[str]:
        """Return default split 'train' for SeePhys dataset."""
        return "train"

    def load(
        self,
        data_dir: Union[str, Path, None] = None,
        variant: Optional[str] = None,
        sample_size: Optional[int] = None,
        split: Optional[str] = None,
        **kwargs,
    ) -> PhysicalDataset:
        """
        Load SeePhys dataset.

        Args:
            data_dir: Path to the data directory containing SeePhys files
            variant: Dataset variant (kept as property but not used for loading)
            sample_size: Number of problems to load
            split: Dataset split to load. Defaults to "train".
            **kwargs: Additional loading parameters

        Returns:
            PhysicalDataset containing SeePhys problems
        """
        # Use defaults if not provided
        if split is None:
            split = self.get_default_split() or "train"
        
        # Validate split
        self.validate_split(split)

        # Resolve data directory with environment variable support
        data_dir = self.resolve_data_dir(data_dir, "SeePhys")
        self.logger.debug(f"Using data directory: {data_dir}")

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Load from split directory
        split_dir = data_dir / split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"SeePhys dataset not found in {data_dir}. "
                f"Expected '{split}/' directory. "
                f"Use the downloader to download the dataset."
            )

        return self._load_from_json_only(
            data_dir, split, sample_size, **kwargs
        )

    @property
    def field_mapping(self) -> Dict[str, str]:
        # Field mapping for SeePhys dataset
        # Fields: question, subject, image_paths, sig_figs, level, language, 
        # index, img_category, vision_relevance, caption, etc.
        return {
            "index": "problem_id",
            "subject": "domain",
        }

    def _load_from_json_only(
        self,
        data_dir: Path,
        split: str,
        sample_size: Optional[int],
        **_kwargs,
    ) -> PhysicalDataset:
        """Load from split directory."""
        split_dir = data_dir / split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Directory not found: {split_dir}. "
                f"Use the downloader to download the dataset."
            )

        # Load from split directory
        self.logger.debug(f"Loading from directory: {split_dir}")
        problems = self._load_from_json_dir(split_dir, data_dir)

        if not problems:
            raise RuntimeError(
                f"No problems loaded from: {data_dir}. "
                f"Check if files exist and are valid."
            )

        # Apply sample_size if specified
        if sample_size is not None and sample_size < len(problems):
            import random
            problems = random.sample(problems, sample_size)
            self.logger.info(f"Sampled {sample_size} problems from {len(problems)} total")

        # Create dataset info
        info = self.get_info()

        # Log final loading result
        self.logger.info(
            f"Successfully loaded {len(problems)} problems from SeePhys dataset"
        )

        return PhysicalDataset(problems, info, split=split)

    def _load_from_json_dir(
        self, split_dir: Path, data_dir: Path
    ) -> List:
        """Load problems from files in a directory."""
        problems = []

        json_files = sorted(split_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No files found in {split_dir}"
            )

        self.logger.info(f"Found {len(json_files)} files")

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    problem_data = json.load(f)

                metadata = self.initialize_metadata(problem_data)
                metadata = self._process_metadata(metadata)

                problem = self.create_physics_problem(
                    metadata=metadata,
                    data_dir=data_dir,
                )
                problems.append(problem)
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(
                    f"Failed to load {json_file}: {e}. Skipping."
                )
                continue

        return problems

    def _process_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process metadata to create standardized problem fields.
        
        Specifically handles:
        - Ensures image_paths are properly formatted
        """
        # Ensure image_paths is properly formatted (base loader now handles image_paths directly)
        if "image_paths" in metadata:
            image_paths = metadata["image_paths"]
            # Ensure it's a list
            if isinstance(image_paths, str):
                metadata["image_paths"] = [image_paths]
            elif not isinstance(image_paths, list):
                if image_paths is None:
                    metadata["image_paths"] = None
                else:
                    # Try to convert to list
                    metadata["image_paths"] = [str(image_paths)]
        
        return metadata

    def load_images_from_paths(
        self,
        image_paths: Union[str, List[str], None],
        data_dir: Optional[Union[str, Path]] = None,
    ) -> List[Any]:
        """
        Load images from image paths using the base loader's image loading functionality.
        
        This method wraps the base class's load_images_from_paths() method, providing
        a convenient way to load images from paths in the SeePhys dataset context.
        
        Args:
            image_paths: Single image path (str) or list of image paths.
                        Can be relative paths (resolved against data_dir) or absolute paths.
            data_dir: Root directory of the SeePhys dataset (for resolving relative paths).
                     If None, will try to resolve using default data directory.

        Returns:
            List of PIL Image objects. Empty list if no images are available or could be loaded.

        Raises:
            ImportError: If PIL/Pillow is not installed

        Example:
            >>> loader = SeePhysLoader()
            >>> # Load images from paths
            >>> images = loader.load_images_from_paths(
            ...     ["images/123_0.png", "images/123_1.jpg"],
            ...     data_dir="/path/to/seephys"
            ... )
            >>> for img in images:
            ...     print(f"Image size: {img.size}")
        """
        # If data_dir is not provided, try to resolve default
        if data_dir is None:
            data_dir = self.resolve_data_dir(None, "SeePhys")
        
        # Use the base class method
        return super().load_images_from_paths(image_paths, data_dir)
