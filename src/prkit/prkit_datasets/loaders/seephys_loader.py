"""
SeePhys dataset loader.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from prkit.prkit_core import PhysKitLogger
from prkit.prkit_core.models import PhysicalDataset

from .base_loader import BaseDatasetLoader


class SeePhysLoader(BaseDatasetLoader):
    """Loader for SeePhys dataset."""

    def __init__(self):
        """Initialize the SeePhys loader with a logger."""
        super().__init__()
        self.logger = PhysKitLogger.get_logger(__name__)

    @property
    def name(self) -> str:
        return "seephys"

    @property
    def description(self) -> str:
        return "SeePhys: A visual physics reasoning dataset with questions, images, and captions"

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "repository_url": "https://huggingface.co/datasets/SeePhys/SeePhys",
            "license": "Research use",
            "homepage": "https://seephys.github.io/",
            "paper_url": "https://openreview.net/pdf?id=APNWmytTCS",
            "languages": ["en", "zh"],
            "splits": ["train"],
            "problem_types": ["OE"],
            "total_problems": "2000",
            "source": "SeePhys dataset from HuggingFace",
        }

    def load(
        self,
        data_dir: Union[str, Path, None] = None,
        variant: Optional[str] = None,
        sample_size: Optional[int] = None,
        split: str = "train",
        **kwargs,
    ) -> PhysicalDataset:
        """
        Load SeePhys dataset from parquet or JSON files.

        Args:
            data_dir: Path to the data directory containing SeePhys files
            variant: Dataset variant (deprecated, kept for backward compatibility)
            sample_size: Number of problems to load
            split: Dataset split to load - "train" (default)
            **kwargs: Additional loading parameters

        Returns:
            PhysicalDataset containing SeePhys problems
        """
        # Resolve data directory with environment variable support
        data_dir = self.resolve_data_dir(data_dir, "seephys")
        self.logger.debug(f"Using data directory: {data_dir}")

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Try loading from new format (parquet/JSON) first
        parquet_file = data_dir / f"{split}.parquet"
        split_dir = data_dir / split

        # Check if new format exists
        if parquet_file.exists() or split_dir.exists():
            return self._load_from_parquet_or_json(
                data_dir, split, sample_size, **kwargs
            )

        # Fall back to old CSV format for backward compatibility
        if variant is not None:
            self.logger.warning(
                "Variant parameter is deprecated. Loading from legacy CSV format."
            )
            return self._load_csv_format(
                data_dir, variant, split, sample_size, **kwargs
            )

        # If no variant specified and new format doesn't exist, try default
        raise FileNotFoundError(
            f"SeePhys dataset not found in {data_dir}. "
            f"Expected either '{split}.parquet' or '{split}/' directory, "
            f"or legacy CSV files. Use the downloader to download the dataset."
        )

    @property
    def field_mapping(self) -> Dict[str, str]:
        # Field mapping for SeePhys dataset
        # Fields: question, subject, image_path, sig_figs, level, language, 
        # index, img_category, vision_relevance, caption, etc.
        return {
            "index": "problem_id",
            "subject": "domain",
        }

    def _load_from_parquet_or_json(
        self,
        data_dir: Path,
        split: str,
        sample_size: Optional[int],
        **_kwargs,
    ) -> PhysicalDataset:
        """Load from parquet or JSON format."""
        parquet_file = data_dir / f"{split}.parquet"
        split_dir = data_dir / split

        problems = []

        # Try loading from parquet file first
        if parquet_file.exists():
            try:
                self.logger.debug("Loading from parquet file: %s", parquet_file)
                df = pd.read_parquet(parquet_file, engine="pyarrow")
                problems = self._load_from_dataframe(df, data_dir)
            except (ValueError, OSError, ImportError) as e:
                self.logger.error(f"Error loading from parquet: {e}")
                self.logger.debug("Falling back to JSON files...")
                try:
                    problems = self._load_from_json_dir(split_dir, data_dir)
                except (json.JSONDecodeError, IOError, ValueError) as json_e:
                    self.logger.error("Error loading from JSON fallback: %s", json_e)
                    raise RuntimeError(
                        f"Failed to load SeePhys dataset. "
                        f"Parquet error: {e}. JSON fallback error: {json_e}"
                    ) from json_e
        elif split_dir.exists():
            # Load from JSON directory
            self.logger.debug(f"Loading from JSON directory: {split_dir}")
            problems = self._load_from_json_dir(split_dir, data_dir)
        else:
            raise FileNotFoundError(
                f"Neither parquet file ({parquet_file}) nor JSON directory "
                f"({split_dir}) found for split '{split}'"
            )

        if not problems:
            raise RuntimeError(
                f"No problems loaded from: {data_dir}. "
                f"Check if data files exist and are valid."
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

    def _load_from_dataframe(
        self, df: pd.DataFrame, data_dir: Path
    ) -> List:
        """Load problems from pandas DataFrame."""
        problems = []

        for _, row in df.iterrows():
            problem_data = row.to_dict()
            metadata = self.initialize_metadata(problem_data)
            metadata = self._process_metadata(metadata)

            problem = self.create_physics_problem(
                metadata=metadata,
                data_dir=data_dir,
            )
            problems.append(problem)

        return problems

    def _load_from_json_dir(
        self, split_dir: Path, data_dir: Path
    ) -> List:
        """Load problems from JSON files in a directory."""
        problems = []

        json_files = sorted(split_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No JSON files found in {split_dir}"
            )

        self.logger.info(f"Found {len(json_files)} JSON files")

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

    def _load_csv_format(
        self,
        data_dir: Path,
        variant: str,
        split: str,
        sample_size: Optional[int],
        **_kwargs,
    ) -> PhysicalDataset:
        """Load from legacy CSV format (for backward compatibility)."""
        if variant == "full":
            seephys_file = data_dir / "seephys_train.csv"
        elif variant == "mini":
            seephys_file = data_dir / "seephys_train_mini.csv"
        else:
            raise ValueError(f"Unknown variant: {variant}. Choose 'full' or 'mini'")

        if not seephys_file.exists():
            raise FileNotFoundError(f"SeePhys CSV file not found: {seephys_file}")

        df = pd.read_csv(seephys_file)

        # Convert to unified format
        problems = []

        for _, row in df.iterrows():
            problem_data = row.to_dict()
            metadata = self.initialize_metadata(problem_data)
            metadata = self._process_metadata(metadata)

            problem = self.create_physics_problem(
                metadata=metadata,
                data_dir=data_dir,
            )
            problems.append(problem)

        # Apply sample_size if specified
        if sample_size is not None and sample_size < len(problems):
            import random
            problems = random.sample(problems, sample_size)

        # Create dataset info
        info = self.get_info()

        # Log final loading result
        self.logger.info(
            f"Successfully loaded {len(problems)} problems from SeePhys dataset (CSV format)"
        )

        return PhysicalDataset(problems, info, split=split)

    def _process_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process metadata to create standardized problem fields."""
        return metadata
