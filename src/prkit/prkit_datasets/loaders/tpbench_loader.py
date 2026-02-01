"""
TPBench Dataset Loader

This module provides a loader for the TPBench dataset, which contains
physics problems requiring Python code implementation.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from prkit.prkit_core import PRKitLogger
from prkit.prkit_core.domain import PhysicsDomain
from prkit.prkit_core.domain import PhysicalDataset
from prkit.prkit_datasets.loaders.base_loader import BaseDatasetLoader


class TPBenchLoader(BaseDatasetLoader):
    """Loader for TPBench dataset."""

    def __init__(self):
        """Initialize the TPBench loader with a logger."""
        super().__init__()
        self.logger = PRKitLogger.get_logger(__name__)

    @property
    def name(self) -> str:
        """Return the name of the dataset."""
        return "tpbench"

    @property
    def description(self) -> str:
        """Return the description of the dataset."""
        return "Physics problems requiring Python code implementation"

    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            "name": self.name,
            "description": self.description,
            "domains": [
                "quantum_mechanics",
                "high_energy_theory",
                "statistical_mechanics",
                "classical_mechanics",
                "cosmology",
            ],
            "languages": ["en"],
            "variants": ["full"],
            "splits": ["public"],  # TPBench only supports public split
            "problem_types": ["OE"],
            "total_problems": "10",
            "difficulty": "1-5 scale",
            "answer_formats": ["symbolic"],
            "modalities": self.modalities,
        }

    @property
    def field_mapping(self) -> Dict[str, str]:
        """
        Define field mapping from TPBench fields to standard PRKit fields.

        Returns:
            Dictionary mapping TPBench field names to standard field names
        """
        return {
            "problem_id": "problem_id",  # Map "problem_id" to "problem_id"
            "problem": "question",  # Map "problem" to "question"
            "solution": "solution",  # Map "solution" to "solution"
            "difficulty_level": "difficulty",  # Map "difficulty_level" to "difficulty"
            "domain": "domain",  # Map "domain" to "domain"
            "answer": "answer",  # Map "answer" to "answer"
        }

    @property
    def DOMAIN_MAPPING(self) -> Dict[str, str]:
        """Mapping of domain abbreviations to full domain names."""
        return {
            "QM": PhysicsDomain.QUANTUM_MECHANICS,
            "HET": PhysicsDomain.HIGH_ENERGY_THEORY,
            "Stat Mech": PhysicsDomain.STATISTICAL_MECHANICS,
            "Classical Mechanics": PhysicsDomain.CLASSICAL_MECHANICS,
            "Cosmology": PhysicsDomain.COSMOLOGY,
        }

    def _process_metadata(self, metadata: Dict[str, Any]):
        """Process metadata to create standardized problem fields."""
        # Set answer type to symbolic
        metadata["answer_type"] = "symbolic"
        domain = metadata.get("domain")
        if domain:
            metadata["domain"] = self.DOMAIN_MAPPING.get(domain, PhysicsDomain.OTHER)

        return metadata

    def load(
        self,
        data_dir: Union[str, Path, None] = None,
        split: Optional[str] = None,
        variant: Optional[str] = None,
        sample_size: Optional[int] = None,
        per_domain: Optional[int] = None,
        language: str = "en",
        **kwargs,
    ) -> PhysicalDataset:
        """
        Load the TPBench dataset.

        Args:
            data_dir: Path to the TPBench dataset (defaults to ~/PHYSICAL_REASONING_DATASETS/TPBench)
            split: Dataset split to load. Defaults to "public" if available.
            variant: Dataset variant. Defaults to "full" if available.
            sample_size: Number of problems to sample (None for all)
            per_domain: Number of problems to sample per domain (None for all)
            language: Language to load ("en" only)

        Returns:
            PhysicalDataset instance

        Raises:
            ValueError: If unsupported split, variant, or language is requested
        """
        # Use defaults if not provided
        if split is None:
            split = self.get_default_split() or "public"
        if variant is None:
            variant = self.get_default_variant() or "full"
        
        # Validate variant and split
        self.validate_variant(variant)
        self.validate_split(split)

        if language != "en":
            raise ValueError("TPBench dataset only supports 'en' language")

        # Resolve data directory with environment variable support
        data_dir = self.resolve_data_dir(data_dir, "TPBench")
        self.logger.debug(f"Using data directory: {data_dir}")

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Try to load from parquet file first
        parquet_file = data_dir / "data" / "public-00000-of-00001.parquet"
        json_file = data_dir / "tpbench_samples.json"

        problems = []
        domain_counts = {}
        domain_problems = {}  # Track problems per domain for sampling

        # Try loading from parquet file
        if parquet_file.exists():
            try:
                self.logger.debug(f"Loading from parquet file: {parquet_file}")
                df = pd.read_parquet(parquet_file, engine="pyarrow")
                problems = self._load_from_dataframe(df, domain_problems, domain_counts)
            except Exception as e:
                self.logger.error(f"Error loading from parquet: {e}")
                self.logger.debug("Falling back to JSON file...")
                try:
                    problems = self._load_from_json(
                        json_file, domain_problems, domain_counts
                    )
                except Exception as json_e:
                    self.logger.error(f"Error loading from JSON fallback: {json_e}")
                    # If both parquet and JSON fail, raise the original parquet error
                    raise RuntimeError(
                        f"Failed to load TPBench dataset. Parquet error: {e}. JSON fallback error: {json_e}"
                    )
        else:
            # Fall back to JSON file
            self.logger.debug(f"Loading from JSON file: {json_file}")
            problems = self._load_from_json(json_file, domain_problems, domain_counts)

        if not problems:
            raise RuntimeError(
                f"No problems loaded from: {data_dir}. Check if data files exist and are valid."
            )

        # Apply per_domain sampling if requested
        if per_domain is not None:
            for domain_name, domain_problem_list in domain_problems.items():
                if len(domain_problem_list) > per_domain:
                    domain_problems[domain_name] = random.sample(
                        domain_problem_list, per_domain
                    )

        # Collect all problems and count by domain
        all_problems = []
        for domain_name, domain_problem_list in domain_problems.items():
            all_problems.extend(domain_problem_list)
            domain_counts[domain_name] = len(domain_problem_list)

        # Apply overall sample_size sampling if requested
        if sample_size is not None and sample_size < len(all_problems):
            all_problems = random.sample(all_problems, sample_size)

        # Create dataset info
        info = self.get_info()
        info["total_problems"] = len(all_problems)
        info["problems_by_domain"] = {
            str(domain): count for domain, count in domain_counts.items()
        }

        # Log final loading result
        self.logger.info(
            f"Successfully loaded {len(all_problems)} problems from TPBench dataset"
        )

        return PhysicalDataset(
            all_problems,
            info,
            split=split,
        )

    def _load_from_dataframe(
        self, df: pd.DataFrame, domain_problems: Dict, domain_counts: Dict
    ) -> List:
        """Load problems from pandas DataFrame."""
        problems = []

        if df.empty:
            raise ValueError("DataFrame is empty")

        for _, row in df.iterrows():
            try:
                # Convert row to dictionary
                data = row.to_dict()

                metadata = self.initialize_metadata(data)
                metadata = self._process_metadata(metadata)

                # Create the physics problem using base class method
                problem = self.create_physics_problem(
                    metadata=metadata,
                )

                domain = metadata.get("domain", "Unknown")
                if domain not in domain_problems:
                    domain_problems[domain] = []
                domain_problems[domain].append(problem)

                problems.append(problem)

            except Exception as e:
                self.logger.error(f"Error loading problem from DataFrame row: {e}")
                continue

        if not problems:
            raise RuntimeError("No problems could be loaded from DataFrame")

        return problems

    def _load_from_json(
        self, json_file: Path, domain_problems: Dict, domain_counts: Dict
    ) -> List:
        """Load problems from JSON file."""
        problems = []

        if not json_file.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file}")

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data_list = json.load(f)

            if not data_list:
                raise ValueError(f"JSON file is empty: {json_file}")

            for data in data_list:
                try:
                    metadata = self.initialize_metadata(data)
                    metadata = self._process_metadata(metadata)

                    # Create the physics problem using base class method
                    problem = self.create_physics_problem(
                        metadata=metadata,
                    )

                    domain = metadata.get("domain", "Unknown")
                    if domain not in domain_problems:
                        domain_problems[domain] = []
                    domain_problems[domain].append(problem)

                    problems.append(problem)

                except Exception as e:
                    self.logger.error(f"Error loading problem from JSON: {e}")
                    continue
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON file {json_file}: {e}")

        if not problems:
            raise RuntimeError("No problems could be loaded from JSON file")

        return problems

    def _get_domains(
        self, data_dir: Union[str, Path], language: str = "en"
    ) -> List[str]:
        """Get list of available domains in the dataset."""
        data_dir = Path(data_dir)
        if not data_dir.exists():
            self.logger.error(f"Data directory does not exist: {data_dir}")
            return []

        self.logger.info(f"Scanning directory: {data_dir}")

        # TPBench has predefined domains
        valid_domains = ["QM", "Cosmology", "Stat Mech", "Classical Mechanics", "HET"]

        return valid_domains
