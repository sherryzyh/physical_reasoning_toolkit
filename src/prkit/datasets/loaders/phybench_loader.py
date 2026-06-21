"""
PHYBench Dataset Loader

This module provides a loader for the PHYBench dataset, which contains
physics reasoning problems across various domains.

For citation information, see prkit.datasets.citations.
"""

import json
import random
from pathlib import Path
from typing import Any

from prkit.core.domain import PhysicalDataset, PhysicsProblem
from prkit.core.domain.physics_domain import PhysicsDomain
from prkit.datasets.license_registry import get_license

from .base_loader import BaseDatasetLoader


class PHYBenchLoader(BaseDatasetLoader):
    """Loader for PHYBench dataset."""

    @property
    def name(self) -> str:
        return "phybench"

    @property
    def description(self) -> str:
        return "PHYBench: A comprehensive physics benchmark dataset with problems across various physics domains"

    def get_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "citation": "See prkit.datasets.citations for BibTeX citation",
            "paper_url": "https://arxiv.org/pdf/2504.16074",
            "homepage": "https://www.phybench.cn/",
            "repository_url": "https://huggingface.co/datasets/Eureka-Lab/PHYBench",
            "license": get_license(self.name).to_info_dict(),
            "license_spdx": get_license(self.name).spdx,
            "domains": [
                "mechanics",
                "electricity",
                "thermodynamics",
                "modern_physics",
                "optics",
                "advanced_physics",
            ],
            "languages": ["en"],
            "variants": ["full", "fullques", "onlyques"],
            "splits": ["train"],
            "problem_types": ["OE"],
            "total_problems": "500",
            "modalities": self.modalities,
        }

    @property
    def field_mapping(self) -> dict[str, str]:
        return {
            "id": "problem_id",
            "tag": "domain",
            "content": "question",
            "answer": "answer",
            "solution": "solution",
        }

    @property
    def DOMAIN_MAPPING(self) -> dict[str, PhysicsDomain]:
        """Mapping of domain abbreviations to full domain names."""
        return {
            "MECHANICS": PhysicsDomain.MECHANICS,
            "ELECTRICITY": PhysicsDomain.ELECTRICITY,
            "THERMODYNAMICS": PhysicsDomain.THERMODYNAMICS,
            "MODERN": PhysicsDomain.MODERN_PHYSICS,
            "OPTICS": PhysicsDomain.OPTICS,
            "ADVANCED": PhysicsDomain.ADVANCED_PHYSICS,
        }

    def _process_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Process metadata to create standardized problem fields."""
        self._map_domain(metadata)
        return metadata

    def load(
        self,
        data_dir: str | Path | None = None,
        variant: str | None = None,
        sample_size: int | None = None,
        split: str | None = None,
        **kwargs: Any,
    ) -> PhysicalDataset:
        """
        Load PHYBench dataset.

        Args:
            data_dir: Path to the PHYBench dataset (defaults to ~/PHYSICAL_REASONING_DATASETS/PHYBench)
            variant: Dataset variant ("full", "fullques", or "onlyques"). Defaults to "full" if available.
            split: Dataset split ("train" is the only available split). Defaults to "train".
            **kwargs: Additional loading parameters (unused, for compatibility)

        Returns:
            PhysicalDataset containing PHYBench problems
        """
        # Use defaults if not provided
        if variant is None:
            variant = self.get_default_variant() or "full"
        if split is None:
            split = self.get_default_split() or "train"

        # Validate variant and split
        self.validate_variant(variant)
        self.validate_split(split)

        # Resolve data directory with environment variable support
        data_dir = self.resolve_data_dir(data_dir, "PHYBench")

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Determine which file to use based on variant
        if variant == "full":
            json_file = data_dir / "PHYBench-questions_v1.json"
        elif variant == "fullques":
            # Try both possible locations for fullques variant
            json_file = data_dir / "PHYBench-fullques_v1.json"
        elif variant == "onlyques":
            json_file = data_dir / "PHYBench-onlyques_v1.json"
        else:
            raise ValueError(
                f"Unknown variant: {variant}. Choose 'full' or 'fullques' or 'onlyques'"
            )

        if not json_file.exists():
            raise FileNotFoundError(f"PHYBench file not found: {json_file}")

        # Load the JSON data
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        # Convert to unified format
        problems: list[PhysicsProblem] = []
        if sample_size:
            data = random.sample(data, sample_size)

        for _, problem_data in enumerate(data):
            metadata = self.initialize_metadata(problem_data)
            metadata = self._process_metadata(metadata)

            problem = self.create_physics_problem(
                metadata=metadata,
            )
            problems.append(problem)

        # Create dataset info
        info = self.get_info()
        info["total_problems"] = len(problems)

        return PhysicalDataset(
            problems,
            info,
            split=split,
        )
