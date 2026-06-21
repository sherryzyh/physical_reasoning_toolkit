"""
PhyX Dataset Loader

This module provides a loader for the PhyX dataset, which contains
physics reasoning problems with visual context.

For citation information, see prkit.datasets.citations.
"""

import json
import random
from pathlib import Path
from typing import Any

from prkit.core import PRKitLogger
from prkit.core.domain import PhysicsDataset, PhysicsProblem
from prkit.core.domain.physics_domain import PhysicsDomain
from prkit.datasets.license_registry import get_license

from .base_loader import BaseDatasetLoader


class PhyXLoader(BaseDatasetLoader):
    """Loader for PhyX dataset."""

    def __init__(self) -> None:
        """Initialize the PhyX loader with a logger."""
        super().__init__()
        self.logger = PRKitLogger.get_logger(__name__)

    @property
    def modalities(self) -> list[str]:
        """PhyX supports both text and image modalities."""
        return ["text", "image"]

    @property
    def name(self) -> str:
        return "phyx"

    @property
    def description(self) -> str:
        return "PhyX: A large-scale benchmark for physics-grounded reasoning in visual scenarios"

    def get_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "citation": "See prkit.datasets.citations for BibTeX citation",
            "paper_url": "https://arxiv.org/pdf/2505.15929v2",
            "homepage": "https://phyx-bench.github.io/",
            "repository_url": "https://huggingface.co/datasets/Cloudriver/PhyX",
            "license": get_license(self.name).to_info_dict(),
            "license_spdx": get_license(self.name).spdx,
            "domains": [
                "mechanics",
                "electromagnetism",
                "thermodynamics",
                "wave_acoustics",
                "optics",
                "modern_physics",
            ],
            "languages": ["en"],
            "variants": ["test_mini"],
            "splits": ["test_mini"],
            "problem_types": ["MC", "OE"],
            "total_problems": "1000 (test_mini)",
            "modalities": self.modalities,
        }

    @property
    def field_mapping(self) -> dict[str, str]:
        return {
            "id": "problem_id",
            "question": "question",
            "answer": "answer",
            "domain": "domain",
            "image": "image_paths",
            "image_path": "image_paths",
            "options": "options",
            "correct_answer": "correct_option",
        }

    @property
    def DOMAIN_MAPPING(self) -> dict[str, PhysicsDomain]:
        """Mapping of domain names to PhysicsDomain enum values."""
        # Note: Wave/Acoustics maps to OTHER since WAVE_ACOUSTICS is not yet in PhysicsDomain
        # This may need to be updated if WAVE_ACOUSTICS is added to the enum
        return {
            "Mechanics": PhysicsDomain.MECHANICS,
            "mechanics": PhysicsDomain.MECHANICS,
            "Electromagnetism": PhysicsDomain.ELECTRICITY,
            "electromagnetism": PhysicsDomain.ELECTRICITY,
            "electricity": PhysicsDomain.ELECTRICITY,
            "Thermodynamics": PhysicsDomain.THERMODYNAMICS,
            "thermodynamics": PhysicsDomain.THERMODYNAMICS,
            "Wave/Acoustics": PhysicsDomain.OTHER,  # TODO: Use WAVE_ACOUSTICS when added to PhysicsDomain
            "wave_acoustics": PhysicsDomain.OTHER,
            "Waves & Acoustics": PhysicsDomain.OTHER,
            "Optics": PhysicsDomain.OPTICS,
            "optics": PhysicsDomain.OPTICS,
            "Modern Physics": PhysicsDomain.MODERN_PHYSICS,
            "modern_physics": PhysicsDomain.MODERN_PHYSICS,
        }

    def _process_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Process metadata to create standardized problem fields."""

        # Combine question and question_description
        question = metadata.get("question", "")
        question_description = metadata.get("question_description", "")
        if question_description and question_description.strip():
            question = question_description.strip() + "\n\n" + question
        metadata["question"] = question

        # Map domain
        self._map_domain(metadata)

        # Determine problem type
        options = metadata.get("options", [])
        if options and len(options) > 1:
            metadata["problem_type"] = "MC"
            metadata["options"] = options
        else:
            metadata["problem_type"] = "OE"

        # Set language
        metadata["language"] = "en"

        # Handle image paths
        image_paths = metadata.get("image_paths")
        if image_paths:
            # Ensure it's a list
            if isinstance(image_paths, str):
                metadata["image_paths"] = [image_paths]
            elif not isinstance(image_paths, list):
                metadata["image_paths"] = [str(image_paths)]
        else:
            # Try alternative field names
            image = metadata.get("image")
            if image:
                if isinstance(image, str):
                    metadata["image_paths"] = [image]
                elif isinstance(image, list):
                    metadata["image_paths"] = image
                else:
                    metadata["image_paths"] = [str(image)]

        return metadata

    def load(
        self,
        data_dir: str | Path | None = None,
        variant: str | None = None,
        sample_size: int | None = None,
        split: str | None = None,
        **kwargs: Any,
    ) -> PhysicsDataset:
        """
        Load PhyX dataset.

        Args:
            data_dir: Path to the PhyX dataset (defaults to ~/PHYSICAL_REASONING_DATASETS/PhyX)
            variant: Dataset variant. Defaults to "test_mini" if available.
            split: Dataset split. Defaults to "test_mini" if available.
            sample_size: Number of problems to load (None = all)
            **kwargs: Additional loading parameters (unused, for compatibility)

        Returns:
            PhysicsDataset containing PhyX problems
        """
        # Use defaults if not provided
        if variant is None:
            variant = self.get_default_variant() or "test_mini"
        if split is None:
            split = self.get_default_split() or "test_mini"

        # Validate variant and split
        self.validate_variant(variant)
        self.validate_split(split)

        # Resolve data directory with environment variable support
        data_dir = self.resolve_data_dir(data_dir, "PhyX")

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Determine which file to use based on split/variant
        json_file = data_dir / f"PhyX-{split}.json"
        if not json_file.exists():
            # Try variant name
            json_file = data_dir / f"PhyX-{variant}.json"
            if not json_file.exists():
                # Try to find any PhyX JSON file
                json_files = list(data_dir.glob("PhyX-*.json"))
                if not json_files:
                    raise FileNotFoundError(
                        f"PhyX JSON file not found in {data_dir}. "
                        f"Expected: PhyX-{split}.json or PhyX-{variant}.json"
                    )
                json_file = json_files[0]

        # Load the JSON data
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        # Convert to unified format
        problems: list[PhysicsProblem] = []
        if sample_size:
            data = random.sample(data, min(sample_size, len(data)))

        for idx, problem_data in enumerate(data):
            try:
                metadata = self.initialize_metadata(problem_data)

                metadata = self._process_metadata(metadata)

                # Ensure problem_id exists
                if "problem_id" not in metadata or not metadata["problem_id"]:
                    metadata["problem_id"] = f"phyx_{split}_{idx}"

                problem = self.create_physics_problem(
                    metadata=metadata,
                    data_dir=data_dir,
                )
                problems.append(problem)
            except Exception as e:
                # Log warning but continue processing
                self.logger.warning(
                    f"Failed to process problem at index {idx}: {e}. Skipping..."
                )
                continue

        # Create dataset info
        info = self.get_info()
        info["total_problems"] = len(problems)
        info["variant"] = variant
        info["split"] = split

        return PhysicsDataset(
            problems,
            info,
            split=split,
        )
