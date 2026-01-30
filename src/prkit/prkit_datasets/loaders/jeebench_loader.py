"""
JEEBench Dataset Loader

This loader supports the JEEBench dataset which contains JEE (Joint Entrance Examination)
Advanced problems across Physics, Chemistry, and Mathematics. The dataset includes:

- Subjects: Physics (phy), Chemistry (chem), Mathematics (math)
- Question Types: MCQ, MCQ(multiple), Integer, Numeric
- Format: JSON with LaTeX-formatted questions and answers
- Source: JEE Advanced exam papers from various years

The loader automatically determines problem types and maps JEEBench fields to standard PRKit fields.

Citation:
@inproceedings{arora-etal-2023-llms,
    title = "Have {LLM}s Advanced Enough? A Challenging Problem Solving Benchmark For Large Language Models",
    author = "Arora, Daman  and
      Singh, Himanshu  and
      {Mausam}",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.468",
    doi = "10.18653/v1/2023.emnlp-main.468",
    pages = "7527--7543",
}
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from prkit.prkit_core import PhysKitLogger
from prkit.prkit_core.models import PhysicalDataset

from .base_loader import BaseDatasetLoader


class JEEBenchLoader(BaseDatasetLoader):
    """Loader for JEEBench dataset with support for multiple subjects and question types."""

    def __init__(self):
        """Initialize the JEEBench loader with a logger."""
        super().__init__()
        self.logger = PhysKitLogger.get_logger(__name__)

    @property
    def name(self) -> str:
        return "jeebench"

    @property
    def description(self) -> str:
        return "JEEBench: JEE Advanced examination problems across Physics, Chemistry, and Mathematics"

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "subjects": ["phy", "chem", "math"],
            "languages": ["en"],
            "variants": ["full"],
            "splits": ["test"],
            "problem_types": ["MC", "OE"],
            "total_problems": "~4000+",
            "difficulty": "JEE Advanced level",
            "source": "JEE Advanced examination papers",
            "citation": "JEEBench dataset for JEE Advanced preparation",
            "license": "Research use",
            "repository": "Local dataset under data/JEEBench/",
        }

    @property
    def field_mapping(self) -> Dict[str, str]:
        """
        Define field mapping from JEEBench fields to standard PRKit fields.

        JEEBench fields:
        - index: problem identifier
        - subject: physics/chemistry/mathematics
        - type: question type (MCQ, MCQ(multiple), Integer, Numeric)
        - question: LaTeX-formatted question text
        - gold: correct answer
        - description: exam paper description
        """
        return {"question": "question", "gold": "answer"}

    def load(
        self,
        data_dir: Union[str, Path, None] = None,
        variant: str = "full",
        split: str = "test",
        sample_size: Optional[int] = None,
        **kwargs,
    ) -> PhysicalDataset:
        """
        Load the JEEBench dataset.

        Args:
            data_dir: Path to the JEEBench dataset (defaults to ~/PHYSICAL_REASONING_DATASETS/JEEBench)
            variant: Dataset variant to load (only "full" is supported)
            split: Dataset split to load (only "test" is supported)
            sample_size: Number of problems to sample (None for all)

            **kwargs: Additional loading parameters

        Returns:
            PhysicalDataset instance

        Raises:
            ValueError: If unsupported split is requested
            FileNotFoundError: If dataset file is not found
        """
        if split != "test":
            raise ValueError("JEEBench dataset only supports 'test' split")

        # Resolve data directory with environment variable support
        data_dir = self.resolve_data_dir(data_dir, "JEEBench")
        self.logger.debug(f"Using data directory: {data_dir}")

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        if variant != "full":
            raise ValueError("JEEBench dataset only supports 'full' variant")

        # Load the main dataset file
        dataset_file = data_dir / "dataset.json"
        if not dataset_file.exists():
            raise FileNotFoundError(f"JEEBench dataset file not found: {dataset_file}")

        # Load and parse the JSON data
        try:
            with open(dataset_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in JEEBench dataset: {e}")
        except Exception as e:
            raise ValueError(f"Error loading JEEBench dataset: {e}")

        # Apply filters
        filtered_data = self._apply_filters(data, subject="phy")

        # Convert to PhysicsProblem instances
        problems = []
        for problem_data in filtered_data:
            try:
                metadata = self.initialize_metadata(problem_data)
                metadata = self._process_metadata(metadata)

                problem = self.create_physics_problem(metadata=metadata)
                problems.append(problem)
            except Exception as e:
                self.logger.warning(
                    f"Skipping problem {problem_data.get('index', 'unknown')}: {e}"
                )
                continue

        # Apply sampling if requested
        if sample_size is not None and sample_size < len(problems):
            problems = random.sample(problems, sample_size)

        # Create dataset info
        info = self.get_info()
        info["total_problems"] = len(problems)

        # Log final loading result
        self.logger.info(
            f"Successfully loaded {len(problems)} problems from JEEBench dataset"
        )

        return PhysicalDataset(
            problems,
            info,
            split=split,
        )

    def _apply_filters(
        self, data: List[Dict[str, Any]], subject: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Apply subject and question type filters to the data."""
        filtered_data = data

        # Filter by subject
        if subject:
            if subject not in ["phy", "chem", "math"]:
                raise ValueError(
                    f"Invalid subject: {subject}. Must be one of: phy, chem, math"
                )
            filtered_data = [p for p in filtered_data if p.get("subject") == subject]
            self.logger.debug(
                f"Filtered to {len(filtered_data)} problems in subject: {subject}"
            )

        return filtered_data

    def _process_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        # "type": "problem_type",

        """Process metadata to create standardized problem fields."""

        # Determine problem type based on JEEBench question type
        problem_type = metadata.get("type", "")
        if problem_type == "MCQ":
            metadata["problem_type"] = "MC"
            # Extract options from question text for MCQ problems
            question_text = metadata.get("question", "")
            options = self._extract_options_from_question(question_text)
            if options:
                metadata["options"] = options
        elif problem_type == "MCQ(multiple)":
            metadata["problem_type"] = "MultipleMC"
            # Extract options from question text for MCQ(multiple) problems
            question_text = metadata.get("question", "")
            options = self._extract_options_from_question(question_text)
            if options:
                metadata["options"] = options
        elif problem_type == "Integer" or problem_type == "Numeric":
            metadata["problem_type"] = "OE"
        metadata.pop("type", None)

        # Construct the problem_id
        metadata["problem_id"] = (
            f"{metadata['subject']}_{metadata['index']}_{metadata['description']}"
        )
        metadata.pop("description", None)
        metadata.pop("index", None)
        metadata.pop("subject", None)

        # Set language to English (JEEBench is primarily in English)
        metadata["language"] = "en"

        # Set answer type based on question type
        if metadata["problem_type"] in ["Integer", "Numeric"]:
            metadata["answer_type"] = "numerical"
        elif metadata["problem_type"] in ["MC", "MultipleMC"]:
            metadata["answer_type"] = "option"  # Multiple choice answers
        else:
            metadata["answer_type"] = "textual"  # Default fallback

        metadata.pop("subject", None)

        return metadata

    def _extract_options_from_question(self, question: str) -> List[str]:
        """
        Extract multiple choice options from the question text.

        JEEBench MCQ questions have options in the format:
        (A) option_text
        (B) option_text
        etc.
        """
        options = []

        # Look for patterns like (A), (B), (C), (D) or (1), (2), (3), (4)
        import re

        # Pattern for (A) option_text or (1) option_text
        option_pattern = r"\(([A-D1-4])\)\s*([^()]+?)(?=\([A-D1-4]\)|$)"
        matches = re.findall(option_pattern, question, re.DOTALL)

        for match in matches:
            option_letter = match[0]
            option_text = match[1].strip()
            if option_text:
                options.append(f"{option_letter}: {option_text}")

        # If no options found with regex, try a simpler approach
        if not options:
            # Look for lines starting with (A), (B), etc.
            lines = question.split("\n")
            for line in lines:
                line = line.strip()
                if re.match(r"^\([A-D1-4]\)", line):
                    options.append(line)

        return options

    def get_subject_statistics(
        self, data_dir: Union[str, Path, None] = None
    ) -> Dict[str, Any]:
        """Get statistics about the JEEBench dataset by subject and question type."""
        try:
            # Load the dataset without filters
            dataset = self.load(data_dir=data_dir, split="test")

            stats = {
                "total_problems": len(dataset),
                "by_subject": {},
                "by_subject_and_type": {},
            }

            for problem in dataset:
                subject = problem.additional_fields.get("subject", "unknown")
                qtype = problem.additional_fields.get("type", "unknown")

                # Count by subject
                if subject not in stats["by_subject"]:
                    stats["by_subject"][subject] = 0
                stats["by_subject"][subject] += 1

                # Count by subject and type combination
                key = f"{subject}_{qtype}"
                if key not in stats["by_subject_and_type"]:
                    stats["by_subject_and_type"][key] = 0
                stats["by_subject_and_type"][key] += 1

            return stats

        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}

    def list_available_subjects(
        self, data_dir: Union[str, Path, None] = None
    ) -> List[str]:
        """List all available subjects in the JEEBench dataset."""
        try:
            dataset = self.load(data_dir=data_dir, split="test")
            subjects = set()
            for problem in dataset:
                subject = problem.additional_fields.get("subject", "unknown")
                if subject != "unknown":
                    subjects.add(subject)
            return sorted(list(subjects))
        except Exception as e:
            self.logger.error(f"Error listing subjects: {e}")
            return []
