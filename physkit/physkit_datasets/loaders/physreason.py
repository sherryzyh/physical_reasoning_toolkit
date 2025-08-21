"""
PhysReason Dataset Loader

This module provides a loader for the PhysReason dataset, which contains
physics reasoning problems with detailed step-by-step solutions.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Union, Optional

from physkit_core.models import PhysicsDomain, PhysicalDataset
from physkit_datasets.loaders.base_loader import BaseDatasetLoader
from physkit_core import PhysKitLogger


# TODO: add support for handling multiple sub-questions
class PhysReasonLoader(BaseDatasetLoader):
    """Loader for PhysReason dataset with support for full and mini variants."""

    def __init__(self):
        """Initialize the PhysReason loader with a logger."""
        super().__init__()
        self.logger = PhysKitLogger.get_logger(__name__)

    @property
    def name(self) -> str:
        return "physreason"

    @property
    def description(self) -> str:
        return "PhysReason: Physics reasoning problems with step-by-step solutions"

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "repository_url": "https://github.com/AI4Phys/PhysReason",
            "license": "Research use",
            "homepage": "https://github.com/AI4Phys/PhysReason",
            "languages": ["en"],
            "variants": ["full", "mini"],
            "splits": ["test"],
            "problem_types": ["OE"],
            "total_problems": "192",
            "difficulty": ["easy", "medium", "hard"],
            "source": "PhysReason dataset with full and mini variants"
        }

    @property
    def field_mapping(self) -> Dict[str, str]:
        """No field mapping needed for PhysReason - use original field names."""
        return {}

    def load(
        self,
        data_dir: Union[str, Path, None] = None,
        variant: str = "full",
        sample_size: Optional[int] = None,
        split: str = "test",
        **kwargs
    ) -> PhysicalDataset:
        """
        Load PhysReason dataset from the specified directory.
        
        Args:
            data_dir: Path to the data directory containing PhysReason files
            variant: Dataset variant - "full" (default) or "mini"
            sample_size: Number of problems to load
            split: Dataset split to load - "test" (only)
            **kwargs: Additional loading parameters (ignored for compatibility)
        
        Returns:
            PhysicalDataset containing PhysReason problems
        """
        # Resolve data directory with environment variable support
        data_dir = self.resolve_data_dir(data_dir, "PhysReason")
        self.logger.info(f"Using data directory: {data_dir}")

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        if split != "test":
            raise ValueError(f"PhysReason dataset only has 'test' split. Got: {split}")

        # Determine variant directory
        if variant == "full":
            variant_dir = data_dir / "PhysReason_full"
        elif variant == "mini":
            variant_dir = data_dir / "PhysReason-mini"
        else:
            raise ValueError(f"Unknown variant: {variant}. Choose 'full' or 'mini'")

        if not variant_dir.exists():
            raise FileNotFoundError(f"Variant directory not found: {variant_dir}")

        self.logger.info(f"Loading {variant} variant from: {variant_dir}")

        # Load all problems from the variant directory
        all_problems = []
        
        # Get all problem directories
        problem_dirs = [d for d in variant_dir.iterdir() if d.is_dir()]
        self.logger.info(f"Found {len(problem_dirs)} problem directories")

        # Process each problem directory
        for problem_dir in problem_dirs:
            try:
                problem_data = self._load_problem_from_directory(problem_dir)
                if problem_data:
                    all_problems.append(problem_data)
            except Exception as e:
                self.logger.warning(f"Failed to load problem from {problem_dir.name}: {e}")
                continue

        # Apply sampling if requested
        if sample_size and len(all_problems) > sample_size:
            import random
            random.seed(42)  # For reproducible sampling
            all_problems = random.sample(all_problems, sample_size)
            self.logger.info(f"Sampled {sample_size} problems from {len(all_problems)} total")

        # Create PhysicsProblem objects
        physics_problems = []
        for problem_data in all_problems:
            try:
                metadata = self.initialize_metadata(problem_data)
                metadata = self._process_metadata(metadata)
                physics_problem = self.create_physics_problem(metadata)
                physics_problems.append(physics_problem)
            except Exception as e:
                self.logger.warning(f"Could not create problem from {problem_data.get('problem_id', 'unknown')}: {e}")
                continue

        self.logger.info(f"Successfully created {len(physics_problems)} PhysicsProblem objects")

        # Create PhysicalDataset
        dataset = PhysicalDataset(
            problems=physics_problems,
            info={
                "name": self.name,
                "description": self.description,
                "variant": variant,
                "total_problems": len(physics_problems),
                "source_directory": str(variant_dir)
            }
        )

        return dataset

    def _load_problem_from_directory(self, problem_dir: Path) -> Optional[Dict[str, Any]]:
        """Load a single problem from its directory."""
        problem_file = problem_dir / "problem.json"
        
        if not problem_file.exists():
            self.logger.warning(f"Problem file not found: {problem_file}")
            return None

        try:
            with open(problem_file, 'r', encoding='utf-8') as f:
                problem_data = json.load(f)
            
            # Add problem_id from directory name
            problem_data['problem_id'] = problem_dir.name
            
            # Check for images
            images_dir = problem_dir / "images"
            if images_dir.exists():
                image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
                problem_data['image_files'] = [str(img.relative_to(problem_dir)) for img in image_files]
            else:
                problem_data['image_files'] = []

            return problem_data

        except Exception as e:
            self.logger.error(f"Error loading problem from {problem_file}: {e}")
            return None

    def _process_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process metadata to create standardized problem fields."""
        # Extract question text from the complex structure
        question_structure = metadata.get('question_structure', {})
        
        # Combine context and sub-questions into a single question
        context = question_structure.get('context', '')
        sub_questions = []
        
        for i in range(1, 10):  # Support up to 9 sub-questions
            sub_q_key = f'sub_question_{i}'
            if sub_q_key in question_structure:
                sub_questions.append(question_structure[sub_q_key])
            else:
                break
        
        # Create combined question text
        if context and sub_questions:
            question_text = context + "\n\n" + "\n".join(sub_questions)
        elif context:
            question_text = context
        elif sub_questions:
            question_text = "\n".join(sub_questions)
        else:
            question_text = "Physics problem"
        
        metadata['question'] = question_text
        
        # Set problem type
        metadata['problem_type'] = 'OE'
        
        # Set language
        metadata['language'] = 'en'
        
        return metadata
