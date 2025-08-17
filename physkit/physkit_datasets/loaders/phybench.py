"""
PHYBench dataset loader.
"""

import json
from pathlib import Path
import random
from typing import Dict, Any, Optional, Union, List

from .base_loader import BaseDatasetLoader
from physkit.models import PhysicalDataset


class PHYBenchLoader(BaseDatasetLoader):
    """Loader for PHYBench dataset."""
    
    @property
    def name(self) -> str:
        return "phybench"
    
    @property
    def description(self) -> str:
        return "PHYBench: A comprehensive physics benchmark dataset with problems across various physics domains"
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": "PHYBench: A comprehensive physics benchmark dataset with problems across various physics domains",
            "citation": "arXiv:2504.16074",
            "paper_url": "https://arxiv.org/pdf/2504.16074",
            "license": "Research use",
            "domains": [],
            "languages": ["en"],
            "variants": ["full", "fullques"],
            "splits": ["test"],
            "problem_types": ["OE"],
            "total_problems": "500",
            "source": "PHYBench-fullques_v1_dict.json"
        }
    
    @property
    def field_mapping(self) -> Dict[str, str]:
        return {
            "id": "problem_id",
            "tag": "domain",
            "content": "question",
            "answer": "answer",
            "solution": "solution"
        }
    
    def load(
        self,
        data_dir: Union[str, Path, None] = None,
        variant: str = "full",
        sample_size: Optional[int] = None,
        split: str = "test",
        **kwargs
    ) -> PhysicalDataset:
        """
        Load PHYBench dataset.
        
        Args:
            data_dir: Path to the PHYBench dataset (defaults to ~/data/PHYBench)
            variant: Dataset variant ("full" or "fullques")
            **kwargs: Additional loading parameters (unused, for compatibility)
        
        Returns:
            PhysicalDataset containing PHYBench problems
        """
        if data_dir is None:
            data_dir = Path.home() / "data" / "PHYBench"
        else:
            data_dir = Path(data_dir)
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        
        if split != "test":
            raise ValueError("PHYBench dataset only supports 'test' split")
        
        # Determine which file to use based on variant
        if variant == "full":
            json_file = data_dir / "PHYBench-questions_v1.json"
        elif variant == "fullques":
            # Try both possible locations for fullques variant
            json_file = data_dir / "PHYBench-fullques_v1.json"
        else:
            raise ValueError(f"Unknown variant: {variant}. Choose 'full' or 'fullques'")
        
        if not json_file.exists():
            raise FileNotFoundError(f"PHYBench file not found: {json_file}")
        
        # Load the JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to unified format
        problems = []
        if sample_size:
            data = random.sample(data, sample_size)
            
        
        for _, problem_data in enumerate(data):
            metadata = self.intiailize_metadata(problem_data)
            
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