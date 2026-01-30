"""
PHYBench dataset loader.
"""

import json
from pathlib import Path
import random
from typing import Dict, Any, Optional, Union, List


from .base_loader import BaseDatasetLoader
from physkit.physkit_core.definitions.physics_domain import PhysicsDomain
from physkit.physkit_core.models import PhysicalDataset


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
            "citation": "PHYBench: A Comprehensive Physics Benchmark Dataset",
            "paper_url": "https://arxiv.org/pdf/2504.16074",
            "domains": [
                "mechanics",
                "electricity",
                "thermodynamics",
                "modern_physics",
                "optics",
                "advanced_physics"
            ],
            "languages": ["en"],
            "variants": ["full", "fullques", "onlyques"],
            "splits": ["test"],
            "problem_types": ["OE"],
            "total_problems": "500",
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
    
    @property
    def DOMAIN_MAPPING(self) -> Dict[str, str]:
        """Mapping of domain abbreviations to full domain names."""
        return {
            "MECHANICS": PhysicsDomain.MECHANICS,
            "ELECTRICITY": PhysicsDomain.ELECTRICITY,
            "THERMODYNAMICS": PhysicsDomain.THERMODYNAMICS,
            "MODERN": PhysicsDomain.MODERN_PHYSICS,
            "OPTICS": PhysicsDomain.OPTICS,
            "ADVANCED": PhysicsDomain.ADVANCED_PHYSICS
        }
    
    def _process_metadata(
        self,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process metadata to create standardized problem fields."""

        metadata['answer_type'] = "symbolic" # according to the paper
        
        domain = metadata.get("domain")
        if domain:
            metadata['domain'] = self.DOMAIN_MAPPING.get(domain, PhysicsDomain.OTHER)
        
        return metadata
        
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
        # Resolve data directory with environment variable support
        data_dir = self.resolve_data_dir(data_dir, "PHYBench")
        
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
        elif variant == "onlyques":
            json_file = data_dir / "PHYBench-onlyques_v1.json"
        else:
            raise ValueError(f"Unknown variant: {variant}. Choose 'full' or 'fullques' or 'onlyques'")
        
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