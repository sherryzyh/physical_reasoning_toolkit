"""
SeePhys dataset loader.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, List, Optional

from .base_loader import BaseDatasetLoader
from physkit.models import PhysicalDataset


class SeePhysLoader(BaseDatasetLoader):
    """Loader for SeePhys dataset."""
    
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
            "repository_url": "https://github.com/AI4Phys/SeePhys",
            "license": "Research use",
            "homepage": "https://github.com/AI4Phys/SeePhys",
            "languages": ["en"],
            "variants": ["full", "mini"],
            "splits": ["test"],
            "problem_types": ["OE"],
            "total_problems": "1000",
            "difficulty": ["easy", "medium", "hard"],
            "source": "SeePhys dataset with full (1000 problems) and mini (100 problems) versions"
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
        Load SeePhys dataset from official CSV files.
        
        Args:
            data_dir: Path to the data directory containing SeePhys files
            variant: Dataset variant - "full" (default) or "mini"
            sample_size: Number of problems to load
            split: Dataset split to load - "test" (only)
            **kwargs: Additional loading parameters (ignored for compatibility)
        
        Returns:
            PhysicalDataset containing SeePhys problems
        """
        # Set default data directory if none provided
        if data_dir is None:
            data_dir = Path.home() / "data" / "SeePhys"
            print(f"🔍 Using default data directory: {data_dir}")
        else:
            data_dir = Path(data_dir)
            print(f"🔍 Using provided data directory: {data_dir}")
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        if split != "test":
            raise ValueError(f"SeePhys dataset only has 'test' split. Got: {split}")
        
        if variant == "full":
            seephys_file = data_dir / "seephys_train.csv"
        elif variant == "mini":
            seephys_file = data_dir / "seephys_train_mini.csv"
        else:
            raise ValueError(f"Unknown variant: {variant}. Choose 'full' or 'mini'")
        
        return self._load_csv_format(seephys_file, variant, sample_size, **kwargs)
    
    @property
    def field_mapping(self) -> Dict[str, str]:
        # question,subject,image_path,sig_figs,level,language,index,img_category,vision_relevance,caption
        return {
            "index": "problem_id",
            "subject": "domain",
        }
    
    def _load_csv_format(
        self,
        seephys_file: Path,
        sample_size: Optional[int],
        split: str,
        **kwargs,
    ) -> PhysicalDataset:
        """Load from official SeePhys CSV files."""
        
        if not seephys_file.exists():
            raise FileNotFoundError(f"SeePhys CSV file not found: {seephys_file}")
        
        df = pd.read_csv(seephys_file)
        
        # Convert to unified format
        problems = []
        
        for _, row in df.iterrows():
            problem_data = row.to_dict()
            metadata = self.intiailize_metadata(problem_data)
            metadata = self._process_metadata(metadata)
            
            problem = self.create_physics_problem(
                metadata=metadata,
            )
            problems.append(problem)
        
        # Create dataset info
        info = self.get_info()
        
        return PhysicalDataset(problems, info, split=split)
    
    def _process_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process metadata to create standardized problem fields."""
        return metadata