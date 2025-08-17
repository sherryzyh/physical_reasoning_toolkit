"""
Dataset hub for managing and loading physical reasoning datasets.

This module provides a clean, simple interface for loading physical reasoning datasets.
"""

from typing import Dict, Any, List, Type, Optional, Union
from pathlib import Path
from physkit_datasets.loaders.base_loader import BaseDatasetLoader
from physkit_datasets.loaders import PHYBenchLoader, SeePhysLoader, UGPhysicsLoader
from physkit.models import PhysicalDataset


class DatasetHub:
    """
    Simple hub for loading physical reasoning datasets.
    
    This class provides a clean, intuitive interface similar to Hugging Face's datasets library.
    It combines registry functionality with a user-friendly API.
    
    Usage:
        # Simple loading
        dataset = DatasetHub.load("ugphysics")
        
        # With options
        dataset = DatasetHub.load("ugphysics", split="test", sample_size=100)
        
        # List available datasets
        print(DatasetHub.list_available())
        
        # Get dataset info
        info = DatasetHub.get_info("ugphysics")
        
        # Register custom loader
        DatasetHub.register("custom", CustomLoader)
    """
    
    # Class-level registry of dataset loaders
    _loaders: Dict[str, Type[BaseDatasetLoader]] = {}
    
    @classmethod
    def _register_default_loaders(cls):
        """Register the default dataset loaders."""
        cls.register("phybench", PHYBenchLoader)
        cls.register("seephys", SeePhysLoader)
        cls.register("ugphysics", UGPhysicsLoader)
    
    @classmethod
    def register(cls, name: str, loader_class: Type[BaseDatasetLoader]):
        """Register a new dataset loader."""
        cls._loaders[name] = loader_class
    
    @classmethod
    def _get_loader(cls, name: str) -> BaseDatasetLoader:
        """Get a dataset loader by name."""
        if not cls._loaders:
            cls._register_default_loaders()
            
        if name not in cls._loaders:
            available = ", ".join(cls._loaders.keys())
            raise ValueError(f"Unknown dataset: {name}. Available datasets: {available}")
        
        return cls._loaders[name]()
    
    @classmethod
    def load(
        cls,
        dataset_name: str,
        data_dir: Union[str, Path] = "/Users/yinghuan/data",
        split: str = "test",
        sample_size: Optional[int] = None,
        **kwargs
    ) -> PhysicalDataset:
        """
        Load a physical reasoning dataset.
        
        Args:
            dataset_name: Name of the dataset ('ugphysics', 'phybench', 'seephys', etc.)
            data_dir: Path to the data directory
            split: Dataset split ('test', 'train', 'val')
            sample_size: Number of problems to load (None = all)
            **kwargs: Additional arguments for the specific loader
            
        Returns:
            PhysicalDataset: Loaded dataset
            
        Raises:
            ValueError: If dataset name is unknown
            FileNotFoundError: If data directory doesn't exist
            
        Examples:
            >>> # Load UGPhysics dataset
            >>> dataset = DatasetHub.load("ugphysics")
            >>> print(f"Loaded {len(dataset)} problems")
            
            >>> # Load with sample size
            >>> dataset = DatasetHub.load("ugphysics", sample_size=50)
            
            >>> # Load specific split
            >>> dataset = DatasetHub.load("ugphysics", split="test")
        """
        # Get the appropriate loader
        loader = cls._get_loader(dataset_name)
        
        # Load the dataset
        dataset = loader.load(data_dir, split=split, sample_size=sample_size, **kwargs)
        
        return dataset
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all available dataset names."""
        if not cls._loaders:
            cls._register_default_loaders()
        return list(cls._loaders.keys())
    
    @classmethod
    def get_info(cls, dataset_name: str) -> Dict[str, Any]:
        """Get information about a specific dataset."""
        loader = cls._get_loader(dataset_name)
        return loader.get_info()


__all__ = [
    "DatasetHub"
]
