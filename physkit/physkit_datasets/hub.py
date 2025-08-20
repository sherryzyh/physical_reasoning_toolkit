"""
Dataset hub for managing and loading physical reasoning datasets.

This module provides a clean, simple interface for loading physical reasoning datasets.
"""

from typing import Dict, Any, List, Type, Optional, Union
from pathlib import Path
from physkit_datasets.loaders.base_loader import BaseDatasetLoader
from physkit_datasets.loaders import (
    PHYBenchLoader, 
    SeePhysLoader, 
    UGPhysicsLoader,
    JEEBenchLoader,
    SciBenchLoader,
    TPBenchLoader,
    PhysReasonLoader
)
from physkit_core.models import PhysicalDataset
from physkit_core import PhysKitLogger


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
        
        # With variant
        dataset = DatasetHub.load("ugphysics", variant="mini", split="test")
        
        # List available datasets
        print(DatasetHub.list_available())
        
        # Get dataset info
        info = DatasetHub.get_info("ugphysics")
        
        # Register custom loader
        DatasetHub.register("custom", CustomLoader)
    """
    
    # Class-level registry of dataset loaders
    _loaders: Dict[str, Type[BaseDatasetLoader]] = {}
    _logger = PhysKitLogger.get_logger(__name__)
    
    @classmethod
    def _register_default_loaders(cls):
        """Register the default dataset loaders."""
        cls.register("phybench", PHYBenchLoader)
        cls.register("seephys", SeePhysLoader)
        cls.register("ugphysics", UGPhysicsLoader)
        cls.register("jeebench", JEEBenchLoader)
        cls.register("scibench", SciBenchLoader)
        cls.register("tpbench", TPBenchLoader)
        cls.register("physreason", PhysReasonLoader)
    
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
        data_dir: Union[str, Path, None] = None,
        sample_size: Optional[int] = None,
        **kwargs
    ) -> PhysicalDataset:
        """
        Load a physical reasoning dataset.
        
        Args:
            dataset_name: Name of the dataset ('ugphysics', 'phybench', 'seephys', etc.)
            data_dir: Path to the data directory (None = auto-detect)
            sample_size: Number of problems to load (None = all)
            **kwargs: Additional arguments for the specific loader (e.g., split, variant, etc.)
            
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
            
            >>> # Load with variant
            >>> dataset = DatasetHub.load("ugphysics", variant="mini", split="test")
        """
        # Get the appropriate loader
        loader = cls._get_loader(dataset_name)
        
        # Prepare load arguments
        load_kwargs = {
            "sample_size": sample_size,
            **kwargs
        }
        
        # Add data_dir if specified
        if data_dir is not None:
            load_kwargs['data_dir'] = data_dir
        
        # Load the dataset
        dataset = loader.load(**load_kwargs)
        
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
    
    @classmethod
    def get_loader_info(cls, dataset_name: str) -> Dict[str, Any]:
        """Get detailed information about a dataset loader including supported parameters."""
        loader = cls._get_loader(dataset_name)
        info = loader.get_info()
        
        # Add loader class information
        info['loader_class'] = loader.__class__.__name__
        info['loader_module'] = loader.__class__.__module__
        
        return info