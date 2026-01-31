"""
PRKit Datasets - A unified dataset loading library for physical reasoning datasets.

This package provides a simple interface for loading various physics and physical
reasoning datasets with a consistent API similar to Hugging Face datasets.

The package uses the unified PhysicsProblem interface from prkit.prkit_core,
providing a consistent experience across all PRKit (physical-reasoning-toolkit) packages.

Usage:
    # Simple loading (recommended)
    from prkit_datasets import DatasetHub

    dataset = DatasetHub.load("ugphysics")
    print(f"Loaded {len(dataset)} problems")

    # With options
    dataset = DatasetHub.load("ugphysics", sample_size=100, split="test")

    # List available datasets
    print(DatasetHub.list_available())

    # Get dataset info
    info = DatasetHub.get_info("ugphysics")

    # All datasets return PhysicsProblem objects
    for problem in dataset:
        print(f"Problem {problem.problem_id}: {problem.question}")
        # Access fields using both attribute and dictionary syntax
        print(f"Domain: {problem.domain}")           # Attribute access
        print(f"Domain: {problem['domain']}")        # Dictionary access
        print(f"Custom field: {problem['source']}")  # Custom fields work too
"""

import os
from pathlib import Path


def _ensure_cache_directory():
    """
    Ensure the default dataset cache directory exists.
    
    This function is called automatically when the package is imported
    to create ~/PHYSICAL_REASONING_DATASETS/ if it doesn't exist.
    """
    # Check if DATASET_CACHE_DIR is set
    cache_dir = os.getenv("DATASET_CACHE_DIR")
    
    if cache_dir:
        # Use the environment variable if set
        cache_path = Path(cache_dir)
    else:
        # Use default: ~/PHYSICAL_REASONING_DATASETS
        cache_path = Path.home() / "PHYSICAL_REASONING_DATASETS"
    
    # Create directory if it doesn't exist
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError):
        # Silently fail if we can't create the directory
        # (e.g., permissions issue, read-only filesystem)
        # The downloader/loader will handle this gracefully
        pass


# Automatically create cache directory on import
_ensure_cache_directory()

from .hub import DatasetHub
from . import citations

__all__ = [
    "DatasetHub",
    "citations",
]
