"""
PhysKit Datasets - A unified dataset loading library for physical reasoning datasets.

This package provides a simple interface for loading various physics and physical
reasoning datasets with a consistent API similar to Hugging Face datasets.

The package now uses the unified PhysicsProblem interface from the core physkit
package, providing a consistent experience across all PhysKit packages.

Usage:
    # Simple loading (recommended)
    from physkit_datasets import DatasetHub
    
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

__version__ = "0.1.0"
__author__ = "Yinghuan Zhang"
__author_email__ = "yinghuan.flash@gmail.com"

from .hub import DatasetHub

__all__ = [
    "DatasetHub",
]
