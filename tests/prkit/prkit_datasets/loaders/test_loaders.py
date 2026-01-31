"""
Tests for dataset loaders.
"""

import pytest

from prkit.prkit_datasets.loaders.base_loader import BaseDatasetLoader
from prkit.prkit_datasets.loaders import SeePhysLoader


class TestBaseDatasetLoader:
    """Test cases for BaseDatasetLoader."""

    def test_base_loader_is_abstract(self):
        """Test that BaseDatasetLoader is abstract."""
        with pytest.raises(TypeError):
            BaseDatasetLoader()

    def test_base_loader_interface(self):
        """Test that BaseDatasetLoader defines required methods."""
        # Check that required methods exist
        assert hasattr(BaseDatasetLoader, "load")
        assert hasattr(BaseDatasetLoader, "get_info")

    def test_create_physics_problem_with_image_paths(self):
        """Test that create_physics_problem handles image_paths correctly."""
        loader = SeePhysLoader()  # Use a concrete loader instance
        
        # Test with image_paths (plural) - preferred format
        metadata = {
            "problem_id": "test_001",
            "question": "What is F = ma?",
            "answer": "F = ma",
            "image_paths": ["images/img1.jpg", "images/img2.png"],
        }
        problem = loader.create_physics_problem(metadata=metadata)
        assert problem is not None
        assert len(problem.image_path) == 2
        assert "img1.jpg" in problem.image_path[0] or "img1.jpg" in str(problem.image_path[0])
        assert "img2.png" in problem.image_path[1] or "img2.png" in str(problem.image_path[1])

    def test_create_physics_problem_with_image_paths_single(self):
        """Test that create_physics_problem handles single image_paths string."""
        loader = SeePhysLoader()
        
        metadata = {
            "problem_id": "test_002",
            "question": "What is E = mc²?",
            "answer": "E = mc²",
            "image_paths": "images/diagram.jpg",
        }
        problem = loader.create_physics_problem(metadata=metadata)
        assert problem is not None
        assert len(problem.image_path) == 1

    def test_create_physics_problem_with_image_paths_empty(self):
        """Test that create_physics_problem handles empty image_paths."""
        loader = SeePhysLoader()
        
        metadata = {
            "problem_id": "test_003",
            "question": "What is the speed of light?",
            "answer": "3e8 m/s",
            "image_paths": [],
        }
        problem = loader.create_physics_problem(metadata=metadata)
        assert problem is not None
        assert problem.image_path == []

    def test_create_physics_problem_with_image_path_legacy(self):
        """Test backward compatibility with image_path (singular) legacy format."""
        loader = SeePhysLoader()
        
        # Test with image_path (singular) - legacy format for backward compatibility
        metadata = {
            "problem_id": "test_004",
            "question": "What is gravity?",
            "answer": "9.8 m/s²",
            "image_path": ["images/gravity.jpg"],
        }
        problem = loader.create_physics_problem(metadata=metadata)
        assert problem is not None
        assert len(problem.image_path) == 1
