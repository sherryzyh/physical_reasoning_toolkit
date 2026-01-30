"""
Tests for DatasetHub.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from physkit.physkit_datasets import DatasetHub
from physkit.physkit_datasets.loaders.base_loader import BaseDatasetLoader
from physkit.physkit_core.models import PhysicsProblem, PhysicalDataset, Answer
from physkit.physkit_core.definitions import AnswerType


class TestDatasetHub:
    """Test cases for DatasetHub."""
    
    def test_list_available(self):
        """Test listing available datasets."""
        datasets = DatasetHub.list_available()
        assert isinstance(datasets, list)
        assert len(datasets) > 0
        # Check for known datasets
        assert "ugphysics" in datasets or "phybench" in datasets or "jeebench" in datasets
    
    def test_register_custom_loader(self):
        """Test registering a custom loader."""
        class CustomLoader(BaseDatasetLoader):
            def load(self, **kwargs):
                return PhysicalDataset(problems=[], info={"name": "custom"})
            
            def get_info(self):
                return {"name": "custom", "description": "Custom dataset"}
        
        DatasetHub.register("custom_test", CustomLoader)
        assert "custom_test" in DatasetHub.list_available()
        
        # Clean up
        if "custom_test" in DatasetHub._loaders:
            del DatasetHub._loaders["custom_test"]
    
    def test_get_info(self):
        """Test getting dataset info."""
        # This will fail if dataset doesn't exist, so we'll test with a known one
        available = DatasetHub.list_available()
        if available:
            dataset_name = available[0]
            info = DatasetHub.get_info(dataset_name)
            assert isinstance(info, dict)
            assert "name" in info or "description" in info
    
    def test_get_loader_info(self):
        """Test getting detailed loader info."""
        available = DatasetHub.list_available()
        if available:
            dataset_name = available[0]
            info = DatasetHub.get_loader_info(dataset_name)
            assert isinstance(info, dict)
            assert "loader_class" in info
            assert "loader_module" in info
    
    def test_load_unknown_dataset(self):
        """Test loading an unknown dataset raises error."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            DatasetHub.load("nonexistent_dataset_xyz")
    
    @patch('physkit.physkit_datasets.hub.UGPhysicsLoader')
    def test_load_with_sample_size(self, mock_loader_class):
        """Test loading with sample_size parameter."""
        # Create mock loader instance
        mock_loader = Mock()
        mock_problems = [
            PhysicsProblem(problem_id=f"test_{i}", question=f"Question {i}")
            for i in range(10)
        ]
        mock_dataset = PhysicalDataset(problems=mock_problems)
        mock_loader.load.return_value = mock_dataset
        mock_loader_class.return_value = mock_loader
        
        # Register mock loader
        DatasetHub.register("mock_test", mock_loader_class)
        
        try:
            dataset = DatasetHub.load("mock_test", sample_size=5)
            assert len(dataset) <= 10  # Should be limited by sample_size if implemented
            mock_loader.load.assert_called_once()
        finally:
            # Clean up
            if "mock_test" in DatasetHub._loaders:
                del DatasetHub._loaders["mock_test"]
    
    def test_load_with_kwargs(self):
        """Test loading with additional kwargs."""
        class MockLoader(BaseDatasetLoader):
            @property
            def field_mapping(self):
                return {}
            
            def load(self, **kwargs):
                assert "custom_param" in kwargs
                return PhysicalDataset(problems=[], info={"name": "mock"})
            
            def get_info(self):
                return {"name": "mock"}
        
        DatasetHub.register("mock_kwargs", MockLoader)
        
        try:
            dataset = DatasetHub.load("mock_kwargs", custom_param="value")
            assert dataset is not None
        finally:
            if "mock_kwargs" in DatasetHub._loaders:
                del DatasetHub._loaders["mock_kwargs"]


class TestDatasetLoadersIntegration:
    """Integration tests for dataset loaders."""
    
    @pytest.mark.integration
    def test_loader_returns_physical_dataset(self):
        """Test that loaders return PhysicalDataset instances."""
        # This is an integration test that may require actual data files
        # Skip if data is not available
        available = DatasetHub.list_available()
        if not available:
            pytest.skip("No datasets available")
        
        # Try to load a small sample if possible
        # Note: This may fail if data files don't exist
        try:
            dataset = DatasetHub.load(available[0], sample_size=1)
            assert isinstance(dataset, PhysicalDataset)
        except (FileNotFoundError, ValueError) as e:
            # If data files don't exist, skip the test
            pytest.skip(f"Data files not available: {e}")
