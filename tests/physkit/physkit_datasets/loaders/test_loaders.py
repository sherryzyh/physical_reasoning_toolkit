"""
Tests for dataset loaders.
"""

import pytest
from physkit.physkit_datasets.loaders.base_loader import BaseDatasetLoader


class TestBaseDatasetLoader:
    """Test cases for BaseDatasetLoader."""
    
    def test_base_loader_is_abstract(self):
        """Test that BaseDatasetLoader is abstract."""
        with pytest.raises(TypeError):
            BaseDatasetLoader()
    
    def test_base_loader_interface(self):
        """Test that BaseDatasetLoader defines required methods."""
        # Check that required methods exist
        assert hasattr(BaseDatasetLoader, 'load')
        assert hasattr(BaseDatasetLoader, 'get_info')
