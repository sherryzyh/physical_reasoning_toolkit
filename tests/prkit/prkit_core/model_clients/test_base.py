"""
Tests for BaseModelClient abstract base class.
"""

import pytest

from prkit.prkit_core.model_clients.base import BaseModelClient


class TestBaseModelClient:
    """Test cases for BaseModelClient."""

    def test_cannot_instantiate_base_class(self):
        """Test that BaseModelClient cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModelClient("test-model")

    def test_base_class_has_abstract_chat_method(self):
        """Test that BaseModelClient defines abstract chat method."""
        # Check that chat is abstract
        assert hasattr(BaseModelClient, "chat")
        # Try to create a concrete implementation without chat
        with pytest.raises(TypeError):
            class IncompleteModel(BaseModelClient):
                pass
            IncompleteModel("test-model")
