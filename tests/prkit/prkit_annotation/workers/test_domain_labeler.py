"""
Tests for DomainLabeler.
"""

from unittest.mock import Mock, patch

import pytest

from prkit.prkit_annotation.annotations.domain import DomainAnnotation
from prkit.prkit_annotation.workers.domain_labeler import DomainLabeler, DomainResponse
from prkit.prkit_core.definitions.physics_domain import PhysicsDomain


class TestDomainLabeler:
    """Test cases for DomainLabeler."""

    @patch("prkit.prkit_annotation.workers.base.LLMClient")
    def test_domain_labeler_initialization(self, mock_llm_class):
        """Test domain labeler initialization."""
        mock_client = Mock()
        mock_llm_class.from_model.return_value = mock_client

        labeler = DomainLabeler(model="gpt-4o")

        assert labeler.model == "gpt-4o"
        assert labeler.llm_client == mock_client

    @patch("prkit.prkit_annotation.workers.base.LLMClient")
    def test_domain_labeler_work_success(self, mock_llm_class):
        """Test domain labeling with successful LLM response."""
        mock_client = Mock()
        mock_response = DomainResponse(
            domains=["classical_mechanics", "mechanics"],
            confidence=0.9,
            reasoning="Test reasoning",
            subdomains=["kinematics"],
        )
        mock_client.chat_structured.return_value = mock_response
        mock_llm_class.from_model.return_value = mock_client

        labeler = DomainLabeler(model="gpt-4o")
        result = labeler.work("What is F=ma?")

        assert isinstance(result, DomainAnnotation)
        assert len(result.domains) > 0
        assert PhysicsDomain.CLASSICAL_MECHANICS in result.domains or PhysicsDomain.MECHANICS in result.domains

    @patch("prkit.prkit_annotation.workers.base.LLMClient")
    def test_domain_labeler_work_with_invalid_domain(self, mock_llm_class):
        """Test domain labeling with invalid domain string."""
        mock_client = Mock()
        mock_response = DomainResponse(
            domains=["invalid_domain", "classical_mechanics"],
            confidence=0.8,
            reasoning="Test",
            subdomains=[],
        )
        mock_client.chat_structured.return_value = mock_response
        mock_llm_class.from_model.return_value = mock_client

        labeler = DomainLabeler(model="gpt-4o")
        result = labeler.work("What is F=ma?")

        # Should handle invalid domain gracefully
        assert isinstance(result, DomainAnnotation)
        assert len(result.domains) >= 1  # At least valid domains should be included

    @patch("prkit.prkit_annotation.workers.base.LLMClient")
    def test_domain_labeler_work_llm_error(self, mock_llm_class):
        """Test domain labeling when LLM call fails."""
        mock_client = Mock()
        mock_client.chat_structured.return_value = None  # Simulate failure
        mock_llm_class.from_model.return_value = mock_client

        labeler = DomainLabeler(model="gpt-4o")
        result = labeler.work("What is F=ma?")

        # Should return empty annotation on failure
        assert isinstance(result, DomainAnnotation)
        assert len(result.domains) == 0

    @patch("prkit.prkit_annotation.workers.base.LLMClient")
    def test_domain_labeler_normalizes_domain_strings(self, mock_llm_class):
        """Test that domain strings are normalized correctly."""
        mock_client = Mock()
        mock_response = DomainResponse(
            domains=["classical-mechanics", "quantum mechanics"],  # With hyphens/spaces
            confidence=0.9,
            reasoning="Test",
            subdomains=[],
        )
        mock_client.chat_structured.return_value = mock_response
        mock_llm_class.from_model.return_value = mock_client

        labeler = DomainLabeler(model="gpt-4o")
        result = labeler.work("Test question")

        # Should normalize and find matching domains
        assert isinstance(result, DomainAnnotation)
        # Should have at least one valid domain after normalization
        assert len(result.domains) >= 1
