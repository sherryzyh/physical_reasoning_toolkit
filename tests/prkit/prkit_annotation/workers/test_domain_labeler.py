"""
Tests for DomainLabeler.
"""

from unittest.mock import Mock, patch

import pytest

from prkit.prkit_annotation.annotations.domain import DomainAnnotation
from prkit.prkit_annotation.workers.domain_labeler import DomainLabeler, DomainResponse
from prkit.prkit_core.domain.physics_domain import PhysicsDomain


class TestDomainLabeler:
    """Test cases for DomainLabeler."""

    @patch("prkit.prkit_annotation.workers.base.create_model_client")
    def test_domain_labeler_initialization(self, mock_create):
        """Test domain labeler initialization."""
        mock_client = Mock()
        mock_create.return_value = mock_client

        labeler = DomainLabeler(model="gpt-5.1")

        assert labeler.model == "gpt-5.1"
        assert labeler.llm_client == mock_client

    @patch("prkit.prkit_annotation.workers.base.create_model_client")
    def test_domain_labeler_work_success(self, mock_create):
        """Test domain labeling with successful LLM response."""
        mock_client = Mock()
        mock_response = DomainResponse(
            domains=["classical_mechanics"],
            confidence=0.9,
            reasoning="Test reasoning",
            subdomains=["kinematics"],
        )
        import json
        # Return JSON string that will be parsed by _call_llm_structured
        mock_client.chat.return_value = json.dumps(mock_response.model_dump())
        mock_create.return_value = mock_client

        labeler = DomainLabeler(model="gpt-5.1")
        result = labeler.work("What is F=ma?")

        assert isinstance(result, DomainAnnotation)
        assert len(result.domains) > 0
        assert PhysicsDomain.CLASSICAL_MECHANICS in result.domains

    @patch("prkit.prkit_annotation.workers.base.create_model_client")
    def test_domain_labeler_work_with_invalid_domain(self, mock_create):
        """Test domain labeling with invalid domain string."""
        mock_client = Mock()
        mock_response = DomainResponse(
            domains=["invalid_domain", "classical_mechanics"],
            confidence=0.8,
            reasoning="Test",
            subdomains=[],
        )
        import json
        # Return JSON string that will be parsed by _call_llm_structured
        mock_client.chat.return_value = json.dumps(mock_response.model_dump())
        mock_create.return_value = mock_client

        labeler = DomainLabeler(model="gpt-5.1")
        result = labeler.work("What is F=ma?")

        # Should handle invalid domain gracefully
        assert isinstance(result, DomainAnnotation)
        assert len(result.domains) >= 1  # At least valid domains should be included

    @patch("prkit.prkit_annotation.workers.base.create_model_client")
    def test_domain_labeler_work_llm_error(self, mock_create):
        """Test domain labeling when LLM call fails."""
        mock_client = Mock()
        mock_client.chat.return_value = None  # Simulate failure
        mock_create.return_value = mock_client

        labeler = DomainLabeler(model="gpt-5.1")
        result = labeler.work("What is F=ma?")

        # Should return empty annotation on failure
        assert isinstance(result, DomainAnnotation)
        assert len(result.domains) == 0

    @patch("prkit.prkit_annotation.workers.base.create_model_client")
    def test_domain_labeler_normalizes_domain_strings(self, mock_create):
        """Test that domain strings are normalized correctly."""
        mock_client = Mock()
        mock_response = DomainResponse(
            domains=["classical-mechanics"],  # With hyphens - should normalize to classical_mechanics
            confidence=0.9,
            reasoning="Test",
            subdomains=[],
        )
        import json
        # Return JSON string that will be parsed by _call_llm_structured
        mock_client.chat.return_value = json.dumps(mock_response.model_dump())
        mock_create.return_value = mock_client

        labeler = DomainLabeler(model="gpt-5.1")
        result = labeler.work("Test question")

        # Should normalize and find matching domains
        assert isinstance(result, DomainAnnotation)
        # Should have at least one valid domain after normalization (classical-mechanics -> classical_mechanics)
        assert len(result.domains) >= 1
        assert PhysicsDomain.CLASSICAL_MECHANICS in result.domains
