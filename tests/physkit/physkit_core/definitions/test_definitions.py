"""
Tests for definitions: PhysicsDomain, AnswerType.
"""

import pytest
from physkit.physkit_core.definitions import PhysicsDomain, AnswerType


class TestPhysicsDomain:
    """Test cases for PhysicsDomain enum."""
    
    def test_domain_enum_values(self):
        """Test that domain enum has expected values."""
        assert PhysicsDomain.CLASSICAL_MECHANICS.value == "classical_mechanics"
        assert PhysicsDomain.QUANTUM_MECHANICS.value == "quantum_mechanics"
        assert PhysicsDomain.THERMODYNAMICS.value == "thermodynamics"
        assert PhysicsDomain.OTHER.value == "other"
    
    def test_domain_from_string_valid(self):
        """Test converting valid string to domain."""
        domain = PhysicsDomain.from_string("classical_mechanics")
        assert domain == PhysicsDomain.CLASSICAL_MECHANICS
        
        domain2 = PhysicsDomain.from_string("QUANTUM_MECHANICS")
        assert domain2 == PhysicsDomain.QUANTUM_MECHANICS
    
    def test_domain_from_string_invalid(self):
        """Test converting invalid string returns OTHER."""
        domain = PhysicsDomain.from_string("invalid_domain")
        assert domain == PhysicsDomain.OTHER
    
    def test_domain_from_string_normalization(self):
        """Test string normalization."""
        # Test with spaces
        domain = PhysicsDomain.from_string("classical mechanics")
        assert domain == PhysicsDomain.CLASSICAL_MECHANICS
        
        # Test with hyphens
        domain2 = PhysicsDomain.from_string("classical-mechanics")
        assert domain2 == PhysicsDomain.CLASSICAL_MECHANICS
    
    def test_domain_from_value(self):
        """Test from_value method (alias for from_string)."""
        domain = PhysicsDomain.from_value("classical_mechanics")
        assert domain == PhysicsDomain.CLASSICAL_MECHANICS
    
    def test_domain_str_repr(self):
        """Test string representations."""
        domain = PhysicsDomain.CLASSICAL_MECHANICS
        assert str(domain) == "classical_mechanics"
        assert "CLASSICAL_MECHANICS" in repr(domain)
    
    def test_all_domains_accessible(self):
        """Test that all defined domains are accessible."""
        domains = [
            PhysicsDomain.SOLID_STATE_PHYSICS,
            PhysicsDomain.RELATIVITY,
            PhysicsDomain.QUANTUM_MECHANICS,
            PhysicsDomain.CLASSICAL_ELECTROMAGNETISM,
            PhysicsDomain.ELECTRODYNAMICS,
            PhysicsDomain.THERMODYNAMICS,
            PhysicsDomain.THEORETICAL_MECHANICS,
            PhysicsDomain.ATOMIC_PHYSICS,
            PhysicsDomain.STATISTICAL_MECHANICS,
            PhysicsDomain.GEOMETRICAL_OPTICS,
            PhysicsDomain.WAVE_OPTICS,
            PhysicsDomain.SEMICONDUCTOR_PHYSICS,
            PhysicsDomain.CLASSICAL_MECHANICS,
            PhysicsDomain.HIGH_ENERGY_THEORY,
            PhysicsDomain.COSMOLOGY,
            PhysicsDomain.FUNDAMENTAL_PHYSICS,
            PhysicsDomain.MECHANICS,
            PhysicsDomain.ELECTRICITY,
            PhysicsDomain.OPTICS,
            PhysicsDomain.MODERN_PHYSICS,
            PhysicsDomain.ADVANCED_PHYSICS,
            PhysicsDomain.OTHER,
        ]
        assert len(domains) > 0
        assert all(isinstance(d, PhysicsDomain) for d in domains)


class TestAnswerType:
    """Test cases for AnswerType enum."""
    
    def test_answer_type_enum_values(self):
        """Test that answer type enum has expected values."""
        assert AnswerType.NUMERICAL.value == "numerical"
        assert AnswerType.SYMBOLIC.value == "symbolic"
        assert AnswerType.TEXTUAL.value == "textual"
        assert AnswerType.OPTION.value == "option"
    
    def test_all_answer_types_accessible(self):
        """Test that all answer types are accessible."""
        types = [
            AnswerType.NUMERICAL,
            AnswerType.SYMBOLIC,
            AnswerType.TEXTUAL,
            AnswerType.OPTION,
        ]
        assert len(types) == 4
        assert all(isinstance(t, AnswerType) for t in types)
    
    def test_answer_type_str(self):
        """Test string representation."""
        assert AnswerType.NUMERICAL.value == "numerical"
        assert AnswerType.SYMBOLIC.value == "symbolic"
