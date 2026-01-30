"""
Physics domain enumeration for PRKit.

This module defines the standard physics domains supported by PRKit (physical-reasoning-toolkit).
"""

from enum import Enum


class PhysicsDomain(Enum):
    
    SOLID_STATE_PHYSICS = "solid_state_physics"                 # UGPhysics
    RELATIVITY = "relativity"                                   # UGPhysics
    QUANTUM_MECHANICS = "quantum_mechanics"                     # UGPhysics # TPBench # SciBench
    CLASSICAL_ELECTROMAGNETISM = "classical_electromagnetism"   # UGPhysics
    ELECTRODYNAMICS = "electrodynamics"                         # UGPhysics
    THERMODYNAMICS = "thermodynamics"                           # UGPhysics # PHYBench
    THEORETICAL_MECHANICS = "theoretical_mechanics"             # UGPhysics
    ATOMIC_PHYSICS = "atomic_physics"                           # UGPhysics
    STATISTICAL_MECHANICS = "statistical_mechanics"             # UGPhysics # TPBench
    GEOMETRICAL_OPTICS = "geometrical_optics"                   # UGPhysics
    WAVE_OPTICS = "wave_optics"                                 # UGPhysics
    SEMICONDUCTOR_PHYSICS = "semiconductor_physics"             # UGPhysics
    CLASSICAL_MECHANICS = "classical_mechanics"                 # UGPhysics # TPBench # SciBench
    HIGH_ENERGY_THEORY = "high_energy_theory"                   # TPBench
    COSMOLOGY = "cosmology"                                     # TPBench
    FUNDAMENTAL_PHYSICS = "fundamental_physics"                 # SciBench
    MECHANICS = "mechanics"                                     # PHYBench
    ELECTRICITY = "electricity"                                 # PHYBench
    OPTICS = "optics"                                           # PHYBench
    MODERN_PHYSICS = "modern_physics"                           # PHYBench
    ADVANCED_PHYSICS = "advanced_physics"                       # PHYBench
    
    OTHER = "other"
    
    
    @classmethod
    def from_string(cls, domain_str: str) -> 'PhysicsDomain':
        """Convert a string to a PhysicsDomain enum value."""
        try:
            return cls(domain_str.lower())
        except ValueError:
            # Try to find a match ignoring case and special characters
            normalized_str = domain_str.lower().replace(' ', '_').replace('-', '_')
            for domain in cls:
                if domain.value == normalized_str:
                    return domain
            return cls.OTHER
        
    @classmethod
    def from_value(cls, domain_str: str) -> 'PhysicsDomain':
        """Convert a string to a PhysicsDomain enum value."""
        try:
            return cls(domain_str.lower())
        except ValueError:
            # Try to find a match ignoring case and special characters
            normalized_str = domain_str.lower().replace(' ', '_').replace('-', '_')
            for domain in cls:
                if domain.value == normalized_str:
                    return domain
            return cls.OTHER
    
    def __str__(self) -> str:
        """Return the string representation of the domain."""
        return self.value
    
    def __repr__(self) -> str:
        """Return the representation of the domain."""
        return f"PhysicsDomain.{self.name}"
