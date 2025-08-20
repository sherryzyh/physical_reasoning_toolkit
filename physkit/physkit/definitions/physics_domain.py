"""
Physics domain enumeration for PhysKit.

This module defines the standard physics domains supported by the toolkit.
"""

from enum import Enum


class PhysicsDomain(Enum):
    """Enumeration of physics domains based on evaluation dataset standards."""
    # Core classical physics domains (from PHYBench, UGPhysics, SeePhys)
    MECHANICS = "mechanics"  # Classical mechanics, kinematics, dynamics, energy, momentum
    ELECTROMAGNETISM = "electromagnetism"  # Electric fields, magnetic fields, circuits, electromagnetic waves
    THERMODYNAMICS = "thermodynamics"  # Heat, temperature, entropy, energy transfer
    OPTICS = "optics"  # Light, reflection, refraction, diffraction, interference, wave optics
    ACOUSTICS = "acoustics"  # Sound waves, wave propagation, resonance, wave acoustics
    
    # Modern physics domains (from UGPhysics, PhysReason)
    QUANTUM_MECHANICS = "quantum_mechanics"  # Wave-particle duality, quantum states, uncertainty
    RELATIVITY = "relativity"  # Special relativity, general relativity, spacetime
    ATOMIC_PHYSICS = "atomic_physics"  # Atomic structure, energy levels, spectroscopy
    
    # Advanced physics domains (from UGPhysics)
    CONDENSED_MATTER = "condensed_matter"  # Materials science, phase transitions, semiconductors
    
    # Fallback
    OTHER = "other"  # If none of the above clearly apply
    
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
    
    def __str__(self) -> str:
        """Return the string representation of the domain."""
        return self.value
    
    def __repr__(self) -> str:
        """Return the representation of the domain."""
        return f"PhysicsDomain.{self.name}"
