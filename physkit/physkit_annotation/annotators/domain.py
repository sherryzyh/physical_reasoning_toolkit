"""
Domain annotator for physics problem classification.
"""

from pydantic import BaseModel
from typing import List
from dataclasses import dataclass, field
from typing import Dict, List, Any


from .base import BaseAnnotator
from physkit_core.models import PhysicsDomain


@dataclass
class DomainAnnotation:
    """Annotation for physics domain classification."""
    primary_domain: PhysicsDomain
    confidence: float = 1.0
    reasoning: str = ""
    subdomains: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "primary_domain": self.primary_domain.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "subdomains": self.subdomains
        }

class DomainResponse(BaseModel):
    primary_domain: PhysicsDomain
    confidence: float
    reasoning: str
    subdomains: List[str]

class DomainAnnotator(BaseAnnotator):
    """Annotator for physics domain classification."""
    
    def annotate(self, question: str, **kwargs) -> DomainAnnotation:
        """Annotate the physics domain of the problem."""
        prompt = f"""
        Analyze the following physics problem and identify the primary physics domain it belongs to.

        Problem: {question}

        Choose from these physics domains based on evaluation dataset standards:

        - mechanics: classical mechanics, kinematics, dynamics, energy, momentum, gravitational forces
        - electromagnetism: electric fields, magnetic fields, circuits, electromagnetic waves, Maxwell's equations
        - thermodynamics: heat, temperature, entropy, energy transfer, work, pressure, volume
        - optics: light, reflection, refraction, diffraction, interference, lenses, mirrors, wave optics
        - acoustics: sound waves, wave propagation, resonance, wave acoustics, ultrasonic waves
        - quantum_mechanics: wave-particle duality, quantum states, uncertainty principle, superposition
        - relativity: special relativity, general relativity, spacetime, time dilation, length contraction
        - atomic_physics: atomic structure, energy levels, spectroscopy, electron transitions
        - condensed_matter: materials science, phase transitions, superconductivity, semiconductors
        - other: if none of the above clearly apply

        Provide your response in this exact JSON format:
        {{
            "primary_domain": "primary domain name from the options above",
            "confidence": "your confidence in the domain choice, between 0 and 1",
            "reasoning": "brief explanation of why this domain was chosen",
            "subdomains": ["specific_subdomain1", "specific_subdomain2"]
        }}
        """
        
        try:
            # Use structured output directly
            result = self._call_llm_structured(
                prompt=prompt,
                response_format=DomainResponse
            )
            if result is not None:
                return DomainAnnotation(
                    primary_domain=result.primary_domain,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    subdomains=result.subdomains
                )
            else:
                # Fallback to basic domain detection
                return self._fallback_domain_detection(question)
        except Exception as e:
            print(f"Error with structured domain annotation: {e}")
            # Fallback to basic domain detection
            return self._fallback_domain_detection(question)
    
    def _fallback_domain_detection(self, question: str) -> DomainAnnotation:
        """Fallback domain detection using keyword matching."""
        question_lower = question.lower()
        
        # Keyword-based domain detection for evaluation dataset domains
        domain_keywords = {
            PhysicsDomain.MECHANICS: ["force", "mass", "velocity", "acceleration", "energy", "momentum", "kinematics", "dynamics", "gravity", "friction", "spring", "pendulum"],
            PhysicsDomain.ELECTROMAGNETISM: ["electric", "magnetic", "field", "charge", "current", "voltage", "circuit", "capacitor", "resistor", "inductor", "maxwell", "electromagnetic"],
            PhysicsDomain.THERMODYNAMICS: ["heat", "temperature", "entropy", "energy", "work", "pressure", "volume", "gas", "adiabatic", "isothermal", "carnot"],
            PhysicsDomain.OPTICS: ["light", "ray", "reflection", "refraction", "lens", "mirror", "interference", "diffraction", "wavelength", "frequency", "optical", "wave optics", "optical physics"],
            PhysicsDomain.ACOUSTICS: ["sound", "acoustic", "ultrasonic", "resonance", "wave acoustics", "frequency", "amplitude", "wave propagation"],
            PhysicsDomain.QUANTUM_MECHANICS: ["quantum", "wave", "particle", "uncertainty", "superposition", "entanglement", "photon", "electron", "wavefunction"],
            PhysicsDomain.RELATIVITY: ["relativity", "spacetime", "light speed", "time dilation", "length contraction", "einstein", "lorentz"],
            PhysicsDomain.ATOMIC_PHYSICS: ["atomic", "spectrum", "energy level", "electron", "ionization", "excitation", "spectroscopy"],
            PhysicsDomain.CONDENSED_MATTER: ["material", "phase transition", "superconductivity", "crystal structure", "semiconductor", "doping", "band gap"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return DomainAnnotation(
                    primary_domain=domain,
                    confidence=0.7,
                    reasoning="Fallback detection using keyword matching",
                    subdomains=[]
                )
        
        return DomainAnnotation(
            primary_domain=PhysicsDomain.OTHER,
            confidence=0.5,
            reasoning="Fallback detection - no clear domain identified",
            subdomains=[]
        )
