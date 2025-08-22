from typing import List
from dataclasses import dataclass, field
from typing import Dict, List, Any


from physkit_core.definitions.physics_domain import PhysicsDomain


@dataclass
class DomainAnnotation:
    """Annotation for physics domain classification."""
    domains: List[PhysicsDomain]
    confidence: float = 1.0
    reasoning: str = ""
    subdomains: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "domains": [domain.value for domain in self.domains],
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "subdomains": self.subdomains,
            "domain_count": len(self.domains)
        }