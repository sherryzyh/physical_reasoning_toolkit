"""Domain subtype of the ``gold`` task: schema + the physics-domain choice list.

A gold-domain record captures the *authoritative* physics domain a human expert assigns
to a problem. It is stored alongside (never inside) the source dataset so the originals
stay untouched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ....core.domain.physics_domain import PhysicsDomain

SUBTYPE = "domain"


@dataclass
class DomainRecord:
    """One expert gold-domain annotation for a single problem."""

    problem_id: str
    gold_domain: str
    suggested_domain: str = ""
    question_preview: str = ""
    annotator: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON storage."""
        return asdict(self)


def domain_choices() -> list[str]:
    """Return the ``PhysicsDomain`` values offered to the annotator, in enum order."""
    return [domain.value for domain in PhysicsDomain]


def normalize_domain(value: str) -> str:
    """Map free-form *value* onto a ``PhysicsDomain`` value, keeping unknown text verbatim.

    Recognized names/aliases collapse to their canonical enum value. Text the taxonomy
    cannot place is returned unchanged (free-form gold label) rather than silently
    becoming ``"other"``.
    """
    text = (value or "").strip()
    if not text:
        return ""
    domain = PhysicsDomain.from_string(text)
    if domain is PhysicsDomain.OTHER and text.lower() not in (
        "other",
        PhysicsDomain.OTHER.value,
    ):
        return text
    return domain.value
