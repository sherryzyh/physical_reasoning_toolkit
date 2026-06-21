"""
Thin observation record for a physics problem's ground-truth answer.

An ``Answer`` captures only what a dataset directly provides:

* ``value``       — verbatim answer string (always ``str``)
* ``unit``        — observed unit string when present (e.g. ``"m/s²"``, ``"N"``)
* ``source_type`` — dataset-native answer-type label, verbatim (e.g. ``"MC"``,
                    ``"NV"``, ``"Integer"``); ``None`` when the dataset provides
                    none.  **Never fabricated from heuristics.**
* ``metadata``    — arbitrary extra fields passed through from the loader.

The canonical answer ontology (``AnswerObjectKind``) and structural
classification live **only** in :mod:`prkit.semantics` (as ``object_kind`` on
:class:`~prkit.semantics.schema.models.PhysicsAnswerSemantics`).  They are
derived on demand by the semantics engine and are **not** stored here.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Answer:
    """Thin observed-data record for a physics answer.

    ``value`` is always a plain string — the verbatim answer text after
    stripping LaTeX wrappers (``\\boxed{}``, ``$…$``, ``$$…$$``).

    ``unit`` is first-class observed data consumed by the equivalence engine;
    it lives *only* here, never duplicated in ``metadata``.

    ``source_type`` is the dataset's own native answer-type label stored
    verbatim (e.g. ``"MC"``, ``"NV"``, ``"EX"``, ``"Integer"``).  The
    semantics engine does **not** read it.  ``None`` when the dataset provides
    no such label.
    """

    value: str
    unit: str | None = None
    source_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}

    # ------------------------------------------------------------------
    # Accessors (kept for convenience; no kind guards)
    # ------------------------------------------------------------------

    def get_value(self) -> str:
        """Return the answer value string."""
        return self.value

    def get_unit(self) -> str | None:
        """Return the unit string, or ``None`` when absent."""
        return self.unit

    def has_unit(self) -> bool:
        """Return ``True`` when a unit is present."""
        return self.unit is not None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        if self.unit:
            return f"{self.value} {self.unit}"
        return str(self.value)

    def __repr__(self) -> str:
        return (
            f"Answer(value={self.value!r}, unit={self.unit!r}, "
            f"source_type={self.source_type!r})"
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict (omits falsy optional fields)."""
        result: dict[str, Any] = {"value": self.value}
        if self.unit:
            result["unit"] = self.unit
        if self.source_type:
            result["source_type"] = self.source_type
        if self.metadata:
            result["metadata"] = self.metadata
        return result
