"""Canonical answer-ontology enumerations for PRKit.

``AnswerObjectKind`` (what kind of object the final answer is) and
``AnswerStructure`` (how the answer is shaped) are the single, version-stable
taxonomy the whole toolkit targets — normalization, the judgement engine, saved
artifacts, and the public contract all agree on these names. They live in
``prkit.core.domain`` (not ``semantics``) because they are *ontology*, not
judgement mechanism; the policy enums (unit policy, comparison mode, bridge tier,
…) stay in ``prkit.semantics.schema``.

See ``../../semantics/comparison/EQUIVALENCE.md`` for the per-kind equivalence
criteria and ``../../semantics/comparison/METHODOLOGY.md`` for the design
discipline behind the taxonomy.
"""

from __future__ import annotations

from enum import StrEnum


class _StrEnum(StrEnum):
    """Enum subclass with string values and friendly ``str()`` output."""

    def __str__(self) -> str:
        return str(self.value)


class AnswerObjectKind(_StrEnum):
    """What kind of answer object the normalized final answer is."""

    NUMBER = "number"
    PHYSICAL_QUANTITY = "physical_quantity"
    EXPRESSION = "expression"
    RELATION = "relation"
    QUALITATIVE_LABEL = "qualitative_label"
    CHOICE = "choice"
    BOOLEAN = "boolean"
    SIGN_DIRECTION = "sign_direction"
    DESCRIPTIVE_TEXT = "descriptive_text"


class AnswerStructure(_StrEnum):
    """How the answer is structured."""

    ATOMIC = "atomic"
    MULTI_PART = "multi_part"
    TUPLE = "tuple"
    SET = "set"
    INTERVAL = "interval"
    VECTOR = "vector"
    MATRIX = "matrix"
    TENSOR = "tensor"
    PIECEWISE = "piecewise"
