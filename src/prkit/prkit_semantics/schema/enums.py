"""Enumerations for physics answer semantics.

See ``TAXONOMY.md`` in this package for the full human-readable taxonomy.
"""

from __future__ import annotations

from enum import Enum


class _StrEnum(str, Enum):
    """Enum subclass with string values and friendly ``str()`` output."""

    def __str__(self) -> str:
        return self.value


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


class QuestionSymbolicMode(_StrEnum):
    """What symbolic form the question admits."""

    EXPRESSION = "expression"
    RELATION = "relation"
    EITHER = "either"


class QuestionUnitPolicy(_StrEnum):
    """How the question expects units to appear."""

    REQUIRED = "required"
    OPTIONAL_IF_QUESTION_FIXED_UNIT = "optional_if_question_fixed_unit"
    FORBIDDEN = "forbidden"
    NOT_APPLICABLE = "not_applicable"


class OrderingPolicy(_StrEnum):
    """Ordering rules for structured answers."""

    ORDERED = "ordered"
    UNORDERED = "unordered"
    PER_PART = "per_part"


class ContractValidationStatus(_StrEnum):
    """Whether an answer is admitted by the evaluation contract."""

    ADMITTED = "admitted"
    COERCIBLE = "coercible"
    VIOLATING = "violating"


class ComparisonPolicyMode(_StrEnum):
    """How strictly the evaluator enforces contract validation and bridges."""

    STRICT = "strict"
    AUDITED = "audited"
    PERMISSIVE = "permissive"


class BridgeTier(_StrEnum):
    """Risk tier for cross-kind or fallback comparison bridges."""

    TIER1 = "tier1"
    TIER2 = "tier2"
    TIER3 = "tier3"
