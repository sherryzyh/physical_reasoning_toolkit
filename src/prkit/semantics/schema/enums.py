"""Enumerations for physics answer semantics.

The answer *ontology* enums (:class:`AnswerObjectKind`, :class:`AnswerStructure`)
and the :class:`_StrEnum` base now live in :mod:`prkit.core.domain.answer_kinds`
as the toolkit's canonical taxonomy; they are re-exported here so existing
``from prkit.semantics.schema import AnswerObjectKind`` import sites keep working.
The *judgement-policy* enums below (unit policy, comparison mode, bridge tier, …)
stay in semantics — they are mechanism, not ontology.

See ``../comparison/EQUIVALENCE.md`` for the detailed equivalence-judgement reference
(object kinds, structures, per-kind criteria, bridges, examples) and
``../comparison/METHODOLOGY.md`` for the design discipline behind it.
"""

from __future__ import annotations

# Re-exported canonical ontology enums (defined in prkit.core.domain). ``_StrEnum``
# is the shared base for the judgement-policy enums defined in this module.
from prkit.core.domain.answer_kinds import (
    AnswerObjectKind,
    AnswerStructure,
    _StrEnum,
)

__all__ = [
    # Re-exported canonical ontology (defined in prkit.core.domain)
    "AnswerObjectKind",
    "AnswerStructure",
    # Judgement-policy enums (defined below)
    "QuestionSymbolicMode",
    "QuestionUnitPolicy",
    "OrderingPolicy",
    "ContractValidationStatus",
    "ComparisonPolicyMode",
    "BridgeTier",
    "SymbolAssumption",
]


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


class SymbolAssumption(_StrEnum):
    """Real-domain a free symbol ranges over during symbolic comparison.

    Physics answers denote real, often nonnegative, quantities; declaring this lets the
    SymPy substrate decide equivalence over the *intended* domain instead of the generic
    complex default (e.g. ``sqrt(a*b) == sqrt(a)*sqrt(b)`` holds for ``a, b >= 0`` but not
    over the complex plane). The judgement stays exact -- it is decided over the declared
    domain, not relaxed.
    """

    COMPLEX = "complex"
    REAL = "real"
    NONZERO = "nonzero"
    NONNEGATIVE = "nonnegative"
    POSITIVE = "positive"
