"""Reference scoring implementations for PRKit's ``Scorer`` contract.

``SemanticsScorer`` is the canonical, version-stamped scorer wrapping the
deterministic (binary) semantics comparison engine. ``PartialCreditScorer`` is its
graded counterpart: an EED/SEED edit-distance scorer that populates
``Verdict.partial_credit``. Both structurally satisfy :class:`prkit.api.Scorer` and
emit :class:`prkit.api.Verdict`.
"""

from .partial_credit_scorer import PartialCreditMode, PartialCreditScorer
from .semantics_scorer import SemanticsScorer

__all__ = ["PartialCreditMode", "PartialCreditScorer", "SemanticsScorer"]
