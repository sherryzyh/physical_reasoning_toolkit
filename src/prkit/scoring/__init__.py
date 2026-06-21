"""Reference scoring implementations for PRKit's ``Scorer`` contract.

``SemanticsScorer`` is the canonical, version-stamped scorer wrapping the
deterministic (binary) semantics comparison engine. ``PartialCreditScorer`` is its
graded counterpart: an EED/SEED edit-distance scorer that populates
``Verdict.partial_credit``. ``LLMJudgeScorer`` is the model-graded scorer wrapping
the ``prkit.evaluation.llm_judge`` engine. All structurally satisfy
:class:`prkit.api.Scorer` and emit :class:`prkit.api.Verdict`.

Import discipline: re-exporting ``LLMJudgeScorer`` here must not pull ``openai`` or
``prkit.evaluation.llm_judge`` onto ``import prkit.scoring`` — its judge imports are
deferred to method bodies (see ``llm_judge_scorer``).
"""

from .llm_judge_scorer import LLMJudgeScorer
from .partial_credit_scorer import PartialCreditMode, PartialCreditScorer
from .semantics_scorer import SemanticsScorer

__all__ = [
    "LLMJudgeScorer",
    "PartialCreditMode",
    "PartialCreditScorer",
    "SemanticsScorer",
]
