"""Reference scoring implementations for PRKit's ``Scorer`` contract.

``SemanticsScorer`` is the canonical, version-stamped scorer wrapping the
deterministic (binary) semantics comparison engine. ``EedScorer`` / ``SeedScorer``
are the faithful PHYBench-EED / CMPhysBench-SEED edit-distance *baselines* (vendor
LaTeX front-end + the front-end-free pure core). ``PartialCreditScorer`` is the
graded EED/SEED scorer over PRKit's own semantics front-end that populates
``Verdict.partial_credit``. ``LLMJudgeScorer`` is the model-graded scorer wrapping
the ``prkit.evaluation.llm_judge`` engine. All structurally satisfy
:class:`prkit.api.Scorer` and emit :class:`prkit.api.Verdict`.

Import discipline: re-exporting these scorers here must not pull ``openai``,
``prkit.evaluation.llm_judge``, ``pint``, or the vendored LaTeX front-end onto
``import prkit.scoring`` — the judge, vendored-core, and front-end imports are all
deferred to method bodies (see ``llm_judge_scorer`` / ``eed_scorer`` / ``seed_scorer``).
"""

from .eed_scorer import EedScorer
from .llm_judge_scorer import LLMJudgeScorer
from .partial_credit_scorer import PartialCreditMode, PartialCreditScorer
from .seed_scorer import SeedScorer
from .semantics_scorer import SemanticsScorer

__all__ = [
    "EedScorer",
    "LLMJudgeScorer",
    "PartialCreditMode",
    "PartialCreditScorer",
    "SeedScorer",
    "SemanticsScorer",
]
