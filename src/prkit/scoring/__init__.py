"""Reference scoring implementations for PRKit's ``Scorer`` contract.

The scorer family is a 2×2 edit-distance matrix (front-end × algorithm) plus two
standalone scorers, all structurally satisfying :class:`prkit.api.Scorer` and
emitting :class:`prkit.api.Verdict`:

- ``SemanticsScorer`` — deterministic binary equivalence (the ``semantics/comparison``
  engine); the canonical, version-stamped scorer.
- ``EedScorer`` / ``SeedScorer`` — faithful PHYBench-EED / CMPhysBench-SEED baselines
  (vendor LaTeX front-end + the front-end-free pure core).
- ``SemanticsEedScorer`` / ``SemanticsSeedScorer`` — the *our-semantics* front-end
  (``normalize_physics_answer``) over those same pure cores; they populate
  ``Verdict.partial_credit`` and ``SemanticsSeedScorer`` backs
  ``verify(..., partial_credit=True)``.
- ``LLMJudgeScorer`` — the model-graded scorer wrapping the ``prkit.evaluation.llm_judge``
  engine.

Import discipline: re-exporting these scorers here must not pull ``openai``,
``prkit.evaluation.llm_judge``, ``pint``, or the vendored LaTeX front-end onto
``import prkit.scoring`` — the judge, vendored-core, and front-end imports are all
deferred to method bodies (see ``llm_judge_scorer`` / ``eed_scorer`` / ``seed_scorer``
/ ``semantics_eed_scorer`` / ``semantics_seed_scorer``).
"""

from .eed_scorer import EedScorer
from .llm_judge_scorer import LLMJudgeScorer
from .seed_scorer import SeedScorer
from .semantics_eed_scorer import SemanticsEedScorer
from .semantics_scorer import SemanticsScorer
from .semantics_seed_scorer import SemanticsSeedScorer

__all__ = [
    "EedScorer",
    "LLMJudgeScorer",
    "SeedScorer",
    "SemanticsEedScorer",
    "SemanticsScorer",
    "SemanticsSeedScorer",
]
