"""
PRKit — Physical Reasoning Toolkit.

A toolkit for physical-reasoning dataset loading, multi-provider LLM inference,
answer evaluation/semantics, and annotation tasks.

PyPI package name: ``physical-reasoning-toolkit``
Import name: ``prkit``

Integration entry point:
    :mod:`prkit.api` is the frozen, semver-governed public contract (the
    ``DatasetProvider``/``ModelClient``/``Scorer``/``Runner`` protocols and the
    ``Verdict`` result). Integrate against it rather than reaching into
    subpackages. See ``prkit/CONTRACT.md`` for the stability and version policy.

Subpackages:
    - :mod:`prkit.api` — frozen public contract (protocols + ``Verdict``).
    - :mod:`prkit.core` — domain models, model clients, logging.
    - :mod:`prkit.datasets` — dataset hub, loaders, downloaders.
    - :mod:`prkit.scoring` — reference scorers (``SemanticsScorer``; the
      ``Eed``/``Seed`` edit-distance baselines + ``Semantics`` front-end variants;
      ``LLMJudgeScorer``).
    - :mod:`prkit.testing` — conformance suite (``check_dataset``/``check_scorer``/…).
    - :mod:`prkit.semantics` — physics-aware answer normalization & comparison.
    - :mod:`prkit.evaluation` — model-graded LLM judge (``llm_judge``). The legacy
      comparator/evaluator stacks were removed while shaping the provisional
      contract; use :mod:`prkit.scoring` for deterministic scoring.
    - :mod:`prkit.annotation` — human annotation tasks (gold, correctness).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("physical-reasoning-toolkit")
except PackageNotFoundError:  # running from a source tree without an installed dist
    __version__ = "0.0.0.dev0"

from .core import PRKitLogger
from .core.domain import (
    AnswerObjectKind,
    AnswerStructure,
    PhysicsDataset,
    PhysicsDomain,
    PhysicsProblem,
)

__all__ = [
    "__version__",
    "PRKitLogger",
    "PhysicsProblem",
    "PhysicsDataset",
    "PhysicsDomain",
    "AnswerObjectKind",
    "AnswerStructure",
]
