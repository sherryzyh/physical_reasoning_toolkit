"""
PRKit — Physical Reasoning Toolkit.

A toolkit for physical-reasoning dataset loading, multi-provider LLM inference,
answer evaluation/semantics, and annotation workflows.

PyPI package name: ``physical-reasoning-toolkit``
Import name: ``prkit``

Subpackages:
    - :mod:`prkit.core` — domain models, model clients, logging.
    - :mod:`prkit.datasets` — dataset hub, loaders, downloaders.
    - :mod:`prkit.evaluation` — comparators, evaluators, LLM judge.
    - :mod:`prkit.semantics` — physics-aware answer normalization & comparison.
    - :mod:`prkit.annotation` — annotation workers and workflows.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("physical-reasoning-toolkit")
except PackageNotFoundError:  # running from a source tree without an installed dist
    __version__ = "0.0.0.dev0"

from .core import PRKitLogger
from .core.domain import AnswerCategory, PhysicalDataset, PhysicsDomain, PhysicsProblem

__all__ = [
    "__version__",
    "PRKitLogger",
    "PhysicsProblem",
    "PhysicalDataset",
    "PhysicsDomain",
    "AnswerCategory",
]
