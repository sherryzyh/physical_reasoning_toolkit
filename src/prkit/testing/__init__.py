"""Public conformance suite for PRKit's contract nouns.

Shipped in the package so downstream loader/scorer/client authors can verify
their implementations satisfy the :mod:`prkit.api` contract::

    from prkit.testing import check_scorer
    check_scorer(MyScorer())

Imports with runtime deps only (``pytest`` is imported lazily by
:class:`ConformanceTestMixin`).
"""

from .conformance import (
    ConformanceTestMixin,
    check_dataset,
    check_model_client,
    check_scorer,
)

__all__ = [
    "check_dataset",
    "check_scorer",
    "check_model_client",
    "ConformanceTestMixin",
]
