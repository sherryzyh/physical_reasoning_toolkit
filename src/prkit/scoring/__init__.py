"""Reference scoring implementations for PRKit's ``Scorer`` contract.

``SemanticsScorer`` is the canonical, version-stamped scorer wrapping the
deterministic semantics comparison engine. It structurally satisfies
:class:`prkit.api.Scorer` and emits :class:`prkit.api.Verdict`.
"""

from .semantics_scorer import SemanticsScorer

__all__ = ["SemanticsScorer"]
