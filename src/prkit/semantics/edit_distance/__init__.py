"""Expression Edit Distance (EED / SEED) partial-credit algorithm on PRKit's substrate.

A self-contained reimplementation of PHYBench's Expression Edit Distance and
CMPhysBench's Scalable EED, run on PRKit's existing SymPy parser, LaTeX normalizer,
and unit backend instead of vendoring ``latex2sympy2`` + ``pint``. The pure pieces
(:mod:`.tree`, :mod:`.zss`, :mod:`.score`, :mod:`.timeout`) depend only on
``sympy`` + stdlib; :mod:`.pipeline` adds the SEED dispatch that reuses the
comparison engine's parsing/unit primitives.

See :class:`prkit.scoring.PartialCreditScorer` for the ``Scorer`` wrapper that maps
:class:`EedResult` onto :class:`prkit.core.verdict.Verdict`.
"""

from __future__ import annotations

from .pipeline import EedConfig, EedResult, eed_compare
from .score import EditCosts, eed_score
from .tree import ExprNode, UnsupportedExpressionError, sympy_to_tree
from .zss import tree_edit_distance

__all__ = [
    "EditCosts",
    "EedConfig",
    "EedResult",
    "ExprNode",
    "UnsupportedExpressionError",
    "eed_compare",
    "eed_score",
    "sympy_to_tree",
    "tree_edit_distance",
]
