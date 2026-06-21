"""Pure Expression Edit Distance (EED / SEED) algorithm core — related-work methods.

A self-contained reimplementation of PHYBench's Expression Edit Distance and
CMPhysBench's Scalable EED *tree-edit* machinery. These modules are deliberately
**free of any PRKit semantics dependency** — they import only ``sympy`` + stdlib:

* :mod:`.tree`    — SymPy expression → :class:`ExprNode` tree builder
* :mod:`.zss`     — extended Zhang-Shasha tree edit distance
* :mod:`.score`   — EED cost model and distance → score map
* :mod:`.timeout` — bounded ``simplify`` helper

The physics-aware *dispatch* that runs these on PRKit's normalized semantics (the
SEED answer-kind/unit/symbolic glue) lives separately in
:mod:`prkit.semantics.edit_distance.pipeline` (``eed_compare``), which imports this
core. Keeping the algorithm here means the related-work method has no dependency on
PRKit's own semantics layer.
"""

from __future__ import annotations

from .score import EditCosts, eed_score
from .timeout import SimplifyTimeout, run_with_timeout
from .tree import ExprNode, UnsupportedExpressionError, sympy_to_tree
from .zss import tree_edit_distance

__all__ = [
    "EditCosts",
    "ExprNode",
    "SimplifyTimeout",
    "UnsupportedExpressionError",
    "eed_score",
    "run_with_timeout",
    "sympy_to_tree",
    "tree_edit_distance",
]
