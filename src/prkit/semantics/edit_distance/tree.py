"""Labeled comparison trees built from SymPy expressions (PHYBench EED grammar).

``sympy_to_tree`` mirrors PHYBench ``EED.py``'s ``sympy_to_tree``: every node gets
a ``"{type}_{value}"`` label with ``type`` in ``{number, symbol, operator,
function}``. Children of commutative operators (``Add``/``Mul``) are sorted by
:func:`sympy.default_sort_key` so the tree is deterministic across SymPy versions
and processes; non-commutative arguments (``Pow`` base/exponent, function
positional args) keep their order.

The grammar is deliberately closed: anything outside the additive / multiplicative
/ power / function vocabulary (``Integral``, ``Sum``, ``Derivative``, matrices,
relations, ...) raises :class:`UnsupportedExpressionError` so the pipeline can
degrade to a binary verdict rather than silently mis-scoring.

This module imports only ``sympy`` + stdlib so the pure algorithm stays reusable
without dragging in the comparison engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sympy import (
    Add,
    Float,
    Integer,
    Mul,
    NumberSymbol,
    Pow,
    Rational,
    Symbol,
    default_sort_key,
)
from sympy.core.numbers import (
    ComplexInfinity,
    Infinity,
    NaN,
    NegativeInfinity,
)


class UnsupportedExpressionError(TypeError):
    """Raised when :func:`sympy_to_tree` meets a node outside the EED grammar."""


@dataclass
class ExprNode:
    """One node of an EED comparison tree.

    Attributes:
        label: ``"{type}_{value}"`` where ``type`` is one of ``number`` /
            ``symbol`` / ``operator`` / ``function`` (e.g. ``"number_2"``,
            ``"symbol_m"``, ``"operator_Add"``, ``"function_sin"``).
        children: ordered child nodes.
    """

    label: str
    children: list[ExprNode] = field(default_factory=list)

    @property
    def node_type(self) -> str:
        """The ``type`` prefix of :attr:`label` (text before the first ``_``)."""
        return self.label.split("_", 1)[0]

    def node_count(self) -> int:
        """Number of nodes in the subtree rooted here (self included)."""
        return 1 + sum(child.node_count() for child in self.children)


def _float_key(value: Any) -> str:
    """Precision-bounded, deterministic string for a SymPy ``Float`` label.

    ``str(Float("3.14"))`` / ``srepr`` leak binary-float noise (``3.1400000000000001``),
    which would make scores depend on print formatting. Rounding to 12 significant
    figures keeps physically meaningful precision while staying stable.
    """
    return format(float(value), ".12g")


def _node_label(expr: Any) -> str:
    """Return the ``"{type}_{value}"`` label for a single SymPy node."""
    # Order matters: ``Integer`` is a subclass of ``Rational``; check it first.
    if isinstance(expr, Symbol):
        return f"symbol_{expr.name}"
    if isinstance(expr, Integer):
        return f"number_{int(expr)}"
    if isinstance(expr, Rational):
        return f"number_{expr.p}/{expr.q}"
    if isinstance(expr, Float):
        return f"number_{_float_key(expr)}"
    if isinstance(expr, NumberSymbol):
        # pi -> Pi, E -> Exp1, GoldenRatio, EulerGamma, Catalan, ...
        return f"number_{type(expr).__name__}"
    if isinstance(expr, (Infinity, NegativeInfinity, ComplexInfinity, NaN)):
        return f"number_{type(expr).__name__}"
    if isinstance(expr, (Add, Mul, Pow)):
        return f"operator_{type(expr).__name__}"
    if getattr(expr, "is_Function", False):
        # sin/cos/exp/log (named) and AppliedUndef f(x) both report is_Function.
        return f"function_{type(expr).__name__}"
    raise UnsupportedExpressionError(
        f"unsupported SymPy node for EED tree: {type(expr).__name__} ({expr!r})"
    )


def _ordered_children(expr: Any) -> list[Any]:
    """Return child args in a deterministic order.

    Commutative operators (``Add`` / ``Mul``) are sorted by
    :func:`sympy.default_sort_key`; everything else keeps positional order so
    ``Pow`` base/exponent and function arguments stay meaningful.
    """
    if isinstance(expr, (Add, Mul)):
        return sorted(expr.args, key=default_sort_key)
    return list(expr.args)


def sympy_to_tree(expr: Any) -> ExprNode:
    """Convert a SymPy expression into a deterministic labeled :class:`ExprNode`.

    Raises:
        UnsupportedExpressionError: on any node outside the EED grammar
            (calculus operators, matrices, relations, sets, ...).
    """
    label = _node_label(expr)
    children = [sympy_to_tree(arg) for arg in _ordered_children(expr)]
    return ExprNode(label=label, children=children)


__all__ = ["ExprNode", "UnsupportedExpressionError", "sympy_to_tree"]
