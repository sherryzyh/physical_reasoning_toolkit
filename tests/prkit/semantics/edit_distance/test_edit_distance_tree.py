"""Unit tests for the SymPy -> ExprNode tree builder (:mod:`...edit_distance.tree`)."""

from __future__ import annotations

import pytest
import sympy as sp

from prkit.semantics.edit_distance.tree import (
    ExprNode,
    UnsupportedExpressionError,
    sympy_to_tree,
)


def _labels(node: ExprNode) -> set[str]:
    """Collect every label in a tree."""
    labels = {node.label}
    for child in node.children:
        labels |= _labels(child)
    return labels


class TestNodeTyping:
    def test_integer_symbol_operator(self) -> None:
        tree = sympy_to_tree(sp.sympify("2*m*g"))
        assert tree.label == "operator_Mul"
        assert "number_2" in _labels(tree)
        assert "symbol_m" in _labels(tree)
        assert "symbol_g" in _labels(tree)

    def test_negative_integer_is_single_number_node(self) -> None:
        tree = sympy_to_tree(sp.Integer(-3))
        assert tree.label == "number_-3"
        assert tree.children == []

    def test_rational_label(self) -> None:
        assert sympy_to_tree(sp.Rational(1, 2)).label == "number_1/2"

    def test_float_label_is_precision_bounded(self) -> None:
        # str()/srepr would leak 3.1400000000000001; we want a stable 3.14.
        assert sympy_to_tree(sp.Float("3.14")).label == "number_3.14"

    def test_number_symbols(self) -> None:
        assert sympy_to_tree(sp.pi).label == "number_Pi"
        assert sympy_to_tree(sp.E).label == "number_Exp1"

    def test_named_and_undefined_functions(self) -> None:
        x = sp.Symbol("x")
        assert sympy_to_tree(sp.sin(x)).label == "function_sin"
        assert sympy_to_tree(sp.Function("f")(x)).label == "function_f"

    def test_pow_keeps_base_exponent_order(self) -> None:
        tree = sympy_to_tree(sp.sympify("v0**2"))
        assert tree.label == "operator_Pow"
        assert [child.label for child in tree.children] == ["symbol_v0", "number_2"]


class TestDeterminism:
    def test_commutative_reorder_yields_identical_tree(self) -> None:
        a = sympy_to_tree(sp.sympify("x + y"))
        b = sympy_to_tree(sp.sympify("y + x"))
        assert _serialize(a) == _serialize(b)

    def test_repeated_builds_are_identical(self) -> None:
        expr = sp.sympify("2*m*g + 2*m*v0**2/l")
        assert _serialize(sympy_to_tree(expr)) == _serialize(sympy_to_tree(expr))


class TestNodeCount:
    def test_counts_all_nodes(self) -> None:
        # Mul(2, g, m): root + 3 leaves == 4 nodes.
        assert sympy_to_tree(sp.sympify("2*m*g")).node_count() == 4

    def test_leaf_count_is_one(self) -> None:
        assert sympy_to_tree(sp.Symbol("x")).node_count() == 1


class TestUnsupported:
    @pytest.mark.parametrize(
        "expr",
        [
            sp.Integral(sp.Symbol("x"), sp.Symbol("x")),
            sp.Derivative(sp.Function("f")(sp.Symbol("x")), sp.Symbol("x")),
            sp.Sum(sp.Symbol("x"), (sp.Symbol("x"), 1, 3)),
            sp.Matrix([[1, 2]]),
            sp.Eq(sp.Symbol("x"), sp.Symbol("y")),
        ],
    )
    def test_unsupported_nodes_raise(self, expr: sp.Basic) -> None:
        with pytest.raises(UnsupportedExpressionError):
            sympy_to_tree(expr)


def _serialize(node: ExprNode) -> str:
    """Stable string form of a tree, for equality assertions."""
    inner = ",".join(_serialize(child) for child in node.children)
    return f"{node.label}({inner})"
