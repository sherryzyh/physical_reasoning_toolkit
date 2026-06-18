"""Unit tests for the extended Zhang-Shasha distance (:mod:`...edit_distance.zss`)."""

from __future__ import annotations

import sympy as sp

from prkit.semantics.edit_distance.score import EditCosts, eed_score
from prkit.semantics.edit_distance.tree import ExprNode, sympy_to_tree
from prkit.semantics.edit_distance.zss import tree_edit_distance

_COSTS = EditCosts()


def _dist(a: str, b: str) -> float:
    return tree_edit_distance(
        sympy_to_tree(sp.sympify(a)), sympy_to_tree(sp.sympify(b)), costs=_COSTS
    )


class TestBasics:
    def test_identical_trees_have_zero_distance(self) -> None:
        assert _dist("2*m*g", "2*m*g") == 0.0

    def test_commutative_equal_trees_zero(self) -> None:
        assert _dist("x + y", "y + x") == 0.0

    def test_single_leaf_relabel_costs_one(self) -> None:
        assert _dist("x", "y") == 1.0

    def test_single_coefficient_change(self) -> None:
        # One leaf differs (4 vs 2); distance is a single update.
        gold = sympy_to_tree(sp.sympify("2*m*g + 2*m*v0**2/l"))
        pred = sympy_to_tree(sp.sympify("2*m*g + 4*m*v0**2/l"))
        distance = tree_edit_distance(pred, gold, costs=_COSTS)
        assert distance == 1.0


class TestLiteratureExample:
    def test_phybench_near_miss_band(self) -> None:
        gold = sympy_to_tree(sp.sympify("2*m*g + 2*m*v0**2/l"))
        pred = sympy_to_tree(sp.sympify("2*m*g + 4*m*v0**2/l"))
        distance = tree_edit_distance(pred, gold, costs=_COSTS)
        score = eed_score(distance, gold.node_count())
        # PHYBench reports ~0.47 for this pair; assert a band (tree size is
        # parser-dependent), not an exact constant.
        assert 0.4 < score < 0.55

    def test_unrelated_expression_scores_zero(self) -> None:
        gold = sympy_to_tree(sp.sympify("2*m*g + 2*m*v0**2/l"))
        pred = sympy_to_tree(sp.sympify("z"))
        distance = tree_edit_distance(pred, gold, costs=_COSTS)
        assert eed_score(distance, gold.node_count()) == 0.0


class TestSubtreeDiscount:
    def test_large_subtree_swap_uses_discount(self) -> None:
        # gold = f(<10 distinct leaves>); pred replaces the whole argument subtree.
        # The discounted whole-subtree edit must beat deleting 10 nodes one by one.
        big = ExprNode(
            "operator_Add",
            [ExprNode(f"symbol_s{i}") for i in range(10)],
        )
        gold = ExprNode("function_f", [big])
        pred = ExprNode("function_f", [ExprNode("symbol_z")])
        distance = tree_edit_distance(pred, gold, costs=_COSTS)
        # Replacing 11 nodes (Add + 10 leaves) with one leaf: discounted, < 11.
        assert distance < 11.0


class TestInfInitRegression:
    def test_distance_above_sentinel_is_exact(self) -> None:
        # PHYBench's extended_zss inits the forest matrix to the sentinel 1000,
        # silently capping any distance above it. We init to math.inf, so a true
        # distance over 1000 must come through exactly. High per-node costs let a
        # tiny tree exceed the sentinel instantly (no slow giant-tree DP).
        big = 300.0
        types = ("number", "symbol", "operator", "function")
        costs = EditCosts(
            insert_cost={t: big for t in types},
            delete_cost={t: big for t in types},
            update_cost={t: big for t in types},
            change_type_cost=big,
        )
        # Same shape (root + 4 leaves), every label differs -> relabel all 5 nodes.
        a = ExprNode(
            "operator_Add",
            [ExprNode(f"symbol_a{i}") for i in range(4)],
        )
        b = ExprNode(
            "operator_Mul",
            [ExprNode(f"symbol_b{i}") for i in range(4)],
        )
        distance = tree_edit_distance(a, b, costs=costs)
        assert distance == 5 * big  # 1500
        assert distance > 1000.0


class TestDeterminism:
    def test_distance_is_stable(self) -> None:
        first = _dist("2*m*g + 4*m*v0**2/l", "2*m*g + 2*m*v0**2/l")
        second = _dist("2*m*g + 4*m*v0**2/l", "2*m*g + 2*m*v0**2/l")
        assert first == second
