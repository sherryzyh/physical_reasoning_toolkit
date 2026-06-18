"""Unit tests for the EED cost model and score map (:mod:`...edit_distance.score`)."""

from __future__ import annotations

from prkit.semantics.edit_distance.score import (
    EditCosts,
    delete_cost,
    eed_score,
    insert_cost,
    subtree_discount,
    update_cost,
)
from prkit.semantics.edit_distance.tree import ExprNode


class TestEditCosts:
    def test_default_unit_costs(self) -> None:
        costs = EditCosts()
        for node_type in ("number", "symbol", "operator", "function"):
            assert costs.insert_cost[node_type] == 1.0
            assert costs.delete_cost[node_type] == 1.0
            assert costs.update_cost[node_type] == 1.0
        assert costs.change_type_cost == 1.0
        assert costs.bar_size == 5
        assert costs.discount_slope == 0.6

    def test_cost_maps_are_independent_instances(self) -> None:
        a = EditCosts()
        b = EditCosts()
        a.insert_cost["number"] = 99.0
        assert b.insert_cost["number"] == 1.0


class TestNodeCosts:
    def test_insert_and_delete_use_node_type(self) -> None:
        costs = EditCosts()
        node = ExprNode("symbol_x")
        assert insert_cost(node, costs) == 1.0
        assert delete_cost(node, costs) == 1.0

    def test_update_identical_labels_is_zero(self) -> None:
        costs = EditCosts()
        assert update_cost(ExprNode("number_2"), ExprNode("number_2"), costs) == 0.0

    def test_update_same_type_uses_type_cost(self) -> None:
        costs = EditCosts()
        assert update_cost(ExprNode("number_2"), ExprNode("number_3"), costs) == 1.0

    def test_update_different_type_uses_change_type_cost(self) -> None:
        costs = EditCosts(change_type_cost=7.0)
        assert update_cost(ExprNode("number_2"), ExprNode("symbol_x"), costs) == 7.0


class TestSubtreeDiscount:
    def test_small_subtrees_get_no_discount(self) -> None:
        costs = EditCosts()
        assert subtree_discount(1, costs) == 1.0
        assert subtree_discount(3, costs) == 3.0
        assert subtree_discount(5, costs) == 5.0

    def test_large_subtrees_are_discounted(self) -> None:
        costs = EditCosts()
        # 0.6 * (15 - 5) + 5 == 11 < 15
        assert subtree_discount(15, costs) == 11.0
        assert subtree_discount(15, costs) < 15.0


class TestEedScore:
    def test_exact_match_is_one(self) -> None:
        assert eed_score(0, 13) == 1.0

    def test_single_edit_formula(self) -> None:
        assert eed_score(1, 13) == 0.6 - 1 / 13

    def test_clamped_to_zero_beyond_threshold(self) -> None:
        assert eed_score(20, 10) == 0.0

    def test_degenerate_gt_size_is_zero(self) -> None:
        assert eed_score(5, 0) == 0.0
        assert eed_score(0, 0) == 0.0

    def test_score_in_unit_interval(self) -> None:
        for distance in range(0, 25):
            value = eed_score(distance, 12)
            assert 0.0 <= value <= 1.0
