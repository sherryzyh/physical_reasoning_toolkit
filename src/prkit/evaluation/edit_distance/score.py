"""Edit costs, subtree cluster discount, and the EED score map (PHYBench parity).

These are the tunable numeric pieces of the algorithm, kept separate from the
tree builder (:mod:`.tree`) and the dynamic program (:mod:`.zss`) so each layer is
independently unit-testable.

Defaults reproduce PHYBench EED: per-type insert/delete/update cost ``1.0``,
``change_type_cost`` ``1.0``, subtree-discount ``bar_size=5`` /
``discount_slope=0.6``, and ``score = max(0, 0.6 - distance/gt_size)`` (returned
as a ``[0, 1]`` fraction rather than ``0..100``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .tree import ExprNode

_TYPES = ("number", "symbol", "operator", "function")


def _unit_costs() -> dict[str, float]:
    """Return a fresh per-type cost map with every node type costing ``1.0``."""
    return {node_type: 1.0 for node_type in _TYPES}


@dataclass(frozen=True)
class EditCosts:
    """Per-type insert/delete/update costs plus subtree-discount parameters.

    Attributes:
        insert_cost / delete_cost / update_cost: maps from node ``type`` prefix
            (``number`` / ``symbol`` / ``operator`` / ``function``) to cost.
        change_type_cost: relabel cost when two nodes have different ``type``
            prefixes (e.g. ``symbol`` -> ``number``).
        bar_size: subtree size below which no cluster discount applies.
        discount_slope: marginal cost per node above ``bar_size`` (and, per
            PHYBench, the score intercept in :func:`eed_score`).
    """

    insert_cost: dict[str, float] = field(default_factory=_unit_costs)
    delete_cost: dict[str, float] = field(default_factory=_unit_costs)
    update_cost: dict[str, float] = field(default_factory=_unit_costs)
    change_type_cost: float = 1.0
    bar_size: int = 5
    discount_slope: float = 0.6


def insert_cost(node: ExprNode, costs: EditCosts) -> float:
    """Cost of inserting a single ``node``."""
    return costs.insert_cost[node.node_type]


def delete_cost(node: ExprNode, costs: EditCosts) -> float:
    """Cost of deleting a single ``node``."""
    return costs.delete_cost[node.node_type]


def update_cost(a: ExprNode, b: ExprNode, costs: EditCosts) -> float:
    """Relabel cost: ``0`` if labels match, per-type cost if the type prefix
    matches, else :attr:`EditCosts.change_type_cost`."""
    if a.label == b.label:
        return 0.0
    if a.node_type == b.node_type:
        return costs.update_cost[a.node_type]
    return costs.change_type_cost


def subtree_discount(total_cost: float, costs: EditCosts) -> float:
    """Discounted cost of inserting/removing a whole subtree.

    ``total_cost`` is the subtree's full per-node insert (or delete) cost. Under
    PHYBench's unit costs this equals the node count, so the curve
    ``min(total_cost, discount_slope*(total_cost - bar_size) + bar_size)`` matches
    upstream exactly; with custom costs it generalizes the discount to the cost
    scale. For ``total_cost <= bar_size`` there is no discount; larger subtrees cost
    less than deleting them node by node. Insert and remove share this function
    (PHYBench ``remove_tree_func == insert_tree_func``).
    """
    discounted = costs.discount_slope * (total_cost - costs.bar_size) + costs.bar_size
    return min(total_cost, discounted)


def eed_score(distance: float, gt_size: int, *, discount_slope: float = 0.6) -> float:
    """Map a tree-edit ``distance`` to a ``[0, 1]`` partial-credit fraction.

    ``1.0`` when ``distance == 0``; otherwise ``max(0, discount_slope -
    distance/gt_size)`` (PHYBench ``score_calc`` rendered as a fraction). The score
    reaches ``0`` once the relative distance exceeds ``discount_slope`` (``0.6``).
    A non-positive ``gt_size`` is degenerate and scores ``0``.
    """
    if gt_size <= 0:
        return 0.0
    if distance == 0:
        return 1.0
    return max(0.0, discount_slope - distance / gt_size)


__all__ = [
    "EditCosts",
    "delete_cost",
    "eed_score",
    "insert_cost",
    "subtree_discount",
    "update_cost",
]
