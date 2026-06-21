"""Extended Zhang-Shasha tree-edit distance with subtree cluster discount.

A clean reimplementation of the ordered-tree edit distance (Zhang & Shasha 1989)
parametrized by :class:`~prkit.evaluation.edit_distance.score.EditCosts`, extended
with PHYBench's whole-subtree "cluster discount": deleting or inserting an entire
subtree can cost less than the per-node sum, so a large wrong sub-formula is not
penalized linearly.

Two deliberate divergences from PHYBench's ``extended_zss.py`` (documented for
cross-checking): the forest-distance matrix is initialized to ``math.inf`` rather
than the upstream sentinel ``1000`` (which silently mis-scores trees whose edit
distance exceeds the sentinel), and the algorithm is pure Python with no global
state so it is deterministic and thread-safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .score import (
    EditCosts,
    delete_cost,
    insert_cost,
    subtree_discount,
    update_cost,
)
from .tree import ExprNode


@dataclass
class _Annotated:
    """Post-order annotation of a tree used by the Zhang-Shasha DP."""

    order: list[ExprNode]  # post-order nodes; node at post-index ``i`` is order[i-1]
    left: dict[int, int]  # post-index -> leftmost-leaf post-index
    del_total: dict[int, float]  # post-index -> total delete cost of the subtree
    ins_total: dict[int, float]  # post-index -> total insert cost of the subtree
    keyroots: list[int]  # ascending keyroot post-indices
    n: int  # total node count


def _annotate(root: ExprNode, costs: EditCosts) -> _Annotated:
    """Compute post-order, leftmost-leaf indices, subtree costs, and keyroots."""
    order: list[ExprNode] = []
    left: dict[int, int] = {}
    del_total: dict[int, float] = {}
    ins_total: dict[int, float] = {}
    index_of: dict[int, int] = {}

    def visit(node: ExprNode) -> None:
        subtree_del = delete_cost(node, costs)
        subtree_ins = insert_cost(node, costs)
        for child in node.children:
            visit(child)
            child_idx = index_of[id(child)]
            subtree_del += del_total[child_idx]
            subtree_ins += ins_total[child_idx]
        order.append(node)
        idx = len(order)  # 1-indexed post-order position
        index_of[id(node)] = idx
        if node.children:
            left[idx] = left[index_of[id(node.children[0])]]
        else:
            left[idx] = idx
        del_total[idx] = subtree_del
        ins_total[idx] = subtree_ins

    visit(root)
    n = len(order)

    # keyroot(i): the largest post-index sharing leftmost-leaf left[i]. Iterating
    # ascending and overwriting keeps exactly that maximum per leftmost value.
    keyroot_by_left: dict[int, int] = {}
    for i in range(1, n + 1):
        keyroot_by_left[left[i]] = i
    keyroots = sorted(keyroot_by_left.values())

    return _Annotated(
        order=order,
        left=left,
        del_total=del_total,
        ins_total=ins_total,
        keyroots=keyroots,
        n=n,
    )


def _forest_distance(
    a: _Annotated,
    b: _Annotated,
    i1: int,
    j1: int,
    treedist: list[list[float]],
    costs: EditCosts,
) -> None:
    """Fill ``treedist`` for the subtree pair rooted at keyroots ``i1`` / ``j1``."""
    la, lb = a.left[i1], b.left[j1]
    rows = i1 - la + 2
    cols = j1 - lb + 2
    fd = [[math.inf] * cols for _ in range(rows)]
    fd[0][0] = 0.0

    for x in range(1, rows):
        node = a.order[la + x - 2]  # post-index (la + x - 1), 0-based list access
        fd[x][0] = fd[x - 1][0] + delete_cost(node, costs)
    for y in range(1, cols):
        node = b.order[lb + y - 2]
        fd[0][y] = fd[0][y - 1] + insert_cost(node, costs)

    for x in range(1, rows):
        i = la + x - 1  # actual post-index in A
        na = a.order[i - 1]
        xa = a.left[i] - la  # forest column just before subtree_i
        for y in range(1, cols):
            j = lb + y - 1
            nb = b.order[j - 1]
            yb = b.left[j] - lb

            del_node = fd[x - 1][y] + delete_cost(na, costs)
            ins_node = fd[x][y - 1] + insert_cost(nb, costs)
            # Whole-subtree discount options: drop subtree_i / add subtree_j as a
            # unit at the (never-larger) discounted price.
            rem_tree = fd[xa][y] + subtree_discount(a.del_total[i], costs)
            ins_tree = fd[x][yb] + subtree_discount(b.ins_total[j], costs)

            if a.left[i] == la and b.left[j] == lb:
                upd = fd[x - 1][y - 1] + update_cost(na, nb, costs)
                best = min(del_node, ins_node, upd, rem_tree, ins_tree)
                fd[x][y] = best
                treedist[i][j] = best
            else:
                match = fd[xa][yb] + treedist[i][j]
                fd[x][y] = min(del_node, ins_node, match, rem_tree, ins_tree)


def tree_edit_distance(a: ExprNode, b: ExprNode, *, costs: EditCosts) -> float:
    """Return the extended Zhang-Shasha edit distance between trees ``a`` and ``b``."""
    ann_a = _annotate(a, costs)
    ann_b = _annotate(b, costs)
    # treedist is 1-indexed in both dimensions; row/col 0 are unused padding.
    treedist = [[0.0] * (ann_b.n + 1) for _ in range(ann_a.n + 1)]
    for i1 in ann_a.keyroots:
        for j1 in ann_b.keyroots:
            _forest_distance(ann_a, ann_b, i1, j1, treedist, costs)
    return treedist[ann_a.n][ann_b.n]


__all__ = ["tree_edit_distance"]
