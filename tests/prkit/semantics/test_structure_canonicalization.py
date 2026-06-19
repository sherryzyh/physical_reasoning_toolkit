"""Structure canonicalization: degeneracy collapses, idempotence, precision guards.

These assert the (degenerate, canonical) pairs collapse to equivalent and that the collapses
are idempotent and precision-safe (they never collapse a non-degenerate structure).
"""

from __future__ import annotations

import pytest

from prkit.semantics import (
    coerce_protocol_answer,
    coerce_question_semantics,
    compare_protocol_answers,
)
from prkit.semantics.comparison.structure_canonicalization import canonicalize_structure
from prkit.semantics.schema.enums import AnswerStructure

_CTX = coerce_question_semantics({})


def _num(value: float) -> dict:
    return {
        "object_kind": "number",
        "structure": "atomic",
        "numeric_value": float(value),
        "numeric_text": str(value),
        "canonical_text": str(value),
    }


def _true() -> dict:
    return {
        "object_kind": "boolean",
        "structure": "atomic",
        "boolean_value": True,
        "canonical_text": "True",
    }


def _one_tuple(value: float) -> dict:
    return {
        "object_kind": "number",
        "structure": "tuple",
        "children": [_num(value)],
        "canonical_text": f"({value})",
    }


def _one_set(value: float) -> dict:
    return {
        "object_kind": "number",
        "structure": "set",
        "children": [_num(value)],
        "canonical_text": f"{{{value}}}",
    }


def _one_vector(value: float) -> dict:
    return {
        "object_kind": "number",
        "structure": "vector",
        "shape": (1,),
        "children": [_num(value)],
        "canonical_text": f"<{value}>",
    }


def _point_interval(value: float, *, open_left=False, open_right=False) -> dict:
    return {
        "object_kind": "number",
        "structure": "interval",
        "children": [_num(value), _num(value)],
        "interval_open_left": open_left,
        "interval_open_right": open_right,
        "canonical_text": f"[{value}, {value}]",
    }


def _single_case_piecewise(value: float, condition: dict) -> dict:
    return {
        "object_kind": "number",
        "structure": "piecewise",
        "canonical_text": str(value),
        "cases": [{"expression": _num(value), "condition": condition}],
    }


# --- collapses reduce to atomic and compare equivalent to the bare value ---


@pytest.mark.parametrize(
    "degenerate",
    [
        _one_tuple(5),
        _one_set(5),
        _one_vector(5),
        _point_interval(5),
        _single_case_piecewise(5, _true()),
        _single_case_piecewise(
            5,
            {
                "object_kind": "qualitative_label",
                "structure": "atomic",
                "canonical_text": "otherwise",
            },
        ),
    ],
    ids=[
        "1-tuple",
        "1-set",
        "1-vector",
        "[a,a]",
        "piecewise-True",
        "piecewise-otherwise",
    ],
)
def test_degeneracy_collapses_to_atomic(degenerate: dict) -> None:
    canon = canonicalize_structure(coerce_protocol_answer(degenerate), context=_CTX)
    assert canon.structure == AnswerStructure.ATOMIC
    assert compare_protocol_answers(degenerate, _num(5)).equivalent is True


def test_collapse_is_idempotent() -> None:
    for degenerate in (
        _one_tuple(7),
        _point_interval(7),
        _single_case_piecewise(7, _true()),
    ):
        once = canonicalize_structure(coerce_protocol_answer(degenerate), context=_CTX)
        twice = canonicalize_structure(once, context=_CTX)
        assert once == twice


# --- precision guards: non-degenerate structures must NOT collapse ---


def test_open_point_interval_does_not_collapse() -> None:
    # (a, a) / [a, a) / (a, a] denote the empty set, not the point a — must not collapse.
    for open_left, open_right in [(True, True), (True, False), (False, True)]:
        canon = canonicalize_structure(
            coerce_protocol_answer(
                _point_interval(5, open_left=open_left, open_right=open_right)
            ),
            context=_CTX,
        )
        assert canon.structure == AnswerStructure.INTERVAL


def test_distinct_endpoint_interval_does_not_collapse() -> None:
    interval = {
        "object_kind": "number",
        "structure": "interval",
        "children": [_num(2), _num(3)],
        "interval_open_left": False,
        "interval_open_right": False,
        "canonical_text": "[2, 3]",
    }
    canon = canonicalize_structure(coerce_protocol_answer(interval), context=_CTX)
    assert canon.structure == AnswerStructure.INTERVAL


def test_multi_element_collection_does_not_collapse() -> None:
    two_tuple = {
        "object_kind": "number",
        "structure": "tuple",
        "children": [_num(1), _num(2)],
        "canonical_text": "(1, 2)",
    }
    canon = canonicalize_structure(coerce_protocol_answer(two_tuple), context=_CTX)
    assert canon.structure == AnswerStructure.TUPLE


def test_one_part_multi_part_does_not_collapse() -> None:
    # A 1-part multi_part may carry a part-structure denotation the contract enforces.
    one_part = {
        "object_kind": "number",
        "structure": "multi_part",
        "children": [_num(5)],
        "canonical_text": "5",
    }
    canon = canonicalize_structure(coerce_protocol_answer(one_part), context=_CTX)
    assert canon.structure == AnswerStructure.MULTI_PART
