"""Non-atomic comparison gating: only proven-sound accepts pass; the rest are TBD.

The equivalence judgement runs for a non-atomic structure only when it provably reduces to
atomic-vs-atomic element comparisons. Everything else returns the ``not_implemented`` (TBD)
sentinel by default, or raises ``NotImplementedError`` under the strict toggle.
"""

from __future__ import annotations

import pytest

from prkit.semantics import compare_protocol_answers
from prkit.semantics.comparison import engine


def _num(value, text=None):
    return {
        "object_kind": "number",
        "structure": "atomic",
        "numeric_value": float(value),
        "numeric_text": text if text is not None else str(value),
        "canonical_text": text if text is not None else str(value),
    }


def _coll(structure, children, **extra):
    return {
        "object_kind": "number",
        "structure": structure,
        "children": children,
        "canonical_text": "",
        **extra,
    }


def _vector(children, **extra):
    return _coll("vector", children, shape=[len(children)], **extra)


# --- ordered: tuple / vector with atomic cells are enabled ---


def test_tuple_atomic_exact_equivalent():
    r = compare_protocol_answers(
        _coll("tuple", [_num(1), _num(2)]), _coll("tuple", [_num(1), _num(2)])
    )
    assert r.equivalent is True and r.comparison_mode == "tuple"


def test_tuple_order_sensitive_real_reject():
    r = compare_protocol_answers(
        _coll("tuple", [_num(1), _num(2)]), _coll("tuple", [_num(2), _num(1)])
    )
    assert r.equivalent is False and r.comparison_mode == "tuple"


def test_tuple_non_atomic_element_is_tbd():
    nested = _coll("tuple", [_coll("tuple", [_num(1), _num(2)]), _num(3)])
    r = compare_protocol_answers(nested, nested)
    assert r.equivalent is False and r.comparison_mode == "not_implemented"


# --- unordered set: only exact multiset matches pass ---


def test_set_exact_match_equivalent():
    r = compare_protocol_answers(
        _coll("set", [_num(1), _num(2)]), _coll("set", [_num(2), _num(1)])
    )
    assert r.equivalent is True and r.comparison_mode == "set"


def test_set_exact_numeric_equivalence_half_vs_decimal():
    # 1/2 and 0.5 are exactly equal numbers (no tolerance) — accepted.
    r = compare_protocol_answers(
        _coll("set", [_num(0), _num(0.5, "1/2")]),
        _coll("set", [_num(0.5, "0.5"), _num(0)]),
    )
    assert r.equivalent is True


def test_set_tolerance_fuzz_is_tbd():
    # The classic non-transitive-tolerance false positive must NOT pass — it is TBD now.
    ctx = {"tolerance": 0.2}
    r = compare_protocol_answers(
        _coll("set", [_num(1.0), _num(1.0)]),
        _coll("set", [_num(1.0), _num(1.1)]),
        context=ctx,
    )
    assert r.equivalent is False and r.comparison_mode == "not_implemented"


def test_set_inexact_elements_is_tbd():
    # Two-element sets (a 1-element set would correctly collapse to its atom): 9.8 vs 9.81
    # are not exactly equal, so the set is TBD rather than tolerance-matched.
    r = compare_protocol_answers(
        _coll("set", [_num(9.8), _num(5)]), _coll("set", [_num(9.81), _num(5)])
    )
    assert r.equivalent is False and r.comparison_mode == "not_implemented"


# --- matrix / tensor are deferred (non-atomic cells) ---


def test_matrix_is_tbd():
    matrix = _coll(
        "matrix",
        [_vector([_num(1), _num(2)]), _vector([_num(3), _num(4)])],
        shape=[2, 2],
    )
    r = compare_protocol_answers(matrix, matrix)
    assert r.equivalent is False and r.comparison_mode == "not_implemented"


# --- vector frames ---


def test_vector_both_unset_frame_equivalent():
    r = compare_protocol_answers(
        _vector([_num(1), _num(2), _num(3)]), _vector([_num(1), _num(2), _num(3)])
    )
    assert r.equivalent is True and r.comparison_mode == "vector"


def test_vector_one_sided_frame_is_tbd():
    r = compare_protocol_answers(
        _vector([_num(1), _num(2)], coordinate_frame="x,y at center"),
        _vector([_num(1), _num(2)]),
    )
    assert r.equivalent is False and r.comparison_mode == "not_implemented"


def test_vector_incompatible_frames_real_reject():
    r = compare_protocol_answers(
        _vector([_num(1), _num(2)], coordinate_frame="polar r,theta"),
        _vector([_num(1), _num(2)], coordinate_frame="cartesian x,y"),
    )
    assert r.equivalent is False and r.comparison_mode == "vector"


# --- per_part: aligned labels only ---


def test_per_part_aligned_labels_equivalent():
    ctx = {"ordering": "per_part", "required_parts": ("a", "b")}
    pred = _coll(
        "multi_part",
        [
            {**_num(1), "part_label": "a"},
            {**_num(2), "part_label": "b"},
        ],
    )
    ref = _coll(
        "multi_part",
        [
            {**_num(2), "part_label": "b"},
            {**_num(1), "part_label": "a"},
        ],
    )
    r = compare_protocol_answers(pred, ref, context=ctx)
    assert r.equivalent is True and r.comparison_mode == "multi_part"


def test_per_part_unaligned_labels_is_tbd():
    ctx = {"ordering": "per_part"}
    pred = _coll("multi_part", [_num(1), _num(2)])
    ref = _coll("multi_part", [_num(1), _num(2)])
    r = compare_protocol_answers(pred, ref, context=ctx)
    assert r.equivalent is False and r.comparison_mode == "not_implemented"


# --- contract reconciliation: a collapsed structure stays admitted ---


@pytest.mark.parametrize("policy", ["strict", "audited", "permissive"])
def test_collapsed_structure_admitted_under_restricted_allowed_structures(policy):
    # A 1-tuple collapses to ATOMIC; with allowed_structures=(tuple,) both the collapsed
    # prediction and the atomic reference must stay admitted (no contract violation).
    one_tuple = _coll("tuple", [_num(5)])
    atomic = _num(5)
    ctx = {"allowed_structures": ["tuple"]}
    result = compare_protocol_answers(
        one_tuple, atomic, context=ctx, policy_mode=policy
    )
    assert result.comparison_mode not in {
        "contract_violation",
        "reference_contract_violation",
    }
    assert result.equivalent is True


def test_strict_mode_raises(monkeypatch):
    monkeypatch.setattr(engine, "STRICT_STRUCTURE_COMPARISON", True)
    # 2-element rows stay non-atomic (a 1-element row would collapse to its atom).
    matrix = _coll(
        "matrix",
        [_vector([_num(1), _num(2)]), _vector([_num(3), _num(4)])],
        shape=[2, 2],
    )
    with pytest.raises(NotImplementedError):
        compare_protocol_answers(matrix, matrix)
