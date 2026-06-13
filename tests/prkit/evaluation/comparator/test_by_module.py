"""Tests for :mod:`prkit.evaluation.comparator.by_module`."""

from __future__ import annotations

import pytest

from prkit.evaluation.comparator.by_module import (
    build_comparator,
    comparator_module_names,
    resolve_comparator_module,
)
from prkit.evaluation.comparator.category_match import CategoryComparator
from prkit.evaluation.comparator.exact_match import ExactMatchComparator
from prkit.evaluation.comparator.record_match import RecordMatchComparator
from prkit.evaluation.comparator.smart_match import SmartMatchComparator
from prkit.evaluation.comparator.typed_llm import TypedLLMComparator


def test_comparator_module_names_sorted_unique() -> None:
    names = comparator_module_names()
    assert names == tuple(sorted(names))
    assert len(names) == len(set(names))


def test_resolve_llm_judge_alias() -> None:
    assert resolve_comparator_module("llm_judge") == "typed_llm"


def test_build_exact_match() -> None:
    c = build_comparator("exact_match")
    assert isinstance(c, ExactMatchComparator)


def test_build_typed_llm_uses_model() -> None:
    c = build_comparator("typed_llm", model="gpt-4.1-mini")
    assert isinstance(c, TypedLLMComparator)
    assert c.model_name == "gpt-4.1-mini"


def test_build_category_match() -> None:
    c = build_comparator("category_match")
    assert isinstance(c, CategoryComparator)


def test_build_smart_match() -> None:
    c = build_comparator("smart_match")
    assert isinstance(c, SmartMatchComparator)


def test_build_record_match() -> None:
    c = build_comparator("record_match")
    assert isinstance(c, RecordMatchComparator)


def test_unknown_module_raises() -> None:
    with pytest.raises(ValueError, match="Unknown comparator"):
        build_comparator("not_a_real_comparator")
