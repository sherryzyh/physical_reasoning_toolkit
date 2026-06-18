"""Robustness tests: timeout helper, number constants, and no-simplify scoring."""

from __future__ import annotations

import time

import pytest
import sympy as sp

from prkit.semantics import normalize_physics_answer
from prkit.semantics.edit_distance import EedConfig, eed_compare, sympy_to_tree
from prkit.semantics.edit_distance.timeout import SimplifyTimeout, run_with_timeout


class TestTimeout:
    def test_returns_value_when_fast(self) -> None:
        assert run_with_timeout(lambda: 1 + 1, timeout_s=5.0) == 2

    def test_raises_on_slow_callable(self) -> None:
        with pytest.raises(SimplifyTimeout):
            run_with_timeout(lambda: time.sleep(2.0), timeout_s=0.05)


class TestNumberConstants:
    def test_infinities_and_constants(self) -> None:
        assert sympy_to_tree(sp.oo).label == "number_Infinity"
        assert sympy_to_tree(sp.S.NegativeInfinity).label == "number_NegativeInfinity"
        assert sympy_to_tree(sp.zoo).label == "number_ComplexInfinity"
        assert sympy_to_tree(sp.nan).label == "number_NaN"
        assert sympy_to_tree(sp.GoldenRatio).label == "number_GoldenRatio"


class TestNoSimplify:
    def test_scores_without_pre_simplify(self) -> None:
        cfg = EedConfig(simplify_before_tree=False)
        result = eed_compare(
            normalize_physics_answer("2*m*g + 4*m*v0**2/l"),
            normalize_physics_answer("2*m*g + 2*m*v0**2/l"),
            config=cfg,
        )
        assert 0.0 < result.score < 1.0
