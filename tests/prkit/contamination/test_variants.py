"""Tests for X3 sub-feature C — parametric variant generation + verification."""

from __future__ import annotations

import pytest

from prkit.contamination.provenance import get_problem_provenance
from prkit.contamination.variants import (
    ParametricTemplate,
    ParamSpec,
    generate_variants,
    variants_to_dataset,
)
from prkit.core.domain.answer import PhysicsAnswer


def _consistent_template() -> ParametricTemplate:
    return ParametricTemplate(
        base_problem_id="kinetic",
        question_template="A mass of {m} kg accelerates at {g} m/s^2; find the force.",
        params={"m": ParamSpec(1, 5, integer=True), "g": ParamSpec(10, 10)},
        answer_fn=lambda p: p["m"] * p["g"],
        answer_expr="m * g",
        unit="N",
    )


class TestGenerateVariants:
    def test_produces_n_synthetic_variants(self) -> None:
        results = generate_variants(_consistent_template(), n=4, seed=7)
        assert len(results) == 4
        for r in results:
            prov = get_problem_provenance(r.problem)
            assert prov is not None
            assert prov.is_synthetic is True
            assert prov.parent_problem_id == "kinetic"
            assert isinstance(r.problem.answer, PhysicsAnswer)
            assert r.problem.answer.unit == "N"

    def test_consistent_template_verifies_true(self) -> None:
        results = generate_variants(_consistent_template(), n=3, seed=1)
        assert all(r.verified for r in results)
        assert all(r.verdict_detail is not None for r in results)
        assert all(r.verdict_detail.correct for r in results)  # type: ignore[union-attr]

    def test_broken_template_verifies_false(self) -> None:
        # answer_fn and answer_expr disagree by a constant → not self-consistent.
        broken = ParametricTemplate(
            base_problem_id="broken",
            question_template="mass {m} at {g}",
            params={"m": ParamSpec(2, 2), "g": ParamSpec(10, 10)},
            answer_fn=lambda p: p["m"] * p["g"],
            answer_expr="m * g + 1",
        )
        results = generate_variants(broken, n=2, seed=3)
        assert all(not r.verified for r in results)
        assert all(r.verdict_detail is not None for r in results)

    def test_deterministic_for_fixed_seed(self) -> None:
        a = generate_variants(_consistent_template(), n=3, seed=42)
        b = generate_variants(_consistent_template(), n=3, seed=42)
        assert [r.params for r in a] == [r.params for r in b]
        assert [r.problem.question for r in a] == [r.problem.question for r in b]

    def test_question_substitution_swaps_numbers(self) -> None:
        results = generate_variants(_consistent_template(), n=3, seed=11)
        for r in results:
            assert (
                str(int(r.params["m"])) in r.problem.question
                or str(r.params["m"]) in r.problem.question
            )

    def test_single_path_template_is_verified_true_without_verdict(self) -> None:
        single = ParametricTemplate(
            base_problem_id="single",
            question_template="value {x}",
            params={"x": ParamSpec(1, 3, integer=True)},
            answer_expr="x",
        )
        results = generate_variants(single, n=2, seed=5)
        assert all(r.verified for r in results)
        assert all(r.verdict_detail is None for r in results)

    def test_requires_a_derivation_path(self) -> None:
        with pytest.raises(ValueError, match="answer_fn / answer_expr"):
            generate_variants(
                ParametricTemplate(
                    base_problem_id="x",
                    question_template="{a}",
                    params={"a": ParamSpec(1, 1)},
                ),
                n=1,
            )


class TestVariantsToDataset:
    def test_packages_into_eval_dataset(self) -> None:
        results = generate_variants(_consistent_template(), n=3, seed=9)
        dataset = variants_to_dataset(results, name="kinetic_variants")
        assert len(dataset) == 3
        assert dataset.split == "eval"
        assert dataset.get_info()["name"] == "kinetic_variants"
        assert get_problem_provenance(dataset[0]) is not None
