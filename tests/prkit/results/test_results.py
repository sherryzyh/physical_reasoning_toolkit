"""Tests for the X4 results store + PhysicsEvalResult schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from prkit.core.domain import PhysicsProblem, PhysicsSolution
from prkit.core.verdict import Verdict
from prkit.cost import CallRecord, TokenUsage
from prkit.results import (
    DatasetProvenance,
    ModelProvenance,
    PhysicsEvalResult,
    ResultStore,
    ScorerProvenance,
    UsageAndCost,
)


def _verdict(equivalent: bool = True) -> Verdict:
    return Verdict(
        equivalent=equivalent,
        score=1.0 if equivalent else 0.0,
        comparison_mode="physical_quantity",
        scorer_version="semantics/1",
        units_ok=True,
        symbolic_equiv=None,
        diagnostics=("unit_match", "numeric_within_tol"),
        details={"bridge_id": "implicit_unit_alias"},
    )


def _model() -> ModelProvenance:
    return ModelProvenance(provider="openai", model_name="gpt-5.1", backend="openai")


def _scorer() -> ScorerProvenance:
    return ScorerProvenance(scorer_name="SemanticsScorer", scorer_version="semantics/1")


def _dataset() -> DatasetProvenance:
    return DatasetProvenance(dataset_name="phybench", split="train")


def _result(**overrides: object) -> PhysicsEvalResult:
    base: dict[str, object] = dict(
        problem_id="p1",
        raw_response="The acceleration is 9.81 m/s^2.",
        extracted_answer="9.81 m/s^2",
        verdict=_verdict(),
        correct=True,
        score=1.0,
        model=_model(),
        scorer=_scorer(),
        dataset=_dataset(),
        usage=UsageAndCost(input_tokens=100, output_tokens=50, cost_usd=0.001),
    )
    base.update(overrides)
    return PhysicsEvalResult(**base)  # type: ignore[arg-type]


class TestSchema:
    def test_jsonl_round_trip_with_verdict(self) -> None:
        record = _result()
        restored = PhysicsEvalResult.from_jsonl_line(record.to_jsonl_line())
        assert restored == record
        assert restored.verdict is not None
        assert restored.verdict.comparison_mode == "physical_quantity"
        assert restored.verdict.details == {"bridge_id": "implicit_unit_alias"}

    def test_schema_version_major_guard(self) -> None:
        tampered = (
            _result()
            .to_jsonl_line()
            .replace("prkit.eval_result/1", "prkit.eval_result/2")
        )
        with pytest.raises(ValueError, match="Unsupported result schema_version"):
            PhysicsEvalResult.from_jsonl_line(tampered)

    def test_carries_all_required_facets(self) -> None:
        r = _result()
        assert r.model.model_name == "gpt-5.1"
        assert r.scorer.scorer_version == "semantics/1"
        assert r.dataset.dataset_name == "phybench"
        assert r.usage.cost_usd == 0.001
        assert r.contaminated is None  # unknown by default
        assert r.result_id  # auto uuid

    def test_failed_row_with_error_and_no_verdict(self) -> None:
        r = _result(verdict=None, correct=None, score=None, error="timeout")
        restored = PhysicsEvalResult.from_jsonl_line(r.to_jsonl_line())
        assert restored.error == "timeout"
        assert restored.verdict is None


class TestUsageBridge:
    def test_from_call_record(self) -> None:
        record = CallRecord(
            provider="openai",
            model="gpt-5.1",
            usage=TokenUsage(input_tokens=100, output_tokens=50, reasoning_tokens=10),
            cost=0.00125,
        )
        usage = UsageAndCost.from_call_record(record)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.reasoning_tokens == 10
        assert usage.cost_usd == 0.00125
        assert usage.pricing_source == "openai:gpt-5.1"


class TestSolutionBridge:
    def test_from_solution_and_back(self) -> None:
        problem = PhysicsProblem(problem_id="p1", question="Find g.")
        solution = PhysicsSolution(
            problem_id="p1",
            problem=problem,
            agent_answer="9.81 m/s^2",
            metadata={"question": "Find g."},
        )
        result = PhysicsEvalResult.from_solution(
            solution,
            model=_model(),
            scorer=_scorer(),
            dataset=_dataset(),
            verdict=_verdict(),
        )
        assert result.problem_id == "p1"
        assert result.raw_response == "9.81 m/s^2"
        assert result.correct is True
        # reverse (lossy) bridge
        back = result.to_solution()
        assert back.problem_id == "p1"
        assert back.agent_answer == "9.81 m/s^2"


class TestResultStore:
    def test_append_iter_len_query(self, tmp_path: Path) -> None:
        store = ResultStore(tmp_path / "results.jsonl")
        store.append(_result(problem_id="a", run_id="r1"))
        store.append(
            _result(problem_id="b", run_id="r1", correct=False, verdict=_verdict(False))
        )
        assert len(store) == 2
        ids = sorted(r.problem_id for r in store)
        assert ids == ["a", "b"]
        assert len(store.query(correct=True)) == 1
        assert len(store.query(run_id="r1")) == 2
        assert len(store.query(model_name="gpt-5.1")) == 2
        assert len(store.query(dataset_name="nope")) == 0

    def test_jsonl_is_append_only_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "r.jsonl"
        store = ResultStore(path)
        store.extend([_result(problem_id="a"), _result(problem_id="b")])
        assert path.read_text(encoding="utf-8").strip().count("\n") == 1  # two lines

    def test_parquet_round_trip_mixed_records(self, tmp_path: Path) -> None:
        store = ResultStore()
        store.append(_result(problem_id="a"))
        store.append(
            _result(
                problem_id="b", verdict=None, correct=None, score=None, error="boom"
            )
        )
        store.append(
            _result(problem_id="c", usage=UsageAndCost(input_tokens=1, output_tokens=1))
        )
        out = store.to_parquet(tmp_path / "r.parquet")
        reloaded = ResultStore.from_parquet(out)
        by_id = {r.problem_id: r for r in reloaded}
        assert set(by_id) == {"a", "b", "c"}
        assert by_id["a"].verdict is not None
        assert by_id["b"].error == "boom" and by_id["b"].verdict is None
        assert by_id["c"].usage.cost_usd is None

    def test_aggregate(self, tmp_path: Path) -> None:
        store = ResultStore(tmp_path / "r.jsonl")
        store.append(_result(problem_id="a", correct=True))
        store.append(_result(problem_id="b", correct=False, verdict=_verdict(False)))
        agg = store.aggregate(("dataset_name", "model_name"))
        assert len(agg) == 1
        row = agg.iloc[0]
        assert row["n"] == 2
        assert row["accuracy"] == pytest.approx(0.5)
        assert row["total_input_tokens"] == 200


class TestInspectBridge:
    def test_to_inspect_score_if_available(self) -> None:
        inspect_scorer = pytest.importorskip("inspect_ai.scorer")
        from prkit.results import to_inspect_score

        score = to_inspect_score(_result())
        assert isinstance(score, inspect_scorer.Score)
        assert score.value is True
        assert score.answer == "9.81 m/s^2"
