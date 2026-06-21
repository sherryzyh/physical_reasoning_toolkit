"""Tests for the model-graded LLMJudgeScorer.

No network and no OpenAI client: every scoring test injects a ``FakeJudgeRunner``
via ``LLMJudgeScorer(runner=...)``. The one default-runner test stubs the OpenAI
client class so construction needs neither an API key nor a live connection.
"""

from __future__ import annotations

import json

import pytest

from prkit.api import Scorer, Verdict
from prkit.core.domain.answer import Answer
from prkit.evaluation.llm_judge.types import LLMJudgeResult
from prkit.scoring import LLMJudgeScorer


class FakeJudgeRunner:
    """Deterministic stand-in for OpenAIJudgeRunner (no OpenAI client / network)."""

    def __init__(
        self, result: LLMJudgeResult, *, model: str = "fake-judge-model"
    ) -> None:
        self._result = result
        self._model_name = model
        self.calls: list[dict] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def judge(self, payload: dict) -> LLMJudgeResult:
        self.calls.append(payload)
        return self._result


def _result(verdict: str, **overrides) -> LLMJudgeResult:
    base = dict(
        verdict=verdict,
        confidence=0.9,
        expected_answer_type="numeric",
        reasoning="because the magnitudes agree",
        raw_response=json.dumps({"verdict": verdict}),
        verdict_type="llm_judge",
    )
    base.update(overrides)
    return LLMJudgeResult(**base)  # type: ignore[arg-type]


def _scorer(runner: FakeJudgeRunner) -> LLMJudgeScorer:
    return LLMJudgeScorer(model="gpt-judge", runner=runner)


class TestProtocolConformance:
    def test_satisfies_scorer_protocol(self):
        assert isinstance(_scorer(FakeJudgeRunner(_result("correct"))), Scorer)

    def test_version_non_empty(self):
        assert isinstance(LLMJudgeScorer.version, str) and LLMJudgeScorer.version

    def test_get_info_matches_version_and_reports_non_deterministic(self):
        s = _scorer(FakeJudgeRunner(_result("correct")))
        info = s.get_info()
        assert info["version"] == s.version
        assert info["name"] == "LLMJudgeScorer"
        assert info["engine"] == "openai_judge"
        assert info["deterministic"] is False
        # Model is pulled from the runner, not the constructor's `model` arg.
        assert info["model"] == "fake-judge-model"


class TestScoring:
    def test_correct_verdict_maps_to_pass(self):
        v = _scorer(FakeJudgeRunner(_result("correct"))).score("3 m/s", "3.0 m/s")
        assert isinstance(v, Verdict)
        assert v.equivalent is True
        assert v.correct is True
        assert v.score == 1.0

    def test_incorrect_verdict_maps_to_fail(self):
        v = _scorer(FakeJudgeRunner(_result("incorrect"))).score("3 m/s", "5 m/s")
        assert v.equivalent is False
        assert v.correct is False
        assert v.score == 0.0

    def test_comparison_mode_embeds_expected_answer_type(self):
        runner = FakeJudgeRunner(_result("correct", expected_answer_type="expression"))
        v = _scorer(runner).score("x+1", "1+x")
        assert v.comparison_mode == "llm_judge:expression"

    def test_rationale_is_the_judge_reasoning(self):
        v = _scorer(FakeJudgeRunner(_result("correct"))).score("a", "a")
        assert v.rationale == "because the magnitudes agree"

    def test_partial_credit_is_none(self):
        v = _scorer(FakeJudgeRunner(_result("correct"))).score("a", "a")
        assert v.partial_credit is None

    def test_scorer_version_propagated(self):
        s = _scorer(FakeJudgeRunner(_result("correct")))
        assert s.score("a", "a").scorer_version == s.version

    def test_details_carry_judge_evidence_and_runner_model(self):
        runner = FakeJudgeRunner(_result("correct"), model="gpt-judge-xl")
        v = _scorer(runner).score("a", "a")
        assert v.details == {
            "confidence": 0.9,
            "expected_answer_type": "numeric",
            "raw_response": json.dumps({"verdict": "correct"}),
            "verdict_type": "llm_judge",
            "model": "gpt-judge-xl",
        }

    def test_details_json_serializable(self):
        v = _scorer(FakeJudgeRunner(_result("correct"))).score("a", "a")
        json.dumps(v.model_dump())

    def test_question_is_threaded_into_the_payload(self):
        runner = FakeJudgeRunner(_result("correct"))
        _scorer(runner).score("3 m/s", "3 m/s", question="What is the speed?")
        assert len(runner.calls) == 1
        assert runner.calls[0]["question"] == "What is the speed?"
        assert runner.calls[0]["model_answer"]["text"] == "3 m/s"
        assert runner.calls[0]["ground_truth"]["text"] == "3 m/s"

    def test_accepts_answer_objects(self):
        runner = FakeJudgeRunner(_result("correct"))
        pred = Answer(value="3.0", unit="m/s")
        ref = Answer(value="3", unit="m/s", source_type="NV")
        v = _scorer(runner).score(pred, ref)
        assert v.equivalent is True
        # Answer.source_type flows into the payload category.
        assert runner.calls[0]["ground_truth"]["category"] == "NV"


class TestConstruction:
    def test_runner_can_be_injected_without_model(self):
        # Matches the §IV fixture form: LLMJudgeScorer(runner=fake).
        s = LLMJudgeScorer(runner=FakeJudgeRunner(_result("correct")))
        assert s.score("a", "a").equivalent is True

    def test_requires_model_when_no_runner_injected(self):
        with pytest.raises(ValueError, match="requires `model`"):
            LLMJudgeScorer()

    def test_default_runner_is_built_lazily_from_model(self, monkeypatch):
        # Stub the OpenAI client class so no API key / network is needed; this also
        # proves the lazy `from prkit.evaluation.llm_judge.runner import ...` path.
        pytest.importorskip("openai")
        import prkit.evaluation.llm_judge.runner as runner_mod

        monkeypatch.setattr(runner_mod, "OpenAI", lambda **kwargs: object())
        s = LLMJudgeScorer(model="gpt-default")
        info = s.get_info()
        assert info["model"] == "gpt-default"
        assert info["deterministic"] is False
