"""
Smart-Match + LLM hybrid comparator.

Runs :func:`~prkit.prkit_evaluation.comparator.smart_pipeline.run_smart_pipeline`
(same-type, RHS, equation-from-text rescue, then cross-type matching).
If cross-type resolution is inconclusive (``None``), optionally calls the shared
LLM judge; otherwise the deterministic outcome is final.
"""

from __future__ import annotations

from typing import Any, Optional, Union, assert_never

from openai import OpenAI

from prkit.prkit_core.domain.answer import Answer
from prkit.prkit_evaluation.llm_judge import (
    DEFAULT_MODEL,
    LLMJudgeResult,
    OpenAIJudgeRunner,
    RESULT_SOURCE_SKIPPED_LLM,
    RESULT_SOURCE_SMART_MATCH,
    build_standard_answer_judge_payload,
)

from .smart_match import SmartMatchComparator
from .smart_pipeline import SmartPipelineResult, run_smart_pipeline


class SmartLLMComparator(SmartMatchComparator):
    """Deterministic SmartMatch pipeline; LLM judge only when cross-type is inconclusive."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        instructions: Optional[str] = None,
        client: Optional[OpenAI] = None,
    ) -> None:
        super().__init__()
        self._runner = OpenAIJudgeRunner(
            model=model,
            instructions=instructions,
            client=client,
            logger=self.logger,
        )
        self._last_result: Optional[LLMJudgeResult] = None

    def _result_smart(self, *, correct: bool) -> LLMJudgeResult:
        return LLMJudgeResult(
            verdict="correct" if correct else "incorrect",
            confidence=1.0,
            expected_answer_type="other",
            reasoning=(
                "SmartMatchComparator deterministic path "
                "(same-type, RHS extraction, equation-from-text rescue, or cross-type)."
            ),
            raw_response="local_smart_match",
            verdict_type=RESULT_SOURCE_SMART_MATCH,
        )

    def compare(
        self,
        answer1: Union[str, Answer],
        answer2: Union[str, Answer],
        *,
        skip_llm: bool = False,
        **kwargs: Any,
    ) -> bool:
        outcome: SmartPipelineResult = run_smart_pipeline(self, answer1, answer2)
        if outcome == "inconclusive":
            if skip_llm:
                self._last_result = LLMJudgeResult(
                    verdict="incorrect",
                    confidence=0.0,
                    expected_answer_type="other",
                    reasoning=(
                        "Cross-type matching returned no deterministic verdict; "
                        "LLM judge skipped (skip_llm=True)."
                    ),
                    raw_response="skipped_llm",
                    verdict_type=RESULT_SOURCE_SKIPPED_LLM,
                )
                return False

            payload = build_standard_answer_judge_payload(
                answer1,
                answer2,
                kwargs.get("question"),
            )
            result = self._runner.judge(payload)
            self._last_result = result
            return result.verdict == "correct"

        if outcome == "match":
            self._last_result = self._result_smart(correct=True)
            return True
        if outcome == "no_match":
            self._last_result = self._result_smart(correct=False)
            return False

        assert_never(outcome)

    def accuracy_score(
        self,
        answer1: Union[str, Answer],
        answer2: Union[str, Answer],
        *,
        skip_llm: bool = False,
        **kwargs: Any,
    ) -> float:
        ok = self.compare(answer1, answer2, skip_llm=skip_llm, **kwargs)
        return 1.0 if ok else 0.0

    @property
    def last_result(self) -> Optional[LLMJudgeResult]:
        return self._last_result

    @property
    def model_name(self) -> str:
        return self._runner.model_name
