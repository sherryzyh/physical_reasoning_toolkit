"""Structured LLM judge outputs and source labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ResultSource = Literal["typed_match", "smart_match", "llm_judge", "skipped_llm"]

RESULT_SOURCE_TYPED_MATCH: ResultSource = "typed_match"
RESULT_SOURCE_SMART_MATCH: ResultSource = "smart_match"
RESULT_SOURCE_LLM_JUDGE: ResultSource = "llm_judge"
RESULT_SOURCE_SKIPPED_LLM: ResultSource = "skipped_llm"


@dataclass
class LLMJudgeResult:
    """Structured judge output.

    ``verdict_type`` is ``typed_match`` or ``smart_match`` (deterministic
    shortcuts), ``llm_judge`` (model verdict), or ``skipped_llm`` (no API call,
    e.g. ``skip_llm=True``).
    """

    verdict: str
    confidence: float
    expected_answer_type: str
    reasoning: str
    raw_response: str
    verdict_type: ResultSource = RESULT_SOURCE_LLM_JUDGE
