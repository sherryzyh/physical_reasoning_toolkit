"""Parse model text into :class:`LLMJudgeResult` (JSON or coarse fallback)."""

from __future__ import annotations

import json
import re
from typing import Any

from prkit.evaluation.llm_judge.schema import EXPECTED_ANSWER_TYPES
from prkit.evaluation.llm_judge.types import RESULT_SOURCE_LLM_JUDGE, LLMJudgeResult

_EXPECTED_SET = frozenset(EXPECTED_ANSWER_TYPES)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first JSON object found in *text*, or ``None``."""
    raw = text.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def reasoning_indicates_incorrect(reasoning: str) -> bool:
    """Heuristic: reasoning text explicitly contradicts a \"correct\" verdict."""
    r = reasoning.lower()
    red_flags = (
        "incorrect",
        "wrong",
        "missing factor",
        "missing sign",
        "wrong sign",
        "does not match",
        "not match",
        "not equivalent",
        "mismatch",
        "fails",
    )
    return any(flag in r for flag in red_flags)


def parse_judge_response(response_text: str) -> LLMJudgeResult:
    """Parse OpenAI (or compatible) judge output into :class:`LLMJudgeResult`."""
    raw = response_text.strip()
    payload = extract_json_object(raw)
    if payload:
        verdict = str(payload.get("verdict", "")).strip().lower()
        if verdict not in ("correct", "incorrect"):
            verdict = "incorrect"
        try:
            confidence = float(payload.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        expected_answer_type = str(payload.get("expected_answer_type", "other")).strip()
        if expected_answer_type not in _EXPECTED_SET:
            expected_answer_type = "other"
        reasoning = str(payload.get("reasoning", "")).strip()
        if verdict == "correct" and reasoning_indicates_incorrect(reasoning):
            verdict = "incorrect"
            confidence = min(confidence, 0.35)
        return LLMJudgeResult(
            verdict=verdict,
            confidence=confidence,
            expected_answer_type=expected_answer_type,
            reasoning=reasoning,
            raw_response=raw,
            verdict_type=RESULT_SOURCE_LLM_JUDGE,
        )

    lower = raw.lower()
    verdict = (
        "correct" if "correct" in lower and "incorrect" not in lower else "incorrect"
    )
    return LLMJudgeResult(
        verdict,
        0.5,
        "other",
        raw[:240],
        raw,
        RESULT_SOURCE_LLM_JUDGE,
    )
