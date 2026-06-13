"""JSON Schema fragments for OpenAI Responses / Chat Completions structured judge output."""

from __future__ import annotations

from typing import Any, Dict, Final, List

# Aligned with the default grading prompt; extend alongside instruction changes.
EXPECTED_ANSWER_TYPES: Final[List[str]] = [
    "numeric_value",
    "physical_quantity",
    "symbolic_expression",
    "textual_concept",
    "multiple_choice",
    "direction_or_sign",
    "multi_part",
    "other",
]

JUDGE_VERDICT_SCHEMA_NAME: Final[str] = "llm_judge_verdict"

RESPONSES_API_JUDGE_TEXT_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "name": JUDGE_VERDICT_SCHEMA_NAME,
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["correct", "incorrect"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "expected_answer_type": {
                "type": "string",
                "enum": EXPECTED_ANSWER_TYPES,
            },
            "reasoning": {"type": "string"},
        },
        "required": ["verdict", "confidence", "expected_answer_type", "reasoning"],
        "additionalProperties": False,
    },
}

# Chat Completions API (json_schema) — used when Responses API rejects the prompt (moderation).
CHAT_COMPLETIONS_JSON_SCHEMA_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": JUDGE_VERDICT_SCHEMA_NAME,
        "strict": True,
        "schema": RESPONSES_API_JUDGE_TEXT_FORMAT["schema"],
    },
}
