"""
Reusable LLM-as-judge building blocks for hybrid comparators.

Includes default grading instructions, JSON schema for structured verdicts,
payload helpers, response parsing, and an :class:`OpenAIJudgeRunner` for OpenAI
Responses API calls with moderation fallbacks.
"""

from prkit.evaluation.llm_judge.instructions import DEFAULT_PHYSICS_GRADING_INSTRUCTIONS
from prkit.evaluation.llm_judge.openai import (
    DEFAULT_JUDGE_MODEL,
    FALLBACK_CHAT_JUDGE_MODEL,
    build_responses_api_judge_body,
)
from prkit.evaluation.llm_judge.parse import (
    extract_json_object,
    parse_judge_response,
    reasoning_indicates_incorrect,
)
from prkit.evaluation.llm_judge.payload import (
    answer_to_text_and_category,
    build_standard_answer_judge_payload,
    clean_answer_text,
    truncate_judge_payload,
)
from prkit.evaluation.llm_judge.runner import OpenAIJudgeRunner
from prkit.evaluation.llm_judge.schema import (
    CHAT_COMPLETIONS_JSON_SCHEMA_RESPONSE_FORMAT,
    EXPECTED_ANSWER_TYPES,
    JUDGE_VERDICT_SCHEMA_NAME,
    RESPONSES_API_JUDGE_TEXT_FORMAT,
)
from prkit.evaluation.llm_judge.types import (
    LLMJudgeResult,
    RESULT_SOURCE_LLM_JUDGE,
    RESULT_SOURCE_SKIPPED_LLM,
    RESULT_SOURCE_SMART_MATCH,
    RESULT_SOURCE_TYPED_MATCH,
    ResultSource,
)

# Backward-compatible alias (matches prior ``typed_llm.DEFAULT_MODEL`` name).
DEFAULT_MODEL = DEFAULT_JUDGE_MODEL

__all__ = [
    "CHAT_COMPLETIONS_JSON_SCHEMA_RESPONSE_FORMAT",
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_MODEL",
    "DEFAULT_PHYSICS_GRADING_INSTRUCTIONS",
    "EXPECTED_ANSWER_TYPES",
    "FALLBACK_CHAT_JUDGE_MODEL",
    "JUDGE_VERDICT_SCHEMA_NAME",
    "LLMJudgeResult",
    "OpenAIJudgeRunner",
    "RESULT_SOURCE_LLM_JUDGE",
    "RESULT_SOURCE_SKIPPED_LLM",
    "RESULT_SOURCE_SMART_MATCH",
    "RESULT_SOURCE_TYPED_MATCH",
    "RESPONSES_API_JUDGE_TEXT_FORMAT",
    "ResultSource",
    "answer_to_text_and_category",
    "build_responses_api_judge_body",
    "build_standard_answer_judge_payload",
    "clean_answer_text",
    "extract_json_object",
    "parse_judge_response",
    "reasoning_indicates_incorrect",
    "truncate_judge_payload",
]
