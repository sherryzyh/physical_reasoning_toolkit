"""Orchestrate OpenAI judge calls (Responses API with moderation fallbacks)."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from openai import BadRequestError, OpenAI

from prkit.core import PRKitLogger
from prkit.core.project_env import ensure_openai_api_key
from prkit.evaluation.llm_judge.instructions import DEFAULT_PHYSICS_GRADING_INSTRUCTIONS
from prkit.evaluation.llm_judge.openai import (
    FALLBACK_CHAT_JUDGE_MODEL,
    build_responses_api_judge_body,
)
from prkit.evaluation.llm_judge.parse import parse_judge_response
from prkit.evaluation.llm_judge.schema import CHAT_COMPLETIONS_JSON_SCHEMA_RESPONSE_FORMAT
from prkit.evaluation.llm_judge.payload import truncate_judge_payload
from prkit.evaluation.llm_judge.types import LLMJudgeResult


class OpenAIJudgeRunner:
    """Run the standard physics judge prompt via Responses API, with truncation + chat fallback."""

    def __init__(
        self,
        *,
        model: str,
        instructions: Optional[str] = None,
        client: Optional[OpenAI] = None,
        fallback_chat_model: str = FALLBACK_CHAT_JUDGE_MODEL,
        logger: Optional[Any] = None,
    ) -> None:
        self._model_name = model
        self._instructions = instructions or DEFAULT_PHYSICS_GRADING_INSTRUCTIONS
        self._fallback_chat_model = fallback_chat_model
        self._client = client or OpenAI(
            api_key=ensure_openai_api_key(__file__, required=False)
        )
        self._logger = logger or PRKitLogger.get_logger(__name__)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def instructions(self) -> str:
        return self._instructions

    def judge_via_responses(self, payload: Dict[str, Any]) -> LLMJudgeResult:
        request_params = build_responses_api_judge_body(
            self._model_name,
            payload,
            instructions=self._instructions,
        )
        response = self._client.responses.create(**request_params)
        response_text = response.output_text if response.output_text else ""
        return parse_judge_response(response_text)

    def judge_via_chat_completions(self, payload: Dict[str, Any]) -> LLMJudgeResult:
        """Fallback when Responses API rejects the prompt (e.g. moderation policy)."""
        response = self._client.chat.completions.create(
            model=self._fallback_chat_model,
            messages=[
                {"role": "system", "content": self._instructions},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            response_format=CHAT_COMPLETIONS_JSON_SCHEMA_RESPONSE_FORMAT,
        )
        content = response.choices[0].message.content or ""
        return parse_judge_response(content)

    def judge(self, payload: Dict[str, Any]) -> LLMJudgeResult:
        """Responses API, then truncated retry, then chat completions."""
        try:
            return self.judge_via_responses(payload)
        except BadRequestError as e:
            self._logger.warning(
                "LLM judge Responses API BadRequest (%s); retrying truncated payload",
                e,
            )
            truncated = truncate_judge_payload(payload)
            try:
                return self.judge_via_responses(truncated)
            except BadRequestError as e2:
                self._logger.warning(
                    "LLM judge truncated Responses BadRequest (%s); chat completions fallback",
                    e2,
                )
                return self.judge_via_chat_completions(truncated)
