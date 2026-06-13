"""OpenAI request bodies for the standard judge schema (Responses + Chat Completions)."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from prkit.prkit_evaluation.llm_judge.instructions import DEFAULT_PHYSICS_GRADING_INSTRUCTIONS
from prkit.prkit_evaluation.llm_judge.schema import RESPONSES_API_JUDGE_TEXT_FORMAT

DEFAULT_JUDGE_MODEL = "gpt-5.4-mini"
FALLBACK_CHAT_JUDGE_MODEL = "gpt-5.4-mini"


def build_responses_api_judge_body(
    model: str,
    payload: Dict[str, Any],
    *,
    instructions: Optional[str] = None,
) -> Dict[str, Any]:
    """JSON body for ``POST /v1/responses`` (sync or OpenAI Batch ``/v1/responses``)."""
    ins = instructions or DEFAULT_PHYSICS_GRADING_INSTRUCTIONS
    return {
        "model": model,
        "instructions": ins,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(payload, ensure_ascii=False),
                    }
                ],
            }
        ],
        "text": {"format": RESPONSES_API_JUDGE_TEXT_FORMAT},
    }
