"""Build and trim judge request payloads (question + ground truth + model answer)."""

from __future__ import annotations

import re
from typing import Any

from prkit.core.domain.answer import Answer


def answer_to_text_and_category(answer: str | Answer) -> tuple[str, str]:
    """Plain text and native-type label for embedding in a judge JSON payload."""
    if isinstance(answer, Answer):
        return str(answer).strip(), answer.source_type or ""
    return str(answer).strip(), ""


def clean_answer_text(answer_text: str) -> str:
    """Normalize whitespace and NBSP for stable judge payloads."""
    text = answer_text.strip()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text


def build_standard_answer_judge_payload(
    predicted: str | Answer,
    ground_truth: str | Answer,
    question: str | None,
) -> dict[str, Any]:
    """Standard physics payload: ``question``, ``ground_truth``, ``model_answer``."""
    pred_text, pred_cat = answer_to_text_and_category(predicted)
    gt_text, gt_cat = answer_to_text_and_category(ground_truth)
    return {
        "question": (question or "").strip(),
        "ground_truth": {
            "text": clean_answer_text(gt_text),
            "category": gt_cat,
        },
        "model_answer": {
            "text": clean_answer_text(pred_text),
            "category": pred_cat,
        },
    }


def truncate_judge_payload(
    payload: dict[str, Any], max_chars: int = 4096
) -> dict[str, Any]:
    """Copy *payload* with truncated question / answer texts (policy / moderation retries)."""

    def _trim(s: str) -> str:
        s = str(s)
        if len(s) <= max_chars:
            return s
        return s[: max_chars - 3] + "..."

    gt = dict(payload.get("ground_truth") or {})
    ma = dict(payload.get("model_answer") or {})
    if "text" in gt:
        gt["text"] = _trim(gt["text"])
    if "text" in ma:
        ma["text"] = _trim(ma["text"])
    return {
        "question": _trim(str(payload.get("question") or "")),
        "ground_truth": gt,
        "model_answer": ma,
    }
