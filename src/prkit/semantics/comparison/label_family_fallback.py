"""Second-stage rescue for bounded label mismatches."""

from __future__ import annotations

import re

from ..schema import (
    AnswerComparison,
    AnswerObjectKind,
    PhysicsAnswerSemantics,
    PhysicsQuestionSemantics,
)
from .semantics import qualitative_label_candidates, sign_direction_candidates

_LABEL_FAMILY_KINDS = frozenset(
    {
        AnswerObjectKind.QUALITATIVE_LABEL,
        AnswerObjectKind.SIGN_DIRECTION,
    }
)
_TERMINAL_POLARITY_RE = re.compile(
    r"(?:^|\b)(?:terminal\s+)?([a-z0-9]{1,3})\s+is\s+(positive|negative)(?:\b|$)"
)


def compare_label_family_fallback(
    pred: PhysicsAnswerSemantics,
    ref: PhysicsAnswerSemantics,
    *,
    context: PhysicsQuestionSemantics,
) -> AnswerComparison | None:
    """Rescue bounded label mismatches when both answers live in the label family."""

    del context

    if not pred.is_atomic or not ref.is_atomic:
        return None
    if (
        pred.object_kind not in _LABEL_FAMILY_KINDS
        or ref.object_kind not in _LABEL_FAMILY_KINDS
    ):
        return None

    if _candidate_overlap(
        _terminal_polarity_candidates(pred), _terminal_polarity_candidates(ref)
    ):
        return AnswerComparison(
            True,
            "label_family_fallback",
            ("normalized_kind=terminal_polarity",),
        )

    if _candidate_overlap(
        _sign_direction_candidates(pred), _sign_direction_candidates(ref)
    ):
        return AnswerComparison(
            True,
            "label_family_fallback",
            ("normalized_kind=sign_direction",),
        )

    if _candidate_overlap(
        _qualitative_label_candidates(pred), _qualitative_label_candidates(ref)
    ):
        return AnswerComparison(
            True,
            "label_family_fallback",
            ("normalized_kind=qualitative_label",),
        )

    return None


def _candidate_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    """Return whether two normalized candidate sets share any label."""

    return bool(set(left) & set(right))


def _qualitative_label_candidates(answer: PhysicsAnswerSemantics) -> tuple[str, ...]:
    """Collect qualitative-label candidates from all answer text surfaces."""

    candidates: list[str] = []
    for text in _answer_surfaces(answer):
        candidates.extend(qualitative_label_candidates(text))
    return _dedupe(candidates)


def _sign_direction_candidates(answer: PhysicsAnswerSemantics) -> tuple[str, ...]:
    """Collect sign/direction candidates from text and explicit sign metadata."""

    candidates: list[str] = []
    if answer.sign_value:
        candidates.append(answer.sign_value)
    for text in _answer_surfaces(answer):
        candidates.extend(sign_direction_candidates(text))
    return _dedupe(candidates)


def _terminal_polarity_candidates(answer: PhysicsAnswerSemantics) -> tuple[str, ...]:
    """Extract ``terminal:polarity`` candidates from answer text surfaces."""

    candidates: list[str] = []
    for text in _answer_surfaces(answer):
        lowered = text.lower()
        for match in _TERMINAL_POLARITY_RE.finditer(lowered):
            candidates.append(f"{match.group(1).upper()}:{match.group(2)}")
    return _dedupe(candidates)


def _answer_surfaces(answer: PhysicsAnswerSemantics) -> tuple[str, ...]:
    """Return answer surfaces in the order the fallback inspects them."""

    surfaces: list[str] = []
    for text in (answer.raw_text, answer.canonical_text):
        if text and text not in surfaces:
            surfaces.append(text)
    return tuple(surfaces)


def _dedupe(candidates: list[str]) -> tuple[str, ...]:
    """Deduplicate candidates while preserving their first-seen order."""

    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return tuple(deduped)
