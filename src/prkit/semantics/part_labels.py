"""Helpers for question-defined multi-part answer labels."""

from __future__ import annotations

import re
from typing import Sequence

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_PART_PHRASE_RE = re.compile(
    r"\b(?:a|an|the)\s+([A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z][A-Za-z0-9-]*){0,2})\s+"
    r"(phase|component|branch|root|mode|state|region|solution|value|current|field|energy|coordinate|case)s?\b",
    re.IGNORECASE,
)
_COORDINATED_PART_PHRASE_RE = re.compile(
    r"\b(?:the\s+)?(?P<left>[A-Za-z][A-Za-z0-9-]*)\s+and\s+(?P<right>[A-Za-z][A-Za-z0-9-]*)\s+"
    r"(?P<head>components?|coordinates?|values?|fields?|energies?|slots?)\b",
    re.IGNORECASE,
)
_COORDINATED_SLOT_RE = re.compile(
    r"\b(?:the\s+)?(?P<left>magnitude|speed|direction|distance|time|force|current|voltage|"
    r"acceleration|velocity|position|displacement|energy|power|pressure|temperature)\s+and\s+"
    r"(?P<right>magnitude|speed|direction|distance|time|force|current|voltage|"
    r"acceleration|velocity|position|displacement|energy|power|pressure|temperature)\b",
    re.IGNORECASE,
)
_GENERIC_PART_TOKENS = frozenset(
    {
        "part",
        "phase",
        "component",
        "branch",
        "root",
        "mode",
        "state",
        "region",
        "solution",
        "value",
        "current",
        "field",
        "energy",
        "coordinate",
        "case",
    }
)
_IGNORED_DESCRIPTOR_TOKENS = frozenset(
    {
        "same",
        "respective",
        "corresponding",
        "following",
        "final",
        "total",
        "both",
        "each",
        "two",
    }
)


def canonicalize_part_label(text: str | None) -> str | None:
    """Return a stable slug-like part label or ``None`` when empty."""

    if text is None:
        return None
    tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
    if not tokens:
        return None
    return "_".join(tokens)


def has_named_part_labels(labels: Sequence[str]) -> bool:
    """Whether the provided labels carry semantic names beyond ``part_n``."""

    return any(label and not re.fullmatch(r"part_\d+", label) for label in labels)


def infer_named_part_labels_from_question(question: str | None) -> tuple[str, ...]:
    """Infer named multi-part slots from repeated noun phrases in the question."""

    if not question:
        return ()

    coordinated = _infer_coordinated_part_labels_from_question(question)
    if coordinated:
        return coordinated

    grouped: dict[str, list[str]] = {}
    first_position: dict[str, int] = {}
    lowered_question = question.lower()

    for match in _PART_PHRASE_RE.finditer(question):
        descriptor_tokens = [
            token.lower()
            for token in _TOKEN_RE.findall(match.group(1))
            if token.lower() not in _IGNORED_DESCRIPTOR_TOKENS
        ]
        if not descriptor_tokens:
            continue
        head = canonicalize_part_label(match.group(2))
        if not head:
            continue
        label = canonicalize_part_label(" ".join(descriptor_tokens + [head]))
        if not label or label == head:
            continue
        grouped.setdefault(head, [])
        if label not in grouped[head]:
            grouped[head].append(label)
        first_position.setdefault(head, match.start())

    candidates = [(head, labels) for head, labels in grouped.items() if len(labels) >= 2]
    if not candidates:
        return ()

    preferred: list[tuple[str, list[str]]] = []
    for head, labels in candidates:
        head_text = head.replace("_", " ")
        if re.search(rf"\b(?:each|both|respectively)\s+{re.escape(head_text)}s?\b", lowered_question):
            preferred.append((head, labels))

    pool = preferred or candidates
    pool.sort(key=lambda item: (-len(item[1]), first_position[item[0]]))
    return tuple(pool[0][1])


def _infer_coordinated_part_labels_from_question(question: str) -> tuple[str, ...]:
    """Infer conservative two-slot labels from explicit coordinated phrasing."""

    for match in _COORDINATED_PART_PHRASE_RE.finditer(question):
        head = _singularize_head(match.group("head"))
        left_label = canonicalize_part_label(f"{match.group('left')} {head}")
        right_label = canonicalize_part_label(f"{match.group('right')} {head}")
        if left_label and right_label and left_label != right_label:
            return (left_label, right_label)

    for match in _COORDINATED_SLOT_RE.finditer(question):
        left_label = canonicalize_part_label(match.group("left"))
        right_label = canonicalize_part_label(match.group("right"))
        if left_label and right_label and left_label != right_label:
            return (left_label, right_label)

    return ()


def _singularize_head(text: str) -> str:
    """Return a stable singularized head token for coordinated slot phrases."""

    lowered = str(text).strip().lower()
    if lowered.endswith("ies") and len(lowered) > 3:
        return lowered[:-3] + "y"
    if lowered.endswith("s") and len(lowered) > 1:
        return lowered[:-1]
    return lowered


def infer_multi_part_part_labels(
    part_texts: Sequence[str | None],
    required_parts: Sequence[str],
) -> tuple[str | None, ...]:
    """Infer child-to-part assignments from answer surfaces and required parts."""

    part_count = len(part_texts)
    if part_count == 0:
        return ()

    canonical_required = tuple(
        label for label in (canonicalize_part_label(part) for part in required_parts) if label
    )
    if not canonical_required or not has_named_part_labels(canonical_required):
        return tuple(None for _ in range(part_count))

    specs = _build_alias_specs(canonical_required)
    assignments: list[str | None] = [None] * part_count
    used_labels: set[str] = set()
    candidates: list[tuple[int, int, str]] = []

    for index, text in enumerate(part_texts):
        surface = str(text or "")
        for label, spec in specs.items():
            score = _match_part_label_score(surface, spec)
            if score > 0:
                candidates.append((score, index, label))

    for score, index, label in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
        if assignments[index] is None and label not in used_labels:
            assignments[index] = label
            used_labels.add(label)

    if len(canonical_required) == part_count:
        remaining = [label for label in canonical_required if label not in used_labels]
        for index in range(part_count):
            if assignments[index] is None and remaining:
                assignments[index] = remaining.pop(0)

    return tuple(assignments)


def _build_alias_specs(required_parts: Sequence[str]) -> dict[str, dict[str, object]]:
    """Build lightweight alias specs used to score part-label matches."""

    content_initials = [
        _content_tokens(label)[0][0]
        for label in required_parts
        if _content_tokens(label)
    ]
    unique_initials = {initial for initial in content_initials if content_initials.count(initial) == 1}

    specs: dict[str, dict[str, object]] = {}
    for label in required_parts:
        content_tokens = _content_tokens(label)
        aliases = {label.replace("_", " "), label}
        aliases.update(token for token in content_tokens if len(token) > 1)

        initial = None
        if content_tokens:
            candidate = content_tokens[0][0]
            if candidate in unique_initials:
                initial = candidate

        specs[label] = {
            "aliases": tuple(sorted(aliases, key=len, reverse=True)),
            "initial": initial,
        }
    return specs


def _content_tokens(label: str) -> list[str]:
    """Return informative tokens from a canonical part label."""

    return [token for token in label.split("_") if token and token not in _GENERIC_PART_TOKENS]


def _match_part_label_score(text: str, spec: dict[str, object]) -> int:
    """Score how strongly one answer surface suggests a required part label.

    Higher scores prefer explicit aliases near the start of the answer,
    because multi-part responses often prefix each clause with its part name
    or symbol before the actual value.
    """

    lowered = text.lower()
    prefix = lowered
    for delimiter in ("=", ":"):
        if delimiter in prefix:
            prefix = prefix.split(delimiter, 1)[0]
    prefix = prefix[:48]

    best = 0
    for alias in spec["aliases"]:
        if _contains_alias(prefix, alias):
            best = max(best, 220 + len(alias))
            continue
        if _contains_alias(lowered[:48], alias):
            best = max(best, 140 + len(alias))
            continue
        if len(alias) > 1 and _contains_alias(lowered, alias):
            best = max(best, 80 + len(alias))

    initial = spec["initial"]
    if initial and _contains_alias(prefix, initial):
        best = max(best, 40)
    return best


def _contains_alias(text: str, alias: str) -> bool:
    """Return whether ``alias`` appears as a standalone token in ``text``."""

    return re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text) is not None


__all__ = [
    "canonicalize_part_label",
    "has_named_part_labels",
    "infer_multi_part_part_labels",
    "infer_named_part_labels_from_question",
]
