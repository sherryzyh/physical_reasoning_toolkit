"""Shared atomic answer-kind definitions for normalization helpers."""

from __future__ import annotations

from enum import Enum


class NormalizedAtomicKind(str, Enum):
    """Atomic answer kinds used by semantics-local normalization."""

    NUMBER = "number"
    PHYSICAL_QUANTITY = "physical_quantity"
    EXPRESSION = "expression"
    RELATION = "relation"
    TEXT = "text"


__all__ = ["NormalizedAtomicKind"]
