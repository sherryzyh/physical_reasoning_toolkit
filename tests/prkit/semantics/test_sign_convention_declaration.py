"""Adversarial tests for the deterministic sign-convention declaration parser.

The parser (``answer_normalization._extract_sign_convention_declaration``, wired into
``normalize_physics_answer``) captures an *explicit* "<dir> as positive" convention onto a
directional scalar's ``sign_convention`` and strips the clause before value parsing. It is
**declaration-only**: a bare signed value or a fully-specifying direction phrase (no "positive")
is never treated as a convention, so the sign-convention lane stays declared-not-derived. The
captured string is read by the judge's ``_convention_orientation`` (shared vocabulary).
"""

from __future__ import annotations

import pytest

from prkit.semantics.build.calls import extract_prediction_answer_semantics
from prkit.semantics.comparison.sign_convention import _convention_orientation
from prkit.semantics.normalization.answer_normalization import (
    _extract_sign_convention_declaration,
    normalize_physics_answer,
)
from prkit.semantics.schema import AnswerObjectKind

# (surface, expected main text after strip, expected orientation the judge reads)
_ACCEPTED = [
    ("-20 m/s (taking rightward as positive)", "-20 m/s", "right"),
    ("-20 m/s, taking rightward as positive", "-20 m/s", "right"),
    ("20 m/s (right-as-positive)", "20 m/s", "right"),
    ("-9.8 m/s^2 (with up as positive)", "-9.8 m/s^2", "up"),
    ("5 (positive direction is left)", "5", "left"),
    ("3 N (+ve = right)", "3 N", "right"),
    ("4 T (into the page as positive)", "4 T", "into_page"),
    ("-12 (taking down as positive)", "-12", "down"),
    ("7 m/s (counterclockwise is positive)", "7 m/s", "counterclockwise"),
]

# Surfaces that must NOT be read as a convention declaration (no capture, no strip).
_REJECTED = [
    "5 N to the right",  # fully specifies the answer; no "positive" -> not a convention
    "+20 m/s",  # bare sign, no stated direction
    "-20 m/s",
    "(3, 4)",  # a tuple, not a declaration
    "{2, -2}",
    "x**2/2",
    "sqrt(E/m)",
    "increases",
    "12 m",
    "F = m*a",
]


@pytest.mark.parametrize("surface, expected_main, expected_orientation", _ACCEPTED)
def test_declaration_captured_and_orientation_readable(
    surface: str, expected_main: str, expected_orientation: str
) -> None:
    main, convention = _extract_sign_convention_declaration(surface)
    assert main == expected_main
    assert convention is not None
    # The captured string round-trips through the judge's orientation reader.
    assert _convention_orientation(convention) == expected_orientation


@pytest.mark.parametrize("surface", _REJECTED)
def test_non_declarations_are_not_captured(surface: str) -> None:
    main, convention = _extract_sign_convention_declaration(surface)
    assert convention is None
    # No declaration -> the surface is returned byte-identical (zero behavior change).
    assert main == surface


@pytest.mark.parametrize("surface, expected_main, expected_orientation", _ACCEPTED)
def test_normalize_applies_convention_to_directional_scalar(
    surface: str, expected_main: str, expected_orientation: str
) -> None:
    answer = normalize_physics_answer(surface)
    assert answer.object_kind in {
        AnswerObjectKind.NUMBER,
        AnswerObjectKind.PHYSICAL_QUANTITY,
    }
    assert answer.sign_convention is not None
    assert _convention_orientation(answer.sign_convention) == expected_orientation
    # The clause was stripped, so the value still parses.
    assert answer.numeric_value is not None


@pytest.mark.parametrize("surface", _REJECTED)
def test_normalize_leaves_non_declarations_convention_free(surface: str) -> None:
    answer = normalize_physics_answer(surface)
    assert answer.sign_convention is None


def test_extract_prediction_value_parses_after_strip() -> None:
    ext = extract_prediction_answer_semantics("-20 m/s (taking rightward as positive)")
    assert ext.object_kind == AnswerObjectKind.PHYSICAL_QUANTITY
    assert ext.numeric_value == -20.0
    assert ext.unit == "m/s"
    assert _convention_orientation(ext.sign_convention) == "right"


def test_declaration_only_surface_is_left_alone() -> None:
    # If the whole surface is the declaration (no value), do not strip to empty.
    main, convention = _extract_sign_convention_declaration("taking right as positive")
    assert main == "taking right as positive"
    assert convention is None
