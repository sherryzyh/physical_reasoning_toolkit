"""Phase-4 intended capability gains (deliberate behavior changes).

These assert the *new* behavior the guarded-full recognition + temperature edits
unlock -- the headline °C/°F offset conversion and the widened unit vocabulary.
Unlike the precision-contract test, these are NOT zero-diff against PRE; they pin
the settled gains (CHANGE 1 + CHANGE 2). Broader D/G/H/I behavior is recorded in
``internal/PHASE4_BEHAVIOR_DIFF.md`` for owner review, not enumerated here.
"""

from __future__ import annotations

import pytest

from prkit.semantics import compare_protocol_answers
from prkit.semantics.normalization.question_inference import _is_supported_question_unit
from prkit.semantics.quantities.surface import _try_parse_physical_quantity
from prkit.semantics.quantities.views import build_quantity_view
from prkit.semantics.schema.enums import AnswerObjectKind, AnswerStructure
from prkit.semantics.schema.models import (
    PhysicsAnswerSemantics,
    PhysicsQuestionSemantics,
)


def _quantity_answer(
    *, numeric_value: float, unit: str, dimension: str
) -> dict[str, object]:
    """Build one minimal physical-quantity protocol answer mapping."""

    return {
        "object_kind": "physical_quantity",
        "canonical_text": f"{numeric_value} {unit}",
        "numeric_value": numeric_value,
        "numeric_text": str(numeric_value),
        "unit": unit,
        "dimension": dimension,
    }


def _canonical_temperature_kelvin(raw_text: str) -> tuple[str | None, float | None]:
    """Canonicalize a temperature surface to its (unit, value) snapshot."""

    answer = PhysicsAnswerSemantics(
        structure=AnswerStructure.ATOMIC,
        object_kind=AnswerObjectKind.PHYSICAL_QUANTITY,
        raw_text=raw_text,
        canonical_text=raw_text,
        dimension="temperature",
    )
    view, _diagnostics = build_quantity_view(answer, context=PhysicsQuestionSemantics())
    assert view is not None
    return view.canonical.unit, view.canonical.numeric_value


# --- CHANGE 2: temperature canonicalizes to Kelvin (offset conversion) ---


@pytest.mark.parametrize(
    ("raw_text", "expected_kelvin"),
    [
        ("20 °C", 293.15),
        ("0 °C", 273.15),
        ("100 °C", 373.15),
        ("-40 °C", 233.15),
        ("98.6 °F", 310.15),
    ],
)
def test_temperature_canonicalizes_to_kelvin(
    raw_text: str, expected_kelvin: float
) -> None:
    unit, value = _canonical_temperature_kelvin(raw_text)
    assert unit == "K"
    assert value == pytest.approx(expected_kelvin, abs=1e-6)


@pytest.mark.parametrize(
    ("celsius", "kelvin", "equivalent"),
    [
        (20.0, 293.15, True),
        (100.0, 373.15, True),
        (20.0, 300.0, False),
    ],
)
def test_celsius_matches_kelvin_reference(
    celsius: float, kelvin: float, equivalent: bool
) -> None:
    pred = _quantity_answer(numeric_value=celsius, unit="°C", dimension="temperature")
    ref = _quantity_answer(numeric_value=kelvin, unit="K", dimension="temperature")
    result = compare_protocol_answers(pred, ref)
    assert result.equivalent is equivalent
    assert result.comparison_mode == "physical_quantity"


@pytest.mark.parametrize(
    "raw_text",
    [
        # LaTeX degree marks (separated/braced/empty-brace forms that used to
        # mis-split into coulomb*degree) across \mathrm / \text / bare wrappers.
        "20^\\circ C",
        "20 ^\\circ C",
        "20^{\\circ} C",
        "20^\\circ\\,C",
        "20\\,^\\circ C",
        "20^{\\circ}C",
        "20^\\circ\\mathrm{C}",
        "20^\\circ\\text{C}",
        # Unicode degree, with lowercase scale letter (case-insensitive).
        "20 °C",
        "20 °c",
        "20°C",
    ],
)
def test_celsius_latex_and_unicode_surfaces_reach_kelvin(raw_text: str) -> None:
    unit, value = _canonical_temperature_kelvin(raw_text)
    assert unit == "K"
    assert value == pytest.approx(293.15, abs=1e-6)


@pytest.mark.parametrize(
    "raw_text",
    ["68^\\circ F", "68^\\circ\\mathrm{F}", "68^{\\circ}F", "68 °F", "68 °f"],
)
def test_fahrenheit_latex_and_unicode_surfaces_reach_kelvin(raw_text: str) -> None:
    unit, value = _canonical_temperature_kelvin(raw_text)
    assert unit == "K"
    assert value == pytest.approx(293.15, abs=1e-6)


def test_latex_and_unicode_degree_surfaces_agree() -> None:
    # The same temperature written as Unicode or LaTeX must canonicalize identically.
    assert _canonical_temperature_kelvin("20 °C") == _canonical_temperature_kelvin(
        "20^\\circ C"
    )


def test_bare_circ_angle_is_not_treated_as_temperature() -> None:
    # The degree-temperature collapse must not hijack a plain angle (no C/F scale).
    assert _try_parse_physical_quantity("30^\\circ") == "30 deg"
    assert _try_parse_physical_quantity("30^{\\circ}") == "30 deg"


# --- CHANGE 1: widened unit vocabulary parses and converts ---


@pytest.mark.parametrize(
    "surface", ["5 daN", "2 cN", "3 dL", "10 ft", "30 mph", "5 lbf"]
)
def test_widened_units_parse(surface: str) -> None:
    assert _try_parse_physical_quantity(surface) is not None


def test_widened_unit_compares_against_si() -> None:
    # 5 daN = 50 N (deka-newton); pint converts the widened unit correctly.
    pred = _quantity_answer(numeric_value=5.0, unit="daN", dimension="force")
    ref = _quantity_answer(numeric_value=50.0, unit="N", dimension="force")
    result = compare_protocol_answers(pred, ref)
    assert result.equivalent is True


def test_si_aliased_gauss_and_metric_ton_convert() -> None:
    # gauss is CGS-Gaussian in pint; prkit pins ``gauss``/``G`` to the SI gauss so
    # 10000 G = 1 T. ``ton`` is pinned to the metric tonne (1000 kg).
    gauss = compare_protocol_answers(
        _quantity_answer(
            numeric_value=10000.0, unit="gauss", dimension="magnetic_flux_density"
        ),
        _quantity_answer(
            numeric_value=1.0, unit="T", dimension="magnetic_flux_density"
        ),
    )
    assert gauss.equivalent is True
    ton = compare_protocol_answers(
        _quantity_answer(numeric_value=1.0, unit="ton", dimension="mass"),
        _quantity_answer(numeric_value=1000.0, unit="kg", dimension="mass"),
    )
    assert ton.equivalent is True


# --- CHANGE 1: the guard still rejects ambiguous tokens (sanity) ---


@pytest.mark.parametrize("surface", ["5 c", "5 rpm", "2 M", "5 S", "3 u"])
def test_guard_keeps_ambiguous_tokens_symbolic(surface: str) -> None:
    assert _try_parse_physical_quantity(surface) is None


@pytest.mark.parametrize(
    "surface",
    [
        # pint-dimensionless constants / counting / ratio units.
        "5 pi",
        "5 count",
        "1 bit",
        "1 byte",
        # non-physical printing-unit dimension.
        "5 dot",
        "2 pixel",
        # CGS-Gaussian fractional-dimension electromagnetic units.
        "5 maxwell",
        "5 oersted",
        "5 statC",
    ],
)
def test_dimensionality_gate_rejects_non_physics_tokens(surface: str) -> None:
    # pint recognizes all of these, but they are not physical quantities prkit
    # should grade; the dimensionality gate keeps them out of quantity parsing.
    assert _try_parse_physical_quantity(surface) is None


# --- CHANGE 1: the question surface widens in lockstep ---


@pytest.mark.parametrize("unit", ["daN", "ft", "mph", "°C", "m/s", "percent"])
def test_question_unit_recognition_widened(unit: str) -> None:
    assert _is_supported_question_unit(unit) is True


@pytest.mark.parametrize("unit", ["rpm", "c", "M", "S"])
def test_question_unit_recognition_denied(unit: str) -> None:
    assert _is_supported_question_unit(unit) is False
