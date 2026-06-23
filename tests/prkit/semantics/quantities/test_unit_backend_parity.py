"""Differential parity net for the unit-conversion backend.

This corpus pins the behaviour of the unit-conversion *factor* computation across
prkit's entire legacy vocabulary. It exists to make the swap from the bespoke
SymPy algebra to a pint backend (the next phase) *provably* behaviour-neutral:
for every ``(from, to)`` pair the legacy factor and the pint factor must agree
(both ``None``, or both finite floats within ``rtol=1e-9``).

Two facts make this test robust to the upcoming backend swap:

* The **legacy oracle** below is a self-contained reimplementation of the
  SymPy factor computation built on the *retained* :data:`UNIT_NAMESPACE` table.
  It deliberately does **not** call ``units.unit_conversion_factor`` so that it
  keeps reporting the *legacy* value even after that function is rewired onto
  pint -- otherwise the differential would degenerate into a pint-vs-pint
  tautology.
* The **backend under test** section is intentionally isolated. The next phase
  replaces its body with imports from ``_pint_backend`` without touching the
  corpus or assertions, so the same corpus then validates the production backend.

Two divergences surfaced while building the corpus and were resolved with the
project owner (see the dedicated tests at the bottom of this module):

* ``G`` (gauss): pint's built-in ``gauss`` is a CGS-Gaussian unit whose
  dimensionality is incompatible with ``tesla``, so the obvious ``G -> gauss``
  alias raises ``DimensionalityError``. prkit treats ``1 G = 1e-4 T`` (SI
  convention) and converts ``G <-> T`` today, so the backend defines a private
  ``prkit_gauss = 1e-4 * tesla`` unit that reproduces the legacy factor exactly.
* ``MHz``: it is recognised (present in ``UNIT_TO_BASE``) but absent from the
  legacy SymPy ``UNIT_NAMESPACE``, so legacy conversion silently returns
  ``None``. pint converts it correctly. The owner approved letting pint's
  correct value win here, so the corpus treats every ``MHz`` cross-conversion as
  an *intentional* divergence rather than a parity failure.
"""

from __future__ import annotations

import math

import pytest
from sympy import Float, Integer, N, Rational, simplify
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from prkit.semantics.quantities._pint_backend import (
    _PRKIT_TO_PINT,
    _prkit_token_to_pint,
)
from prkit.semantics.quantities._pint_backend import (
    unit_conversion_factor as _pint_unit_conversion_factor,
)
from prkit.semantics.quantities.units import (
    UNIT_NAMESPACE,
    UNIT_TO_BASE,
    normalize_unit_text,
)

# ---------------------------------------------------------------------------
# Legacy oracle: a faithful copy of the SymPy factor computation, built on the
# *retained* UNIT_NAMESPACE table so it survives the backend swap unchanged.
# Verified byte-identical to ``units.unit_conversion_factor`` over the corpus.
# ---------------------------------------------------------------------------

_LEGACY_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)
_LEGACY_GLOBALS = {"Integer": Integer, "Float": Float, "Rational": Rational}


def _legacy_parse_unit_expression(text: str):
    if not text:
        return Integer(1)
    try:
        return simplify(
            parse_expr(
                text,
                local_dict=dict(UNIT_NAMESPACE),
                global_dict=_LEGACY_GLOBALS,
                transformations=_LEGACY_TRANSFORMATIONS,
                evaluate=True,
            )
        )
    except Exception:
        return None


def _legacy_unit_conversion_factor(
    from_unit: str | None, to_unit: str | None
) -> float | None:
    normalized_from = normalize_unit_text(from_unit)
    normalized_to = normalize_unit_text(to_unit)
    if normalized_from == normalized_to:
        return 1.0
    from_expr = _legacy_parse_unit_expression(normalized_from)
    to_expr = _legacy_parse_unit_expression(normalized_to)
    if from_expr is None or to_expr is None:
        return None
    ratio = simplify(from_expr / to_expr)
    if ratio.free_symbols:
        return None
    try:
        return float(N(ratio))
    except (TypeError, ValueError):
        return None


# The backend under test is imported above:
# ``_pint_unit_conversion_factor`` is the production
# ``_pint_backend.unit_conversion_factor``, and ``_prkit_token_to_pint`` /
# ``_PRKIT_TO_PINT`` are its translation map. Both consume normalized prkit
# tokens, so the corpus passes ``normalize_unit_text`` output to them.


# ---------------------------------------------------------------------------
# Corpus construction.
# ---------------------------------------------------------------------------


def _same_base_pairs() -> list[tuple[str, str]]:
    """Every ordered (from, to) pair within each legacy base-dimension group."""

    groups: dict[str, list[str]] = {}
    for token, (base, _scale) in UNIT_TO_BASE.items():
        groups.setdefault(base, []).append(token)
    pairs: list[tuple[str, str]] = []
    for tokens in groups.values():
        for a in tokens:
            for b in tokens:
                pairs.append((a, b))
    return pairs


# Compound units whose self-conversion must round-trip cleanly through pint.
_COMPOUND_UNITS = [
    "m/s",
    "m/s^2",
    "kg*m/s^2",
    "N*m",
    "J/kg",
    "V/m",
    "kg/m^3",
    "rad/s",
    "mL/s",
    "rad s^-1",
    "cm s^-2",
    "N/m^2",
    "m^3",
    "m^3/s",
    "kg*m/s",
]

# Cross-base dimensional bridges (different UNIT_TO_BASE groups, same physical
# dimension) plus the compound conversions pinned in test_quantity_views.py /
# test_protocol_comparison.py.
_CROSS_BASE_PAIRS = [
    ("cm s^-2", "m/s^2"),
    ("rad s^-1", "rad/s"),
    ("N/m^2", "Pa"),
    ("mL/s", "m^3/s"),
    ("mL", "m^3"),
    ("L", "m^3"),
    ("km/h", "m/s"),
    ("eV", "J"),
    ("keV", "J"),
    ("MeV", "J"),
    ("cal", "J"),
    ("kcal", "J"),
    ("atm", "Pa"),
    ("bar", "Pa"),
    ("mbar", "Pa"),
    ("mmHg", "Pa"),
    ("torr", "Pa"),
    ("hPa", "Pa"),
    ("angstrom", "m"),
    ("deg", "rad"),
    ("erg", "J"),
    ("dyn", "N"),
    # Pinned protocol/view cases.
    ("torr", "torr"),
    ("Pa", "Pa"),
    ("kohm", "ohm"),
    ("cm", "m"),
]


def _corpus() -> list[tuple[str, str]]:
    pairs = _same_base_pairs()
    pairs.extend((c, c) for c in _COMPOUND_UNITS)
    pairs.extend(_CROSS_BASE_PAIRS)
    return pairs


def _is_intentional_divergence(from_unit: str, to_unit: str) -> bool:
    """Owner-approved: MHz is recognised but absent from the legacy table."""

    return from_unit != to_unit and "MHz" in (from_unit, to_unit)


@pytest.mark.parametrize("from_unit,to_unit", _corpus())
def test_pint_factor_matches_legacy(from_unit: str, to_unit: str) -> None:
    legacy = _legacy_unit_conversion_factor(from_unit, to_unit)
    new = _pint_unit_conversion_factor(
        normalize_unit_text(from_unit), normalize_unit_text(to_unit)
    )

    if _is_intentional_divergence(from_unit, to_unit):
        assert (
            legacy is None
        ), f"{from_unit!r}->{to_unit!r}: expected legacy gap (None), got {legacy!r}"
        assert new is not None and math.isfinite(
            new
        ), f"{from_unit!r}->{to_unit!r}: pint should convert, got {new!r}"
        return

    if legacy is None or new is None:
        assert (
            legacy is None and new is None
        ), f"{from_unit!r}->{to_unit!r}: legacy={legacy!r} new={new!r}"
    else:
        assert math.isclose(
            legacy, new, rel_tol=1e-9
        ), f"{from_unit!r}->{to_unit!r}: legacy={legacy!r} new={new!r}"


def test_every_legacy_unit_is_pint_recognized() -> None:
    """Each legacy unit must translate to a pint-recognised unit (self-factor 1.0)."""

    unmapped = []
    for token in UNIT_TO_BASE:
        normalized = normalize_unit_text(token)
        factor = _pint_unit_conversion_factor(normalized, normalized)
        if factor is None or not math.isclose(factor, 1.0, rel_tol=1e-12):
            unmapped.append((token, normalized, factor))
    assert not unmapped, f"units without a value-matching pint mapping: {unmapped}"


def test_mmHg_and_torr_alias_to_pint_torr() -> None:
    expected = 101325.0 / 760.0
    assert _legacy_unit_conversion_factor("mmHg", "Pa") == pytest.approx(expected)
    assert _legacy_unit_conversion_factor("torr", "Pa") == pytest.approx(expected)
    assert _pint_unit_conversion_factor("mmHg", "Pa") == pytest.approx(expected)
    assert _pint_unit_conversion_factor("torr", "Pa") == pytest.approx(expected)


def test_gauss_alias_preserves_legacy_factor() -> None:
    # 1 G = 1e-4 T; pint's CGS ``gauss`` would raise DimensionalityError instead.
    assert _legacy_unit_conversion_factor("G", "T") == pytest.approx(1e-4)
    assert _pint_unit_conversion_factor("G", "T") == pytest.approx(1e-4)
    assert _pint_unit_conversion_factor("T", "G") == pytest.approx(1e4)


def test_mhz_is_an_intentional_documented_divergence() -> None:
    # Legacy cannot convert MHz (missing from UNIT_NAMESPACE); pint can. The
    # owner approved letting pint's correct value win.
    assert _legacy_unit_conversion_factor("MHz", "Hz") is None
    assert _pint_unit_conversion_factor("MHz", "Hz") == pytest.approx(1e6)
    assert _pint_unit_conversion_factor("MHz", "kHz") == pytest.approx(1e3)


def test_incompatible_dimensions_return_none() -> None:
    # A dimensional mismatch must be ``None`` on both sides (no spurious factor).
    assert _legacy_unit_conversion_factor("m", "s") is None
    assert _pint_unit_conversion_factor("m", "s") is None


def test_unparseable_or_empty_tokens_return_none() -> None:
    assert _pint_unit_conversion_factor("", "m") is None
    assert _pint_unit_conversion_factor("m", "") is None
    assert _pint_unit_conversion_factor("definitely_not_a_unit", "m") is None


def test_token_translation_handles_micro_prefix_and_aliases() -> None:
    # Micro prefix: ASCII ``u`` leading a unit-letter run becomes ``µ``.
    assert _prkit_token_to_pint("um") == "µm"
    assert _prkit_token_to_pint("uA") == "µA"
    assert _prkit_token_to_pint("kg*um/s") == "kg*µm/s"
    # Irregular tokens route through the alias map.
    assert _prkit_token_to_pint("mmHg") == _PRKIT_TO_PINT["mmHg"] == "torr"
    assert _prkit_token_to_pint("G") == _PRKIT_TO_PINT["G"] == "prkit_gauss"
    # Empty / whitespace-only input is guarded before pint sees it.
    assert _prkit_token_to_pint("") is None
    assert _prkit_token_to_pint("   ") is None
    assert _prkit_token_to_pint(None) is None
