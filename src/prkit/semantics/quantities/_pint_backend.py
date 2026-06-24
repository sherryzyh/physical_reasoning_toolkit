"""Private pint-backed adapter for unit recognition and conversion.

This module is the *only* place pint is touched for prkit's physical-quantity
pipeline. It exposes a thin, ``None``-returning adapter that consumes
prkit-canonical unit tokens (the output of
:func:`prkit.semantics.quantities.units.normalize_unit_text`) and translates
them to pint, never the reverse: pint quantities never leave this module, so
stored snapshots stay ``float`` + ``str`` and the registry's pickling/registry
serialization traps cannot bite.

Design notes:

* The registry is a *private* module-level :class:`pint.UnitRegistry` -- not the
  application registry. We never pickle quantities, so a hermetic registry
  avoids global-state coupling, and every operation stays within one registry
  (pint raises on cross-registry arithmetic).
* It is warmed once (see :func:`warm`, driven from ``units.py`` at import) by
  parsing the known vocabulary, which primes pint's parse cache before any
  concurrent N4 batch worker can race on it (the cache is not documented as
  thread-safe).
* Every pint exception is swallowed to ``None`` / ``False``; empty or
  whitespace-only input is guarded to ``None`` before pint sees it (``ureg("")``
  silently yields ``1 dimensionless``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import pint

_UREG: pint.UnitRegistry = pint.UnitRegistry()

# pint's built-in ``gauss`` is a CGS-Gaussian unit whose dimensionality is
# incompatible with ``tesla`` (``Q(1, "T").to("gauss")`` raises
# DimensionalityError). prkit treats ``1 G = 1e-4 T`` (the SI convention) and
# converts ``G <-> T`` today, so define a private SI-convention gauss that
# reproduces the legacy factor exactly.
_UREG.define("prkit_gauss = 1e-4 * tesla = _prkit_gauss")

# Irregular prkit-canonical token -> pint name. Each value's pint definition
# matches the legacy numeric value exactly (parity by alias, not allow-list).
_PRKIT_TO_PINT: dict[str, str] = {
    # prkit defines mmHg == torr == 101325/760 Pa exactly; pint's own ``mmHg``
    # is 133.322387 Pa and would diverge, so route both to pint ``torr``.
    "mmHg": "torr",
    "torr": "torr",
    # SI-convention gauss, see registry definition above.
    "G": "prkit_gauss",
}

# Matches a micro-prefix ``u`` that leads a unit-letter run (``um`` -> ``µm``,
# ``uA`` -> ``µA``, ``kg*um/s`` -> ``kg*µm/s``). pint does not accept ASCII
# ``u`` for the micro prefix.
_MICRO_PREFIX_RE = re.compile(r"(?<![A-Za-zµ])u(?=[A-Za-z])")


def _prkit_token_to_pint(token: str | None) -> str | None:
    """Translate a normalized prkit unit token to a pint-parseable token.

    ``token`` is expected to be the output of ``normalize_unit_text`` (``**``
    powers, ``Ω`` -> ``ohm``, ``%`` -> ``percent``, ``μ`` -> ``u``). Returns
    ``None`` for empty/whitespace-only input.
    """

    if token is None:
        return None
    stripped = token.strip()
    if not stripped:
        return None
    if stripped in _PRKIT_TO_PINT:
        return _PRKIT_TO_PINT[stripped]
    return _MICRO_PREFIX_RE.sub("µ", stripped)


def unit_conversion_factor(from_unit: str | None, to_unit: str | None) -> float | None:
    """Multiplicative factor converting ``from_unit`` -> ``to_unit``.

    Inputs are normalized prkit tokens. Returns ``None`` when either token is
    unmappable/unrecognized, the dimensions are incompatible, or an offset
    (affine) unit is involved -- a single factor cannot express an offset.
    """

    from_pint = _prkit_token_to_pint(from_unit)
    to_pint = _prkit_token_to_pint(to_unit)
    if from_pint is None or to_pint is None:
        return None
    try:
        factor = float(_UREG.Quantity(1.0, from_pint).to(to_pint).magnitude)
        offset = float(_UREG.Quantity(0.0, from_pint).to(to_pint).magnitude)
    except Exception:
        return None
    if abs(offset) > 1e-12 * (abs(factor) + 1.0):
        return None
    return factor


def convert_numeric_value(
    value: float, from_unit: str | None, to_unit: str | None
) -> float | None:
    """Convert ``value`` from ``from_unit`` to ``to_unit`` (true conversion).

    Unlike :func:`unit_conversion_factor`, this performs the full conversion, so
    it handles offset units (e.g. ``degC`` -> ``K``) correctly. Inputs are
    normalized prkit tokens. Returns ``None`` on any failure.
    """

    from_pint = _prkit_token_to_pint(from_unit)
    to_pint = _prkit_token_to_pint(to_unit)
    if from_pint is None or to_pint is None:
        return None
    try:
        return float(_UREG.Quantity(value, from_pint).to(to_pint).magnitude)
    except Exception:
        return None


def is_recognized_unit(token: str | None) -> bool:
    """Return whether ``token`` is a unit pint can parse.

    Inputs are normalized prkit tokens. Returns ``False`` for empty input or any
    parse failure.
    """

    pint_token = _prkit_token_to_pint(token)
    if pint_token is None:
        return False
    try:
        _UREG.Unit(pint_token)
    except Exception:
        return False
    return True


def warm(tokens: Iterable[str | None]) -> None:
    """Prime the registry parse cache for ``tokens`` (best-effort, None-safe)."""

    for token in tokens:
        pint_token = _prkit_token_to_pint(token)
        if pint_token is None:
            continue
        try:
            _UREG.Unit(pint_token)
        except Exception:
            continue


def _warm_known_vocabulary() -> None:
    """Warm the registry over prkit's known unit vocabulary (best-effort).

    Runs in this module's body, which Python's import lock executes exactly once
    before any worker can race on the (undocumented-thread-safety) parse cache.
    ``units`` is imported lazily here rather than warming from ``units.py`` at its
    import time, so the conversion front-end stays pint-free for the light
    ``prkit.verify`` path until a real conversion pulls this backend in. By then
    ``units`` is fully initialized, so this import is not circular.
    """

    try:
        from .units import UNIT_TO_BASE, normalize_unit_text
    except Exception:
        return
    warm(normalize_unit_text(token) for token in UNIT_TO_BASE)


_warm_known_vocabulary()


# Authoritative denylist of pint-recognized tokens that prkit must NOT treat as
# units when widening parse recognition beyond ``UNIT_TO_BASE``. Two families:
#
# * Ambiguous single letters that double as symbolic variables / physics
#   constants (``c``, ``e``, ``k``, ...) -- pint reads them as units (``c`` =
#   speed-of-light unit, ``e`` = elementary charge, ``M`` = molar, ...), but in
#   free-form physics answers they are far more often variables, so accepting
#   them would hijack symbolic answers into spurious quantities.
# * The revolution family (``rpm``/``rps``/``revolution``/``cycle``/``turn``),
#   which pint couples to radians through a 2pi factor -- a conversion trap.
#
# This is the *authoritative constant*; the Phase-4 contract test derives the
# equivalent set from the installed pint + ``UNIT_TO_BASE`` (literal membership)
# and asserts equality, so a pint upgrade that shifts single-letter recognition
# fails loudly in CI rather than silently changing parse behavior.
PARSE_DENYLIST: frozenset[str] = frozenset(
    {
        # lowercase single letters
        "a",
        "b",
        "c",
        "d",
        "e",
        "k",
        "l",
        "t",
        "u",
        # uppercase single letters
        "B",
        "D",
        "M",
        "P",
        "R",
        "S",
        "U",
        # revolution family (2pi rev<->rad trap)
        "rpm",
        "rps",
        "revolution",
        "cycle",
        "turn",
    }
)


def is_recognized_unit_for_parse(token: str | None) -> bool:
    """Guarded-full recognition gate deciding whether ``token`` may parse as a unit.

    A token is accepted iff it is a known prkit unit (``UNIT_TO_BASE``) or pint
    recognizes it *and* it is not denylisted. Inputs are normalized prkit tokens
    (the output of ``normalize_unit_text`` / ``canonicalize_unit_alias``).

    The ``UNIT_TO_BASE`` fast path means callers that have already cleared known
    units never reach pint; this keeps the light ``prkit.verify`` path off pint
    for the common case (see ``tests/prkit/verify/test_import_isolation.py``).
    """

    if token is None:
        return False
    from .units import UNIT_TO_BASE

    if token in UNIT_TO_BASE:
        return True
    return is_recognized_unit(token) and token not in PARSE_DENYLIST


__all__ = [
    "PARSE_DENYLIST",
    "convert_numeric_value",
    "is_recognized_unit",
    "is_recognized_unit_for_parse",
    "unit_conversion_factor",
    "warm",
]
