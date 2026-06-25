"""Reference-conditioned unit-alias resolution for the implicit-unit-alias bridge.

Phase 4 deliberately keeps ambiguous bare tokens (``S``, ``M``, ``c`` ...) out of
standalone quantity parsing -- in isolation they are far more often symbolic
variables than units, so accepting them would hijack symbolic answers. But when a
prediction like ``"5 S"`` is compared against a *reference* of ``"5 siemens"``, the
reference fixes the unit and ``"5 S"`` almost certainly means 5 siemens.

This module supplies the small, curated, **case-insensitive** alias map (keyed by
each unit's full name, normalized to the gold's canonical form) and the helpers the
two comparison bridges use to recover that recall at comparison time only --
``different_object_kind._compare_implicit_unit_alias`` (one side is descriptive
text) and ``same_object_kind.compare_quantity_implicit_unit_alias`` (both parse as
quantities with incompatible dimensions). Standalone parsing is never affected.
"""

from __future__ import annotations

import re

from ..quantities.units import canonicalize_unit_alias

# Curated alias map: reference unit -> bare tokens that may denote it in a
# prediction. Keys are the unit's **full name** for readability; they are
# normalized through ``canonicalize_unit_alias`` at import (see
# ``_NORMALIZED_ALIASES``) so a full-name key matches however the gold parses
# (e.g. ``"5 seconds"`` parses to unit ``"s"``, and ``"seconds"`` canonicalizes
# to ``"s"``). Values are matched **case-insensitively** (listing ``"s"`` covers
# ``"s"`` and ``"S"``), and the *reference's* unit selects the entry, so a bare
# ``S`` resolves to siemens next to a siemens reference and to second next to a
# second reference. Opt-in: only units listed here are ever rescued.
#
# To extend: add ``"<full unit name>": {<short/symbol surfaces>}``. List only
# short/symbol forms a prediction might write (including the symbol itself, e.g.
# ``seconds <- {sec, s}``); never list full words (``"second"`` under
# ``siemens``) -- they are unambiguous and would wrongly rescue ``"5 second"``
# against a siemens gold. pint can bootstrap-suggest a unit's symbol
# (``ureg.get_name``). High-ambiguity letters (``c`` speed-of-light, ``k``
# Boltzmann/kilo, ``a`` are/annum) are intentionally excluded.
IMPLICIT_UNIT_ALIASES: dict[str, frozenset[str]] = {
    "seconds": frozenset({"sec", "s"}),  # time: S/s/sec written for seconds
    "siemens": frozenset({"s"}),  # conductance: bare S/s
    "molar": frozenset({"m"}),  # concentration (molarity): bare M/m
    "mol/L": frozenset({"m"}),
    "poise": frozenset({"p"}),  # dynamic viscosity: bare P/p
    "debye": frozenset({"d"}),  # dipole moment: bare D/d
}

# Built once at import, keyed by the canonical form of each full-name key, so a
# lookup on the gold's parsed unit (also canonicalized) matches.
_NORMALIZED_ALIASES: dict[str, frozenset[str]] = {
    canonicalize_unit_alias(key): tokens
    for key, tokens in IMPLICIT_UNIT_ALIASES.items()
}

# A single ``<number> <bare-token>`` surface (e.g. ``"5 S"``). The token is a run
# of unit-like letters with no embedded whitespace, so multi-word prose,
# expressions, and relations decline.
_NUMBER_TOKEN_RE = re.compile(
    r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s+([A-Za-zµμΩ°%]+)\s*$"
)


def split_number_and_token(raw_text: str | None) -> tuple[str, str] | None:
    """Split ``"5 S"`` into ``("5", "S")``; return ``None`` if not numberish."""

    if not raw_text:
        return None
    match = _NUMBER_TOKEN_RE.match(raw_text)
    if match is None:
        return None
    return match.group(1), match.group(2)


def resolve_alias_unit(token: str, ref_unit: str | None) -> str | None:
    """Return ``ref_unit`` when ``token`` is a case-insensitive alias of it.

    The reference unit is canonicalized to the (also-canonicalized) alias-map key,
    so a full-name key such as ``"seconds"`` matches a gold answer that parses to
    ``"s"``. Returns ``None`` when the reference unit is not in the map or the
    token is not one of its aliases.
    """

    if not ref_unit:
        return None
    aliases = _NORMALIZED_ALIASES.get(canonicalize_unit_alias(ref_unit))
    if not aliases:
        return None
    folded = token.casefold()
    if folded in {alias.casefold() for alias in aliases}:
        return ref_unit
    return None


__all__ = [
    "IMPLICIT_UNIT_ALIASES",
    "resolve_alias_unit",
    "split_number_and_token",
]
