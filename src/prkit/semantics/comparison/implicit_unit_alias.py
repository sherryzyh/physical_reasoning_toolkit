"""Reference-conditioned unit-alias resolution for the implicit-unit-alias bridge.

Phase 4 deliberately keeps ambiguous bare tokens (``S``, ``M``, ``c`` ...) out of
standalone quantity parsing -- in isolation they are far more often symbolic
variables than units, so accepting them would hijack symbolic answers. But when a
prediction like ``"5 S"`` is compared against a *reference* of ``"5 siemens"``, the
reference fixes the unit and ``"5 S"`` almost certainly means 5 siemens.

This module supplies the small, curated, **case-insensitive** alias map (keyed by
the reference's canonical unit) and the helpers the comparison bridge
(``different_object_kind._compare_implicit_unit_alias``) uses to recover that
recall at comparison time only. Standalone parsing is never affected.
"""

from __future__ import annotations

import re

from ..quantities.units import canonicalize_unit_alias

# Curated alias map: reference canonical unit -> bare tokens that may denote it in
# a prediction. Matching is **case-insensitive** (listing ``"s"`` covers ``"s"``
# and ``"S"``), and the *reference's* unit selects the target, so a bare ``S``
# resolves to siemens next to a siemens reference and to second next to a second
# reference. Opt-in: only units listed here are ever rescued. The owner curates
# this; pint can bootstrap-suggest a unit's symbols.
IMPLICIT_UNIT_ALIASES: dict[str, frozenset[str]] = {
    "siemens": frozenset({"s"}),  # bare S/s -> siemens (conductance)
    "s": frozenset({"s"}),  # bare S/s -> second (time); canonical unit token "s"
    "molar": frozenset({"m"}),  # bare M/m -> molar (concentration)
    "mol/L": frozenset({"m"}),  # bare M/m -> mol/L (concentration)
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

    The reference unit is canonicalized to the alias-map key, so ``"siemens"`` and
    a stored ``"siemens"`` agree. Returns ``None`` when the reference unit is not
    in the map or the token is not one of its aliases.
    """

    if not ref_unit:
        return None
    aliases = IMPLICIT_UNIT_ALIASES.get(canonicalize_unit_alias(ref_unit))
    if aliases is None:
        aliases = IMPLICIT_UNIT_ALIASES.get(ref_unit)
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
