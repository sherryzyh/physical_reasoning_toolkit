"""Shared unit definitions and conversion helpers for PRKit semantics."""

from __future__ import annotations

import re

from sympy import Float, Integer, Rational, Symbol, pi

_TEXT_WRAPPER_RE = re.compile(r"\\(?:mathrm|text|operatorname)\{([^{}]+)\}")
_BOXED_RE = re.compile(r"^\\boxed\{(.+)\}$", re.DOTALL)
_DOLLAR_RE = re.compile(r"^\$(.*)\$$", re.DOTALL)

_U_M = Symbol("unit_meter")
_U_KG = Symbol("unit_kilogram")
_U_S = Symbol("unit_second")
_U_A = Symbol("unit_ampere")
_U_K = Symbol("unit_kelvin")
_U_MOL = Symbol("unit_mole")
_PASCAL_EXPR = _U_KG / (_U_M * _U_S**2)
_JOULE_EXPR = _U_KG * _U_M**2 / _U_S**2

UNIT_ALIASES: dict[str, str] = {
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "sec": "s",
    "secs": "s",
    "second": "s",
    "seconds": "s",
    "minute": "min",
    "minutes": "min",
    "hr": "h",
    "hrs": "h",
    "hour": "h",
    "hours": "h",
    "gram": "g",
    "grams": "g",
    "amp": "A",
    "amps": "A",
    "ampere": "A",
    "amperes": "A",
    "current": "A",
    "volt": "V",
    "volts": "V",
    "newton": "N",
    "newtons": "N",
    "joule": "J",
    "joules": "J",
    "watt": "W",
    "watts": "W",
    "pascal": "Pa",
    "pascals": "Pa",
    "coulomb": "C",
    "coulombs": "C",
    "farad": "F",
    "farads": "F",
    "henry": "H",
    "henries": "H",
    "tesla": "T",
    "teslas": "T",
    "weber": "Wb",
    "webers": "Wb",
    "hertz": "Hz",
    "ohm": "ohm",
    "ohms": "ohm",
    "radian": "rad",
    "radians": "rad",
    "degree": "deg",
    "degrees": "deg",
    "kelvin": "K",
    "kelvins": "K",
    "liter": "L",
    "liters": "L",
    "litre": "L",
    "litres": "L",
    "mole": "mol",
    "moles": "mol",
    "candela": "cd",
    "candelas": "cd",
    "lumen": "lm",
    "lumens": "lm",
    "lux": "lx",
    "angstrom": "angstrom",
    "å": "angstrom",
    "°": "deg",
    "l": "L",
}

UNIT_TO_BASE: dict[str, tuple[str, float]] = {
    "m": ("m", 1.0),
    "cm": ("m", 1e-2),
    "mm": ("m", 1e-3),
    "km": ("m", 1e3),
    "um": ("m", 1e-6),
    "μm": ("m", 1e-6),
    "nm": ("m", 1e-9),
    "pm": ("m", 1e-12),
    "fm": ("m", 1e-15),
    "s": ("s", 1.0),
    "ms": ("s", 1e-3),
    "us": ("s", 1e-6),
    "μs": ("s", 1e-6),
    "ns": ("s", 1e-9),
    "ps": ("s", 1e-12),
    "min": ("s", 60.0),
    "h": ("s", 3600.0),
    "kg": ("kg", 1.0),
    "g": ("kg", 1e-3),
    "mg": ("kg", 1e-6),
    "ug": ("kg", 1e-9),
    "μg": ("kg", 1e-9),
    "A": ("A", 1.0),
    "mA": ("A", 1e-3),
    "uA": ("A", 1e-6),
    "μA": ("A", 1e-6),
    "kA": ("A", 1e3),
    "K": ("K", 1.0),
    "V": ("V", 1.0),
    "mV": ("V", 1e-3),
    "kV": ("V", 1e3),
    "MV": ("V", 1e6),
    "N": ("N", 1.0),
    "kN": ("N", 1e3),
    "MN": ("N", 1e6),
    "mN": ("N", 1e-3),
    "dyn": ("N", 1e-5),
    "J": ("J", 1.0),
    "kJ": ("J", 1e3),
    "MJ": ("J", 1e6),
    "GJ": ("J", 1e9),
    "mJ": ("J", 1e-3),
    "erg": ("J", 1e-7),
    "cal": ("J", 4.184),
    "kcal": ("J", 4184.0),
    "eV": ("eV", 1.0),
    "keV": ("eV", 1e3),
    "MeV": ("eV", 1e6),
    "GeV": ("eV", 1e9),
    "TeV": ("eV", 1e12),
    "W": ("W", 1.0),
    "kW": ("W", 1e3),
    "MW": ("W", 1e6),
    "GW": ("W", 1e9),
    "mW": ("W", 1e-3),
    "Pa": ("Pa", 1.0),
    "kPa": ("Pa", 1e3),
    "MPa": ("Pa", 1e6),
    "GPa": ("Pa", 1e9),
    "hPa": ("Pa", 1e2),
    "atm": ("atm", 1.0),
    "bar": ("bar", 1.0),
    "mbar": ("bar", 1e-3),
    "torr": ("torr", 1.0),
    "mmHg": ("torr", 1.0),
    "C": ("C", 1.0),
    "mC": ("C", 1e-3),
    "μC": ("C", 1e-6),
    "uC": ("C", 1e-6),
    "nC": ("C", 1e-9),
    "pC": ("C", 1e-12),
    "F": ("F", 1.0),
    "mF": ("F", 1e-3),
    "μF": ("F", 1e-6),
    "uF": ("F", 1e-6),
    "nF": ("F", 1e-9),
    "pF": ("F", 1e-12),
    "H": ("H", 1.0),
    "mH": ("H", 1e-3),
    "μH": ("H", 1e-6),
    "uH": ("H", 1e-6),
    "T": ("T", 1.0),
    "mT": ("T", 1e-3),
    "μT": ("T", 1e-6),
    "uT": ("T", 1e-6),
    "G": ("T", 1e-4),
    "Wb": ("Wb", 1.0),
    "Hz": ("Hz", 1.0),
    "kHz": ("Hz", 1e3),
    "MHz": ("Hz", 1e6),
    "GHz": ("Hz", 1e9),
    "THz": ("Hz", 1e12),
    "ohm": ("ohm", 1.0),
    "kohm": ("ohm", 1e3),
    "Mohm": ("ohm", 1e6),
    "mohm": ("ohm", 1e-3),
    "deg": ("deg", 1.0),
    "rad": ("rad", 1.0),
    "mrad": ("rad", 1e-3),
    "angstrom": ("m", 1e-10),
    "L": ("L", 1.0),
    "mL": ("L", 1e-3),
    "mol": ("mol", 1.0),
    "mmol": ("mol", 1e-3),
    "μmol": ("mol", 1e-6),
    "umol": ("mol", 1e-6),
    "cd": ("cd", 1.0),
    "lm": ("lm", 1.0),
    "lx": ("lx", 1.0),
}

UNIT_NAMESPACE = {
    "1": Integer(1),
    "A": _U_A,
    "C": _U_A * _U_S,
    "F": _U_A**2 * _U_S**4 / (_U_KG * _U_M**2),
    "G": Float("1e-4") * _U_KG / (_U_A * _U_S**2),
    "GHz": Float("1e9") / _U_S,
    "GJ": Float("1e9") * _JOULE_EXPR,
    "GPa": Float("1e9") * _PASCAL_EXPR,
    "GW": Float("1e9") * _JOULE_EXPR / _U_S,
    "GeV": Float("1e9") * Float("1.602176634e-19") * _JOULE_EXPR,
    "H": _U_KG * _U_M**2 / (_U_A**2 * _U_S**2),
    "Hz": Integer(1) / _U_S,
    "J": _JOULE_EXPR,
    "K": _U_K,
    "L": Rational(1, 1000) * _U_M**3,
    "MJ": Float("1e6") * _JOULE_EXPR,
    "MN": Float("1e6") * _U_KG * _U_M / _U_S**2,
    "MPa": Float("1e6") * _PASCAL_EXPR,
    "MV": Float("1e6") * _U_KG * _U_M**2 / (_U_A * _U_S**3),
    "MW": Float("1e6") * _JOULE_EXPR / _U_S,
    "MeV": Float("1e6") * Float("1.602176634e-19") * _JOULE_EXPR,
    "Mohm": Float("1e6") * _U_KG * _U_M**2 / (_U_A**2 * _U_S**3),
    "N": _U_KG * _U_M / _U_S**2,
    "Pa": _PASCAL_EXPR,
    "T": _U_KG / (_U_A * _U_S**2),
    "THz": Float("1e12") / _U_S,
    "TeV": Float("1e12") * Float("1.602176634e-19") * _JOULE_EXPR,
    "V": _U_KG * _U_M**2 / (_U_A * _U_S**3),
    "W": _JOULE_EXPR / _U_S,
    "Wb": _U_KG * _U_M**2 / (_U_A * _U_S**2),
    "angstrom": Float("1e-10") * _U_M,
    "atm": Float("101325") * _PASCAL_EXPR,
    "bar": Float("100000") * _PASCAL_EXPR,
    "cal": Float("4.184") * _JOULE_EXPR,
    "cd": Symbol("unit_candela"),
    "cm": Rational(1, 100) * _U_M,
    "deg": pi / Integer(180),
    "dyn": Float("1e-5") * _U_KG * _U_M / _U_S**2,
    "eV": Float("1.602176634e-19") * _JOULE_EXPR,
    "erg": Float("1e-7") * _JOULE_EXPR,
    "fm": Float("1e-15") * _U_M,
    "g": Rational(1, 1000) * _U_KG,
    "h": Integer(3600) * _U_S,
    "hPa": Float("100") * _PASCAL_EXPR,
    "kA": Float("1000") * _U_A,
    "kHz": Float("1000") / _U_S,
    "kJ": Float("1000") * _JOULE_EXPR,
    "kN": Float("1000") * _U_KG * _U_M / _U_S**2,
    "kPa": Float("1000") * _PASCAL_EXPR,
    "kV": Float("1000") * _U_KG * _U_M**2 / (_U_A * _U_S**3),
    "kW": Float("1000") * _JOULE_EXPR / _U_S,
    "kcal": Float("4184") * _JOULE_EXPR,
    "keV": Float("1000") * Float("1.602176634e-19") * _JOULE_EXPR,
    "kg": _U_KG,
    "km": Float("1000") * _U_M,
    "kohm": Float("1000") * _U_KG * _U_M**2 / (_U_A**2 * _U_S**3),
    "lm": Symbol("unit_lumen"),
    "lx": Symbol("unit_lux"),
    "m": _U_M,
    "mA": Rational(1, 1000) * _U_A,
    "mC": Rational(1, 1000) * _U_A * _U_S,
    "mF": Rational(1, 1000) * _U_A**2 * _U_S**4 / (_U_KG * _U_M**2),
    "mH": Rational(1, 1000) * _U_KG * _U_M**2 / (_U_A**2 * _U_S**2),
    "mJ": Rational(1, 1000) * _JOULE_EXPR,
    "mL": Rational(1, 1000000) * _U_M**3,
    "mN": Rational(1, 1000) * _U_KG * _U_M / _U_S**2,
    "mT": Rational(1, 1000) * _U_KG / (_U_A * _U_S**2),
    "mV": Rational(1, 1000) * _U_KG * _U_M**2 / (_U_A * _U_S**3),
    "mW": Rational(1, 1000) * _JOULE_EXPR / _U_S,
    "mbar": Float("100") * _PASCAL_EXPR,
    "mg": Float("1e-6") * _U_KG,
    "min": Integer(60) * _U_S,
    "mm": Rational(1, 1000) * _U_M,
    "mmHg": Rational(101325, 760) * _PASCAL_EXPR,
    "mmol": Rational(1, 1000) * _U_MOL,
    "mohm": Rational(1, 1000) * _U_KG * _U_M**2 / (_U_A**2 * _U_S**3),
    "mol": _U_MOL,
    "ms": Rational(1, 1000) * _U_S,
    "mrad": Rational(1, 1000),
    "nC": Float("1e-9") * _U_A * _U_S,
    "nF": Float("1e-9") * _U_A**2 * _U_S**4 / (_U_KG * _U_M**2),
    "nm": Float("1e-9") * _U_M,
    "ns": Float("1e-9") * _U_S,
    "ohm": _U_KG * _U_M**2 / (_U_A**2 * _U_S**3),
    "pC": Float("1e-12") * _U_A * _U_S,
    "pF": Float("1e-12") * _U_A**2 * _U_S**4 / (_U_KG * _U_M**2),
    "percent": Rational(1, 100),
    "pm": Float("1e-12") * _U_M,
    "ps": Float("1e-12") * _U_S,
    "rad": Integer(1),
    "s": _U_S,
    "torr": Rational(101325, 760) * _PASCAL_EXPR,
    "uA": Float("1e-6") * _U_A,
    "uC": Float("1e-6") * _U_A * _U_S,
    "uF": Float("1e-6") * _U_A**2 * _U_S**4 / (_U_KG * _U_M**2),
    "uH": Float("1e-6") * _U_KG * _U_M**2 / (_U_A**2 * _U_S**2),
    "uT": Float("1e-6") * _U_KG / (_U_A * _U_S**2),
    "ug": Float("1e-9") * _U_KG,
    "um": Float("1e-6") * _U_M,
    "umol": Float("1e-6") * _U_MOL,
    "us": Float("1e-6") * _U_S,
}

PREFERRED_UNITS_BY_DIMENSION: dict[str, str] = {
    "length": "m",
    "area": "m^2",
    "volume": "m^3",
    "time": "s",
    "mass": "kg",
    "electric_current": "A",
    "temperature": "K",
    "amount_of_substance": "mol",
    "luminous_intensity": "cd",
    "velocity": "m/s",
    "acceleration": "m/s^2",
    "angle": "deg",
    "angular_velocity": "rad/s",
    "angular_acceleration": "rad/s^2",
    "frequency": "Hz",
    "period": "s",
    "wavelength": "m",
    "force": "N",
    "momentum": "kg*m/s",
    "impulse": "kg*m/s",
    "energy": "J",
    "power": "W",
    "pressure": "Pa",
    "torque": "N*m",
    "mass_density": "kg/m^3",
    "surface_tension": "N/m",
    "volume_flow_rate": "m^3/s",
    "mass_flow_rate": "kg/s",
    "charge": "C",
    "voltage": "V",
    "resistance": "ohm",
    "capacitance": "F",
    "inductance": "H",
    "magnetic_flux_density": "T",
    "magnetic_flux": "Wb",
    "electric_field": "V/m",
    "luminous_flux": "lm",
    "illuminance": "lx",
}

DIMENSION_ALIASES: dict[str, str] = {
    "length": "length",
    "distance": "length",
    "position": "length",
    "displacement": "length",
    "area": "area",
    "surface_area": "area",
    "cross_sectional_area": "area",
    "volume": "volume",
    "capacity": "volume",
    "time": "time",
    "duration": "time",
    "period": "period",
    "mass": "mass",
    "electric_current": "electric_current",
    "current": "electric_current",
    "temperature": "temperature",
    "amount_of_substance": "amount_of_substance",
    "moles": "amount_of_substance",
    "luminous_intensity": "luminous_intensity",
    "velocity": "velocity",
    "speed": "velocity",
    "acceleration": "acceleration",
    "angle": "angle",
    "angular_velocity": "angular_velocity",
    "angular_speed": "angular_velocity",
    "angular_frequency": "angular_velocity",
    "angular_acceleration": "angular_acceleration",
    "frequency": "frequency",
    "wavelength": "wavelength",
    "force": "force",
    "momentum": "momentum",
    "impulse": "impulse",
    "energy": "energy",
    "work": "energy",
    "heat": "energy",
    "power": "power",
    "pressure": "pressure",
    "stress": "pressure",
    "torque": "torque",
    "moment_of_force": "torque",
    "density": "mass_density",
    "mass_density": "mass_density",
    "surface_tension": "surface_tension",
    "volume_flow_rate": "volume_flow_rate",
    "flow_rate": "volume_flow_rate",
    "mass_flow_rate": "mass_flow_rate",
    "charge": "charge",
    "voltage": "voltage",
    "potential_difference": "voltage",
    "emf": "voltage",
    "resistance": "resistance",
    "capacitance": "capacitance",
    "inductance": "inductance",
    "magnetic_flux_density": "magnetic_flux_density",
    "magnetic_flux": "magnetic_flux",
    "electric_field": "electric_field",
    "luminous_flux": "luminous_flux",
    "illuminance": "illuminance",
}


def _strip_text_wrappers(text: str | None) -> str:
    """Peel common math wrappers such as ``\\boxed{}`` and ``$...$``."""

    if text is None:
        return ""
    stripped = str(text).strip()
    while True:
        boxed_match = _BOXED_RE.match(stripped)
        if boxed_match:
            stripped = boxed_match.group(1).strip()
            continue
        dollar_match = _DOLLAR_RE.match(stripped)
        if dollar_match:
            stripped = dollar_match.group(1).strip()
            continue
        if stripped.startswith(r"\(") and stripped.endswith(r"\)"):
            stripped = stripped[2:-2].strip()
            continue
        if stripped.startswith(r"\[") and stripped.endswith(r"\]"):
            stripped = stripped[2:-2].strip()
            continue
        break
    return stripped


def canonicalize_unit_alias(token: str) -> str:
    """Map one unit token alias onto the shared canonical symbol."""

    token = token.strip().replace("µ", "μ")
    normalized_token = normalize_unit_text(token)
    if normalized_token in UNIT_TO_BASE:
        return normalized_token
    lowered = token.lower()
    if lowered in UNIT_ALIASES:
        return UNIT_ALIASES[lowered]
    return UNIT_ALIASES.get(token, token)


def normalize_dimension_name(dimension: str | None) -> str | None:
    """Normalize a free-form dimension label onto a canonical dimension key."""

    if not dimension:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", str(dimension).strip().lower()).strip("_")
    if not normalized:
        return None
    return DIMENSION_ALIASES.get(normalized, normalized)


def preferred_unit_for_dimension(dimension: str | None) -> str | None:
    """Return the preferred readable canonical unit for a normalized dimension."""

    normalized = normalize_dimension_name(dimension)
    if normalized is None:
        return None
    return PREFERRED_UNITS_BY_DIMENSION.get(normalized)


def normalize_unit_text(text: str | None) -> str:
    """Canonicalize unit text into a parser-friendly symbolic form."""

    if not text:
        return ""
    normalized = _strip_text_wrappers(text)
    normalized = re.sub(
        r"\\mathring\s*\{\s*\\mathrm\s*\{\s*A\s*\}\s*\}", "angstrom", normalized
    )
    normalized = re.sub(r"\\mathring\s*\{\s*A\s*\}", "angstrom", normalized)
    normalized = normalized.replace("\\AA", "angstrom").replace(
        "\\angstrom", "angstrom"
    )
    normalized = _TEXT_WRAPPER_RE.sub(r"\1", normalized)
    normalized = (
        normalized.replace("Ω", "ohm").replace("°", "deg").replace("Å", "angstrom")
    )
    normalized = normalized.replace("μ", "u").replace("µ", "u")
    normalized = normalized.replace("%", "percent")
    normalized = normalized.replace("·", "*").replace("×", "*")
    normalized = re.sub(r"\bper\b", "/", normalized, flags=re.IGNORECASE)
    for alias, canonical in UNIT_ALIASES.items():
        normalized = re.sub(
            rf"\b{re.escape(alias)}\b",
            canonical,
            normalized,
            flags=re.IGNORECASE,
        )
    normalized = re.sub(r"(?<=[A-Za-z0-9\)])\s+(?=[A-Za-z(])", "*", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized.replace("^", "**")


def unit_conversion_factor(from_unit: str | None, to_unit: str | None) -> float | None:
    """Return the multiplicative factor needed to convert between two units."""

    normalized_from = normalize_unit_text(from_unit)
    normalized_to = normalize_unit_text(to_unit)
    if normalized_from == normalized_to:
        return 1.0
    # Imported lazily so the conversion front-end stays pint-free until a real
    # (non-identity) conversion is needed -- keeps the light prkit.verify path
    # off pint (see tests/prkit/verify/test_import_isolation.py).
    from . import _pint_backend

    return _pint_backend.unit_conversion_factor(normalized_from, normalized_to)


def convert_numeric_value(
    value: float, from_unit: str | None, to_unit: str | None
) -> float | None:
    """Convert a numeric value between units when the conversion is dimensionally valid."""

    normalized_from = normalize_unit_text(from_unit)
    normalized_to = normalize_unit_text(to_unit)
    if normalized_from == normalized_to:
        return value
    from . import _pint_backend

    return _pint_backend.convert_numeric_value(value, normalized_from, normalized_to)


__all__ = [
    "DIMENSION_ALIASES",
    "PREFERRED_UNITS_BY_DIMENSION",
    "UNIT_ALIASES",
    "UNIT_NAMESPACE",
    "UNIT_TO_BASE",
    "canonicalize_unit_alias",
    "convert_numeric_value",
    "normalize_dimension_name",
    "normalize_unit_text",
    "preferred_unit_for_dimension",
    "unit_conversion_factor",
]
