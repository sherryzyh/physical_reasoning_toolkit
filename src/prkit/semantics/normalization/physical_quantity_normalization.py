"""Numeric and physical-quantity normalization helpers."""

from __future__ import annotations

import math
import re

from ..units import UNIT_ALIASES as _UNIT_ALIASES
from ..units import UNIT_TO_BASE as _UNIT_TO_BASE
from ..units import canonicalize_unit_alias
from .math_text_normalization import _UNICODE_WHITESPACE, _extract_math_content

_FRAC_LATEX_PATTERN = re.compile(
    r"^\s*\\frac\s*\{(-?\d+(?:\.\d+)?)\}\s*\{(-?\d+(?:\.\d+)?)\}\s*$"
)
_NUM_TOKEN = r"(?:\d+(?:\.\d*)?|\.\d+)"
_SIGNED_NUM_TOKEN = rf"[+-]?{_NUM_TOKEN}"
_SIGNED_NUM_OR_E_TOKEN = rf"{_SIGNED_NUM_TOKEN}(?:e[+-]?\d+)?"
_SCI_E_TOKEN = rf"{_SIGNED_NUM_TOKEN}e[+-]?\d+"
_SCI_10_TOKEN = rf"{_SIGNED_NUM_TOKEN}\*10\^[+-]?\d+"
_FRACTION_TOKEN = rf"{_SIGNED_NUM_OR_E_TOKEN}/{_SIGNED_NUM_OR_E_TOKEN}"
_POWER_TOKEN = rf"{_SIGNED_NUM_TOKEN}(?:\*\*|\^)[\d\s()+*\-]+"

_SIMPLE_NUMBER_RE = re.compile(rf"^{_SIGNED_NUM_OR_E_TOKEN}$", re.IGNORECASE)
_FRACTION_RE = re.compile(
    rf"^({_SIGNED_NUM_OR_E_TOKEN})/({_SIGNED_NUM_OR_E_TOKEN})$",
    re.IGNORECASE,
)
_SCI_10_RE = re.compile(
    rf"^({_SIGNED_NUM_TOKEN})\*10\^([+-]?\d+)$",
    re.IGNORECASE,
)
_POWER_RE = re.compile(
    rf"^({_SIGNED_NUM_TOKEN})(?:\*\*|\^)([\d\s()+*\-]+)$",
    re.IGNORECASE,
)
_SUPERSCRIPT_TRANSLATION = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁺": "+",
        "⁻": "-",
    }
)
_QUANTITY_PATTERN = re.compile(
    rf"^(?:{_SCI_10_TOKEN}|{_FRACTION_TOKEN}|{_POWER_TOKEN}|{_SCI_E_TOKEN}|{_SIGNED_NUM_OR_E_TOKEN})([A-Za-zΩμµ°/%].*)$",
    re.IGNORECASE,
)
_NUMERIC_PREFIX_RE = re.compile(
    rf"^\s*({_SCI_10_TOKEN}|{_FRACTION_TOKEN}|{_POWER_TOKEN}|{_SCI_E_TOKEN}|{_SIGNED_NUM_OR_E_TOKEN})\s*(.*)$",
    re.DOTALL | re.IGNORECASE,
)


def _replace_superscript_exponents(text: str) -> str:
    """Rewrite Unicode superscript exponents into ASCII ``^`` notation."""

    return re.sub(
        r"(?<=\w|[/%)])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
        lambda match: "^" + match.group(1).translate(_SUPERSCRIPT_TRANSLATION),
        text,
    )


def _canonicalize_quantity_string(
    text: str, *, normalize_ohm_symbol: bool = True
) -> str:
    """Normalize Unicode and lightweight LaTeX variants in quantity text."""

    text = text.strip()
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = _UNICODE_WHITESPACE.sub(" ", text).strip()
    text = text.replace(r"\times", "*").replace(r"\cdot", "*").replace(r"\div", "/")
    text = re.sub(r"\^\{([^{}]+)\}", r"^\1", text)
    text = _replace_superscript_exponents(text)
    text = re.sub(r"(?<=\d)\s*[×·]\s*(?=10\^[-+]?\d+)", "*", text)
    text = re.sub(r"(?<=\d)\s*[xX]\s*(?=10\^[-+]?\d+)", "*", text)
    text = re.sub(r"\s*\*\s*", "*", text)
    text = text.replace("µ", "μ")
    if normalize_ohm_symbol:
        text = text.replace("Ω", "ohm")
    return text.strip()


def _parse_numeric_base(text: str) -> float | None:
    """Parse the scalar portion of a quantity before unit handling is applied."""

    cleaned = _canonicalize_quantity_string(text).replace(" ", "").replace(",", "")

    sci_10_match = _SCI_10_RE.fullmatch(cleaned)
    if sci_10_match:
        try:
            return float(sci_10_match.group(1)) * (10 ** int(sci_10_match.group(2)))
        except ValueError:
            return None

    fraction_match = _FRACTION_RE.fullmatch(cleaned)
    if fraction_match:
        try:
            numerator = float(fraction_match.group(1))
            denominator = float(fraction_match.group(2))
            if denominator == 0:
                return None
            return numerator / denominator
        except ValueError:
            return None

    if _SIMPLE_NUMBER_RE.fullmatch(cleaned):
        try:
            return float(cleaned)
        except ValueError:
            return None

    return None


def _format_numeric_value(value: float, source_str: str | None = None) -> str:
    """Render a numeric value while preserving obvious source precision cues."""

    if not math.isfinite(value):
        return str(value)
    if source_str is not None and "." in source_str:
        source_clean = source_str.strip().lstrip("-").lstrip("+")
        if re.fullmatch(r"\d+\.\d+", source_clean):
            decimal_places = len(source_clean.split(".")[1])
            return f"{value:.{decimal_places}f}"
    if value == int(value):
        return str(int(value))
    return format(value, ".15g")


def normalize_number(answer_str: str) -> float:
    """Normalize a number-like answer surface into ``float`` or ``nan``."""

    cleaned, _ = _extract_math_content(answer_str)
    cleaned = _canonicalize_quantity_string(cleaned)

    fraction_match = _FRAC_LATEX_PATTERN.fullmatch(cleaned.strip())
    if fraction_match:
        fraction_str = (
            f"{fraction_match.group(1).strip()}/{fraction_match.group(2).strip()}"
        )
        value = _parse_numeric_base(fraction_str)
        return value if value is not None else float("nan")

    value = _parse_numeric_base(cleaned)
    return value if value is not None else float("nan")


def _parse_exponent(exp_str: str) -> int | None:
    """Parse a simple integer exponent from text, optionally evaluating arithmetic."""

    if exp_str is None:
        return None
    text = _canonicalize_quantity_string(exp_str).strip()
    while text.startswith("(") and text.endswith(")") and len(text) > 2:
        inner = text[1:-1].strip()
        if inner.count("(") != inner.count(")"):
            break
        text = inner
    try:
        return int(text)
    except ValueError:
        pass
    if re.match(r"^[\d\s+\-*()]+$", text):
        try:
            return int(eval(text))
        except (ValueError, TypeError, ZeroDivisionError, SyntaxError):
            return None
    return None


def _canonicalize_unit_alias(token: str) -> str:
    """Map unit aliases onto the canonical symbol used by PRKit semantics."""

    return canonicalize_unit_alias(token)


def _split_unit_exponent(token: str) -> tuple[str, int]:
    """Split one unit token into its base symbol and integer exponent."""

    token = _replace_superscript_exponents(token.strip())
    token = re.sub(r"\^\{([^{}]+)\}", r"^\1", token)
    match = re.fullmatch(r"(.+?)\^([+-]?\d+)", token)
    if match:
        return match.group(1).strip(), int(match.group(2))
    return token, 1


def _parse_unit_expression(unit_raw: str) -> list[tuple[int, str, int]]:
    """Parse a compound unit string into signed unit factors."""

    text = _canonicalize_quantity_unit_text(unit_raw)
    text = _canonicalize_quantity_string(text).strip()
    text = text.replace("·", "*")
    text = re.sub(r"\s+", "*", text)
    text = re.sub(r"\*+", "*", text).strip("*")
    if not text:
        return []

    terms: list[tuple[int, str, int]] = []
    for index, side in enumerate(text.split("/")):
        sign = 1 if index == 0 else -1
        raw_tokens = [
            raw_token.strip() for raw_token in side.split("*") if raw_token.strip()
        ]
        collapsed_tokens: list[str] = []
        token_index = 0
        while token_index < len(raw_tokens):
            if token_index + 1 < len(raw_tokens):
                joined = f"{raw_tokens[token_index]}{raw_tokens[token_index + 1]}"
                if _canonicalize_unit_alias(joined) in _UNIT_TO_BASE:
                    collapsed_tokens.append(joined)
                    token_index += 2
                    continue
            collapsed_tokens.append(raw_tokens[token_index])
            token_index += 1

        for raw_token in collapsed_tokens:
            token = raw_token.strip()
            if not token:
                continue
            symbol, exponent = _split_unit_exponent(token)
            symbol = _canonicalize_unit_alias(symbol)
            if symbol:
                terms.append((sign, symbol, exponent))
    return terms


def _normalize_unit_expression(unit_raw: str) -> tuple[float, str]:
    """Project a unit expression onto canonical base symbols and a scale factor."""

    terms = _parse_unit_expression(unit_raw)
    if not terms:
        return 1.0, re.sub(r"\s+", " ", unit_raw).strip()

    total_scale = 1.0
    net_exponents: dict[str, int] = {}
    for sign, symbol, exponent in terms:
        base_symbol, factor = _UNIT_TO_BASE.get(symbol, (symbol, 1.0))
        signed_exponent = exponent if sign > 0 else -exponent
        total_scale *= factor**signed_exponent
        net_exponents[base_symbol] = net_exponents.get(base_symbol, 0) + signed_exponent

    numerator_parts: list[str] = []
    denominator_parts: list[str] = []
    for symbol in sorted(net_exponents):
        net = net_exponents[symbol]
        if net > 0:
            numerator_parts.append(symbol if net == 1 else f"{symbol}^{net}")
        elif net < 0:
            denom_exp = -net
            denominator_parts.append(
                symbol if denom_exp == 1 else f"{symbol}^{denom_exp}"
            )

    if not numerator_parts:
        numerator_parts = ["1"]
    unit_str = "*".join(numerator_parts)
    if denominator_parts:
        unit_str += "/" + "*".join(denominator_parts)
    unit_str = unit_str.replace("1/", "/")
    if unit_str.startswith("/"):
        unit_str = "1" + unit_str
    return total_scale, unit_str


def _split_numeric_and_unit(text: str) -> tuple[str, str] | None:
    """Split a quantity surface into its numeric and unit substrings."""

    compact = _canonicalize_quantity_string(text).strip()
    match = _NUMERIC_PREFIX_RE.match(compact)
    if not match:
        return None
    numeric_part = match.group(1).strip()
    unit_part = match.group(2).strip()
    if not unit_part:
        return None
    if not re.match(r"^[A-Za-zμµ°/%]", unit_part.replace(" ", "")):
        return None
    return numeric_part, unit_part


def _evaluate_numeric_expression(numeric_part: str) -> float | None:
    """Evaluate the numeric portion of a quantity surface."""

    text = _canonicalize_quantity_string(numeric_part).replace(" ", "")
    base_value = _parse_numeric_base(text)
    if base_value is not None:
        return base_value

    power_match = _POWER_RE.fullmatch(text)
    if not power_match:
        return None
    base_str, exp_str = power_match.groups()
    base_value = _parse_numeric_base(base_str)
    exponent = _parse_exponent(exp_str)
    if base_value is None or exponent is None:
        return None
    try:
        sign = -1 if base_value < 0 else 1
        return sign * (abs(base_value) ** exponent)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None


_SYMBOLIC_UNIT_REJECT = re.compile(r"[_(){}\\]")

_QUANTITY_APPROX_PREFIX_RE = re.compile(
    r"^\s*(?:\\approx|≈|~|\\sim|approx(?:imately)?)\s*",
    re.IGNORECASE,
)
_RELATION_WRAPPED_QUANTITY_RE = re.compile(
    r"^(?P<lhs>.+?)\s*(?P<op>=|\\approx|≈)\s*(?P<rhs>.+?)\s*$",
    re.DOTALL,
)
_RELATION_WRAPPED_TARGET_RE = re.compile(
    r"^(?:[A-Za-z\u0370-\u03FF][A-Za-z0-9_]*|\\[A-Za-z]+(?:_[A-Za-z0-9]+|_\{[^{}]+\})?)$"
)


def _canonicalize_quantity_unit_text(
    text: str,
    *,
    normalize_ohm_symbol: bool = True,
) -> str:
    """Normalize LaTeX-heavy unit text into a parser-friendly unit surface."""

    normalized = str(text)
    normalized = re.sub(
        r"\\mathring\s*\{\s*\\mathrm\s*\{\s*A\s*\}\s*\}", " angstrom ", normalized
    )
    normalized = re.sub(r"\\mathring\s*\{\s*A\s*\}", " angstrom ", normalized)
    normalized = re.sub(r"\\AA\b", " angstrom ", normalized)
    normalized = re.sub(r"\\angstrom\b", " angstrom ", normalized)
    normalized = normalized.replace("Å", " angstrom ")
    ohm_surface = " ohm " if normalize_ohm_symbol else " Ω "
    normalized = re.sub(r"\\Omega\b", ohm_surface, normalized)
    normalized = re.sub(r"\\ohm\b", ohm_surface, normalized)
    normalized = normalized.replace("Ω", ohm_surface)
    normalized = re.sub(r"\^\s*\\circ\b", " deg", normalized)
    normalized = re.sub(r"\^\s*\{\\circ\}", " deg", normalized)
    normalized = re.sub(r"\\circ\b", " deg", normalized)
    normalized = re.sub(r"\\(?:,|;|:|!| |quad|qquad)\s*", " ", normalized)
    normalized = normalized.replace("~", " ")
    normalized = re.sub(r"\\mu(?=\s*[A-Za-z])", "μ", normalized)
    normalized = re.sub(r"(?<![A-Za-z])(μ|u)\s+(?=[A-Za-z])", r"\1", normalized)
    normalized = re.sub(
        r"\\(?:mathrm|textrm|text|operatorname)\s*\{([^{}]*)\}", r"\1", normalized
    )
    return re.sub(r"\s+", " ", normalized).strip()


def _strip_quantity_approximation_prefix(text: str) -> str:
    """Remove leading approximation markers from quantity text."""

    return _QUANTITY_APPROX_PREFIX_RE.sub("", text, count=1).strip()


def _looks_like_relation_wrapped_target(text: str) -> bool:
    """Return whether a relation-wrapped quantity has a plausible LHS target."""

    compact = re.sub(r"\s+", "", text.strip())
    if (
        not compact
        or len(compact) > 64
        or any(token in compact for token in (",", ";", "\n", "<", ">"))
    ):
        return False
    if _try_parse_physical_quantity(compact) is not None:
        return False
    return bool(_RELATION_WRAPPED_TARGET_RE.fullmatch(compact))


def _try_parse_relation_wrapped_quantity(
    clean_str: str,
) -> tuple[str, str | None] | None:
    """Parse quantities embedded in relation wrappers such as ``V = 5 V``."""

    text, _ = _extract_math_content(clean_str)
    text = _UNICODE_WHITESPACE.sub(" ", text).strip()
    text = _canonicalize_quantity_unit_text(text)

    stripped_prefix = _strip_quantity_approximation_prefix(text)
    if stripped_prefix and stripped_prefix != text:
        parsed = _try_parse_physical_quantity(stripped_prefix)
        if parsed is not None:
            return parsed, None

    match = _RELATION_WRAPPED_QUANTITY_RE.match(text)
    if match is None:
        return None
    lhs_text = match.group("lhs").strip()
    rhs_text = match.group("rhs").strip()
    if not _looks_like_relation_wrapped_target(lhs_text):
        return None
    parsed_rhs = _try_parse_physical_quantity(
        _strip_quantity_approximation_prefix(rhs_text)
    )
    if parsed_rhs is None:
        return None
    return parsed_rhs, lhs_text


def extract_relation_wrapped_quantity_target(text: str) -> str | None:
    """Extract the target variable from a relation-wrapped quantity, if present."""

    parsed = _try_parse_relation_wrapped_quantity(text)
    if parsed is None:
        return None
    return parsed[1]


def _is_valid_unit_string(unit_raw: str) -> bool:
    """Return whether a unit string can be parsed without symbolic leftovers."""

    if _SYMBOLIC_UNIT_REJECT.search(unit_raw):
        return False
    terms = _parse_unit_expression(unit_raw)
    if not terms:
        return False
    return all(
        re.fullmatch(r"[A-Za-zΩμµ°/%]+", symbol) and symbol in _UNIT_TO_BASE
        for _sign, symbol, _exp in terms
    )


def _try_parse_physical_quantity(clean_str: str) -> str | None:
    """Normalize a raw quantity surface into canonical ``value unit`` text."""

    text = _UNICODE_WHITESPACE.sub(" ", clean_str).strip()
    text = re.sub(r"\s+", " ", text).strip()
    text = _canonicalize_quantity_unit_text(text)
    text = text.replace(r"\cdot", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = _strip_quantity_approximation_prefix(text)

    def _replace_frac_unit(match: re.Match) -> str:
        return f"{match.group(1).strip()}/{match.group(2).strip()}"

    text = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", _replace_frac_unit, text)
    text = _canonicalize_quantity_string(text)

    parts = _split_numeric_and_unit(text)
    if parts is None:
        return None
    numeric_part, unit_raw = parts
    if not _is_valid_unit_string(unit_raw):
        return None

    numeric_value = _evaluate_numeric_expression(numeric_part)
    if numeric_value is None:
        return None

    unit_scale, canonical_unit = _normalize_unit_expression(unit_raw)
    numeric_value *= unit_scale
    source_hint = numeric_part if unit_scale == 1.0 else None
    return (
        f"{_format_numeric_value(numeric_value, source_hint)} {canonical_unit}".strip()
    )


def _try_parse_number_only(clean_math: str) -> float | None:
    """Parse numeric math text that does not carry an explicit unit."""

    text = _canonicalize_quantity_string(clean_math)
    text = text.replace(" ", "").replace(",", "")
    value = _parse_numeric_base(text)
    if value is not None:
        return value
    return _evaluate_numeric_expression(text)


def _normalize_physical_quantity(clean_str: str) -> str:
    """Best-effort quantity normalization with relation-wrapped fallback."""

    parsed = _try_parse_physical_quantity(clean_str)
    if parsed is not None:
        return parsed
    relation_wrapped = _try_parse_relation_wrapped_quantity(clean_str)
    if relation_wrapped is not None:
        return relation_wrapped[0]
    return re.sub(r"\s+", " ", _canonicalize_quantity_string(clean_str)).strip()


def parse_physical_quantity(text: str) -> tuple[float | None, str, str]:
    """Parse a normalized physical quantity as ``(numeric_value, unit, numeric_text)``."""

    stripped = str(text).strip()
    parts = stripped.split(None, 1)
    numeric_text = parts[0]
    unit = parts[1].strip() if len(parts) > 1 else ""
    try:
        if "/" in numeric_text and numeric_text.count("/") == 1:
            numerator, denominator = numeric_text.split("/", 1)
            numeric_value = float(numerator.strip()) / float(denominator.strip())
        else:
            numeric_value = float(numeric_text.replace(",", ""))
        return numeric_value, unit, numeric_text
    except (ValueError, ZeroDivisionError):
        return None, stripped, ""


def synthesize_quantity_latex(
    *,
    raw_text: str | None,
    numeric_text: str | None,
    numeric_value: float | None,
    unit: str | None,
) -> str | None:
    """Synthesize a stable LaTeX surface for a normalized physical quantity."""

    raw = None if raw_text is None else str(raw_text).strip()
    if raw and any(token in raw for token in ("\\", "$")):
        return raw

    unit_text = None if unit is None else str(unit).strip()
    if not unit_text:
        return None

    numeric = None if numeric_text is None else str(numeric_text).strip()
    if not numeric and numeric_value is not None:
        numeric = _format_numeric_value(float(numeric_value))
    if not numeric:
        return None

    if unit_text == "deg":
        return f"{numeric}^\\circ"
    if unit_text == "ohm":
        return f"{numeric}\\,\\Omega"

    escaped_unit = unit_text.replace("%", r"\%")
    scientific_match = re.fullmatch(
        r"([+-]?\d+(?:\.\d+)?)e([+-]?\d+)", numeric, re.IGNORECASE
    )
    if scientific_match:
        coefficient = scientific_match.group(1)
        exponent = int(scientific_match.group(2))
        numeric = f"{coefficient}\\times 10^{{{exponent}}}"
    return f"{numeric}\\ \\mathrm{{{escaped_unit}}}"


__all__ = [
    "_FRAC_LATEX_PATTERN",
    "_FRACTION_RE",
    "_FRACTION_TOKEN",
    "_NUMERIC_PREFIX_RE",
    "_NUM_TOKEN",
    "_POWER_RE",
    "_POWER_TOKEN",
    "_QUANTITY_APPROX_PREFIX_RE",
    "_QUANTITY_PATTERN",
    "_RELATION_WRAPPED_QUANTITY_RE",
    "_RELATION_WRAPPED_TARGET_RE",
    "_SCI_10_RE",
    "_SCI_10_TOKEN",
    "_SCI_E_TOKEN",
    "_SIGNED_NUM_OR_E_TOKEN",
    "_SIGNED_NUM_TOKEN",
    "_SIMPLE_NUMBER_RE",
    "_SUPERSCRIPT_TRANSLATION",
    "_SYMBOLIC_UNIT_REJECT",
    "_UNIT_ALIASES",
    "_UNIT_TO_BASE",
    "_canonicalize_quantity_string",
    "_canonicalize_quantity_unit_text",
    "_canonicalize_unit_alias",
    "_evaluate_numeric_expression",
    "_format_numeric_value",
    "_is_valid_unit_string",
    "_looks_like_relation_wrapped_target",
    "_normalize_physical_quantity",
    "_normalize_unit_expression",
    "_parse_exponent",
    "_parse_numeric_base",
    "_parse_unit_expression",
    "_replace_superscript_exponents",
    "_split_numeric_and_unit",
    "_split_unit_exponent",
    "_strip_quantity_approximation_prefix",
    "_try_parse_number_only",
    "_try_parse_physical_quantity",
    "_try_parse_relation_wrapped_quantity",
    "extract_relation_wrapped_quantity_target",
    "normalize_number",
    "parse_physical_quantity",
    "synthesize_quantity_latex",
]
