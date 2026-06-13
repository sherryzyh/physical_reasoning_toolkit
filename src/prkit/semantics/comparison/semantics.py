"""Self-contained semantic comparison helpers."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from sympy import (
    Abs,
    And,
    Basic,
    Eq,
    Float,
    Ge,
    Gt,
    Integer,
    Le,
    Lt,
    Max,
    Min,
    N,
    Piecewise,
    Rational,
    Symbol,
    acos,
    asin,
    atan,
    cos,
    cosh,
    exp,
    false,
    log,
    oo,
    pi,
    simplify,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    trigsimp,
    true,
)
from sympy.core.relational import Relational
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from ..units import UNIT_NAMESPACE as _SHARED_UNIT_NAMESPACE
from ..units import convert_numeric_value as _shared_convert_numeric_value
from ..units import normalize_unit_text as _shared_normalize_unit_text
from ..units import unit_conversion_factor as _shared_unit_conversion_factor

_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)
_PARSE_GLOBALS = {"Integer": Integer, "Float": Float, "Rational": Rational}
_EXPRESSION_FUNCTIONS = {
    "Abs": Abs,
    "And": And,
    "Eq": Eq,
    "Ge": Ge,
    "Gt": Gt,
    "Le": Le,
    "Lt": Lt,
    "Max": Max,
    "Min": Min,
    "Piecewise": Piecewise,
    "abs": Abs,
    "acos": acos,
    "asin": asin,
    "atan": atan,
    "cos": cos,
    "cosh": cosh,
    "exp": exp,
    "false": false,
    "ln": log,
    "log": log,
    "oo": oo,
    "pi": pi,
    "sin": sin,
    "sinh": sinh,
    "sqrt": sqrt,
    "tan": tan,
    "tanh": tanh,
    "true": true,
    "True": true,
    "False": false,
}
_SYMBOL_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SIMPLE_FRAC_RE = re.compile(r"\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}")
_COMPACT_FRAC_RE = re.compile(r"\\(?:dfrac|tfrac|frac)\s*([A-Za-z0-9])\s*([A-Za-z0-9])")
_FRACTION_COMMAND_RE = re.compile(r"\\(?:dfrac|tfrac|frac)\b")
_SIMPLE_SQRT_RE = re.compile(r"\\sqrt\{([^{}]+)\}")
_INDEXED_SQRT_RE = re.compile(r"\\sqrt\[(?P<index>[^\[\]{}]+)\]\{(?P<body>[^{}]+)\}")
_TEXT_WRAPPER_RE = re.compile(r"\\(?:mathrm|text|operatorname)\{([^{}]+)\}")
_BOXED_RE = re.compile(r"^\\boxed\{(.+)\}$", re.DOTALL)
_DOLLAR_RE = re.compile(r"^\$(.*)\$$", re.DOTALL)
_LATEX_SPACING_RE = re.compile(
    r"\\(?:,|;|:|!|quad|qquad|enspace|thinspace|medspace|thickspace)\b"
)
_LATEX_SUBSCRIPT_BRACED_RE = re.compile(
    r"(?P<base>\\[A-Za-z]+|[A-Za-z][A-Za-z0-9_]*)\s*_\s*\{\s*(?P<sub>[^{}]+)\s*\}"
)
_LATEX_SUBSCRIPT_PLAIN_RE = re.compile(
    r"(?P<base>\\[A-Za-z]+|[A-Za-z][A-Za-z0-9_]*)\s*_\s*(?P<sub>[A-Za-z0-9]+)"
)
_LATEX_ACCENT_RE = re.compile(
    r"\\(?P<decor>dot|ddot|hat|vec|bar|tilde)\s*(?:\{\s*(?P<braced>[^{}]+)\s*\}|(?P<plain>\\[A-Za-z]+|[A-Za-z][A-Za-z0-9_]*))"
    r"(?:\s*_\s*\{\s*(?P<sub>[^{}]+)\s*\}|\s*_\s*(?P<sub_plain>[A-Za-z0-9]+))?"
)
_COMPACT_DECORATED_TOKEN_RE = re.compile(
    r"\b(?P<base>[A-Za-z][A-Za-z0-9]*?)(?P<decor>ddot|dot|hat|vec|bar|tilde)\b"
)
_SUPERSCRIPT_BRACED_RE = re.compile(r"\^\{([^{}]+)\}")
_SUPERSCRIPT_PLAIN_RE = re.compile(r"\^(?P<exp>[A-Za-z0-9_]+)")
_COMMAND_TOKEN_RE = re.compile(r"\\([A-Za-z]+)\b")
_TOP_LEVEL_AND_RE = re.compile(r"\band\b", re.IGNORECASE)
_SIMPLE_SYMBOLIC_LABEL_RE = re.compile(r"[^\W\d]\w*", re.UNICODE)
_RELATION_EQUIVALENCE_CUE_RE = re.compile(
    r"(?i)\b(?:or\s+)?(?:equivalently|i\.?\s*e\.?|that\s+is)\b"
)
_ALTERNATE_FORM_PREFIX_RE = re.compile(
    r"(?i)^(?:or\s+)?(?:equivalently|i\.?\s*e\.?|that\s+is)\b[\s,:;-]*"
)
_RELATION_CONDITION_PREFIX_RE = re.compile(
    r"(?i)^(?:subject\s+to|where|when|for|valid\s+for)\s*[:,-]?\s*"
)
_RELATION_CONSTRAINT_TOKEN_RE = re.compile(
    r"(?:<=|>=|<|>|≤|≥|!=|≠|≈|∈|∉|\\(?:leq|geq|neq|approx|in|notin)\b)"
)
_BARE_FUNCTION_NAMES = (
    "sinh",
    "cosh",
    "tanh",
    "asin",
    "acos",
    "atan",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "log",
    "ln",
    "exp",
    "abs",
    "Abs",
)
_BARE_FUNCTION_POWER_RE = re.compile(
    rf"\b(?P<func>{'|'.join(_BARE_FUNCTION_NAMES)})"
    r"\s*\^\s*(?P<power>\{[^{}]+\}|\([^()]+\)|[A-Za-z0-9_]+)\s*"
    r"(?P<arg>\([^()]+\)|\{[^{}]+\}|[A-Za-z0-9_]+)"
)
_BARE_FUNCTION_ARG_RE = re.compile(
    rf"\b(?P<func>{'|'.join(_BARE_FUNCTION_NAMES)})"
    r"\s+(?P<arg>\{[^{}]+\}|[A-Za-z0-9_]+)"
)
_FUNCTION_COMMANDS = frozenset(_BARE_FUNCTION_NAMES) | {"ln"}
_SHORT_SYMBOL_RUN_RE = re.compile(r"(?<!\\)\b[A-Za-z]{2,}\b")
_TRIG_FUNCTION_RE = re.compile(r"\b(?:sin|cos|tan|asin|acos|atan|sinh|cosh|tanh)\b")
_ALIAS_BOUNDARY_TOKEN_RE = re.compile(r"[A-Za-z0-9_]")
_DEFINITION_LIKE_TOKEN_RE = re.compile(r"(?<![<>=!])\b[\w]+\s*=")
_DECORATION_SUFFIXES = ("_ddot", "_dot", "_hat", "_vec", "_bar", "_tilde")
_TOP_LEVEL_EQUALITY_LIKE_MARKERS = (
    r"\approx",
    r"\simeq",
    r"\equiv",
    "≈",
    "==",
    "=",
)
_PARSE_RESERVED_SYMBOLS = {
    "lambda": "lambda_symbol",
}
_SAFE_SYMBOL_FAMILY_CANONICAL = {
    "eps0": "eps0",
    "epsilon0": "eps0",
    "varepsilon0": "eps0",
    "mu0": "mu0",
}
_ZERO_SUBSCRIPT_TOKEN_RE = re.compile(r"^(?P<base>[A-Za-z][A-Za-z0-9]*)_0$")
_LATEX_SYMBOL_ALIASES = {
    "alpha": "alpha",
    "beta": "beta",
    "gamma": "gamma",
    "Gamma": "Gamma",
    "delta": "delta",
    "Delta": "Delta",
    "epsilon": "epsilon",
    "varepsilon": "varepsilon",
    "zeta": "zeta",
    "eta": "eta",
    "theta": "theta",
    "vartheta": "vartheta",
    "Theta": "Theta",
    "iota": "iota",
    "kappa": "kappa",
    "lambda": "lambda",
    "Lambda": "Lambda",
    "mu": "mu",
    "nu": "nu",
    "xi": "xi",
    "Xi": "Xi",
    "pi": "pi",
    "Pi": "Pi",
    "rho": "rho",
    "varrho": "varrho",
    "sigma": "sigma",
    "varsigma": "varsigma",
    "Sigma": "Sigma",
    "tau": "tau",
    "upsilon": "upsilon",
    "Upsilon": "Upsilon",
    "phi": "phi",
    "varphi": "varphi",
    "Phi": "Phi",
    "chi": "chi",
    "psi": "psi",
    "Psi": "Psi",
    "omega": "omega",
    "Omega": "Omega",
    "partial": "partial",
    "nabla": "nabla",
    "infty": "oo",
}
_PROTECTED_SYMBOL_RUNS = frozenset(
    {
        *(_name for _name in _EXPRESSION_FUNCTIONS if _name.isalpha()),
        "alpha",
        "and",
        "beta",
        "chi",
        "cosh",
        "delta",
        "Delta",
        "eps",
        "epsilon",
        "eta",
        "gamma",
        "Gamma",
        "hbar",
        "lambda",
        "Lambda",
        "nabla",
        "omega",
        "Omega",
        "partial",
        "phi",
        "pi",
        "Pi",
        "psi",
        "Psi",
        "rho",
        "sigma",
        "Sigma",
        "sinh",
        "sqrt",
        "tanh",
        "tau",
        "theta",
        "Theta",
        "not",
        "or",
        "true",
        "false",
        "varepsilon",
        "varphi",
        "vartheta",
        "xi",
        "Xi",
        "zeta",
    }
)

_BOOLEAN_CANONICAL = {
    "true": True,
    "false": False,
    "yes": True,
    "no": False,
}

_SIGN_DIRECTION_CANONICAL = {
    "+": "positive",
    "positive": "positive",
    "positive sign": "positive",
    "-": "negative",
    "negative": "negative",
    "negative sign": "negative",
    "clockwise": "clockwise",
    "counterclockwise": "counterclockwise",
    "anticlockwise": "counterclockwise",
    "to the right": "right",
    "right": "right",
    "rightward": "right",
    "to the left": "left",
    "left": "left",
    "leftward": "left",
    "up": "up",
    "upward": "up",
    "down": "down",
    "downward": "down",
    "into the page": "into_page",
    "into page": "into_page",
    "out of the page": "out_of_page",
    "out of page": "out_of_page",
    "inward": "inward",
    "outward": "outward",
    "up the plane of the page": "up_in_plane",
    "down the plane of the page": "down_in_plane",
}

_QUALITATIVE_ALIAS_GROUPS = {
    "constant_temperature": {
        "temperature stays constant",
        "temperature is constant",
        "constant temperature",
        "isothermal",
    },
    "mechanical_energy_conserved": {
        "mechanical energy is conserved",
        "mechanical energy conserved",
        "conservation of mechanical energy",
    },
    "increase": {"increases", "goes up", "becomes larger"},
    "decrease": {"decreases", "goes down", "becomes smaller"},
    "no_change": {"no change", "unchanged", "stays the same", "remains unchanged"},
}

_RELATION_REVERSED = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "=": "="}


@dataclass(frozen=True)
class RelationClause:
    """Canonical binary relation extracted from a relation answer."""

    lhs_text: str
    operator: str
    rhs_text: str


def normalize_plain_text(text: str | None) -> str:
    """Lowercase and strip lightweight punctuation for label-style comparisons."""

    stripped = _strip_text_wrappers(text)
    stripped = stripped.replace("_", " ").replace("-", " ")
    stripped = re.sub(r"[^\w\s]+", " ", stripped.lower())
    return re.sub(r"\s+", " ", stripped).strip()


def canonicalize_choice_label(label: str) -> str:
    """Collapse a choice label to its canonical uppercase token."""

    stripped = normalize_plain_text(label).replace(" ", "")
    return stripped.upper()


def canonicalize_boolean_value(text: str) -> bool | None:
    """Map a free-form boolean answer onto the canonical truth value."""

    return _BOOLEAN_CANONICAL.get(normalize_plain_text(text))


def canonicalize_sign_direction(text: str) -> str | None:
    """Normalize directional labels onto the controlled sign-direction vocabulary."""

    return _SIGN_DIRECTION_CANONICAL.get(normalize_plain_text(text))


_SIGN_DIRECTION_SUBSTRING_PHRASES = frozenset(
    {
        "clockwise",
        "counterclockwise",
        "anticlockwise",
        "to the right",
        "rightward",
        "to the left",
        "leftward",
        "upward",
        "downward",
        "into the page",
        "into page",
        "out of the page",
        "out of page",
        "inward",
        "outward",
        "up the plane of the page",
        "down the plane of the page",
        "positive sign",
        "negative sign",
    }
)


def sign_direction_candidates(text: str) -> tuple[str, ...]:
    """Return exact and phrase-level sign-direction interpretations for ``text``."""

    normalized = normalize_plain_text(text)
    if not normalized:
        return ()

    candidates: list[str] = []
    canonical = canonicalize_sign_direction(text)
    if canonical is not None:
        candidates.append(canonical)

    for phrase in _SIGN_DIRECTION_SUBSTRING_PHRASES:
        canonical = _SIGN_DIRECTION_CANONICAL.get(phrase)
        if canonical is None:
            continue
        if (
            normalized == phrase
            or normalized.endswith(f" {phrase}")
            or _contains_normalized_phrase(normalized, phrase)
        ):
            candidates.append(canonical)

    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return tuple(deduped)


def canonicalize_qualitative_label(text: str) -> str:
    """Normalize a qualitative label using the curated alias groups."""

    normalized = normalize_plain_text(text)
    for canonical, aliases in _QUALITATIVE_ALIAS_GROUPS.items():
        if normalized in aliases:
            return canonical
    return normalized


def qualitative_label_candidates(text: str) -> tuple[str, ...]:
    """Return canonical qualitative labels that are explicitly asserted in ``text``."""

    normalized = normalize_plain_text(text)
    if not normalized:
        return ()

    candidates: list[str] = []
    canonical = canonicalize_qualitative_label(text)
    if canonical in _QUALITATIVE_ALIAS_GROUPS:
        candidates.append(canonical)

    for label, aliases in _QUALITATIVE_ALIAS_GROUPS.items():
        phrases = {label.replace("_", " "), *aliases}
        if any(
            normalized == phrase
            or normalized.endswith(f" {phrase}")
            or re.search(rf"\boverall\b.*\b{re.escape(phrase)}\b", normalized)
            or _contains_normalized_phrase(normalized, phrase)
            for phrase in phrases
        ):
            candidates.append(label)

    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return tuple(deduped)


def parse_numeric_value(value: object) -> float | None:
    """Parse a numeric literal or constant symbolic expression into a float."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    expr = parse_scalar_symbolic_expression(str(value))
    if expr is None or expr.free_symbols:
        return None
    if expr == oo:
        return float("inf")
    if expr == -oo:
        return float("-inf")
    try:
        return float(N(expr))
    except (TypeError, ValueError):
        return None


def numbers_close(left: float, right: float, tolerance: float) -> bool:
    """Check relative numeric agreement while handling ``NaN`` and infinities."""

    if left == right:
        return True
    if any(value != value for value in (left, right)):
        return False
    if left in {float("inf"), float("-inf")} or right in {float("inf"), float("-inf")}:
        return left == right
    if left == 0.0 or right == 0.0:
        return abs(left - right) <= tolerance
    if (left < 0.0 < right) or (right < 0.0 < left):
        return False
    allowed = tolerance * max(abs(left), abs(right))
    return abs(left - right) <= allowed


def numbers_match_with_reference_precision(
    *,
    pred_value: float,
    pred_text: str | None,
    ref_value: float,
    ref_text: str | None,
    tolerance: float,
    allow_decimal_place_fallback: bool = True,
) -> bool:
    """Compare numbers using exact closeness first, then reference-side printed precision."""

    if numbers_close(pred_value, ref_value, tolerance):
        return True

    pred_sig_figs = _significant_figures(
        pred_text if pred_text is not None else pred_value
    )
    ref_sig_figs = _significant_figures(ref_text if ref_text is not None else ref_value)
    if (
        ref_sig_figs is not None
        and pred_sig_figs is not None
        and pred_sig_figs >= ref_sig_figs
    ):
        if _numbers_match_at_significant_figures(
            pred_value,
            ref_value,
            significant_figures=ref_sig_figs,
            tolerance=tolerance,
        ):
            return True

    pred_is_fixed_point = _is_fixed_point_numeric_text(pred_text)
    ref_is_fixed_point = _is_fixed_point_numeric_text(ref_text)
    if allow_decimal_place_fallback:
        if pred_is_fixed_point and ref_is_fixed_point:
            pred_decimal_places = _decimal_places(
                pred_text if pred_text is not None else pred_value
            )
            ref_decimal_places = _decimal_places(
                ref_text if ref_text is not None else ref_value
            )
            if pred_decimal_places >= ref_decimal_places:
                return _numbers_match_at_decimal_places(
                    pred_value,
                    ref_value,
                    decimal_places=ref_decimal_places,
                    tolerance=tolerance,
                )

    if _reference_uses_exact_nonterminating_rational(ref_text):
        if pred_sig_figs is not None and _numbers_match_at_significant_figures(
            pred_value,
            ref_value,
            significant_figures=pred_sig_figs,
            tolerance=tolerance,
        ):
            return True
        if allow_decimal_place_fallback and pred_is_fixed_point:
            return _numbers_match_at_decimal_places(
                pred_value,
                ref_value,
                decimal_places=_decimal_places(
                    pred_text if pred_text is not None else pred_value
                ),
                tolerance=tolerance,
            )
    return False


def _reference_uses_exact_nonterminating_rational(text: str | None) -> bool:
    """Return whether one reference text is an exact rational with a repeating decimal."""

    if not isinstance(text, str) or not text.strip():
        return False
    expr = parse_scalar_symbolic_expression(text)
    if expr is None or expr.free_symbols or not isinstance(expr, Rational):
        return False

    denominator = abs(int(expr.q))
    if denominator in {0, 1}:
        return False
    for factor in (2, 5):
        while denominator % factor == 0:
            denominator //= factor
    return denominator != 1


def _numbers_match_at_significant_figures(
    left: float,
    right: float,
    *,
    significant_figures: int,
    tolerance: float,
) -> bool:
    """Round both values to a shared significant-figure budget before comparison."""

    rounded_left = _round_to_significant_figures(left, significant_figures)
    rounded_right = _round_to_significant_figures(right, significant_figures)
    if rounded_left is None or rounded_right is None:
        return False
    if not numbers_close(rounded_left, rounded_right, tolerance):
        return False

    quantum = _significant_figure_quantum(left, right, significant_figures)
    return _difference_is_strictly_within_half_quantum(
        left,
        right,
        quantum=quantum,
        tolerance=tolerance,
    )


def _numbers_match_at_decimal_places(
    left: float,
    right: float,
    *,
    decimal_places: int,
    tolerance: float,
) -> bool:
    """Round both values to a shared decimal-place budget before comparison."""

    rounded_left = _round_to_decimal_places(left, decimal_places)
    rounded_right = _round_to_decimal_places(right, decimal_places)
    if rounded_left is None or rounded_right is None:
        return False
    if not numbers_close(rounded_left, rounded_right, tolerance):
        return False

    return _difference_is_strictly_within_half_quantum(
        left,
        right,
        quantum=10.0 ** (-decimal_places),
        tolerance=tolerance,
    )


def _difference_is_strictly_within_half_quantum(
    left: float,
    right: float,
    *,
    quantum: float | None,
    tolerance: float,
) -> bool:
    """Require the raw values to sit strictly inside the rounding interval."""

    if quantum is None or quantum <= 0:
        return False
    difference = abs(left - right)
    if difference == 0:
        return True
    if left == 0.0 or right == 0.0:
        allowed = tolerance
    elif (left < 0.0 < right) or (right < 0.0 < left):
        allowed = 0.0
    else:
        allowed = tolerance * max(abs(left), abs(right))
    return difference + allowed < (quantum / 2.0)


def _significant_figure_quantum(
    left: float,
    right: float,
    significant_figures: int,
) -> float | None:
    """Return the step size implied by ``significant_figures`` at this magnitude."""

    if significant_figures <= 0:
        return None
    magnitude = max(abs(left), abs(right))
    if (
        magnitude == 0
        or magnitude in {float("inf"), float("-inf")}
        or magnitude != magnitude
    ):
        return None
    exponent = math.floor(math.log10(magnitude)) - significant_figures + 1
    return 10.0**exponent


def _significant_figures(value: str | float) -> int | None:
    """Estimate how many significant figures a rendered numeric value communicates."""

    if isinstance(value, str):
        mantissa = _numeric_mantissa_text(value)
        if mantissa is None:
            return None
        digits = re.sub(r"[^0-9]", "", mantissa)
        significant = digits.lstrip("0")
        if significant:
            return len(significant)
        if digits:
            return len(digits)
        return None

    if value != value or value in {float("inf"), float("-inf")}:
        return None
    rendered = format(value, ".15g")
    mantissa = rendered.split("e", 1)[0].split("E", 1)[0]
    digits = re.sub(r"[^0-9]", "", mantissa)
    significant = digits.lstrip("0")
    if significant:
        return len(significant)
    if digits:
        return len(digits)
    return None


def _numeric_mantissa_text(text: str) -> str | None:
    """Extract the mantissa portion of a numeric string, including scientific notation."""

    normalized = text.strip()
    if not normalized:
        return None

    normalized = normalized.replace("−", "-")
    normalized = normalized.replace("·", "*").replace("×", "*")
    normalized = normalized.replace(" ", "")
    normalized = normalized.replace("{", "").replace("}", "")

    scientific_match = re.fullmatch(
        r"[+-]?(?P<mantissa>\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+|(?:\*|x)10\^?\(?[+-]?\d+\)?)?",
        normalized,
    )
    if scientific_match is not None:
        return scientific_match.group("mantissa")
    return None


def _is_fixed_point_numeric_text(text: str | None) -> bool:
    """Return ``True`` when ``text`` is a plain fixed-point decimal literal."""

    if not isinstance(text, str):
        return False
    normalized = text.strip().replace("−", "-").replace(" ", "")
    return bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", normalized))


def _round_to_significant_figures(
    value: float, significant_figures: int
) -> float | None:
    """Round ``value`` using decimal arithmetic to avoid binary float drift."""

    if significant_figures <= 0:
        return None
    if value != value or value in {float("inf"), float("-inf")}:
        return value
    if value == 0:
        return 0.0
    try:
        decimal_value = Decimal(str(value))
    except InvalidOperation:
        return None
    exponent = decimal_value.adjusted() - significant_figures + 1
    quantum = Decimal(f"1e{exponent}")
    try:
        return float(decimal_value.quantize(quantum, rounding=ROUND_HALF_UP))
    except InvalidOperation:
        return None


def _round_to_decimal_places(value: float, decimal_places: int) -> float | None:
    """Round ``value`` to ``decimal_places`` using ``Decimal`` semantics."""

    if decimal_places < 0:
        return None
    if value != value or value in {float("inf"), float("-inf")}:
        return value
    try:
        decimal_value = Decimal(str(value))
    except InvalidOperation:
        return None
    quantum = Decimal("1").scaleb(-decimal_places)
    try:
        return float(decimal_value.quantize(quantum, rounding=ROUND_HALF_UP))
    except InvalidOperation:
        return None


def _decimal_places(value: str | float) -> int:
    """Count the explicit decimal places represented by ``value``."""

    if isinstance(value, str):
        stripped = value.strip()
        sci_match = re.fullmatch(
            r"[+-]?(?P<mantissa>\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
            stripped,
        )
        if sci_match:
            mantissa = sci_match.group("mantissa")
            if "." in mantissa:
                return len(mantissa.split(".", 1)[1])
            return 0
        if "." in stripped:
            return len(stripped.split(".", 1)[1])
        return 0

    if value == 0 or value != value or value in {float("inf"), float("-inf")}:
        return 0
    rendered = format(value, ".15g")
    if "e" in rendered.lower():
        mantissa = rendered.split("e", 1)[0].split("E", 1)[0]
        if "." in mantissa:
            return len(mantissa.split(".", 1)[1].rstrip("0"))
        return 0
    rendered = rendered.rstrip("0").rstrip(".")
    if "." in rendered:
        return len(rendered.split(".", 1)[1])
    return 0


def parse_symbolic_expression(
    text: str,
    *,
    alias_map: Mapping[str, str] | None = None,
) -> Any | None:
    """Parse algebraic text into a SymPy expression after light canonicalization."""

    candidate = preprocess_symbolic_text(text, alias_map=alias_map)
    if not candidate:
        return None
    parse_candidate = _sanitize_parse_candidate(
        _prepare_symbolic_parse_candidate(candidate)
    )
    if not parse_candidate:
        return None

    local_dict = dict(_EXPRESSION_FUNCTIONS)
    for token in _SYMBOL_TOKEN_RE.findall(parse_candidate):
        if token not in local_dict:
            local_dict[token] = Symbol(token)

    try:
        return parse_expr(
            parse_candidate,
            local_dict=local_dict,
            global_dict=_PARSE_GLOBALS,
            transformations=_TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception:
        try:
            return parse_expr(
                parse_candidate,
                local_dict=local_dict,
                global_dict=_PARSE_GLOBALS,
                evaluate=True,
            )
        except Exception:
            return None


def parse_scalar_symbolic_expression(
    text: str,
    *,
    alias_map: Mapping[str, str] | None = None,
) -> Any | None:
    """Parse one scalar symbolic expression and reject tuple/set-like parses."""

    expression = parse_symbolic_expression(text, alias_map=alias_map)
    if not _is_scalar_symbolic_object(expression):
        return None
    return expression


def expressions_equivalent(
    left_text: str | None,
    right_text: str | None,
    tolerance: float,
    *,
    alias_map: Mapping[str, str] | None = None,
) -> bool:
    """Check symbolic equivalence between two expression texts."""

    if not left_text or not right_text:
        return False
    if preprocess_symbolic_text(
        left_text, alias_map=alias_map
    ) == preprocess_symbolic_text(
        right_text,
        alias_map=alias_map,
    ):
        return True
    if re.sub(
        r"\s+", "", preprocess_symbolic_text(left_text, alias_map=alias_map)
    ) == re.sub(
        r"\s+",
        "",
        preprocess_symbolic_text(right_text, alias_map=alias_map),
    ):
        return True

    left_expr = parse_symbolic_expression(left_text, alias_map=alias_map)
    right_expr = parse_symbolic_expression(right_text, alias_map=alias_map)
    if left_expr is None or right_expr is None:
        return normalize_plain_text(left_text) == normalize_plain_text(right_text)
    if isinstance(left_expr, Relational) or isinstance(right_expr, Relational):
        return relations_equivalent(
            left_text,
            right_text,
            tolerance,
            alias_map=alias_map,
        )
    if not _is_scalar_symbolic_object(left_expr) or not _is_scalar_symbolic_object(
        right_expr
    ):
        return normalize_plain_text(left_text) == normalize_plain_text(right_text)

    try:
        diff = simplify(left_expr - right_expr)
    except Exception:
        return normalize_plain_text(left_text) == normalize_plain_text(right_text)
    if diff == 0 or diff.is_zero is True:
        return True
    if _TRIG_FUNCTION_RE.search(left_text) or _TRIG_FUNCTION_RE.search(right_text):
        try:
            trig_diff = trigsimp(left_expr - right_expr)
            if trig_diff == 0 or trig_diff.is_zero is True:
                return True
        except Exception:
            pass
        try:
            if trigsimp(left_expr) == trigsimp(right_expr):
                return True
        except Exception:
            pass
    try:
        if diff.equals(0):
            return True
    except Exception:
        pass
    if diff.is_number:
        try:
            return numbers_close(float(N(diff)), 0.0, tolerance)
        except (TypeError, ValueError):
            return False
    return False


def parse_relation_clauses(
    text: str | None,
    *,
    alias_map: Mapping[str, str] | None = None,
) -> tuple[RelationClause, ...] | None:
    """Parse relation text into a flat tuple of binary clauses."""

    candidate = preprocess_symbolic_text(text, alias_map=alias_map)
    if not candidate:
        return None

    relation_object = _try_parse_relation_object(candidate)
    if relation_object is not None:
        relation_clauses = _clauses_from_relation_object(relation_object)
        if relation_clauses:
            return relation_clauses

    clauses: list[RelationClause] = []
    for segment in _split_top_level_conjunctions(candidate):
        parsed = _parse_relation_segment(segment)
        if parsed is None:
            return None
        clauses.extend(parsed)
    return tuple(clauses) if clauses else None


def relations_equivalent(
    left_text: str | None,
    right_text: str | None,
    tolerance: float,
    *,
    alias_map: Mapping[str, str] | None = None,
) -> bool:
    """Check whether two relation strings encode the same constraint set."""

    if not left_text or not right_text:
        return False
    if re.sub(
        r"\s+", "", preprocess_symbolic_text(left_text, alias_map=alias_map)
    ) == re.sub(
        r"\s+",
        "",
        preprocess_symbolic_text(right_text, alias_map=alias_map),
    ):
        return True

    left_clauses = parse_relation_clauses(left_text, alias_map=alias_map)
    right_clauses = parse_relation_clauses(right_text, alias_map=alias_map)
    if left_clauses is None or right_clauses is None:
        return normalize_plain_text(left_text) == normalize_plain_text(right_text)
    if len(left_clauses) != len(right_clauses):
        return False

    unused = list(right_clauses)
    for left_clause in left_clauses:
        matched_index = None
        for index, right_clause in enumerate(unused):
            if _relation_clause_equivalent(
                left_clause,
                right_clause,
                tolerance,
                alias_map=alias_map,
            ):
                matched_index = index
                break
        if matched_index is None:
            return False
        unused.pop(matched_index)
    return True


def relation_compare_candidates(
    text: str | None,
    *,
    alias_map: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return candidate relation surfaces, including explicit equivalent rewrites.

    Some model answers emit multiple equivalent equation forms in one atomic
    relation surface, for example ``A = B ; equivalently C = 0``. The strict
    relation parser treats that as one malformed chain, so same-kind relation
    comparison benefits from trying each explicitly signposted equivalent form.
    """

    normalized = preprocess_symbolic_text(text, alias_map=alias_map)
    if not normalized:
        return ()

    candidates: list[str] = [normalized]
    split_chain = _split_top_level_equality_like_chain(normalized)
    if split_chain is None or len(split_chain[1]) < 2:
        return tuple(candidates)
    if _RELATION_EQUIVALENCE_CUE_RE.search(normalized) is None:
        return tuple(candidates)

    for piece in _RELATION_EQUIVALENCE_CUE_RE.split(normalized):
        for candidate in _clean_relation_candidate_variants(piece):
            if candidate in candidates:
                continue
            if parse_relation_clauses(candidate) is None:
                continue
            candidates.append(candidate)
    return tuple(candidates)


def _clean_relation_candidate_variants(text: str) -> tuple[str, ...]:
    """Return cleaned candidate variants after splitting explicit equivalence cues."""

    stripped = text.strip(" \t\r\n;,:.")
    if not stripped:
        return ()

    variants = [stripped]
    if stripped.startswith("(") and stripped.endswith(")"):
        inner = stripped[1:-1].strip(" \t\r\n;,:.")
        if inner:
            variants.append(inner)
    if stripped.endswith(")") and "(" in stripped:
        trimmed = stripped.rstrip(")")
        trimmed = trimmed.strip(" \t\r\n;,:.")
        if trimmed:
            variants.append(trimmed)

    deduped: list[str] = []
    for variant in variants:
        if variant and variant not in deduped:
            deduped.append(variant)
    return tuple(deduped)


def relation_to_expression_text(
    text: str | None,
    target_variable: str | None,
    *,
    alias_map: Mapping[str, str] | None = None,
) -> str | None:
    """Project ``x = ...`` style relations down to the solved expression side."""

    if not text:
        return None
    clauses = parse_relation_clauses(text, alias_map=alias_map)
    if not clauses:
        return None
    if len(clauses) != 1:
        return None
    clause = clauses[0]
    if clause.operator != "=":
        return None
    if target_variable:
        normalized_target = preprocess_symbolic_text(
            target_variable,
            alias_map=alias_map,
        )
        if (
            preprocess_symbolic_text(clause.lhs_text, alias_map=alias_map)
            == normalized_target
        ):
            return clause.rhs_text
        if (
            preprocess_symbolic_text(clause.rhs_text, alias_map=alias_map)
            == normalized_target
        ):
            return clause.lhs_text
    return clause.rhs_text


def equality_like_rhs_expression_text(
    text: str | None,
    target_variable: str | None,
    *,
    alias_map: Mapping[str, str] | None = None,
) -> str | None:
    """Extract the terminal RHS from a top-level equality-like chain."""

    split_chain = _split_top_level_equality_like_chain(text)
    if split_chain is None:
        return None

    segments, _markers = split_chain
    if not all(
        _is_rhs_extractable_symbolic_label(
            segment,
            target_variable,
            alias_map=alias_map,
        )
        for segment in segments[:-1]
    ):
        return None
    return segments[-1]


def normalize_unit_text(text: str | None) -> str:
    """Canonicalize unit text into a parser-friendly symbolic form."""

    return _shared_normalize_unit_text(text)


def _split_top_level_equality_like_chain(
    text: str | None,
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Split a top-level equality-like chain without descending into groups."""

    if not text:
        return None

    stripped = _strip_text_wrappers(text)
    if not stripped:
        return None

    segments: list[str] = []
    markers: list[str] = []
    current: list[str] = []
    depth = 0
    index = 0

    while index < len(stripped):
        char = stripped[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)

        if depth == 0:
            marker = _equality_like_marker_at(stripped, index)
            if marker is not None:
                segment = "".join(current).strip()
                if not segment:
                    return None
                segments.append(segment)
                markers.append(marker)
                current = []
                index += len(marker)
                continue

        current.append(char)
        index += 1

    segment = "".join(current).strip()
    if not markers or not segment:
        return None
    segments.append(segment)
    return tuple(segments), tuple(markers)


def _equality_like_marker_at(text: str, index: int) -> str | None:
    """Return the supported equality-like marker that starts at ``index``."""

    for marker in _TOP_LEVEL_EQUALITY_LIKE_MARKERS:
        if text.startswith(marker, index):
            return marker
    return None


def _is_rhs_extractable_symbolic_label(
    text: str,
    target_variable: str | None,
    *,
    alias_map: Mapping[str, str] | None = None,
) -> bool:
    """Whether ``text`` is a safe symbolic label for RHS extraction."""

    normalized = preprocess_symbolic_text(text, alias_map=alias_map)
    if not normalized:
        return False
    if target_variable:
        normalized_target = preprocess_symbolic_text(
            target_variable,
            alias_map=alias_map,
        )
        if normalized == normalized_target:
            return True
    return _SIMPLE_SYMBOLIC_LABEL_RE.fullmatch(normalized) is not None


def unit_conversion_factor(from_unit: str | None, to_unit: str | None) -> float | None:
    """Return the multiplicative factor needed to convert between two units."""

    return _shared_unit_conversion_factor(from_unit, to_unit)


def convert_numeric_value(
    value: float, from_unit: str | None, to_unit: str | None
) -> float | None:
    """Convert a numeric value between units when the conversion is dimensionally valid."""

    return _shared_convert_numeric_value(value, from_unit, to_unit)


def preprocess_symbolic_text(
    text: str | None,
    *,
    alias_map: Mapping[str, str] | None = None,
) -> str:
    """Normalize LaTeX-ish math text into a form SymPy can usually parse."""

    if not text:
        return ""
    normalized = _strip_text_wrappers(text)
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\leqslant", "<=").replace("\\geqslant", ">=")
    normalized = normalized.replace("\\leq", "<=").replace("\\geq", ">=")
    normalized = normalized.replace("\\lt", "<").replace("\\gt", ">")
    normalized = normalized.replace("≤", "<=").replace("≥", ">=")
    normalized = normalized.replace("−", "-").replace("∞", "oo")
    normalized = normalized.replace("\\infty", "oo")
    normalized = normalized.replace("\\cdot", "*").replace("\\times", "*")
    normalized = normalized.replace("\\neq", "!=").replace("\\ne", "!=")
    normalized = normalized.replace("\\approx", "~=").replace("\\sim", "~")
    normalized = normalized.replace("\\pm", " pm ").replace("\\mp", " mp ")
    normalized = normalized.replace("\\to", " -> ")
    normalized = re.sub(r"(?<=[A-Za-z0-9_\)\}])(?=\\[A-Za-z]+\b)", "*", normalized)
    normalized = _LATEX_SPACING_RE.sub(" ", normalized)
    normalized = normalized.replace(" true", " True").replace(" false", " False")
    normalized = normalized.replace("true", "True").replace("false", "False")
    normalized = _replace_simple_latex(normalized)
    normalized = _normalize_latex_accents(normalized)
    normalized = _replace_latex_symbol_commands(normalized)
    normalized = _normalize_latex_subscripts(normalized)
    normalized = _normalize_compact_decorated_tokens(normalized)
    normalized = _normalize_superscripts(normalized)
    normalized = _canonicalize_notation_safe_symbol_variants(normalized)
    normalized = _canonicalize_symbol_alias_surfaces(normalized, alias_map=alias_map)
    normalized = _canonicalize_symbol_aliases(normalized, alias_map=alias_map)
    normalized = _normalize_symbol_products(normalized)
    normalized = _normalize_bare_function_calls(normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _sanitize_parse_candidate(text: str) -> str:
    """Rewrite parser-hostile symbol names into safe temporary identifiers."""

    return _SYMBOL_TOKEN_RE.sub(
        lambda match: _PARSE_RESERVED_SYMBOLS.get(match.group(0), match.group(0)),
        text,
    )


def _prepare_symbolic_parse_candidate(text: str) -> str:
    """Trim parse-only annotations and normalize grouping before SymPy parsing."""

    normalized = text.strip().replace("[", "(").replace("]", ")")
    normalized = _strip_top_level_with_annotation(normalized)
    normalized = _strip_trailing_definition_parenthetical(normalized)
    normalized = _strip_top_level_definition_suffix(normalized)
    return normalized.strip()


def _strip_top_level_with_annotation(text: str) -> str:
    """Drop trailing ``with ...`` definitions that are not part of the formula."""

    lowered = text.lower()
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        elif depth == 0 and lowered.startswith(" with ", index):
            suffix = text[index + len(" with ") :]
            if _looks_like_symbolic_definition_suffix(suffix):
                return text[:index].rstrip()
    return text


def _strip_trailing_definition_parenthetical(text: str) -> str:
    """Drop one trailing parenthetical note when it only defines parameters."""

    stripped = text.rstrip()
    if not stripped.endswith(")"):
        return stripped

    depth = 0
    open_index = None
    for index in range(len(stripped) - 1, -1, -1):
        char = stripped[index]
        if char == ")":
            depth += 1
        elif char == "(":
            depth -= 1
            if depth == 0:
                open_index = index
                break

    if open_index is None or open_index == 0 or not stripped[open_index - 1].isspace():
        return stripped

    suffix = stripped[open_index + 1 : -1]
    if _looks_like_symbolic_definition_suffix(suffix):
        return stripped[:open_index].rstrip()
    return stripped


def _strip_top_level_definition_suffix(text: str) -> str:
    """Drop comma-led top-level definition lists appended to an expression."""

    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        elif char == "," and depth == 0:
            suffix = text[index + 1 :]
            if _looks_like_symbolic_definition_suffix(suffix):
                return text[:index].rstrip()
    return text


def _looks_like_symbolic_definition_suffix(text: str) -> bool:
    """Return whether ``text`` looks like parameter definitions, not formula content."""

    return bool(_DEFINITION_LIKE_TOKEN_RE.search(text))


def _canonicalize_symbol_aliases(
    text: str,
    *,
    alias_map: Mapping[str, str] | None,
) -> str:
    """Rewrite identifier aliases while preserving decoration suffixes."""

    if not alias_map:
        return text
    return _SYMBOL_TOKEN_RE.sub(
        lambda match: _canonicalize_symbol_token(match.group(0), alias_map=alias_map),
        text,
    )


def _canonicalize_symbol_alias_surfaces(
    text: str,
    *,
    alias_map: Mapping[str, str] | None,
) -> str:
    """Rewrite explicit question-scoped alias surfaces before token parsing."""

    if not alias_map:
        return text

    replacements: list[tuple[str, str]] = []
    seen_aliases: set[str] = set()
    for alias, canonical in alias_map.items():
        normalized_alias = _normalize_alias_surface(alias)
        normalized_canonical = _normalize_alias_surface(canonical)
        if (
            not normalized_alias
            or not normalized_canonical
            or normalized_alias == normalized_canonical
            or normalized_alias in seen_aliases
        ):
            continue
        seen_aliases.add(normalized_alias)
        replacements.append((normalized_alias, normalized_canonical))

    normalized = text
    for alias, canonical in sorted(
        replacements, key=lambda item: (-len(item[0]), item[0])
    ):
        normalized = _replace_alias_surface_occurrences(
            normalized,
            alias=alias,
            canonical=canonical,
        )
    return normalized


def _normalize_alias_surface(text: str | None) -> str:
    """Normalize an alias surface with the same symbolic cleanup used pre-parse."""

    if not text:
        return ""

    normalized = _strip_text_wrappers(text)
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\leqslant", "<=").replace("\\geqslant", ">=")
    normalized = normalized.replace("\\leq", "<=").replace("\\geq", ">=")
    normalized = normalized.replace("\\lt", "<").replace("\\gt", ">")
    normalized = normalized.replace("≤", "<=").replace("≥", ">=")
    normalized = normalized.replace("−", "-").replace("∞", "oo")
    normalized = normalized.replace("\\infty", "oo")
    normalized = normalized.replace("\\cdot", "*").replace("\\times", "*")
    normalized = normalized.replace("\\neq", "!=").replace("\\ne", "!=")
    normalized = normalized.replace("\\approx", "~=").replace("\\sim", "~")
    normalized = normalized.replace("\\pm", " pm ").replace("\\mp", " mp ")
    normalized = normalized.replace("\\to", " -> ")
    normalized = re.sub(r"(?<=[A-Za-z0-9_\)\}])(?=\\[A-Za-z]+\b)", "*", normalized)
    normalized = _LATEX_SPACING_RE.sub(" ", normalized)
    normalized = normalized.replace(" true", " True").replace(" false", " False")
    normalized = normalized.replace("true", "True").replace("false", "False")
    normalized = _replace_simple_latex(normalized)
    normalized = _normalize_latex_accents(normalized)
    normalized = _replace_latex_symbol_commands(normalized)
    normalized = _normalize_latex_subscripts(normalized)
    normalized = _normalize_compact_decorated_tokens(normalized)
    normalized = _normalize_superscripts(normalized)
    normalized = _canonicalize_notation_safe_symbol_variants(normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _replace_alias_surface_occurrences(
    text: str,
    *,
    alias: str,
    canonical: str,
) -> str:
    """Replace boundary-safe surface alias occurrences with the canonical form."""

    if not alias or alias not in text:
        return text

    pieces: list[str] = []
    cursor = 0
    while True:
        index = text.find(alias, cursor)
        if index < 0:
            pieces.append(text[cursor:])
            return "".join(pieces)

        end = index + len(alias)
        if _has_alias_boundary(text, index - 1) and _has_alias_boundary(text, end):
            pieces.append(text[cursor:index])
            pieces.append(canonical)
        else:
            pieces.append(text[cursor:end])
        cursor = end


def _has_alias_boundary(text: str, index: int) -> bool:
    """Return whether ``index`` lies on a safe alias-replacement boundary."""

    if index < 0 or index >= len(text):
        return True
    return _ALIAS_BOUNDARY_TOKEN_RE.fullmatch(text[index]) is None


def _canonicalize_symbol_token(
    token: str,
    *,
    alias_map: Mapping[str, str],
) -> str:
    """Canonicalize one identifier token against the provided alias map."""

    if token in alias_map:
        return alias_map[token]
    for suffix in _DECORATION_SUFFIXES:
        if token.endswith(suffix):
            base = token[: -len(suffix)]
            canonical_base = alias_map.get(base)
            if canonical_base:
                return canonical_base + suffix
    return token


def _canonicalize_notation_safe_symbol_variants(text: str) -> str:
    """Normalize notation-only symbol variants before question-scoped aliasing."""

    return _SYMBOL_TOKEN_RE.sub(
        lambda match: _canonicalize_notation_safe_symbol_token(match.group(0)),
        text,
    )


def _canonicalize_notation_safe_symbol_token(token: str) -> str:
    """Collapse standard notation variants without inferring new symbol identity."""

    decoration_suffix = ""
    base_token = token
    for suffix in _DECORATION_SUFFIXES:
        if token.endswith(suffix):
            decoration_suffix = suffix
            base_token = token[: -len(suffix)]
            break

    zero_subscript_match = _ZERO_SUBSCRIPT_TOKEN_RE.fullmatch(base_token)
    if zero_subscript_match is not None:
        base_token = f"{zero_subscript_match.group('base')}0"

    base_token = _SAFE_SYMBOL_FAMILY_CANONICAL.get(base_token, base_token)
    return base_token + decoration_suffix


def _normalize_symbol_products(text: str) -> str:
    """Insert explicit multiplication for compact symbol runs like ``AB``."""

    return _SHORT_SYMBOL_RUN_RE.sub(_rewrite_symbol_run, text)


def _rewrite_symbol_run(match: re.Match[str]) -> str:
    """Expand an ambiguous symbol run unless it is a protected function/constant name."""

    token = match.group(0)
    if token in _PROTECTED_SYMBOL_RUNS or token.lower() in _PROTECTED_SYMBOL_RUNS:
        return token
    if token[0].islower() and len(token) > 3:
        return token
    return "*".join(token)


def _normalize_bare_function_calls(text: str) -> str:
    """Rewrite bare trig/log calls into explicit function-call syntax."""

    normalized = text
    while True:
        updated = _BARE_FUNCTION_POWER_RE.sub(_rewrite_bare_function_power, normalized)
        updated = _BARE_FUNCTION_ARG_RE.sub(_rewrite_bare_function_arg, updated)
        if updated == normalized:
            return updated
        normalized = updated


def _rewrite_bare_function_power(match: re.Match[str]) -> str:
    """Rewrite forms like ``sin^2 x`` into explicit exponentiated calls."""

    func = match.group("func")
    arg = _wrap_function_argument(match.group("arg"))
    power = match.group("power").strip()
    if (power.startswith("{") and power.endswith("}")) or (
        power.startswith("(") and power.endswith(")")
    ):
        power = power[1:-1].strip()
    return f"({func}{arg})**({power})"


def _rewrite_bare_function_arg(match: re.Match[str]) -> str:
    """Rewrite forms like ``sin x`` into ``sin(x)``."""

    func = match.group("func")
    return f"{func}{_wrap_function_argument(match.group('arg'))}"


def _wrap_function_argument(arg: str) -> str:
    """Wrap a bare function argument in parentheses when needed."""

    stripped = arg.strip()
    if stripped.startswith("(") and stripped.endswith(")"):
        return stripped
    if stripped.startswith("{") and stripped.endswith("}"):
        return f"({stripped[1:-1].strip()})"
    return f"({stripped})"


def _strip_text_wrappers(text: str | None) -> str:
    """Peel common math wrappers such as ``\\boxed{}``, ``$...$``, and display math."""

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


def _replace_simple_latex(text: str) -> str:
    """Expand a small LaTeX subset into parser-friendly ASCII math."""

    replaced = text
    while True:
        next_text = _TEXT_WRAPPER_RE.sub(r"\1", replaced)
        next_text = _replace_fraction_commands(next_text)
        next_text = _COMPACT_FRAC_RE.sub(r"((\1)/(\2))", next_text)
        next_text = _SIMPLE_FRAC_RE.sub(r"((\1)/(\2))", next_text)
        next_text = _INDEXED_SQRT_RE.sub(r"((\g<body>)**(1/(\g<index>)))", next_text)
        next_text = _SIMPLE_SQRT_RE.sub(r"sqrt(\1)", next_text)
        next_text = _replace_exponential_constants(next_text)
        if next_text == replaced:
            return next_text
        replaced = next_text


def _contains_normalized_phrase(text: str, phrase: str) -> bool:
    """Look for an affirmed phrase while filtering simple negations like ``not``."""

    for match in re.finditer(rf"\b{re.escape(phrase)}\b", text):
        prefix_tokens = text[: match.start()].strip().split()
        if prefix_tokens and prefix_tokens[-1] in {"no", "not"}:
            continue
        return True
    return False


def _replace_fraction_commands(text: str) -> str:
    """Recursively rewrite ``\\frac``-style commands into explicit division."""

    pieces: list[str] = []
    cursor = 0

    while True:
        match = _FRACTION_COMMAND_RE.search(text, cursor)
        if match is None:
            pieces.append(text[cursor:])
            return "".join(pieces)

        pieces.append(text[cursor : match.start()])
        replacement, next_index = _rewrite_fraction_command(text, match.start())
        if replacement is None:
            pieces.append(text[match.start() : match.end()])
            cursor = match.end()
            continue

        pieces.append(replacement)
        cursor = next_index


def _rewrite_fraction_command(text: str, start_index: int) -> tuple[str | None, int]:
    """Rewrite one fraction command starting at ``start_index`` if it is well-formed."""

    match = _FRACTION_COMMAND_RE.match(text, start_index)
    if match is None:
        return None, start_index

    cursor = _skip_whitespace(text, match.end())
    numerator, cursor = _extract_braced_group(text, cursor)
    if numerator is None:
        return None, match.end()

    cursor = _skip_whitespace(text, cursor)
    denominator, cursor = _extract_braced_group(text, cursor)
    if denominator is None:
        return None, match.end()

    numerator = _replace_fraction_commands(numerator)
    denominator = _replace_fraction_commands(denominator)
    return f"(({numerator})/({denominator}))", cursor


def _replace_exponential_constants(text: str) -> str:
    """Rewrite textual ``e^...`` constants into ``exp(...)`` when safe to do so."""

    pieces: list[str] = []
    cursor = 0

    while cursor < len(text):
        exp_index = text.find("e^", cursor)
        if exp_index < 0:
            pieces.append(text[cursor:])
            return "".join(pieces)

        pieces.append(text[cursor:exp_index])
        exponent, next_index = _consume_exponent_group(text, exp_index + 2)
        if exponent is None or not _should_normalize_exponential(exponent):
            pieces.append(text[exp_index:next_index])
            cursor = next_index
            continue

        prefix = "".join(pieces).rstrip()
        separator = " " if prefix and prefix[-1] not in "([{=,+-*/^<> " else ""
        pieces.append(f"{separator}exp({exponent})")
        cursor = next_index

    return "".join(pieces)


def _consume_exponent_group(text: str, start_index: int) -> tuple[str | None, int]:
    """Consume the exponent token or grouped expression after ``e^``."""

    cursor = _skip_whitespace(text, start_index)
    if cursor >= len(text):
        return None, start_index

    if text[cursor] == "{":
        return _extract_braced_group(text, cursor)
    if text[cursor] == "(":
        return _extract_parenthesized_group(text, cursor)

    token_match = re.match(r"[A-Za-z0-9_]+", text[cursor:])
    if token_match is None:
        return None, start_index
    return token_match.group(0), cursor + len(token_match.group(0))


def _should_normalize_exponential(exponent: str) -> bool:
    """Skip trivial ``e^x`` symbols and only normalize clearly structured exponents."""

    stripped = exponent.strip()
    return any(char in stripped for char in "\\+-*/() ")


def _extract_braced_group(text: str, start_index: int) -> tuple[str | None, int]:
    """Extract a balanced braced group starting at ``start_index``."""

    if start_index >= len(text) or text[start_index] != "{":
        return None, start_index

    depth = 0
    content_start = start_index + 1
    for index in range(start_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index], index + 1
    return None, start_index


def _extract_parenthesized_group(text: str, start_index: int) -> tuple[str | None, int]:
    """Extract a balanced parenthesized group starting at ``start_index``."""

    if start_index >= len(text) or text[start_index] != "(":
        return None, start_index

    depth = 0
    content_start = start_index + 1
    for index in range(start_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[content_start:index], index + 1
    return None, start_index


def _skip_whitespace(text: str, start_index: int) -> int:
    """Advance ``start_index`` past any ASCII whitespace."""

    cursor = start_index
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return cursor


def _normalize_latex_accents(text: str) -> str:
    """Rewrite LaTeX accents such as ``\\dot{x}`` into suffix-style identifiers."""

    normalized = text
    while True:
        updated = _LATEX_ACCENT_RE.sub(_rewrite_latex_accent, normalized)
        if updated == normalized:
            return updated
        normalized = updated


def _rewrite_latex_accent(match: re.Match[str]) -> str:
    """Render one LaTeX accent match as a canonical identifier token."""

    decoration = match.group("decor")
    base = _normalize_latex_identifier(match.group("braced") or match.group("plain"))
    sub = _normalize_latex_identifier(match.group("sub") or match.group("sub_plain"))
    if not base:
        return ""
    token = f"{base}_{sub}" if sub else base
    return f" {token}_{decoration} "


def _normalize_latex_subscripts(text: str) -> str:
    """Rewrite simple LaTeX subscripts into ``base_sub`` identifiers."""

    normalized = _LATEX_SUBSCRIPT_BRACED_RE.sub(_rewrite_subscript_token, text)
    return _LATEX_SUBSCRIPT_PLAIN_RE.sub(_rewrite_subscript_token, normalized)


def _rewrite_subscript_token(match: re.Match[str]) -> str:
    """Render one subscript token in canonical ``base_sub`` form."""

    base = _normalize_latex_identifier(match.group("base"))
    sub = _normalize_latex_identifier(match.group("sub"))
    if not base:
        return ""
    return f"{base}_{sub}" if sub else base


def _normalize_latex_identifier(value: str | None) -> str:
    """Normalize a LaTeX identifier fragment to the plain token used downstream."""

    if not value:
        return ""
    normalized = _TEXT_WRAPPER_RE.sub(r"\1", value.strip())
    normalized = _replace_latex_symbol_commands(normalized)
    normalized = normalized.replace(r"\{", "{").replace(r"\}", "}")
    normalized = normalized.strip("{}()[] ")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.lstrip("\\")
    return normalized


def _replace_latex_symbol_commands(text: str) -> str:
    """Map LaTeX symbol commands like ``\\alpha`` to parser-friendly tokens."""

    def replacer(match: re.Match[str]) -> str:
        command = match.group(1)
        replacement = _LATEX_SYMBOL_ALIASES.get(command, command)
        if replacement in _FUNCTION_COMMANDS:
            return f" {replacement} "
        return replacement

    return _COMMAND_TOKEN_RE.sub(replacer, text)


def _normalize_compact_decorated_tokens(text: str) -> str:
    """Rewrite compact tokens like ``vhat`` into ``v_hat``."""

    return _COMPACT_DECORATED_TOKEN_RE.sub(_rewrite_compact_decorated_token, text)


def _rewrite_compact_decorated_token(match: re.Match[str]) -> str:
    """Render one compact decoration token in suffix form."""

    return f"{match.group('base')}_{match.group('decor')}"


def _normalize_superscripts(text: str) -> str:
    """Rewrite simple superscripts into explicit exponent syntax."""

    normalized = _SUPERSCRIPT_BRACED_RE.sub(r"^(\1)", text)
    return _SUPERSCRIPT_PLAIN_RE.sub(r"^(\g<exp>)", normalized)


def _try_parse_relation_object(text: str) -> Any | None:
    """Parse canonical relation constructor text such as ``Eq(...)`` or ``And(...)``."""

    if not text.startswith(("Eq(", "Le(", "Lt(", "Ge(", "Gt(", "And(")):
        return None
    return parse_symbolic_expression(text)


def _clauses_from_relation_object(obj: Any) -> tuple[RelationClause, ...]:
    """Flatten SymPy relation objects into canonical binary relation clauses."""

    if isinstance(obj, And):
        clauses: list[RelationClause] = []
        for arg in obj.args:
            clauses.extend(_clauses_from_relation_object(arg))
        return tuple(clauses)
    if not isinstance(obj, Relational):
        return ()
    if isinstance(obj, Eq):
        operator = "="
    elif isinstance(obj, Le):
        operator = "<="
    elif isinstance(obj, Lt):
        operator = "<"
    elif isinstance(obj, Ge):
        operator = ">="
    elif isinstance(obj, Gt):
        operator = ">"
    else:
        return ()
    return (RelationClause(str(obj.lhs), operator, str(obj.rhs)),)


def _split_top_level_conjunctions(text: str) -> list[str]:
    """Split relation conjunctions without breaking nested expressions."""

    segments: list[str] = []
    current: list[str] = []
    depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)

        if depth == 0 and text[index] == "&":
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            index += 1
            continue

        if depth == 0:
            if char in ",;":
                remaining = text[index + 1 :]
                if _looks_like_relation_constraint_segment(remaining):
                    segment = "".join(current).strip()
                    if segment:
                        segments.append(segment)
                    current = []
                    index += 1
                    continue
            match = _TOP_LEVEL_AND_RE.match(text, index)
            if match:
                segment = "".join(current).strip()
                if segment:
                    segments.append(segment)
                current = []
                index = match.end()
                continue

        current.append(char)
        index += 1

    segment = "".join(current).strip()
    if segment:
        segments.append(segment)
    return segments or [text.strip()]


def _parse_relation_segment(text: str) -> tuple[RelationClause, ...] | None:
    """Split a chained relation like ``a < b < c`` into pairwise clauses."""

    parts: list[str] = []
    operators: list[str] = []
    current: list[str] = []
    depth = 0
    index = 0
    stripped = _strip_relation_condition_prefix(text)

    while index < len(stripped):
        char = stripped[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)

        if depth == 0:
            if stripped.startswith("<=", index) or stripped.startswith(">=", index):
                parts.append("".join(current).strip())
                operators.append(stripped[index : index + 2])
                current = []
                index += 2
                continue
            if stripped.startswith("==", index):
                parts.append("".join(current).strip())
                operators.append("=")
                current = []
                index += 2
                continue
            if char in "<>=":
                parts.append("".join(current).strip())
                operators.append(char)
                current = []
                index += 1
                continue

        current.append(char)
        index += 1

    parts.append("".join(current).strip())
    if (
        not operators
        or len(parts) != len(operators) + 1
        or any(not part for part in parts)
    ):
        return None

    return tuple(
        RelationClause(parts[idx], operators[idx], parts[idx + 1])
        for idx in range(len(operators))
    )


def _relation_clause_equivalent(
    left: RelationClause,
    right: RelationClause,
    tolerance: float,
    *,
    alias_map: Mapping[str, str] | None = None,
) -> bool:
    """Compare two relation clauses, including reversed and scaled formulations."""

    if (
        left.operator == right.operator
        and expressions_equivalent(
            left.lhs_text,
            right.lhs_text,
            tolerance,
            alias_map=alias_map,
        )
        and expressions_equivalent(
            left.rhs_text,
            right.rhs_text,
            tolerance,
            alias_map=alias_map,
        )
    ):
        return True

    reversed_operator = _RELATION_REVERSED.get(right.operator)
    if (
        reversed_operator == left.operator
        and expressions_equivalent(
            left.lhs_text,
            right.rhs_text,
            tolerance,
            alias_map=alias_map,
        )
        and expressions_equivalent(
            left.rhs_text,
            right.lhs_text,
            tolerance,
            alias_map=alias_map,
        )
    ):
        return True

    left_lhs = parse_symbolic_expression(left.lhs_text, alias_map=alias_map)
    left_rhs = parse_symbolic_expression(left.rhs_text, alias_map=alias_map)
    right_lhs = parse_symbolic_expression(right.lhs_text, alias_map=alias_map)
    right_rhs = parse_symbolic_expression(right.rhs_text, alias_map=alias_map)
    if left_lhs is None or left_rhs is None or right_lhs is None or right_rhs is None:
        return False
    if not all(
        _is_scalar_symbolic_object(expr)
        for expr in (left_lhs, left_rhs, right_lhs, right_rhs)
    ):
        return False

    left_residual = simplify(left_lhs - left_rhs)
    right_residual = simplify(right_lhs - right_rhs)
    ratio = _proportional_ratio(left_residual, right_residual, tolerance)
    if ratio is None:
        return False

    if left.operator == "=" and right.operator == "=":
        return True

    if ratio > 0 and left.operator == right.operator:
        return True

    return ratio < 0 and left.operator == _RELATION_REVERSED.get(right.operator)


def _proportional_ratio(
    left_expr: Any, right_expr: Any, tolerance: float
) -> float | None:
    """Return the scalar relating two residual expressions when one exists."""

    same = simplify(left_expr - right_expr)
    if same == 0 or same.is_zero is True:
        return 1.0
    negated = simplify(left_expr + right_expr)
    if negated == 0 or negated.is_zero is True:
        return -1.0
    try:
        ratio = simplify(left_expr / right_expr)
    except Exception:
        return None
    if ratio.free_symbols:
        return None
    try:
        ratio_value = float(N(ratio))
    except (TypeError, ValueError):
        return None
    if abs(ratio_value) <= tolerance:
        return None
    residual = simplify(left_expr - ratio * right_expr)
    if residual == 0 or residual.is_zero is True:
        return ratio_value
    try:
        if residual.equals(0):
            return ratio_value
    except Exception:
        return None
    return None


def _strip_relation_condition_prefix(text: str) -> str:
    """Remove lightweight prose prefixes that introduce one relation segment."""

    return _RELATION_CONDITION_PREFIX_RE.sub("", text.strip()).strip()


def _looks_like_relation_constraint_segment(text: str) -> bool:
    """Return whether one top-level trailing segment looks like a constraint."""

    stripped = _strip_relation_condition_prefix(text).strip(" \t\r\n,;")
    if not stripped:
        return False
    if _ALTERNATE_FORM_PREFIX_RE.match(stripped):
        return False

    relation_object = _try_parse_relation_object(stripped)
    if relation_object is not None:
        clauses = _clauses_from_relation_object(relation_object)
        return bool(clauses) and any(clause.operator != "=" for clause in clauses)

    if _RELATION_CONSTRAINT_TOKEN_RE.search(stripped):
        return True

    parsed = _parse_relation_segment(stripped)
    return parsed is not None and any(clause.operator != "=" for clause in parsed)


def _is_scalar_symbolic_object(value: Any) -> bool:
    """Return whether ``value`` is one scalar SymPy expression-like object."""

    return isinstance(value, Basic) and not isinstance(value, Relational)


def _parse_unit_expression(text: str) -> Any | None:
    """Parse a canonicalized unit expression into a simplified SymPy quantity."""

    if not text:
        return Integer(1)
    local_dict = dict(_SHARED_UNIT_NAMESPACE)
    try:
        return simplify(
            parse_expr(
                text,
                local_dict=local_dict,
                global_dict=_PARSE_GLOBALS,
                transformations=_TRANSFORMATIONS,
                evaluate=True,
            )
        )
    except Exception:
        return None
