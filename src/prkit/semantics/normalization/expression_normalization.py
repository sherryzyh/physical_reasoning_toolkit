"""Expression and relation normalization helpers."""

from __future__ import annotations

import re

from latex2sympy2_extended import latex2sympy

from .atomic_kinds import NormalizedAtomicKind
from .math_text_normalization import _LATEX_BINARY_RELATION_MARKERS
from .physical_quantity_normalization import (
    _QUANTITY_PATTERN,
    _canonicalize_quantity_string,
    _try_parse_physical_quantity,
    _try_parse_relation_wrapped_quantity,
)

_PROTECTED_PHYSICS_SYMBOLS = {
    r"\hbar": "hbar",
    r"\mu_0": "mu0",
    r"\epsilon_0": "eps0",
    r"\varepsilon_0": "eps0",
    r"\ell": "ell",
    r"\square": "dalembert",
    r"\angstrom": "angstrom",
    r"\degree": "deg",
}


def _preprocess_latex(latex_str: str) -> str:
    """Simplify LaTeX before handing it to ``latex2sympy``.

    The preprocessing step strips spacing commands and cosmetic decorations
    that frequently cause parser instability while preserving the physics
    symbols that matter for comparison.
    """

    if not latex_str:
        return ""

    processed = latex_str
    for spacing in (r"\,", r"\:", r"\;", r"\!", r"\quad", r"\qquad"):
        processed = processed.replace(spacing, " ")

    for command, name in sorted(
        _PROTECTED_PHYSICS_SYMBOLS.items(), key=lambda item: -len(item[0])
    ):
        replacement = f"\\mathrm{{{name}}}"
        if command in processed:
            processed = processed.replace(command, replacement)
        if "_" in command:
            base, sub = command.rsplit("_", 1)
            braced = f"{base}_{{{sub}}}"
            if braced in processed:
                processed = processed.replace(braced, replacement)

    for decoration in (
        r"\vec",
        r"\hat",
        r"\dot",
        r"\ddot",
        r"\bar",
        r"\tilde",
        r"\overline",
        r"\underline",
    ):
        processed = re.sub(re.escape(decoration) + r"\{([^}]*)\}", r"\1", processed)

    processed = processed.replace(r"\mathrm{d}", " d ").replace(r"\text{d}", " d ")
    return re.sub(r"\s+", " ", processed).strip()


def _normalize_symbolic_expression(
    clean_math: str, had_latex_patterns: bool
) -> tuple[str, bool]:
    """Normalize a symbolic surface into a stable comparison string."""

    if had_latex_patterns:
        preprocessed = _preprocess_latex(clean_math)
        try:
            symbolic_expr = latex2sympy(preprocessed)
            normalized = re.sub(r"\s+", " ", str(symbolic_expr)).strip()
            return normalized, True
        except Exception:
            return preprocessed, False

    normalized = re.sub(r"\s+", " ", clean_math).strip()
    return normalized, True


def classify_expression(clean_str: str) -> NormalizedAtomicKind:
    """Classify a math-like surface as quantity, relation, or expression."""

    canonical = _canonicalize_quantity_string(clean_str)
    if _try_parse_relation_wrapped_quantity(canonical) is not None:
        return NormalizedAtomicKind.PHYSICAL_QUANTITY
    if _try_parse_physical_quantity(clean_str) is not None:
        return NormalizedAtomicKind.PHYSICAL_QUANTITY
    if "=" in canonical:
        eq_count = canonical.count("=")
        if eq_count == 1 or (eq_count >= 2 and "," not in canonical):
            return NormalizedAtomicKind.RELATION
    elif any(marker in canonical for marker in _LATEX_BINARY_RELATION_MARKERS):
        return NormalizedAtomicKind.RELATION

    quantity_candidate = canonical.replace(" ", "")
    quantity_candidate = re.sub(
        r"\\(?:mathrm|textrm|text|operatorname)\s*\{[^{}]*\}",
        "X",
        quantity_candidate,
    )
    quantity_candidate = quantity_candidate.replace("~", "")
    if (
        _QUANTITY_PATTERN.match(quantity_candidate)
        and _try_parse_physical_quantity(quantity_candidate) is not None
    ):
        return NormalizedAtomicKind.PHYSICAL_QUANTITY

    return NormalizedAtomicKind.EXPRESSION


__all__ = [
    "_PROTECTED_PHYSICS_SYMBOLS",
    "_normalize_symbolic_expression",
    "_preprocess_latex",
    "classify_expression",
]
