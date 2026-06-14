"""LaTeX preprocessing to protect physics symbols that break ``latex2sympy2_extended``."""

import re

# ONLY include commands that are NOT natively handled as atomic symbols
# or those that include subscripts/decorations that break the parser.
PROTECTED_PHYSICS_SYMBOLS = {
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
    """Protect physics symbols known to break ``latex2sympy2_extended`` before parsing."""
    if not latex_str:
        return ""

    processed = latex_str

    # 1. Clean spacing (Essential: these often cause "Unexpected Token" errors)
    spacings = [r"\,", r"\:", r"\;", r"\!", r"\quad", r"\qquad"]
    for space in spacings:
        processed = processed.replace(space, " ")

    # 2. Protect specific breaking symbols
    # Standard Greek (\alpha, \omega, etc.) are REMOVED from here
    # because the parser handles them natively.
    # Sort by length descending so longer keys (e.g. \varepsilon_0) are
    # replaced before shorter prefixes (e.g. \varepsilon).
    for cmd, name in sorted(
        PROTECTED_PHYSICS_SYMBOLS.items(), key=lambda x: -len(x[0])
    ):
        replacement = f"\\mathrm{{{name}}}"
        if cmd in processed:
            processed = processed.replace(cmd, replacement)
        # Also handle braced subscript form: \foo_{0} alongside \foo_0
        if "_" in cmd:
            base, sub = cmd.rsplit("_", 1)
            braced = f"{base}_{{{sub}}}"
            if braced in processed:
                processed = processed.replace(braced, replacement)

    # 3. Strip decoration commands that latex2sympy doesn't reduce to the
    #    inner symbol (\dot{r} -> "dot{r}" instead of "r").  Stripping the
    #    wrapper lets the underlying variable parse cleanly.
    for deco in (
        r"\vec",
        r"\hat",
        r"\dot",
        r"\ddot",
        r"\bar",
        r"\tilde",
        r"\overline",
        r"\underline",
    ):
        processed = re.sub(re.escape(deco) + r"\{([^}]*)\}", r"\1", processed)

    # 4. Standardize differentials
    # Many physics equations use \text{d}x; converting to 'd x' helps SymPy.
    processed = processed.replace(r"\mathrm{d}", " d ").replace(r"\text{d}", " d ")

    # 5. Final cleanup
    processed = re.sub(r"\s+", " ", processed).strip()

    return processed
