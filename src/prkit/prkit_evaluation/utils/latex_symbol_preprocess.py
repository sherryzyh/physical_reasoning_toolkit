import re

# ONLY include commands that are NOT natively handled as atomic symbols 
# or those that include subscripts/decorations that break the parser.
PROTECTED_PHYSICS_SYMBOLS = {
    r'\hbar': 'hbar',
    r'\mu_0': 'mu0',
    r'\epsilon_0': 'eps0',
    r'\ell': 'ell',
    r'\square': 'dalembert',
    r'\angstrom': 'angstrom',
    r'\degree': 'deg',
}

def _preprocess_latex(latex_str: str) -> str:
    """
    Minimally preprocesses LaTeX by protecting only the symbols 
    known to break latex2sympy2_extended.
    """
    if not latex_str:
        return ""

    processed = latex_str

    # 1. Clean spacing (Essential: these often cause "Unexpected Token" errors)
    spacings = [r'\,', r'\:', r'\;', r'\!', r'\quad', r'\qquad']
    for space in spacings:
        processed = processed.replace(space, ' ')

    # 2. Protect specific breaking symbols
    # Standard Greek (\alpha, \omega, etc.) are REMOVED from here 
    # because the parser handles them natively.
    for cmd, name in PROTECTED_PHYSICS_SYMBOLS.items():
        if cmd in processed:
            processed = processed.replace(cmd, f'\\mathrm{{{name}}}')

    # 3. Handle Vector/Hat decorations
    # Native parsers often fail to relate \vec{v} to v; stripping is safer.
    processed = re.sub(r'\\vec\{([^}]*)\}', r'\1', processed)
    processed = re.sub(r'\\hat\{([^}]*)\}', r'\1', processed)
    
    # 4. Standardize differentials
    # Many physics equations use \text{d}x; converting to 'd x' helps SymPy.
    processed = processed.replace(r'\mathrm{d}', ' d ').replace(r'\text{d}', ' d ')

    # 5. Final cleanup
    processed = re.sub(r'\s+', ' ', processed).strip()
    
    return processed