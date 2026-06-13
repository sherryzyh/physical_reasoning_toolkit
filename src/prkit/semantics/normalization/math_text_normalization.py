"""Math-text cleanup helpers used before deeper answer normalization."""

from __future__ import annotations

import re
from typing import Dict, Tuple

_UNICODE_FRACTIONS: Dict[str, str] = {
    "\u00bc": "1/4",
    "\u00bd": "1/2",
    "\u00be": "3/4",
    "\u2150": "1/7",
    "\u2151": "1/9",
    "\u2152": "1/10",
    "\u2153": "1/3",
    "\u2154": "2/3",
    "\u2155": "1/5",
    "\u2156": "2/5",
    "\u2157": "3/5",
    "\u2158": "4/5",
    "\u2159": "1/6",
    "\u215a": "5/6",
    "\u215b": "1/8",
    "\u215c": "3/8",
    "\u215d": "5/8",
    "\u215e": "7/8",
}

_SUBSCRIPT_TRANSLATION = str.maketrans(
    {
        "\u2080": "0",
        "\u2081": "1",
        "\u2082": "2",
        "\u2083": "3",
        "\u2084": "4",
        "\u2085": "5",
        "\u2086": "6",
        "\u2087": "7",
        "\u2088": "8",
        "\u2089": "9",
        "\u208a": "+",
        "\u208b": "-",
    }
)

_FULLWIDTH_PUNCT: Dict[int, str] = {
    0xFF08: "(",
    0xFF09: ")",
    0xFF0B: "+",
    0xFF0C: ",",
    0xFF0E: ".",
    0xFF0F: "/",
    0xFF1A: ":",
    0xFF1B: ";",
    0xFF1D: "=",
    0xFF3B: "[",
    0xFF3D: "]",
    0xFF5B: "{",
    0xFF5D: "}",
}


def _normalize_unicode(text: str) -> str:
    """Replace common Unicode with ASCII or LaTeX-compatible math tokens."""

    text = text.replace("\u2212", "-")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = text.replace("\u2010", "-")
    text = text.replace("\u2011", "-")
    text = text.replace("\uFF0D", "-")

    text = text.replace("\u00d7", r" \times ")
    text = text.replace("\u00b7", r" \cdot ")
    text = text.replace("\u22c5", r" \cdot ")
    text = text.replace("\u2219", r" \cdot ")
    text = text.replace("\u00f7", r" \div ")

    for unicode_fraction, fraction in _UNICODE_FRACTIONS.items():
        text = text.replace(unicode_fraction, fraction)

    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")
    text = text.replace("\u00ab", '"')
    text = text.replace("\u00bb", '"')
    text = text.replace("\u00b5", "\u03bc")

    output: list[str] = []
    for char in text:
        codepoint = ord(char)
        if 0xFF10 <= codepoint <= 0xFF19:
            output.append(chr(codepoint - 0xFF10 + ord("0")))
        elif 0xFF21 <= codepoint <= 0xFF3A:
            output.append(chr(codepoint - 0xFF21 + ord("A")))
        elif 0xFF41 <= codepoint <= 0xFF5A:
            output.append(chr(codepoint - 0xFF41 + ord("a")))
        elif codepoint in _FULLWIDTH_PUNCT:
            output.append(_FULLWIDTH_PUNCT[codepoint])
        else:
            output.append(char)
    text = "".join(output)

    text = re.sub(
        r"(?<=[A-Za-z\d)])([₀₁₂₃₄₅₆₇₈₉₊₋]+)",
        lambda match: "_" + match.group(1).translate(_SUBSCRIPT_TRANSLATION),
        text,
    )

    text = text.replace("\u2264", r" \leq ")
    text = text.replace("\u2265", r" \geq ")
    text = text.replace("\u2260", r" \neq ")
    text = text.replace("\u2248", r"\approx ")
    text = text.replace("\u221e", r"\infty")
    text = text.replace("\u221d", r" \propto ")
    text = text.replace("\u00b1", r" \pm ")
    text = text.replace("\u00b0", " deg")

    return text


_UNICODE_GREEK_TO_LATEX: Dict[str, str] = {
    "\u03b1": r"\alpha ",
    "\u03b2": r"\beta ",
    "\u03b3": r"\gamma ",
    "\u03b4": r"\delta ",
    "\u03b5": r"\varepsilon ",
    "\u03b6": r"\zeta ",
    "\u03b7": r"\eta ",
    "\u03b8": r"\theta ",
    "\u03b9": r"\iota ",
    "\u03ba": r"\kappa ",
    "\u03bb": r"\lambda ",
    "\u03bd": r"\nu ",
    "\u03be": r"\xi ",
    "\u03c0": r"\pi ",
    "\u03c1": r"\rho ",
    "\u03c2": r"\varsigma ",
    "\u03c3": r"\sigma ",
    "\u03c4": r"\tau ",
    "\u03c5": r"\upsilon ",
    "\u03c6": r"\phi ",
    "\u03c7": r"\chi ",
    "\u03c8": r"\psi ",
    "\u03c9": r"\omega ",
    "\u0393": r"\Gamma ",
    "\u0394": r"\Delta ",
    "\u0398": r"\Theta ",
    "\u039b": r"\Lambda ",
    "\u039e": r"\Xi ",
    "\u03a0": r"\Pi ",
    "\u03a3": r"\Sigma ",
    "\u03a5": r"\Upsilon ",
    "\u03a6": r"\Phi ",
    "\u03a8": r"\Psi ",
    "\u03d1": r"\vartheta ",
    "\u03d5": r"\varphi ",
    "\u03d6": r"\varpi ",
    "\u03f0": r"\varkappa ",
    "\u03f1": r"\varrho ",
    "\u03f5": r"\epsilon ",
}

_UNICODE_SUPERSCRIPT_TO_CARET: Dict[str, str] = {
    "\u2070": "0",
    "\u00b9": "1",
    "\u00b2": "2",
    "\u00b3": "3",
    "\u2074": "4",
    "\u2075": "5",
    "\u2076": "6",
    "\u2077": "7",
    "\u2078": "8",
    "\u2079": "9",
    "\u207a": "+",
    "\u207b": "-",
}

_UNICODE_SUBSCRIPT_CHARS: Dict[str, str] = {
    "\u2080": "0",
    "\u2081": "1",
    "\u2082": "2",
    "\u2083": "3",
    "\u2084": "4",
    "\u2085": "5",
    "\u2086": "6",
    "\u2087": "7",
    "\u2088": "8",
    "\u2089": "9",
    "\u208a": "+",
    "\u208b": "-",
}

_UNICODE_GREEK_CHARS = frozenset(_UNICODE_GREEK_TO_LATEX)
_SUPERSCRIPT_RUN_RE = re.compile(
    r"[" + re.escape("".join(_UNICODE_SUPERSCRIPT_TO_CARET)) + r"]+"
)
_SUBSCRIPT_RUN_RE = re.compile(
    r"[" + re.escape("".join(_UNICODE_SUBSCRIPT_CHARS)) + r"]+"
)


def _convert_sqrt_unicode(text: str) -> str:
    """Replace Unicode square roots with LaTeX ``\\sqrt{...}``."""

    result: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        if text[index] != "\u221a":
            result.append(text[index])
            index += 1
            continue

        index += 1
        while index < length and text[index] == " ":
            index += 1
        if index < length and text[index] == "(":
            depth = 1
            start = index + 1
            index += 1
            while index < length and depth > 0:
                if text[index] == "(":
                    depth += 1
                elif text[index] == ")":
                    depth -= 1
                index += 1
            result.append(r"\sqrt{")
            result.append(text[start : index - 1])
            result.append("}")
            continue
        if index < length and text[index] == "[":
            depth = 1
            start = index + 1
            index += 1
            while index < length and depth > 0:
                if text[index] == "[":
                    depth += 1
                elif text[index] == "]":
                    depth -= 1
                index += 1
            result.append(r"\sqrt{")
            result.append(text[start : index - 1])
            result.append("}")
            continue
        if index < length and (text[index].isdigit() or text[index].isalpha()):
            result.append(r"\sqrt{")
            result.append(text[index])
            result.append("}")
            index += 1
            continue
        result.append(r"\sqrt ")
    return "".join(result)


def _unicode_math_to_latex(text: str) -> str:
    """Convert bare Unicode math symbols to LaTeX commands."""

    if not text:
        return text
    if (
        not any(char in _UNICODE_GREEK_CHARS for char in text)
        and "\u221a" not in text
        and not _SUPERSCRIPT_RUN_RE.search(text)
        and not _SUBSCRIPT_RUN_RE.search(text)
    ):
        return text

    for char, latex in _UNICODE_GREEK_TO_LATEX.items():
        if char in text:
            text = text.replace(char, latex)

    def _subscript_repl(match: re.Match) -> str:
        digits = "".join(_UNICODE_SUBSCRIPT_CHARS[ch] for ch in match.group())
        return "_{" + digits + "}"

    def _superscript_repl(match: re.Match) -> str:
        digits = "".join(_UNICODE_SUPERSCRIPT_TO_CARET[ch] for ch in match.group())
        return "^{" + digits + "}"

    text = _SUBSCRIPT_RUN_RE.sub(_subscript_repl, text)
    text = _SUPERSCRIPT_RUN_RE.sub(_superscript_repl, text)
    if "\u221a" in text:
        text = _convert_sqrt_unicode(text)
    return re.sub(r"  +", " ", text)


def _match_balanced_braces(
    text: str,
    start_pos: int,
    open_char: str = "{",
    close_char: str = "}",
) -> int:
    """Return the matching close-brace position or ``-1`` when unbalanced."""

    if start_pos >= len(text) or text[start_pos] != open_char:
        return -1

    depth = 1
    pos = start_pos + 1
    while pos < len(text) and depth > 0:
        if text[pos] == open_char:
            depth += 1
        elif text[pos] == close_char:
            depth -= 1
        pos += 1
    return pos - 1 if depth == 0 else -1


def _extract_math_content(text: str) -> Tuple[str, bool]:
    """Strip common math delimiters and formatting wrappers from text."""

    normalized = _normalize_unicode(text).strip()
    had_latex_patterns = False

    simple_patterns = [
        (r"\$\$(.*?)\$\$", r"\1"),
        (r"\$(.*?)\$", r"\1"),
        (r"\\\[(.*?)\\\]", r"\1"),
        (r"\\\((.*?)\\\)", r"\1"),
    ]
    brace_patterns = [
        r"\\operatorname",
        r"\\displaystyle",
        r"\\textrm",
        r"\\textit",
        r"\\textbf",
        r"\\textsl",
        r"\\textsf",
        r"\\mathit",
        r"\\mathbf",
        r"\\mathsf",
        r"\\mathbb",
        r"\\mathcal",
        r"\\boxed",
        r"\\mathrm",
    ]

    for _ in range(20):
        previous = normalized
        for pattern, replacement in simple_patterns:
            normalized = re.sub(pattern, replacement, normalized, flags=re.DOTALL).strip()
            if normalized != previous:
                had_latex_patterns = True
                break

        for pattern_prefix in brace_patterns:
            matches = list(re.finditer(pattern_prefix + r"\{", normalized))
            if not matches:
                continue
            for match in reversed(matches):
                brace_start = match.end() - 1
                end_pos = _match_balanced_braces(normalized, brace_start)
                if end_pos <= brace_start:
                    continue
                content = normalized[brace_start + 1 : end_pos]
                normalized = normalized[: match.start()] + content + normalized[end_pos + 1 :]
                had_latex_patterns = True
                break
            if normalized != previous:
                break

        if normalized == previous:
            break

    normalized = re.sub(r"\\(?:[;,:!]| )", " ", normalized)
    normalized = re.sub(r"\\(?:left|right|[bB]ig[glr]?)\b", "", normalized)
    normalized = normalized.replace(r"\dfrac", r"\frac")
    normalized = re.sub(r"\^\s*\\circ\b", " deg", normalized)
    normalized = re.sub(r"\^\s*\{\\circ\}", " deg", normalized)
    return normalized, had_latex_patterns


_UNICODE_WHITESPACE = re.compile(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000\s]+")
_LATEX_DELIMITER_START = re.compile(
    r"^\s*(?:\$\$|\$|\\\[|\\\(|\\boxed\s*\{|\\frac\s*\{|\\text\s*\{|\\mathrm\s*\{)"
)
_LATEX_MATH_COMMANDS = re.compile(
    r"\\(?:frac|dfrac|sqrt|sum|prod|int|oint|partial|nabla|infty|pm|mp|"
    r"sin|cos|tan|cot|sec|csc|arcsin|arccos|arctan|sinh|cosh|tanh|"
    r"log|ln|exp|lim|max|min|sup|inf|det|"
    r"alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|"
    r"iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|varphi|"
    r"chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega|"
    r"vec|hat|dot|ddot|bar|tilde|overline|underline|"
    r"cdot|times|div|circ|otimes|oplus|"
    r"leq|geq|neq|approx|equiv|propto|sim|simeq|"
    r"rightarrow|leftarrow|Rightarrow|Leftarrow)"
    r"(?![a-zA-Z])"
)
_LATEX_BINARY_RELATION_MARKERS: Tuple[str, ...] = (
    r"\leq",
    r"\geq",
    r"\neq",
    r"\equiv",
    r"\propto",
)
_PROSE_MARKERS = frozenset(
    {
        "the",
        "an",
        "is",
        "if",
        "in",
        "of",
        "on",
        "to",
        "by",
        "for",
        "from",
        "into",
        "with",
        "about",
        "between",
        "through",
        "above",
        "below",
        "during",
        "after",
        "before",
        "within",
        "without",
        "under",
        "over",
        "around",
        "across",
        "along",
        "among",
        "against",
        "toward",
        "towards",
        "upon",
        "until",
        "onto",
        "near",
        "and",
        "but",
        "yet",
        "nor",
        "because",
        "since",
        "although",
        "therefore",
        "however",
        "while",
        "whereas",
        "unless",
        "though",
        "hence",
        "thus",
        "this",
        "that",
        "these",
        "those",
        "which",
        "where",
        "when",
        "what",
        "how",
        "who",
        "its",
        "their",
        "they",
        "them",
        "our",
        "his",
        "her",
        "she",
        "you",
        "your",
        "are",
        "was",
        "were",
        "does",
        "did",
        "has",
        "have",
        "had",
        "would",
        "could",
        "should",
        "will",
        "can",
        "may",
        "might",
        "must",
        "been",
        "being",
        "let",
        "yes",
        "no",
        "not",
        "true",
        "false",
        "then",
        "than",
        "also",
        "only",
        "just",
        "very",
        "here",
        "there",
        "always",
        "never",
        "answer",
        "option",
        "correct",
        "incorrect",
        "equals",
        "equal",
        "result",
        "solution",
        "given",
        "means",
        "implies",
        "represents",
    }
)
_ALPHA_TOKEN_RE = re.compile(r"[A-Za-z]+(?:_[A-Za-z]+)*")


def _starts_with_latex_delimiter(text: str) -> bool:
    """Return whether a text surface begins with a LaTeX math wrapper."""

    return bool(_LATEX_DELIMITER_START.match(text))


def _contains_latex_commands(text: str) -> bool:
    """Return whether a text surface still contains recognizable LaTeX commands."""

    return bool(_LATEX_MATH_COMMANDS.search(text))


def _looks_like_math_expression(text: str) -> bool:
    """Heuristically separate math-like text from prose-heavy text."""

    tokens = _ALPHA_TOKEN_RE.findall(text.lower())
    return not any(token in _PROSE_MARKERS for token in tokens)


def normalize_text(answer_str: str) -> str:
    """Normalize generic answer text by collapsing Unicode and whitespace noise."""

    return _UNICODE_WHITESPACE.sub(" ", _normalize_unicode(answer_str)).strip()


__all__ = [
    "_ALPHA_TOKEN_RE",
    "_FULLWIDTH_PUNCT",
    "_LATEX_BINARY_RELATION_MARKERS",
    "_LATEX_DELIMITER_START",
    "_LATEX_MATH_COMMANDS",
    "_PROSE_MARKERS",
    "_SUBSCRIPT_TRANSLATION",
    "_SUPERSCRIPT_RUN_RE",
    "_SUBSCRIPT_RUN_RE",
    "_UNICODE_FRACTIONS",
    "_UNICODE_GREEK_CHARS",
    "_UNICODE_GREEK_TO_LATEX",
    "_UNICODE_SUBSCRIPT_CHARS",
    "_UNICODE_SUPERSCRIPT_TO_CARET",
    "_UNICODE_WHITESPACE",
    "_contains_latex_commands",
    "_convert_sqrt_unicode",
    "_extract_math_content",
    "_looks_like_math_expression",
    "_match_balanced_braces",
    "_normalize_unicode",
    "_starts_with_latex_delimiter",
    "_unicode_math_to_latex",
    "normalize_text",
]
