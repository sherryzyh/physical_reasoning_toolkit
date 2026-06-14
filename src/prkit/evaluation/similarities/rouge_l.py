"""
Word-level ROUGE-L (F1) using longest common subsequence length.

Reference: Lin, ROUGE (2004). Suitable for comparing predicted vs reference text
without extra dependencies.
"""

from __future__ import annotations

import re


def _tokenize(text: str) -> list[str]:
    """Lowercase and whitespace-split *text* into word tokens."""
    s = text.strip().lower()
    s = re.sub(r"\s+", " ", s)
    if not s:
        return []
    return s.split()


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of longest common subsequence (dynamic programming)."""
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    # Two rows to limit memory
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[m]


def rouge_l_f1(candidate: str, reference: str) -> float:
    """
    ROUGE-L F1 between candidate and reference (word-level).

    Returns a value in [0, 1]. Empty candidate or reference yields 0.0.
    """
    c = _tokenize(candidate)
    r = _tokenize(reference)
    if not c or not r:
        return 0.0
    lcs = _lcs_length(c, r)
    if lcs == 0:
        return 0.0
    p = lcs / len(c)
    rec = lcs / len(r)
    if p + rec == 0:
        return 0.0
    return 2.0 * p * rec / (p + rec)
