#!/usr/bin/env python3
"""Generate the LaTeX-surface canonicalization corpus fixture (deterministic).

Emits ``tests/fixtures/latex_surface_corpus.jsonl`` -- the fixed input set for
``tests/prkit/semantics/test_surface_repair.py``.

This corpus exists because ``tests/fixtures/phase4_corpus.jsonl`` cannot cover this
lane: its rows are plain text, so ``had_latex_patterns`` is false and ``latex2sympy``
never runs. Zero of its 291 rows change under a latex2sympy config change, which left
the ``canonical_text`` writer effectively untested.

Row schema (one JSON object per line):
    {"id","category","kind","input"|("pred","ref"),"expectation"}
  - kind        : "single" | "pair"
  - expectation :
      MATCH     -> compare_protocol_answers(pred, ref).equivalent is True
      NOMATCH   -> compare_protocol_answers(pred, ref).equivalent is False
      ROUNDTRIP -> the comparison surface reparses to the same symbols as the source

Categories:
  CASE      symbol-case collapse (``\\frac{E}{T}`` -> ``e/t``)
  SINGLETON SymPy singleton capture (``\\gamma``/``\\Gamma`` -> EulerGamma, ``I``, ``e``)
  EXPONENT  exponent dropped on a captured singleton (``I^{2}`` -> ``I``)
  RELATION  the same defects inside a relation, where ``Eq(...)`` hides them
  PRESERVE  genuinely equivalent pairs that rely on the text surface (recall lock)
  ROUNDTRIP writer/reader agreement on a single surface

Run from the repo root:  ``.venv/bin/python tools/build_latex_surface_corpus.py``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "latex_surface_corpus.jsonl"
)

# --- Adversarial rejects: distinct physics that a collapsed surface conflates -------
# Each pair is two *different* answers whose stored canonical_text is byte-identical.
_REJECT_PAIRS: list[tuple[str, str, str, str]] = [
    # (category, id-suffix, pred, ref)
    ("CASE", "energy-vs-charge", r"$\frac{E}{T}$", r"$\frac{e}{t}$"),
    ("CASE", "charge-mass-ratio", r"$\frac{Q}{M}$", r"$\frac{q}{m}$"),
    ("CASE", "boltzmann", r"$N k_B T$", r"$n k_B t$"),
    ("CASE", "capacitance", r"$\frac{Q}{V}$", r"$\frac{q}{v}$"),
    ("SINGLETON", "gamma-vs-Gamma", r"$\Gamma$", r"$\gamma$"),
    ("SINGLETON", "adiabatic-index", r"$P V^{\gamma}$", r"$P V^{\Gamma}$"),
    ("SINGLETON", "lorentz-vs-reflection", r"$\gamma m c^2$", r"$\Gamma m c^2$"),
    ("EXPONENT", "joule-heating", r"$I^2 R$", r"$I R$"),
    ("EXPONENT", "current-cubed", r"$I^3 R$", r"$I R$"),
    ("RELATION", "ohm-vs-joule", r"$V = I R$", r"$V = I^2 R$"),
    (
        "RELATION",
        "coulomb-distinct-charges",
        r"$F = \frac{Q q}{4\pi\epsilon_0 r^2}$",
        r"$F = \frac{q^2}{4\pi\epsilon_0 r^2}$",
    ),
]

# --- Recall lock: equivalent pairs whose accept currently rides on the text surface --
# These are the reason the fix repairs the surface rather than distrusting it; dropping
# canonical_text instead of recomputing it turns every one of these into a false negative.
_ACCEPT_PAIRS: list[tuple[str, str, str, str]] = [
    ("PRESERVE", "latex-vs-plain", r"$\frac{E}{T}$", r"E/T"),
    ("PRESERVE", "factor-order", r"$\frac{1}{2}mv^2$", r"$\frac{mv^2}{2}$"),
    ("PRESERVE", "lorentz-commuted", r"$\gamma m c^2$", r"$m c^2 \gamma$"),
    ("PRESERVE", "lorentz-spacing", r"$\gamma m c^2$", r"$\gamma\, m c^{2}$"),
    ("PRESERVE", "joule-commuted", r"$I^2 R$", r"$R I^2$"),
    ("PRESERVE", "joule-spacing", r"$I^{2}R$", r"$I^{2}\,R$"),
    ("PRESERVE", "boltzmann-commuted", r"$N k_B T$", r"$k_B N T$"),
    ("PRESERVE", "boltzmann-spacing", r"$N k_B T$", r"$N\,k_{B}\,T$"),
    ("PRESERVE", "adiabatic-spacing", r"$P V^{\gamma}$", r"$PV^{\gamma}$"),
    ("PRESERVE", "gamma-ratio", r"$\frac{\gamma}{\Gamma}$", r"$\gamma / \Gamma$"),
    ("PRESERVE", "gamma-identical", r"$\gamma$", r"$\gamma$"),
    ("PRESERVE", "euler-vs-exp", r"$e^{-t/\tau}$", r"$\exp(-t/\tau)$"),
    ("PRESERVE", "euler-imaginary", r"$e^{i\theta}$", r"$\exp{(i\theta)}$"),
    (
        "PRESERVE",
        "coulomb-varepsilon",
        r"$\frac{e^2}{4\pi\epsilon_0 r}$",
        r"$\frac{e^2}{4\pi \varepsilon_0 r}$",
    ),
    (
        "PRESERVE",
        "lorentz-half",
        r"$\frac{1}{2}\gamma m v^2$",
        r"$\frac{\gamma m v^{2}}{2}$",
    ),
    ("PRESERVE", "mass-energy", r"$E = mc^2$", r"$E = m c^{2}$"),
    ("PRESERVE", "mass-energy-commuted", r"$E = mc^2$", r"$E = c^{2} m$"),
    ("PRESERVE", "ohm-commuted", r"$V = I R$", r"$V = R I$"),
    (
        "PRESERVE",
        "adiabatic-relation-flip",
        r"$P V^{\gamma} = C$",
        r"$C = P V^{\gamma}$",
    ),
    (
        "PRESERVE",
        "coulomb-varepsilon-relation",
        r"$F = \frac{Q q}{4\pi\epsilon_0 r^2}$",
        r"$F = \frac{Q q}{4\pi \varepsilon_0 r^{2}}$",
    ),
]

# --- Writer/reader agreement: the comparison surface must reparse to the same symbols --
_ROUNDTRIP_INPUTS: list[tuple[str, str]] = [
    ("lorentz", r"$\gamma m c^2$"),
    ("adiabatic", r"$P V^{\gamma}$"),
    ("joule", r"$I^2 R$"),
    ("energy-period", r"$\frac{E}{T}$"),
    ("boltzmann", r"$N k_B T$"),
    ("charge-mass", r"$\frac{Q}{M}$"),
    ("coulomb", r"$\frac{Q q}{4\pi\epsilon_0 r^2}$"),
    ("capital-gamma", r"$\Gamma$"),
    ("kinetic", r"$\frac{1}{2}mv^2$"),
    ("decay", r"$e^{-t/\tau}$"),
]


def _build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for category, suffix, pred, ref in _REJECT_PAIRS:
        rows.append(
            {
                "id": f"{category}-{suffix}",
                "category": category,
                "kind": "pair",
                "pred": pred,
                "ref": ref,
                "expectation": "NOMATCH",
            }
        )
    for category, suffix, pred, ref in _ACCEPT_PAIRS:
        rows.append(
            {
                "id": f"{category}-{suffix}",
                "category": category,
                "kind": "pair",
                "pred": pred,
                "ref": ref,
                "expectation": "MATCH",
            }
        )
    for suffix, text in _ROUNDTRIP_INPUTS:
        rows.append(
            {
                "id": f"ROUNDTRIP-{suffix}",
                "category": "ROUNDTRIP",
                "kind": "single",
                "input": text,
                "expectation": "ROUNDTRIP",
            }
        )
    return rows


def _serialize(rows: list[dict[str, object]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed fixture matches the generator; exit 1 on drift",
    )
    args = parser.parse_args()

    rendered = _serialize(_build_rows())
    if args.check:
        current = _FIXTURE.read_text(encoding="utf-8") if _FIXTURE.exists() else ""
        if current != rendered:
            print(
                f"latex surface corpus fixture is stale -- run `python {Path(__file__).name}`"
                " to regenerate",
                file=sys.stderr,
            )
            return 1
        print(f"latex surface corpus fixture is up to date ({_FIXTURE})")
        return 0

    _FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    _FIXTURE.write_text(rendered, encoding="utf-8")
    print(f"wrote {len(_build_rows())} rows to {_FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
