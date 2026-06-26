#!/usr/bin/env python3
"""Generate the Phase-4 adversarial corpus fixture (deterministic).

Emits ``tests/fixtures/phase4_corpus.jsonl`` -- the fixed input set the
recognition-contract test (``tests/.../test_phase4_recognition_contract.py``) and
the behavior-diff harness (``tools/phase4_diff.py``) both consume. The companion
spec is ``internal/PHASE4_ADVERSARIAL_CORPUS.md``; this script is its runnable form.

Row schema (one JSON object per line):
    {"id","category","kind","input"|("pred","ref"),"expectation"}
  - kind     : "single" | "pair"
  - expectation:
      SYMBOLIC  -> _try_parse_physical_quantity(input) is None
      QUANTITY  -> _try_parse_physical_quantity(input) is not None
      MATCH     -> compare_protocol_answers(pred, ref).equivalent is True
      NOMATCH   -> compare_protocol_answers(pred, ref).equivalent is False
      OBSERVE   -> diff-only; no contract assertion (owner ratifies)

Single-letter expectations are derived from the design predicate
``normalize_unit_text(letter) in UNIT_TO_BASE`` (NOT from the parser, so the
contract test is a genuine cross-check, not a tautology). This correctly handles
aliases such as ``l`` -> ``L`` (liter), which the spec doc mislabels.

Run from the repo root:  ``.venv/bin/python tools/build_phase4_corpus.py``
"""

from __future__ import annotations

import argparse
import json
import string
import sys
from pathlib import Path

from prkit.semantics.quantities.units import UNIT_TO_BASE, normalize_unit_text

_FIXTURE = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "phase4_corpus.jsonl"
)


def _is_known_base_unit(token: str) -> bool:
    """Design predicate: does this canonical token map to a known prkit base unit?"""

    return normalize_unit_text(token) in UNIT_TO_BASE or token in UNIT_TO_BASE


def _single_letter_expectation(unit_token: str) -> str:
    """SYMBOLIC unless the unit token resolves to a known base unit (then QUANTITY)."""

    return "QUANTITY" if _is_known_base_unit(unit_token) else "SYMBOLIC"


# --- curated rows transcribed from internal/PHASE4_ADVERSARIAL_CORPUS.md -------

# A: bare/spaced denylisted single letters (expectation derived per letter, so the
# alias case "l" -> "L" is classified QUANTITY rather than the doc's SYMBOLIC).
_A_SINGLES: list[tuple[str, str]] = [
    ("5c", "c"),
    ("5 c", "c"),
    ("3u", "u"),
    ("3 u", "u"),
    ("2a", "a"),
    ("7 d", "d"),
    ("4b", "b"),
    ("9 e", "e"),
    ("6k", "k"),
    ("8 l", "l"),
    ("5 t", "t"),
    ("2 B", "B"),
    ("3 M", "M"),
    ("4 P", "P"),
    ("5 R", "R"),
    ("6 S", "S"),
    ("7 U", "U"),
]
# A controls: legacy-unit letters that must stay QUANTITY.
_A_CONTROLS: list[tuple[str, str]] = [
    ("5 m", "m"),
    ("5 s", "s"),
    ("5 g", "g"),
    ("5 A", "A"),
    ("5 N", "N"),
    ("5 J", "J"),
    ("5 V", "V"),
    ("5 K", "K"),
    ("5 T", "T"),
]

# B: single letters inside expressions / relations (all SYMBOLIC).
_B_SINGLES: list[str] = [
    "v = 5a",
    "E = m c^2",
    "F = q v B",
    "x = 5 d + 2",
    "a = b c",
    "P = I V",
    "5 a + 3 b",
    "omega = 2 c",
    "k = 1/2 m v^2",
    "2 u + 5",
    "E_k = 1/2 M u^2",
]

# C: physics-constant letters / symbolic relations (all SYMBOLIC).
_C_SINGLES: list[str] = ["c", "e", "k", "R", "answer is c", "P = nRT/V"]

# D: multi-char unguarded pint tokens (OBSERVE -- owner reviews keep/deny).
_D_SINGLES: list[str] = [
    "5 au",
    "3 year",
    "2 barn",
    "5 are",
    "7 day",
    "5 cycle",
    "2 week",
    "1 month",
    "4 sr",
    "3 dB",
    "2 Np",
    "5 gal",
    "3 cup",
    "6 pt",
]

# E: rev/rpm family (all SYMBOLIC -- denylisted, 2pi trap).
_E_SINGLES: list[str] = [
    "5 rpm",
    "60 rpm",
    "3 rps",
    "2 revolution",
    "omega = 5 rpm",
    "1 turn",
    "4 cycle",
]

# F: genuine quantities, parity anchor (all QUANTITY).
_F_SINGLES: list[str] = [
    "1 km",
    "1.1 kΩ",
    "2.8e-4 Pa",
    "3e-4 N/m^2",
    "0.11°",
    "16 rad s^-1",
    "3.20×10^2 cm s^-2",
    "530 pF",
    "9.81 m/s^2",
    "5 eV",
    "2 kcal",
    "760 mmHg",
    "1000 Å",
    "4.5×10^3 N",
    "1.752 m",
    r"5\,\mathrm{m}",
    r"1.2\times10^{3}\,\mathrm{J}",
    "30^\\circ",
    "2.5×10⁻³ m",
]

# G: temperature (CHANGE -> converts; intended gain).
_G_SINGLES: list[str] = [
    "20 °C",
    "20 degC",
    "-40 °C",
    "100 °C",
    "0 °C",
    "98.6 °F",
    "37 ^\\circ C",
]
_G_PAIRS: list[tuple[str, str, str]] = [
    ("20 °C", "293.15 K", "MATCH"),
    ("20 °C", "300 K", "NOMATCH"),
    ("100 °C", "373.15 K", "MATCH"),
    ("-40 °C", "-40 °F", "MATCH"),
]

# H: newly-enabled legit units (CHANGE -> QUANTITY; intended gain).
_H_SINGLES: list[str] = [
    "5 daN",
    "2 cN",
    "3 dL",
    "10 ft",
    "30 mph",
    "5 lbf",
    "2 Mg",
    "7 nN",
    "4 µPa",
    "1 in",
    "12 in",
    "3 yd",
    "2 mile",
    "5 hp",
    "1 cal_th",
]

# I: uppercase unit-vs-variable borderline (singles denied -> SYMBOLIC; pairs OBSERVE).
_I_SINGLES: list[tuple[str, str]] = [
    ("5 S", "S"),
    ("2 M", "M"),
    ("4 P", "P"),
    ("3 R", "R"),
    ("7 B", "B"),
    ("2 D", "D"),
    ("5 U", "U"),
]
_I_PAIRS: list[tuple[str, str]] = [("5 S", "5 siemens"), ("2 M", "2 mol/L")]

# J: implicit unit-alias pairs (Phase 5). MATCH/NOMATCH are the intended bridge
# gains; OBSERVE rows are decline cases. These are *not* precision-contract rows --
# the bridge flips former object_kind_mismatch verdicts, recorded for owner review.
_J_PAIRS: list[tuple[str, str, str]] = [
    ("5 siemens", "5 S", "MATCH"),  # symmetric: terse gold
    ("5 S", "5 seconds", "MATCH"),  # same token, reference = second
    ("5 S", "5 s", "MATCH"),
    ("2 M", "2 molar", "MATCH"),
    ("5 S", "3 siemens", "NOMATCH"),  # numeric gate
    ("5 X", "5 siemens", "OBSERVE"),  # token not in alias map -> declines
    ("5 c", "5 m/s", "OBSERVE"),  # reference unit not in map -> declines
    # Phase 5b: same-kind (both parse) + new dictionary entries (poise, debye).
    ("5 s", "5 siemens", "MATCH"),  # "s" parses as second; rescued to siemens
    ("5 sec", "5 siemens", "OBSERVE"),  # original surface "sec" -> declines
    ("5 siemens", "5 s", "OBSERVE"),  # asymmetric: reference unit authoritative
    ("5 P", "5 poise", "MATCH"),  # new entry: dynamic viscosity
    ("5 D", "5 debye", "MATCH"),  # new entry: dipole moment
]

# Greek-letter controls (all SYMBOLIC).
_GREEK: list[str] = ["5\\omega", "5\\theta", "5\\alpha", "5\\beta", "5\\gamma"]


def _build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add_single(cat: str, idx: int, text: str, expectation: str) -> None:
        rows.append(
            {
                "id": f"{cat}-{idx:03d}",
                "category": cat,
                "kind": "single",
                "input": text,
                "expectation": expectation,
            }
        )

    def add_pair(cat: str, idx: int, pred: str, ref: str, expectation: str) -> None:
        rows.append(
            {
                "id": f"{cat}-P{idx:03d}",
                "category": cat,
                "kind": "pair",
                "pred": pred,
                "ref": ref,
                "expectation": expectation,
            }
        )

    n = 0
    for text, letter in _A_SINGLES + _A_CONTROLS:
        n += 1
        add_single("A", n, text, _single_letter_expectation(letter))
    for i, text in enumerate(_B_SINGLES, 1):
        add_single("B", i, text, "SYMBOLIC")
    for i, text in enumerate(_C_SINGLES, 1):
        add_single("C", i, text, "SYMBOLIC")
    for i, text in enumerate(_D_SINGLES, 1):
        add_single("D", i, text, "OBSERVE")
    for i, text in enumerate(_E_SINGLES, 1):
        add_single("E", i, text, "SYMBOLIC")
    for i, text in enumerate(_F_SINGLES, 1):
        add_single("F", i, text, "QUANTITY")
    for i, text in enumerate(_G_SINGLES, 1):
        add_single("G", i, text, "QUANTITY")
    for i, (pred, ref, exp) in enumerate(_G_PAIRS, 1):
        add_pair("G", i, pred, ref, exp)
    for i, text in enumerate(_H_SINGLES, 1):
        add_single("H", i, text, "QUANTITY")
    for i, (text, letter) in enumerate(_I_SINGLES, 1):
        add_single("I", i, text, _single_letter_expectation(letter))
    for i, (pred, ref) in enumerate(_I_PAIRS, 1):
        add_pair("I", i, pred, ref, "OBSERVE")
    for i, (pred, ref, exp) in enumerate(_J_PAIRS, 1):
        add_pair("J", i, pred, ref, exp)
    for i, text in enumerate(_GREEK, 1):
        add_single("C", 100 + i, text, "SYMBOLIC")

    # Single-letter sweep: every ascii letter (both cases) x {5{L}, {L}, 2{L}^2}.
    # Categorized A (SYMBOLIC) / F (QUANTITY) by the design predicate so the
    # contract test asserts them. Bare "{L}" carries no number -> always SYMBOLIC.
    for letter in string.ascii_letters:
        for tag, text, has_number in (
            ("5x", f"5{letter}", True),
            ("bare", letter, False),
            ("sq", f"2{letter}^2", True),
        ):
            expectation = (
                _single_letter_expectation(letter) if has_number else "SYMBOLIC"
            )
            cat = "F" if expectation == "QUANTITY" else "A"
            rows.append(
                {
                    "id": f"SWEEP-{tag}-{letter}",
                    "category": cat,
                    "kind": "single",
                    "input": text,
                    "expectation": expectation,
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
                f"phase4 corpus fixture is stale -- run `python {Path(__file__).name}`"
                " to regenerate",
                file=sys.stderr,
            )
            return 1
        print(f"phase4 corpus fixture is up to date ({_FIXTURE})")
        return 0

    _FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    _FIXTURE.write_text(rendered, encoding="utf-8")
    print(f"wrote {rendered.count(chr(10))} rows -> {_FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
