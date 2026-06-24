"""Phase-4 precision contract: the zero-diff guarantee of the recognition widening.

Drives ``tests/fixtures/phase4_corpus.jsonl`` and asserts that every
precision-contract category (A, B, C, E, F, I) classifies exactly as expected on
the current tree -- the machine-checkable statement that guarded-full recognition
introduced no symbolic-hijack regression. Categories D/G/H carry the *intended*
changes and are reviewed via ``internal/PHASE4_BEHAVIOR_DIFF.md``, so they are not
asserted here.
"""

from __future__ import annotations

import json
import string
from pathlib import Path

import pytest

from prkit.semantics.quantities import _pint_backend
from prkit.semantics.quantities.surface import _try_parse_physical_quantity
from prkit.semantics.quantities.units import UNIT_TO_BASE

_FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "phase4_corpus.jsonl"
_PRECISION_CONTRACT = frozenset({"A", "B", "C", "E", "F", "I"})


def _load_rows() -> list[dict[str, object]]:
    text = _FIXTURE.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


_ROWS = _load_rows()
_CONTRACT_SINGLES = [
    row
    for row in _ROWS
    if row["category"] in _PRECISION_CONTRACT
    and row["kind"] == "single"
    and row["expectation"] in {"SYMBOLIC", "QUANTITY"}
]


def test_corpus_fixture_shape() -> None:
    assert len(_ROWS) >= 250
    assert _PRECISION_CONTRACT <= {row["category"] for row in _ROWS}
    assert _CONTRACT_SINGLES, "no precision-contract single rows loaded"


@pytest.mark.parametrize(
    "row", _CONTRACT_SINGLES, ids=[str(row["id"]) for row in _CONTRACT_SINGLES]
)
def test_precision_contract_single(row: dict[str, object]) -> None:
    parsed = _try_parse_physical_quantity(str(row["input"]))
    if row["expectation"] == "SYMBOLIC":
        assert (
            parsed is None
        ), f"{row['id']} {row['input']!r} expected SYMBOLIC, parsed {parsed!r}"
    else:
        assert (
            parsed is not None
        ), f"{row['id']} {row['input']!r} expected QUANTITY, parsed None"


def test_explicit_recognition_anchors() -> None:
    # Guard holds: ambiguous single letters / the revolution family stay symbolic.
    assert _try_parse_physical_quantity("5 c") is None
    assert _try_parse_physical_quantity("5 rpm") is None
    assert _try_parse_physical_quantity("2 M") is None
    assert _try_parse_physical_quantity("5 S") is None
    # Genuine quantities are byte-identical to the pre-widening behavior.
    assert _try_parse_physical_quantity("1 km") == "1000 m"
    assert _try_parse_physical_quantity("16 rad s^-1") == "16 rad/s"
    assert _try_parse_physical_quantity("9.81 m/s^2") == "9.81 m/s^2"


def test_parse_denylist_matches_installed_pint() -> None:
    """Derive the denylist from the installed pint and assert it equals the constant.

    ``{ascii letters pint recognizes AND not a literal UNIT_TO_BASE key} U rev``.
    Literal membership is required: ``l`` is denylisted because ``"l"`` is not a
    key (only ``"L"`` is) even though it aliases to liter.
    """

    rev = {"rpm", "rps", "revolution", "cycle", "turn"}
    derived = frozenset(
        {
            ch
            for ch in string.ascii_letters
            if _pint_backend.is_recognized_unit(ch) and ch not in UNIT_TO_BASE
        }
        | rev
    )
    assert _pint_backend.PARSE_DENYLIST == derived
    assert _pint_backend.PARSE_DENYLIST == frozenset(
        set("abcdekltu") | set("BDMPRSU") | rev
    )
