"""Surface-form canonicalization: the stored ``canonical_text`` cannot decide a verdict.

``canonical_text`` is written by ``latex2sympy`` but read back by this package's own SymPy
substrate, and the two disagree. latex2sympy resolves a few tokens to SymPy *singletons*
before it looks at symbol casing, and lowercases the rest, so the stored surface stops
denoting what the answer said::

    \\gamma, \\Gamma -> EulerGamma    I -> ImaginaryUnit    e, e^2 -> E, exp(2)
    \\frac{E}{T} -> e/t               N k_B T -> k_b*n*t    Q q -> q*q

Because that surface is consulted as an OR-branch, every collapse is a *precision* leak:
``$I^2 R$`` and ``$I R$`` both stored ``I*r`` and compared equal. The engine now recomputes
the surface from ``canonical_latex`` before using it
(``prkit.semantics.comparison.common.effective_canonical_text``).

Three batteries, each proving a different thing:

- NOMATCH rows: the precision leak is closed.
- MATCH rows: it was not closed by over-blocking. Note what these do *not* prove -- simply
  dropping a corrupted ``canonical_text`` instead of recomputing it also passes every row
  here, because the primary ``canonical_latex`` surface carries these pairs on its own.
- ROUNDTRIP rows: the surface the engine compares still denotes the answer. This is the
  battery that separates repair from dropping: dropping leaves no surface at all, so these
  rows fail under it while every comparison row still passes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from prkit.semantics import compare_protocol_answers, normalize_physics_answer
from prkit.semantics.comparison.common import (
    _needs_surface_repair,
    effective_canonical_text,
)
from prkit.semantics.comparison.semantics import (
    parse_symbolic_expression,
    preprocess_symbolic_text,
)

_FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "latex_surface_corpus.jsonl"
)
_GENERATOR = (
    Path(__file__).resolve().parents[3] / "tools" / "build_latex_surface_corpus.py"
)


def _load_rows() -> list[dict[str, object]]:
    text = _FIXTURE.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


_ROWS = _load_rows()
_PAIRS = [row for row in _ROWS if row["kind"] == "pair"]
_REJECTS = [row for row in _PAIRS if row["expectation"] == "NOMATCH"]
_ACCEPTS = [row for row in _PAIRS if row["expectation"] == "MATCH"]
_SINGLES = [row for row in _ROWS if row["kind"] == "single"]


def test_corpus_fixture_matches_generator() -> None:
    """The committed fixture must equal its generator output (no hand-edit drift)."""

    spec = importlib.util.spec_from_file_location(
        "build_latex_surface_corpus", _GENERATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = module._serialize(module._build_rows())
    assert _FIXTURE.read_text(encoding="utf-8") == expected, (
        "latex_surface_corpus.jsonl is out of sync with "
        "tools/build_latex_surface_corpus.py; regenerate with "
        "`python tools/build_latex_surface_corpus.py`"
    )


def test_corpus_has_both_batteries() -> None:
    """A reject battery without its matched accept battery cannot detect over-blocking."""

    assert _REJECTS, "no adversarial reject rows loaded"
    assert _ACCEPTS, "no recall-preservation rows loaded"
    assert _SINGLES, "no round-trip rows loaded"


@pytest.mark.parametrize("row", _REJECTS, ids=[str(row["id"]) for row in _REJECTS])
def test_collapsed_surface_does_not_accept(row: dict[str, object]) -> None:
    """Distinct physics that shared a collapsed ``canonical_text`` must not compare equal."""

    pred = normalize_physics_answer(str(row["pred"]))
    ref = normalize_physics_answer(str(row["ref"]))
    result = compare_protocol_answers(pred, ref)
    assert not result.equivalent, (
        f"{row['pred']!r} and {row['ref']!r} are different answers but compared equal; "
        f"stored canonical_text was {pred.canonical_text!r} / {ref.canonical_text!r}"
    )


@pytest.mark.parametrize("row", _ACCEPTS, ids=[str(row["id"]) for row in _ACCEPTS])
def test_repair_preserves_equivalent_answers(row: dict[str, object]) -> None:
    """Equivalent answers involving a repaired surface must still compare equal.

    The over-blocking lock: a guard that suppressed too much would fail here. Measured
    scope -- these pairs are also carried by the primary ``canonical_latex`` surface, so
    they detect over-blocking rather than proving the text surface was load-bearing.
    """

    pred = normalize_physics_answer(str(row["pred"]))
    ref = normalize_physics_answer(str(row["ref"]))
    result = compare_protocol_answers(pred, ref)
    assert result.equivalent, (
        f"{row['pred']!r} and {row['ref']!r} are equivalent but compared unequal "
        f"(mode={result.comparison_mode}, diagnostics={result.diagnostics})"
    )


@pytest.mark.parametrize("row", _SINGLES, ids=[str(row["id"]) for row in _SINGLES])
def test_comparison_surface_round_trips(row: dict[str, object]) -> None:
    """The surface the engine compares must reparse to the symbols the answer actually has.

    Without repair this fails loudly: ``$\\gamma m c^2$`` stores ``EulerGamma*c**2*m``, which
    the reader splits into nine invented symbols (``E*G*a**2*e*l*m**2*r*u``).
    """

    source = str(row["input"])
    answer = normalize_physics_answer(source)
    surface = effective_canonical_text(answer)
    assert surface, f"no comparison surface for {source!r}"

    expected = parse_symbolic_expression(preprocess_symbolic_text(source))
    actual = parse_symbolic_expression(surface)
    if expected is None:
        pytest.skip(f"{source!r} is not a bare expression")
    assert actual is not None, f"comparison surface {surface!r} no longer parses"
    assert {str(s) for s in actual.free_symbols} == {
        str(s) for s in expected.free_symbols
    }, f"surface {surface!r} does not carry the symbols of {source!r}"


def test_repair_is_scoped_to_latex_written_surfaces() -> None:
    """Only symbolic answers built from LaTeX are candidates for repair.

    Numbers, quantities and labels never pass through latex2sympy, so probing them would
    misread an ordinary unit (``\\AA`` -> ``angstrom``) as a collapsed symbol.
    """

    for source in ("42", "1 km", r"1000 \AA", "Yes", "counterclockwise", "E/T"):
        assert not _needs_surface_repair(normalize_physics_answer(source)), source


def test_repair_is_reported_in_diagnostics() -> None:
    """A repaired surface is recorded so an audit replay can count it."""

    repaired = compare_protocol_answers(
        normalize_physics_answer(r"$I^2 R$"), normalize_physics_answer(r"$R I^2$")
    )
    assert "canonical_text_repaired" in repaired.diagnostics

    untouched = compare_protocol_answers(
        normalize_physics_answer(r"$\frac{1}{2}mv^2$"),
        normalize_physics_answer(r"$\frac{mv^2}{2}$"),
    )
    assert "canonical_text_repaired" not in untouched.diagnostics
