"""Tests for the gold/domain subtype: schema, normalization, terminal loop, task wiring."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from prkit.annotation.base import load_json
from prkit.annotation.tasks.gold import task as gold_task
from prkit.annotation.tasks.gold.domain import domain_choices, normalize_domain
from prkit.annotation.tasks.gold.task import GoldAnnotationTask
from prkit.annotation.tasks.gold.terminal import run_terminal_domain_annotation
from prkit.core.domain.physics_domain import PhysicsDomain
from prkit.core.domain.physics_problem import PhysicsProblem


def _scripted(responses: Iterable[str]):
    iterator = iter(responses)

    def _input(_prompt: str = "") -> str:
        return next(iterator)

    return _input


def _silent(_message: str) -> None:
    return None


def test_domain_choices_match_enum():
    choices = domain_choices()
    assert choices == [d.value for d in PhysicsDomain]
    assert "classical_mechanics" in choices


def test_normalize_domain_variants():
    assert normalize_domain("mechanics") == "mechanics"
    assert normalize_domain("Classical Mechanics") == "classical_mechanics"
    assert normalize_domain("") == ""
    # Unrecognized free-form text is kept verbatim, not collapsed to "other".
    assert normalize_domain("plasma turbulence") == "plasma turbulence"


def test_terminal_writes_records_and_skips(tmp_path):
    problems = [
        PhysicsProblem(problem_id="1", question="Q1", domain="mechanics"),
        PhysicsProblem(problem_id="2", question="Q2", domain="optics"),
        PhysicsProblem(problem_id="3", question="Q3"),
    ]
    # P1: accept suggested; P2: free text; P3: skip (no record).
    code = run_terminal_domain_annotation(
        problems,
        tmp_path,
        annotator="yh",
        input_fn=_scripted(["", "optics", "s"]),
        print_fn=_silent,
        now_fn=lambda: "2026-01-01T00:00:00",
    )
    assert code == 0
    assert load_json(tmp_path / "problem_1.json")["gold_domain"] == "mechanics"
    rec2 = load_json(tmp_path / "problem_2.json")
    assert rec2["gold_domain"] == "optics"
    assert rec2["annotator"] == "yh"
    assert not (tmp_path / "problem_3.json").exists()


def test_terminal_resume_skips_already_done(tmp_path):
    problems = [
        PhysicsProblem(problem_id="1", question="Q1", domain="mechanics"),
        PhysicsProblem(problem_id="2", question="Q2", domain="optics"),
        PhysicsProblem(problem_id="3", question="Q3"),
    ]
    run_terminal_domain_annotation(
        problems, tmp_path, input_fn=_scripted(["", "optics", "s"]), print_fn=_silent
    )
    # Re-run: P1/P2 are done; a single input must land on P3 (else StopIteration).
    run_terminal_domain_annotation(
        problems, tmp_path, input_fn=_scripted(["1"]), print_fn=_silent
    )
    assert load_json(tmp_path / "problem_3.json")["gold_domain"] == domain_choices()[0]
    # Earlier records untouched.
    assert load_json(tmp_path / "problem_1.json")["gold_domain"] == "mechanics"


def test_terminal_back_then_overwrite(tmp_path):
    problems = [
        PhysicsProblem(problem_id="1", question="Q1", domain="mechanics"),
        PhysicsProblem(problem_id="2", question="Q2", domain="optics"),
    ]
    # P1 accept -> P2 'b' back -> P1 pick #1 (overwrite) -> P2 free text.
    run_terminal_domain_annotation(
        problems,
        tmp_path,
        input_fn=_scripted(["", "b", "1", "optics"]),
        print_fn=_silent,
    )
    assert load_json(tmp_path / "problem_1.json")["gold_domain"] == domain_choices()[0]
    assert load_json(tmp_path / "problem_2.json")["gold_domain"] == "optics"


def test_task_rejects_bad_subtype():
    with pytest.raises(ValueError, match="gold subtype"):
        GoldAnnotationTask().run(subtype="laws", dataset="x")
    with pytest.raises(ValueError, match="needs a subtype"):
        GoldAnnotationTask().run(subtype=None)


def test_task_requires_a_source():
    with pytest.raises(ValueError, match="dataset name or a problems directory"):
        GoldAnnotationTask().run(subtype="domain")


def test_task_loads_problems_dir_and_uses_default_output(tmp_path, monkeypatch):
    folder = tmp_path / "seephys_gpt"
    folder.mkdir()
    for pid in ("1", "2"):
        (folder / f"problem_{pid}.json").write_text(
            f'{{"problem_id": "{pid}", "question": "Q{pid}", "domain": "mechanics"}}',
            encoding="utf-8",
        )

    captured = {}

    def fake_terminal(problems, output_dir, annotator=""):
        captured["n"] = len(problems)
        captured["output_dir"] = output_dir
        captured["annotator"] = annotator
        return 0

    monkeypatch.setattr(gold_task, "run_terminal_domain_annotation", fake_terminal)
    monkeypatch.chdir(tmp_path)

    code = GoldAnnotationTask().run(
        subtype="domain", problems_dir=folder, annotator="yh"
    )
    assert code == 0
    assert captured["n"] == 2
    assert captured["annotator"] == "yh"
    # Default output store is derived from the source directory name.
    assert captured["output_dir"].parts[-3:] == ("gold", "domain", "seephys_gpt")
