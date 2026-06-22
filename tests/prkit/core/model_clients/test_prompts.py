"""Tests for the core-layer physics prompt builders."""

from prkit.core.domain import PhysicsProblem
from prkit.core.model_clients.prompts import (
    build_plain_question_prompt,
    format_problem_context,
)


def test_format_problem_context_includes_core_fields():
    problem = PhysicsProblem(
        problem_id="p1",
        question="  What is the net force?  ",
        problem_type="MC",
        domain="mechanics",
        options=["1 N", "2 N", "3 N"],
    )
    text = format_problem_context(problem)

    assert "Problem ID: p1" in text
    assert "Problem type: MC" in text
    assert "Language: en" in text
    assert "A. 1 N" in text and "B. 2 N" in text and "C. 3 N" in text
    # Question is stripped and rendered last.
    assert text.endswith("Question:\nWhat is the net force?")


def test_format_problem_context_notes_attached_images(tmp_path):
    image = tmp_path / "fig.png"
    image.write_bytes(b"x")
    problem = PhysicsProblem(
        problem_id="p2",
        question="Describe the figure.",
        image_path=[str(image)],
    )
    text = format_problem_context(problem)
    assert "Attached images: 1 image(s)" in text


def test_format_problem_context_omits_optional_sections_when_absent():
    problem = PhysicsProblem(problem_id="p3", question="Define momentum.")
    text = format_problem_context(problem)
    assert "Options:" not in text
    assert "Attached images" not in text


def test_build_plain_question_prompt_matches_context():
    problem = PhysicsProblem(problem_id="p4", question="State Hooke's law.")
    assert build_plain_question_prompt(problem) == format_problem_context(problem)


def test_header_parity_with_semantics_format_problem():
    """The core header must match the semantics layer's prediction header."""
    from prkit.semantics.build.prompts import _format_problem

    problem = PhysicsProblem(
        problem_id="p5",
        question="A 2 kg block accelerates at 3 m/s^2. Net force?",
        problem_type="OE",
        options=None,
    )
    assert format_problem_context(problem) == _format_problem(
        problem, include_reference_context=False
    )
