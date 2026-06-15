"""Core-only prompt builders for asking a model to solve a physics problem.

These helpers depend only on :mod:`prkit.core.domain` so the model-client layer
stays self-contained (no dependency on the higher-level ``prkit.semantics``
package). The semantics package reuses :func:`format_problem_context` for the
shared problem-context header.
"""

from __future__ import annotations

from ..domain import PhysicsProblem


def format_problem_context(problem: PhysicsProblem) -> str:
    """Render the shared problem-context header embedded in solver prompts.

    Includes the problem id, type, domain, language, an attached-images notice,
    multiple-choice options, and the question text. It deliberately excludes any
    reference answer or worked solution so it is safe to send to a model that is
    being asked to solve the problem.
    """
    lines = [
        f"Problem ID: {problem.problem_id}",
        f"Problem type: {problem.problem_type or 'unspecified'}",
        f"Domain: {problem.get_domain_name()}",
        f"Language: {problem.language}",
    ]

    if problem.image_path:
        lines.append(
            f"Attached images: {len(problem.image_path)} image(s) are provided "
            "separately with this request."
        )

    if problem.options:
        option_lines = []
        for index, option in enumerate(problem.options):
            label = chr(ord("A") + index)
            option_lines.append(f"{label}. {option}")
        lines.append("Options:\n" + "\n".join(option_lines))

    lines.append("Question:\n" + (problem.question or "").strip())

    return "\n".join(lines)


def build_plain_question_prompt(problem: PhysicsProblem) -> str:
    """Build the prompt for solving *problem* from its plain question text."""
    return format_problem_context(problem)
