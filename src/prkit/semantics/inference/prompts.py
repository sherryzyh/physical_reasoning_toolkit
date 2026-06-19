"""Reusable prompt builders for semantics inference workflows."""

from __future__ import annotations

from prkit.core.domain import Answer, PhysicsProblem
from prkit.core.model_clients.prompts import format_problem_context

from ..normalization import (
    infer_prediction_question_semantics,
    infer_reference_question_semantics,
    normalize_physics_answer,
)
from ..schema import PhysicsAnswerSemantics, PhysicsQuestionSemantics

REFERENCE_PROMPT_NAME = "reference_semantics"
REFERENCE_PROMPT_VERSION = "v4"
PREDICTION_PROMPT_NAME = "prediction_semantics"
PREDICTION_PROMPT_VERSION = "v3"

# Shared instructions keep the two prompt families aligned on the protocol
# schema and on the constraint that only the final answer object should be
# represented semantically.
_COMMON_ROLE = """You are generating canonical PRKit physics semantics.

The goal is stable answer comparison, not prose explanation.
Return semantics for the final answer object only.

Rules:
- `question_semantics` describes what forms of final answers the question allows.
- Add `question_semantics.symbol_aliases` when the problem and answer use different names for the same symbol, such as `y_s` versus `y`.
- In `symbol_aliases`, use plain token-style symbol names like `y`, `y_s`, `theta_dot`, not full equations or wrapped LaTeX snippets.
- `reference_answer_semantics` or `prediction_answer_semantics` must represent only the final answer.
- Use `object_kind` from: number, physical_quantity, expression, relation, qualitative_label, choice, boolean, sign_direction.
- Use `structure` from: atomic, multi_part, tuple, set, interval, vector, matrix, tensor, piecewise. Decide structure by denotation, not surface punctuation:
  - `atomic` = one indivisible value (the default). Prefer it: a single coordinate is atomic, not a 1-tuple; a closed point-range `[a, a]` is the atom `a`.
  - `tuple` = one ordered coordinate of a single object, `(x, y)`; a bare finite `(a, b)` is a tuple, NOT an interval.
  - `interval` = a connected range; use it only for bracket forms `[a, b]`/`(a, b]`/… or a range containing `∞`.
  - `set` = an unordered collection of distinct solutions `{x1, x2}`.
  - `multi_part` = answers to several question-defined sub-questions; use it only when the question defines parts (`required_parts`) or the answer is explicitly enumerated.
  - `vector`/`matrix`/`tensor` = a shaped array of rank 1/2/≥3; keep a `(n,)` vector distinct from an `(n, 1)` matrix.
  - `piecewise` = a function with (expression, condition) branches.
- Fill `numeric_value`, `numeric_text`, and `unit` for physical quantities when possible.
- Use `children` for structured answers and `cases` only for true piecewise answers.
- Use `subject_to` for global constraints on one answer object, such as `x>0`, `a<r<b`, or `n∈Z`.
- Keep `canonical_text` and `canonical_latex` focused on the main answer object; `raw_text` may preserve the full original surface including constraints.
- For `multi_part`, set each child `part_label` when the question defines named answer slots.
- Use `choice_label` for multiple-choice answers.
- Keep `canonical_text` stable and concise.
- Keep `diagnostics` empty unless there is real uncertainty.
"""


def build_reference_semantics_prompt(
    problem: PhysicsProblem,
    *,
    draft_question_semantics: PhysicsQuestionSemantics | None = None,
    draft_reference_answer_semantics: PhysicsAnswerSemantics | None = None,
) -> str:
    """Build the prompt for reference-semantics generation."""

    if problem.answer is None:
        raise ValueError(
            f"Problem {problem.problem_id} does not provide `problem.answer`."
        )

    question_draft = draft_question_semantics or infer_reference_question_semantics(
        problem
    )
    reference_draft = draft_reference_answer_semantics or normalize_physics_answer(
        problem.answer,
        context=question_draft,
    )

    sections = [
        _COMMON_ROLE,
        "Task: produce canonical reference semantics for the problem and its ground-truth answer.",
        _format_problem(problem, include_reference_context=True),
    ]

    sections.extend(
        [
            "Toolkit heuristic draft question semantics:",
            question_draft.model_dump_json(indent=2),
            "Toolkit heuristic draft reference answer semantics:",
            reference_draft.model_dump_json(indent=2),
            "Return the final corrected `question_semantics` and `reference_answer_semantics`.",
        ]
    )

    return "\n\n".join(sections)


def build_prediction_semantics_prompt(
    problem: PhysicsProblem,
    *,
    draft_question_semantics: PhysicsQuestionSemantics | None = None,
    include_prediction_answer_semantics: bool = True,
) -> str:
    """Build the prompt for prediction-semantics generation."""

    question_draft = draft_question_semantics or infer_prediction_question_semantics(
        problem
    )

    sections = [
        _COMMON_ROLE,
        (
            "Task: solve the physics problem, then return a concise reasoning summary, "
            "the final answer surface, and its prediction semantics."
            if include_prediction_answer_semantics
            else (
                "Task: solve the physics problem, then return only a concise reasoning "
                "summary and the final answer surface."
            )
        ),
        _format_problem(problem, include_reference_context=False),
        "Toolkit heuristic draft question semantics:",
        question_draft.model_dump_json(indent=2),
        "Your `final_answer` must be only the final answer text.",
    ]
    if include_prediction_answer_semantics:
        sections.append(
            "Make `prediction_answer_semantics` match that final answer exactly."
        )
    else:
        sections.append(
            "Do not restate the derivation in `reasoning`; keep it to 1-3 short sentences."
        )
    return "\n\n".join(sections)


def answer_like_to_text(answer: object) -> str:
    """Convert an answer-like object into a single answer surface string."""

    if answer is None:
        return ""
    if isinstance(answer, PhysicsAnswerSemantics):
        return answer.raw_text or answer.canonical_text
    if isinstance(answer, Answer):
        value_text = str(answer.value).strip()
        unit_text = "" if answer.unit is None else str(answer.unit).strip()
        if value_text and unit_text:
            return f"{value_text} {unit_text}"
        return value_text or unit_text
    return str(answer).strip()


def _format_problem(
    problem: PhysicsProblem,
    *,
    include_reference_context: bool = True,
) -> str:
    """Render the problem context block that is embedded in prompts.

    The shared header (id/type/domain/language/images/options/question) is
    produced by the core-layer ``format_problem_context``; the reference-context
    (answer/solution) block is appended here only when requested.
    """

    sections = [format_problem_context(problem)]

    if include_reference_context:
        answer_text = answer_like_to_text(problem.answer)
        if answer_text:
            sections.append("Answer:\n" + answer_text)

        solution_text = _problem_solution_text(problem)
        if solution_text:
            sections.append("Solution:\n" + solution_text)

    return "\n".join(sections)


def _problem_solution_text(problem: PhysicsProblem) -> str:
    """Collect any available worked-solution text without duplicates."""

    parts: list[str] = []
    seen: set[str] = set()

    for value in (
        problem.solution,
        problem.get("reason"),
        problem.get("reasoning"),
    ):
        text = "" if value is None else str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        parts.append(text)

    return "\n\n".join(parts)


__all__ = [
    "PREDICTION_PROMPT_NAME",
    "PREDICTION_PROMPT_VERSION",
    "REFERENCE_PROMPT_NAME",
    "REFERENCE_PROMPT_VERSION",
    "answer_like_to_text",
    "build_prediction_semantics_prompt",
    "build_reference_semantics_prompt",
]
