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
# v6: answer-level sign-convention declaration — Call A sets a_ref's expressed convention when the
# problem leaves the axis free; Call B sets q_ref's convention only when the problem text fixes one.
# v5: staged 3-call build (answer-surface cleanup / question policy / symbol assumptions)
# replaces the single fused reference call; structure/kind are deterministically pinned.
REFERENCE_PROMPT_VERSION = "v6"
PREDICTION_PROMPT_NAME = "prediction_semantics"
# v5: directional answers declare their sign convention — a_pred_llm via
# prediction_answer_semantics.sign_convention, a_pred_ext by stating it in the final-answer surface.
# v4: isolated problem-only solve flag (suppresses the embedded question-semantics draft) +
# STRUCTURE.md section-2 surface conventions added to the answer-format guidance.
PREDICTION_PROMPT_VERSION = "v5"
PROBLEM_PROMPT_NAME = "problem_semantics"
# Problem-only (answer-blind) q_prob build, shares the staged question-policy / assumption
# prompts with no answer context.
PROBLEM_PROMPT_VERSION = "v1"

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

# STRUCTURE.md section-2 surface conventions, restated for the answer-format guidance so a
# plain `final_answer` surface is unambiguous to the deterministic parser (a_pred_ext). These
# are the same boundary tie-break rules the parser/canonicalizer share; stating them in the
# prompt keeps the model's surface and the parser's reading of it aligned.
_ANSWER_SURFACE_CONVENTIONS = """Write the final-answer surface using these conventions so it parses unambiguously:
- One indivisible value: write it bare (e.g. `5 m`, `x**2/2`). A single coordinate is one value, not a 1-tuple.
- Ordered coordinate of one object `(x, y)`: parentheses with >=2 finite parts, e.g. `(3, 4)`. A bare finite `(a, b)` is a tuple, NOT an interval.
- Unordered set of distinct solutions: braces, e.g. `{2, -2}`.
- Connected range (interval): bracket form only, e.g. `[a, b]`, `(a, b]`, or a range containing `inf`/`-inf`. Never use bare parentheses for a range.
- Several question-defined parts: label them, e.g. `(a) 5 m; (b) 2 s`.
- Shaped array: a vector as `<...>` or `[a, b, c]` (depth 1); a matrix as nested brackets (depth 2); keep a `(n,)` vector distinct from an `(n, 1)` matrix.
- Piecewise function: use `\\begin{cases}...\\end{cases}` or `Piecewise(...)`.
- Equation/relation: write the full relation, e.g. `F = m*a`, `v >= 0`.
- Free-axis sign convention: if your answer is a directional quantity (signed scalar, vector, or direction) whose sign depends on a positive-direction choice the problem left free, state that choice explicitly in the surface, e.g. `-20 m/s (taking rightward as positive)`.
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
    suppress_question_semantics_draft: bool = False,
) -> str:
    """Build the prompt for prediction-semantics generation.

    By default the prompt injects a toolkit draft of the question semantics as a hint (the
    fused path). For the isolated problem-only solve (``a_pred_llm`` / ``a_pred_ext``), pass
    ``suppress_question_semantics_draft=True``: the solve prompt is then problem text +
    options + context only, with no embedded question-semantics draft — closing a leakage
    surface (the draft is reference-conditioned policy the solver should not see) and keeping
    the contract side strictly separate from the solve side. The STRUCTURE.md section-2
    surface conventions are added to the answer-format guidance so a plain ``final_answer``
    surface is unambiguous to the deterministic parser regardless of the draft hint.
    """

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
    ]

    if not suppress_question_semantics_draft:
        question_draft = (
            draft_question_semantics or infer_prediction_question_semantics(problem)
        )
        sections.extend(
            [
                "Toolkit heuristic draft question semantics:",
                question_draft.model_dump_json(indent=2),
            ]
        )

    sections.extend(
        [
            "Your `final_answer` must be only the final answer text.",
            _ANSWER_SURFACE_CONVENTIONS,
        ]
    )
    if include_prediction_answer_semantics:
        sections.append(
            "Make `prediction_answer_semantics` match that final answer exactly. When your answer "
            "is directional and the problem fixed no positive direction, set "
            "`prediction_answer_semantics.sign_convention` to the convention you used (e.g. "
            "`right-as-positive`)."
        )
    else:
        sections.append(
            "Do not restate the derivation in `reasoning`; keep it to 1-3 short sentences."
        )
    return "\n\n".join(sections)


# ----------------------------------------------------------------------------------------
# Staged build prompts (3 focused calls). Each is advisory: the deterministic backbone pins
# structure/object_kind, tolerance, and allowed_*; these calls only clean/declare fields.
# ----------------------------------------------------------------------------------------
_STAGED_BUILD_NOTE = (
    "The toolkit decides `structure`, `object_kind`, `tolerance`, and `allowed_*` "
    "deterministically; do not try to change them. Only provide the fields this task asks for."
)


def build_answer_surface_cleanup_prompt(
    problem: PhysicsProblem,
    *,
    golden_text: str,
    draft_answer: PhysicsAnswerSemantics,
) -> str:
    """Call A: clean the golden answer surface; structure/object_kind are pinned."""

    return "\n\n".join(
        [
            _COMMON_ROLE,
            "Task: clean the canonical surface of the ground-truth answer semantics below.",
            _STAGED_BUILD_NOTE,
            "Keep `structure` and `object_kind` EXACTLY as in the draft (pinned by the toolkit). "
            "Only improve `canonical_text`, `canonical_latex`, `unit`, `numeric_text`, "
            "`choice_label`, and similar surface fields; preserve the answer's printed numeric "
            "precision (do not round or add digits).",
            "If the golden is a directional quantity (a signed scalar, a vector, or a directional "
            "sign) whose sign depends on a positive-direction / axis choice the problem did NOT fix, "
            "set `sign_convention` to the convention this answer is expressed in (e.g. "
            "`right-as-positive`, `up-as-positive`), inferred from the problem text, any figure, and "
            "the golden's sign. Otherwise leave `sign_convention` null — for a non-directional "
            "quantity, or when the problem itself fixes the axis (that is a question policy, captured "
            "separately, not an answer attribute).",
            _format_problem(problem, include_reference_context=True),
            "Ground-truth answer surface:\n" + golden_text,
            "Toolkit deterministic draft answer semantics (authoritative for structure/kind):",
            draft_answer.model_dump_json(indent=2),
        ]
    )


def build_question_policy_prompt(
    problem: PhysicsProblem,
    *,
    answer_draft: PhysicsAnswerSemantics | None = None,
) -> str:
    """Call B: question-side policy fields (q_ref when answer_draft given, else q_prob)."""

    sections = [
        _COMMON_ROLE,
        "Task: return the question-side policy semantics that constrain acceptable answers.",
        _STAGED_BUILD_NOTE,
        "Provide `target_variable`, `symbol_aliases`, `question_unit_policy`, `question_unit`, "
        "`dimension`, `ordering`, `required_parts`, and `choice_space` when applicable. Leave "
        "`symbol_assumptions` empty here (declared separately).",
        "Set `coordinate_frame` / `sign_convention` ONLY if the problem statement itself fixes a "
        "convention every answer must follow (e.g. it says 'take rightward as positive', or it "
        "defines the axes). If the problem leaves the positive direction / axis free, leave both "
        "null — the convention the golden answer happens to use is captured on the answer, not here.",
        _format_problem(problem, include_reference_context=answer_draft is not None),
    ]
    if answer_draft is not None:
        sections.append(
            "Cleaned answer semantics for context (do not restate it; infer policy against it):"
        )
        sections.append(answer_draft.model_dump_json(indent=2))
    return "\n\n".join(sections)


def build_symbol_assumptions_prompt(
    problem: PhysicsProblem,
    *,
    candidate_symbols: tuple[str, ...],
    answer_draft: PhysicsAnswerSemantics | None = None,
) -> str:
    """Call C: justified real-domain declarations for the answer's free symbols."""

    symbols_line = (
        ", ".join(candidate_symbols)
        if candidate_symbols
        else "(infer the free symbols from the problem)"
    )
    sections = [
        _COMMON_ROLE,
        "Task: declare the real-domain assumption for each free symbol, with justification.",
        "Use ONLY canonical (post-alias) symbol tokens. Allowed assumptions: "
        "real, nonzero, nonnegative, positive, complex.",
        "Declare a stronger-than-real domain (positive/nonnegative/nonzero) ONLY when the "
        "problem text explicitly justifies it (e.g. a stated constraint, a physical bound). "
        "Never infer positivity from surface form; when unsure, declare `real` or omit the symbol.",
        f"Candidate canonical symbols: {symbols_line}",
        _format_problem(problem, include_reference_context=answer_draft is not None),
    ]
    if answer_draft is not None:
        sections.append("Answer semantics for context:")
        sections.append(answer_draft.model_dump_json(indent=2))
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
    "PROBLEM_PROMPT_NAME",
    "PROBLEM_PROMPT_VERSION",
    "REFERENCE_PROMPT_NAME",
    "REFERENCE_PROMPT_VERSION",
    "answer_like_to_text",
    "build_answer_surface_cleanup_prompt",
    "build_prediction_semantics_prompt",
    "build_question_policy_prompt",
    "build_reference_semantics_prompt",
    "build_symbol_assumptions_prompt",
]
