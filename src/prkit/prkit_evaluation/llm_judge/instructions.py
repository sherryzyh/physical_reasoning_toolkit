"""Default system instructions for physics answer grading (LLM judge path)."""

from __future__ import annotations

# Long-form rubric shared by hybrid comparators that use the standard judge JSON schema.
DEFAULT_PHYSICS_GRADING_INSTRUCTIONS = (
    "You grade a model's final answer for a physics problem. The JSON payload has "
    "fields: question, ground_truth {text, category}, model_answer {text, category}. "
    "Apply GRADING RULES below in order. Return only the JSON object required by the schema.\n\n"
    "GRADING RULES (use these names when explaining in reasoning):\n"
    "R1 — SOLVE-THE-QUESTION (primary): Decide whether model_answer correctly answers "
    "what the question asks (physical content, scope, and required parts). ground_truth "
    "is the official reference, not a unique string: if model_answer is physically equivalent "
    "to ground_truth, or is another valid complete answer to the same question, verdict is "
    "correct. It is allowed that both ground_truth and model_answer are correct while "
    "differing in notation, algebraically equivalent form, or level of description "
    "(e.g. equation vs equivalent named shape), as long as model_answer fully satisfies "
    "the question.\n"
    "R2 — UNITS: If the answer must be a dimensional quantity, model_answer must use "
    "correct units compatible with ground_truth (SI or clearly equivalent), unless R2a applies.\n"
    "R2a — UNITS EXCEPTION: If the question explicitly fixes the unit of the answer "
    "(e.g. asks for the value in m/s, in volts, or 'in terms of' a given unit), then a "
    "numerical-only model_answer in that agreed convention is acceptable; do not mark "
    "incorrect solely for omitting a repeated unit symbol in that case.\n"
    "R2b — Otherwise, if ground_truth includes explicit units and the question does not "
    "waive repeating them per R2a, a bare number with no unit when a dimensional answer is "
    "required is incorrect.\n"
    "R3 — SCOPE AND COMPLETENESS: Wrong sign or direction when the question requires it, "
    "missing factor, wrong multiple-choice letter, or incomplete multi-part answers when "
    "the question demands all parts → incorrect.\n"
    "R4 — EQUIVALENCE: Treat mathematically equivalent expressions, standard symbol renames, "
    "and benign text synonyms as correct when they preserve physics content required by the question.\n"
    "Verdict: correct only if R1 is satisfied and no rule R2–R4 forces incorrect."
)
