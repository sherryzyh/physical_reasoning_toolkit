"""Top-level normalization exports for question and answer semantics.

Question inference derives the semantic constraints implied by the prompt.
Answer normalization converts raw answer surfaces into structured,
object-kind-aware protocol objects with canonical physical-quantity
payloads and deterministic quantity views.
``enrich_answer_quantity_views`` remains available as a compatibility
helper for saved artifacts or externally supplied semantics records that
need quantity backfill.
"""

from .answer_normalization import normalize_physics_answer, normalize_problem_answer
from .quantity_views import enrich_answer_quantity_views, materialize_quantity_view
from .question_inference import (
    infer_prediction_question_semantics,
    infer_question_semantics,
    infer_reference_question_semantics,
)

__all__ = [
    "enrich_answer_quantity_views",
    "infer_prediction_question_semantics",
    "infer_question_semantics",
    "infer_reference_question_semantics",
    "materialize_quantity_view",
    "normalize_physics_answer",
    "normalize_problem_answer",
]
