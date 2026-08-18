"""Top-level normalization exports for question and answer semantics.

Question inference derives the semantic constraints implied by the prompt.
Answer normalization converts raw answer surfaces into structured,
object-kind-aware protocol objects with canonical physical-quantity
payloads and deterministic quantity views.
``enrich_answer_quantity_views`` remains available as a compatibility
helper for saved artifacts or externally supplied semantics records that
need quantity backfill.
"""

from ..quantities.views import enrich_answer_quantity_views, materialize_quantity_view
from .answer_normalization import normalize_physics_answer, normalize_problem_answer
from .question_inference import (
    infer_prediction_question_semantics,
    infer_question_semantics,
    infer_reference_question_semantics,
)

#: Revision of the deterministic surface canonicalization that writes ``canonical_text``.
#: Bump when a change alters the stored canonical surface, so an artifact built by an older
#: revision is identifiable rather than silently reused. Records written at ``"1"`` carry
#: latex2sympy singleton capture and are repaired at read time by
#: ``prkit.semantics.comparison.common.effective_canonical_text``.
NORMALIZATION_VERSION = "1"

__all__ = [
    "NORMALIZATION_VERSION",
    "enrich_answer_quantity_views",
    "infer_prediction_question_semantics",
    "infer_question_semantics",
    "infer_reference_question_semantics",
    "materialize_quantity_view",
    "normalize_physics_answer",
    "normalize_problem_answer",
]
