"""Public entry points for contract-driven semantics comparison.

This subpackage separates four concerns:

- ``coercion`` turns loose dictionaries or persisted JSON payloads into
  structured schema objects.
- ``contract`` builds and validates ``PhysicsEvaluationContract`` objects.
- ``engine`` orchestrates recursive comparison over atomic and structured
  answers under a policy-controlled contract.
- helper modules such as ``same_object_kind`` and
  ``different_object_kind`` implement the object-kind-aware comparison
  rules that the engine dispatches to.
"""

from .coercion import (
    coerce_evaluation_contract,
    coerce_protocol_answer,
    coerce_question_semantics,
)
from .contract import (
    build_evaluation_contract,
    coerce_policy_mode,
    validate_answer_against_contract,
)
from .engine import (
    compare_physics_answers,
    compare_predictions,
    compare_protocol_answers,
    compare_protocol_answers_legacy,
)

__all__ = [
    "build_evaluation_contract",
    "coerce_protocol_answer",
    "coerce_evaluation_contract",
    "coerce_policy_mode",
    "coerce_question_semantics",
    "compare_physics_answers",
    "compare_predictions",
    "compare_protocol_answers",
    "compare_protocol_answers_legacy",
    "validate_answer_against_contract",
]
