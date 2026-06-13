"""Convenience exports for semantics-generation workflows.

The :mod:`prkit.prkit_semantics.inference` package wraps three related
tasks:

- building stable prompts for reference and prediction semantics calls,
- validating and persisting the resulting artifacts, and
- comparing saved artifacts with the protocol comparator.
"""

from .artifacts import (
    PredictionSemanticsArtifact,
    PredictionSemanticsResponse,
    ReferenceSemanticsArtifact,
    ReferenceSemanticsResponse,
    SemanticsArtifact,
    SemanticsComparisonInputs,
    SemanticsEvaluationRecord,
    SemanticsGeneratorInfo,
    SemanticsProblemRecord,
    load_prediction_semantics_artifact,
    load_reference_semantics_artifact,
    load_semantics_artifact,
    load_semantics_evaluation_record,
    save_semantics_json,
)
from .calls import (
    PredictionSemanticsInferenceSpec,
    build_prediction_semantics_artifact,
    compare_saved_semantics,
    evaluate_saved_semantics,
    infer_prediction_semantics,
    infer_reference_semantics,
    parse_prediction_semantics_response_text,
    parse_reference_semantics_response_text,
    prepare_prediction_semantics_inference_spec,
    prepare_semantics_comparison,
)
from .prompts import (
    PREDICTION_PROMPT_NAME,
    PREDICTION_PROMPT_VERSION,
    REFERENCE_PROMPT_NAME,
    REFERENCE_PROMPT_VERSION,
    answer_like_to_text,
    build_prediction_semantics_prompt,
    build_reference_semantics_prompt,
)

__all__ = [
    "PREDICTION_PROMPT_NAME",
    "PREDICTION_PROMPT_VERSION",
    "PredictionSemanticsArtifact",
    "PredictionSemanticsInferenceSpec",
    "PredictionSemanticsResponse",
    "REFERENCE_PROMPT_NAME",
    "REFERENCE_PROMPT_VERSION",
    "ReferenceSemanticsArtifact",
    "ReferenceSemanticsResponse",
    "SemanticsArtifact",
    "SemanticsComparisonInputs",
    "SemanticsEvaluationRecord",
    "SemanticsGeneratorInfo",
    "SemanticsProblemRecord",
    "answer_like_to_text",
    "build_prediction_semantics_prompt",
    "build_reference_semantics_prompt",
    "build_prediction_semantics_artifact",
    "compare_saved_semantics",
    "evaluate_saved_semantics",
    "infer_prediction_semantics",
    "infer_reference_semantics",
    "load_prediction_semantics_artifact",
    "load_reference_semantics_artifact",
    "load_semantics_artifact",
    "load_semantics_evaluation_record",
    "parse_prediction_semantics_response_text",
    "parse_reference_semantics_response_text",
    "prepare_prediction_semantics_inference_spec",
    "prepare_semantics_comparison",
    "save_semantics_json",
]
