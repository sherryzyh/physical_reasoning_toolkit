"""Physics-aware final-answer semantics APIs for PRKit.

The implemented near-term method is ``PASEC-Base``:

1. infer question semantics as auxiliary interpretation context
2. normalize final answers into structured semantics objects
3. build a reference-assisted evaluation contract
4. compare answers under a policy-controlled contract

For the public schema, read ``TAXONOMY.md``.
For the operational comparison flow, read ``PROTOCOL_COMPARISON.md``.
"""

from .comparison import (
    build_evaluation_contract,
    coerce_evaluation_contract,
    coerce_policy_mode,
    coerce_protocol_answer,
    coerce_question_semantics,
    compare_physics_answers,
    compare_protocol_answers,
    compare_protocol_answers_legacy,
    validate_answer_against_contract,
)
from .inference import (
    PREDICTION_PROMPT_NAME,
    PREDICTION_PROMPT_VERSION,
    REFERENCE_PROMPT_NAME,
    REFERENCE_PROMPT_VERSION,
    PredictionSemanticsArtifact,
    PredictionSemanticsInferenceSpec,
    ReferenceSemanticsArtifact,
    SemanticsComparisonInputs,
    SemanticsEvaluationRecord,
    SemanticsGeneratorInfo,
    SemanticsProblemRecord,
    build_prediction_semantics_artifact,
    compare_saved_semantics,
    evaluate_saved_semantics,
    infer_prediction_semantics,
    infer_reference_semantics,
    load_prediction_semantics_artifact,
    load_reference_semantics_artifact,
    load_semantics_artifact,
    parse_prediction_semantics_response_text,
    parse_reference_semantics_response_text,
    prepare_prediction_semantics_inference_spec,
    prepare_semantics_comparison,
    save_semantics_json,
)
from .normalization import (
    infer_prediction_question_semantics,
    infer_question_semantics,
    infer_reference_question_semantics,
    normalize_physics_answer,
    normalize_problem_answer,
)
from .schema import (
    AnswerComparison,
    AnswerObjectKind,
    AnswerStructure,
    BridgeTier,
    ComparisonPolicyMode,
    ContractValidationResult,
    ContractValidationStatus,
    OrderingPolicy,
    PhysicalQuantitySnapshot,
    PhysicalQuantityView,
    PhysicsAnswerCaseSemantics,
    PhysicsAnswerSemantics,
    PhysicsEvaluationContract,
    PhysicsQuestionSemantics,
    PhysicsSymbolAliasSemantics,
    QuestionSymbolicMode,
    QuestionUnitPolicy,
)

QuestionContext = PhysicsQuestionSemantics
infer_question_context = infer_question_semantics
infer_reference_question_context = infer_reference_question_semantics
infer_prediction_question_context = infer_prediction_question_semantics

__all__ = [
    "AnswerComparison",
    "AnswerObjectKind",
    "AnswerStructure",
    "BridgeTier",
    "ComparisonPolicyMode",
    "ContractValidationResult",
    "ContractValidationStatus",
    "OrderingPolicy",
    "PhysicalQuantitySnapshot",
    "PhysicalQuantityView",
    "PREDICTION_PROMPT_NAME",
    "PREDICTION_PROMPT_VERSION",
    "PredictionSemanticsArtifact",
    "PredictionSemanticsInferenceSpec",
    "PhysicsAnswerCaseSemantics",
    "PhysicsAnswerSemantics",
    "PhysicsEvaluationContract",
    "PhysicsQuestionSemantics",
    "PhysicsSymbolAliasSemantics",
    "build_evaluation_contract",
    "QuestionContext",
    "QuestionSymbolicMode",
    "QuestionUnitPolicy",
    "REFERENCE_PROMPT_NAME",
    "REFERENCE_PROMPT_VERSION",
    "ReferenceSemanticsArtifact",
    "SemanticsComparisonInputs",
    "SemanticsEvaluationRecord",
    "SemanticsGeneratorInfo",
    "SemanticsProblemRecord",
    "compare_physics_answers",
    "compare_protocol_answers",
    "compare_protocol_answers_legacy",
    "compare_saved_semantics",
    "coerce_evaluation_contract",
    "coerce_policy_mode",
    "coerce_protocol_answer",
    "coerce_question_semantics",
    "evaluate_saved_semantics",
    "infer_prediction_question_context",
    "infer_prediction_question_semantics",
    "infer_question_semantics",
    "infer_prediction_semantics",
    "infer_reference_question_context",
    "infer_reference_question_semantics",
    "infer_reference_semantics",
    "infer_question_context",
    "build_prediction_semantics_artifact",
    "load_prediction_semantics_artifact",
    "load_reference_semantics_artifact",
    "load_semantics_artifact",
    "normalize_physics_answer",
    "normalize_problem_answer",
    "parse_prediction_semantics_response_text",
    "parse_reference_semantics_response_text",
    "prepare_prediction_semantics_inference_spec",
    "prepare_semantics_comparison",
    "save_semantics_json",
    "validate_answer_against_contract",
]
