"""Artifact models and JSON helpers for semantics inference workflows."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..schema import (
    AnswerComparison,
    ComparisonPolicyMode,
    PhysicsAnswerSemantics,
    PhysicsEvaluationContract,
    PhysicsQuestionSemantics,
    SymbolAssumption,
)


def _utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp string for saved artifacts."""

    return datetime.now(UTC).isoformat()


class _InferenceModel(BaseModel):
    """Shared strict base model for semantics inference records."""

    model_config = ConfigDict(extra="forbid")


class SemanticsProblemRecord(_InferenceModel):
    """Serializable problem snapshot used across semantics artifacts."""

    problem_id: str = Field(description="Problem identifier.")
    question: str = Field(description="Problem question text.")
    problem_type: str | None = Field(
        default=None,
        description="Problem type such as MC or OE.",
    )
    domain: str | None = Field(
        default=None,
        description="Physics domain label when available.",
    )
    language: str | None = Field(
        default=None,
        description="Problem language code when available.",
    )
    options: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Multiple-choice options when present.",
    )
    correct_option: int | None = Field(
        default=None,
        description="Zero-based correct-option index when available.",
    )
    image_paths: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Attached image paths that were available during inference.",
    )
    answer: str | dict[str, Any] | None = Field(
        default=None,
        description=(
            "Dataset ground-truth answer when embedded, typically as plain source text "
            "and optionally as a legacy structured dict in older exports."
        ),
    )


class SemanticsGeneratorInfo(_InferenceModel):
    """Metadata describing how an artifact was generated."""

    provider: str | None = Field(
        default=None,
        description="Model-provider identifier such as openai or google.",
    )
    model_name: str | None = Field(
        default=None,
        description="Concrete model name used for generation.",
    )
    prompt_name: str = Field(description="Prompt family name used for the call.")
    prompt_version: str = Field(description="Prompt version string.")
    normalization_version: str | None = Field(
        default=None,
        description=(
            "Revision of the deterministic surface canonicalization that produced the "
            "stored canonical_text. Optional so artifacts written before the stamp existed "
            "still load; None means 'unknown, predates the stamp'."
        ),
    )
    structured_output_mode: str = Field(
        description="json_schema, json_object, or prompt_only.",
    )
    structured_output_strategy: str | None = Field(
        default=None,
        description="Provider-specific structured output strategy identifier.",
    )


class SymbolAssumptionProvenance(_InferenceModel):
    """Provenance for one synthesized symbol assumption in a build report."""

    symbol: str = Field(description="Canonical (post-alias) symbol token.")
    assumption: SymbolAssumption = Field(
        description="The real-domain assumption adopted for the symbol.",
    )
    source: str = Field(
        description="Where the assumption came from: subject_to, llm_declared, or merged.",
    )
    justification: str | None = Field(
        default=None,
        description="LLM-stated justification for a declared assumption, when available.",
    )


class SemanticsBuildReport(_InferenceModel):
    """Objective provenance + confidence record for one semantics build.

    Attached additively to reference/problem artifacts so a build is auditable and
    reproducible: which fields are deterministic vs LLM-advisory, why each symbol
    assumption was adopted, what disagreements/cross-check reverts occurred, and whether a
    human should review the result.
    """

    build_method: str = Field(
        description="Identifier for the build pipeline, e.g. reference_3call or problem_3call.",
    )
    temperature: float = Field(
        default=0.0,
        description="LLM sampling temperature used for advisory calls (0 for reproducibility).",
    )
    field_provenance: dict[str, str] = Field(
        default_factory=dict,
        description="Per-field source: deterministic, subject_to, llm_declared, or default.",
        json_schema_extra={"additionalProperties": False},
    )
    assumption_provenance: tuple[SymbolAssumptionProvenance, ...] = Field(
        default_factory=tuple,
        description="Provenance for each adopted symbol assumption.",
    )
    flags: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Disagreement, strengthening, and cross-check-revert flags for review.",
    )
    cross_checks_passed: bool = Field(
        default=True,
        description="Whether round-trip, contract self-consistency, and pair-consistency held.",
    )
    review_required: bool = Field(
        default=False,
        description="Whether the build surfaced a low-confidence signal warranting review.",
    )


class ReferenceSemanticsResponse(_InferenceModel):
    """Structured model output for reference-semantics generation."""

    question_semantics: PhysicsQuestionSemantics = Field(
        description="Question-side semantics that define acceptable reference answers.",
    )
    reference_answer_semantics: PhysicsAnswerSemantics = Field(
        description="Canonical reference answer semantics for the ground-truth final answer.",
    )


class PredictionSemanticsResponse(_InferenceModel):
    """Structured model output for prediction-semantics generation."""

    reasoning: str = Field(
        description="Concise reasoning summary used to produce the final answer.",
    )
    final_answer: str = Field(
        description="Final answer surface form only.",
    )
    question_semantics: PhysicsQuestionSemantics = Field(
        description="Question-side semantics inferred while solving the problem.",
    )
    prediction_answer_semantics: PhysicsAnswerSemantics = Field(
        description="Canonical semantics for the predicted final answer.",
    )


class ReferenceSemanticsArtifact(_InferenceModel):
    """Saved JSON artifact for a problem's reference semantics."""

    artifact_type: str = Field(
        default="reference_semantics",
        description="Artifact discriminator.",
    )
    created_at: str = Field(
        default_factory=_utc_now_iso,
        description="UTC timestamp for artifact creation.",
    )
    problem: SemanticsProblemRecord = Field(
        description="Problem snapshot used for the reference semantics call.",
    )
    ground_truth_answer: str = Field(
        description="Ground-truth answer surface supplied to the model.",
    )
    question_semantics: PhysicsQuestionSemantics = Field(
        description="Reference-conditioned question semantics (q_ref), built from problem + golden.",
    )
    reference_answer_semantics: PhysicsAnswerSemantics = Field(
        description="Reference answer semantics (a_ref) for the ground-truth final answer.",
    )
    generator: SemanticsGeneratorInfo = Field(
        description="Generation metadata.",
    )
    build_report: SemanticsBuildReport | None = Field(
        default=None,
        description="Objective build provenance/confidence record when built by the staged pipeline.",
    )


class ProblemSemanticsArtifact(_InferenceModel):
    """Saved JSON artifact for a problem-only question semantics (q_prob).

    Built from the problem text alone (answer-blind), this is the contract for the
    reference-free clustering judgement ``Eq(a_pred_i, a_pred_j; q_prob)``. It carries no
    answer semantics: there is no golden answer to realize a structure/kind, so the contract
    only declares admissible answer forms and policy.
    """

    artifact_type: str = Field(
        default="problem_semantics",
        description="Artifact discriminator.",
    )
    created_at: str = Field(
        default_factory=_utc_now_iso,
        description="UTC timestamp for artifact creation.",
    )
    problem: SemanticsProblemRecord = Field(
        description="Problem snapshot used for the problem-only semantics call.",
    )
    question_semantics: PhysicsQuestionSemantics = Field(
        description="Problem-only question semantics (q_prob), built answer-blind.",
    )
    generator: SemanticsGeneratorInfo = Field(
        description="Generation metadata.",
    )
    build_report: SemanticsBuildReport | None = Field(
        default=None,
        description="Objective build provenance/confidence record when built by the staged pipeline.",
    )


class PredictionSemanticsArtifact(_InferenceModel):
    """Saved JSON artifact for a problem's predicted answer semantics."""

    artifact_type: str = Field(
        default="prediction_semantics",
        description="Artifact discriminator.",
    )
    created_at: str = Field(
        default_factory=_utc_now_iso,
        description="UTC timestamp for artifact creation.",
    )
    problem: SemanticsProblemRecord = Field(
        description="Problem snapshot used for the prediction call.",
    )
    reasoning: str = Field(
        description="Reasoning summary returned by the model.",
    )
    final_answer: str = Field(
        description="Predicted final answer surface.",
    )
    question_semantics: PhysicsQuestionSemantics = Field(
        description="Question semantics returned by the model.",
    )
    prediction_answer_semantics: PhysicsAnswerSemantics = Field(
        description="Prediction answer semantics returned by the model.",
    )
    generator: SemanticsGeneratorInfo = Field(
        description="Generation metadata.",
    )
    build_report: SemanticsBuildReport | None = Field(
        default=None,
        description=(
            "Objective build provenance/confidence record. For isolated solves it carries "
            "the a_pred_llm-vs-a_pred_ext structure disagreement audit; for the extracted "
            "path it records deterministic provenance."
        ),
    )


class SemanticsComparisonInputs(_InferenceModel):
    """Comparison-ready semantics bundle loaded from saved artifacts."""

    problem: SemanticsProblemRecord = Field(
        description="Problem snapshot shared by the compared artifacts.",
    )
    question_semantics: PhysicsQuestionSemantics = Field(
        description="Question semantics used as the comparison context.",
    )
    evaluation_contract: PhysicsEvaluationContract = Field(
        description="Reference-assisted contract used during comparison.",
    )
    reference_answer_semantics: PhysicsAnswerSemantics = Field(
        description="Reference answer semantics loaded from the reference artifact.",
    )
    prediction_answer_semantics: PhysicsAnswerSemantics = Field(
        description="Prediction answer semantics loaded from the prediction artifact.",
    )


class SemanticsEvaluationRecord(_InferenceModel):
    """Saved evaluation result built from two semantics artifacts."""

    created_at: str = Field(
        default_factory=_utc_now_iso,
        description="UTC timestamp for evaluation record creation.",
    )
    problem: SemanticsProblemRecord = Field(
        description="Problem snapshot shared by the compared artifacts.",
    )
    question_semantics: PhysicsQuestionSemantics = Field(
        description="Question semantics used during comparison.",
    )
    evaluation_contract: PhysicsEvaluationContract = Field(
        description="Reference-assisted contract used during comparison.",
    )
    policy_mode: ComparisonPolicyMode = Field(
        description="Comparison policy mode used for this evaluation record.",
    )
    reference_answer_semantics: PhysicsAnswerSemantics = Field(
        description="Reference answer semantics used during comparison.",
    )
    prediction_answer_semantics: PhysicsAnswerSemantics = Field(
        description="Prediction answer semantics used during comparison.",
    )
    comparison: AnswerComparison = Field(
        description="Comparison result.",
    )


SemanticsArtifact = (
    ReferenceSemanticsArtifact | PredictionSemanticsArtifact | ProblemSemanticsArtifact
)


def save_semantics_json(record: BaseModel, path: str | Path) -> Path:
    """Persist a Pydantic semantics record as indented JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return target


def load_reference_semantics_artifact(path: str | Path) -> ReferenceSemanticsArtifact:
    """Load a reference-semantics artifact from JSON."""

    return ReferenceSemanticsArtifact.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def load_prediction_semantics_artifact(path: str | Path) -> PredictionSemanticsArtifact:
    """Load a prediction-semantics artifact from JSON."""

    return PredictionSemanticsArtifact.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def load_problem_semantics_artifact(path: str | Path) -> ProblemSemanticsArtifact:
    """Load a problem-only (q_prob) semantics artifact from JSON."""

    return ProblemSemanticsArtifact.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def load_semantics_artifact(path: str | Path) -> SemanticsArtifact:
    """Load a reference, prediction, or problem semantics artifact from JSON."""

    raw_text = Path(path).read_text(encoding="utf-8")
    payload = json.loads(raw_text)
    artifact_type = payload.get("artifact_type")

    if artifact_type == "reference_semantics":
        return ReferenceSemanticsArtifact.model_validate(payload)
    if artifact_type == "prediction_semantics":
        return PredictionSemanticsArtifact.model_validate(payload)
    if artifact_type == "problem_semantics":
        return ProblemSemanticsArtifact.model_validate(payload)

    raise ValueError(
        "Semantics artifact JSON must define artifact_type as "
        "'reference_semantics', 'prediction_semantics', or 'problem_semantics'."
    )


def load_semantics_evaluation_record(path: str | Path) -> SemanticsEvaluationRecord:
    """Load an evaluation record from JSON."""

    return SemanticsEvaluationRecord.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


__all__ = [
    "PredictionSemanticsArtifact",
    "PredictionSemanticsResponse",
    "ProblemSemanticsArtifact",
    "ReferenceSemanticsArtifact",
    "ReferenceSemanticsResponse",
    "SemanticsArtifact",
    "SemanticsBuildReport",
    "SemanticsComparisonInputs",
    "SemanticsEvaluationRecord",
    "SemanticsGeneratorInfo",
    "SemanticsProblemRecord",
    "SymbolAssumptionProvenance",
    "load_prediction_semantics_artifact",
    "load_problem_semantics_artifact",
    "load_reference_semantics_artifact",
    "load_semantics_artifact",
    "load_semantics_evaluation_record",
    "save_semantics_json",
]
