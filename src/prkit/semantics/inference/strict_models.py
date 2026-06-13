"""Internal provider-facing semantics schemas for native structured output.

These models intentionally omit canonical fields that are not authored by the
LLM during semantics inference, such as ``metadata``, ``provenance``, and
``quantity_view``. They are used only as the strict response contract passed to
providers that support native JSON-schema enforcement, then deterministically
mapped back into the public canonical semantics models.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..schema import (
    AnswerObjectKind,
    AnswerStructure,
    OrderingPolicy,
    PhysicsAnswerSemantics,
    PhysicsQuestionSemantics,
    PhysicsSymbolAliasSemantics,
    QuestionSymbolicMode,
    QuestionUnitPolicy,
)


class _StrictInferenceModel(BaseModel):
    """Shared strict base model for provider-facing semantics schemas."""

    model_config = ConfigDict(extra="forbid")


class StrictPhysicsQuestionSemantics(_StrictInferenceModel):
    """Provider-facing question semantics used during inference."""

    target_variable: str | None = Field(
        default=None,
        description="Primary target variable the question asks the solver to determine.",
    )
    symbol_aliases: tuple[PhysicsSymbolAliasSemantics, ...] = Field(
        default_factory=tuple,
        description="Question-conditioned symbol alias groups.",
    )
    allowed_object_kinds: tuple[AnswerObjectKind, ...] = Field(
        default_factory=lambda: tuple(AnswerObjectKind),
        description="Semantic answer kinds admitted by the question.",
    )
    allowed_structures: tuple[AnswerStructure, ...] = Field(
        default_factory=lambda: tuple(AnswerStructure),
        description="Structured answer forms admitted by the question.",
    )
    question_symbolic_mode: QuestionSymbolicMode = Field(
        default=QuestionSymbolicMode.EITHER,
        description="Whether the question expects a symbolic expression, relation, or either.",
    )
    question_unit_policy: QuestionUnitPolicy = Field(
        default=QuestionUnitPolicy.NOT_APPLICABLE,
        description="How explicitly units must appear in the final answer.",
    )
    question_unit: str | None = Field(
        default=None,
        description="Question-fixed output unit when the task specifies one.",
    )
    dimension: str | None = Field(
        default=None,
        description="Expected physical dimension metadata for the answer.",
    )
    ordering: OrderingPolicy = Field(
        default=OrderingPolicy.ORDERED,
        description="Whether multi-part answer order is semantically significant.",
    )
    required_parts: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Named answer parts required by the question.",
    )
    coordinate_frame: str | None = Field(
        default=None,
        description="Coordinate-frame convention that answers should follow.",
    )
    sign_convention: str | None = Field(
        default=None,
        description="Sign convention that answers should follow.",
    )
    tolerance: float = Field(
        default=1e-10,
        description="Default numeric comparison tolerance for semantic equivalence.",
    )
    choice_space: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Allowed multiple-choice labels when the question is choice-based.",
    )

    def to_canonical(self) -> PhysicsQuestionSemantics:
        """Return the public canonical question semantics model."""

        return PhysicsQuestionSemantics.model_validate(self.model_dump(mode="python"))


class StrictPhysicsAnswerCaseSemantics(_StrictInferenceModel):
    """Provider-facing piecewise branch for answer semantics."""

    expression: StrictPhysicsAnswerSemantics = Field(
        description="The answer expression produced when the condition holds.",
    )
    condition: StrictPhysicsAnswerSemantics = Field(
        description="The condition that activates the paired expression.",
    )


class StrictPhysicsAnswerSemantics(_StrictInferenceModel):
    """Provider-facing answer semantics used during inference."""

    canonical_text: str = Field(
        description="Canonical text representation of the normalized answer.",
    )
    object_kind: AnswerObjectKind = Field(
        description="Atomic semantic kind assigned to the normalized answer.",
    )
    structure: AnswerStructure = Field(
        default=AnswerStructure.ATOMIC,
        description="Structural form of the normalized answer.",
    )
    canonical_latex: str | None = Field(
        default=None,
        description="Canonical LaTeX surface when the answer is symbolic.",
    )
    raw_text: str | None = Field(
        default=None,
        description="Original text surface used to derive this semantics object.",
    )
    numeric_value: float | None = Field(
        default=None,
        description="Parsed numeric magnitude when available.",
    )
    numeric_text: str | None = Field(
        default=None,
        description="Canonical numeric text surface when available.",
    )
    unit: str | None = Field(
        default=None,
        description="Canonical unit string for physical-quantity answers.",
    )
    dimension: str | None = Field(
        default=None,
        description="Physical dimension metadata propagated onto the answer.",
    )
    target_variable: str | None = Field(
        default=None,
        description="Target variable bound to the answer when known.",
    )
    part_label: str | None = Field(
        default=None,
        description="Named slot label for a child of a multi-part answer.",
    )
    shape: tuple[int, ...] = Field(
        default_factory=tuple,
        description="Tensor-like shape metadata for structured numeric answers.",
    )
    interval_open_left: bool | None = Field(
        default=None,
        description="Whether an interval answer is open on the left endpoint.",
    )
    interval_open_right: bool | None = Field(
        default=None,
        description="Whether an interval answer is open on the right endpoint.",
    )
    children: tuple[StrictPhysicsAnswerSemantics, ...] = Field(
        default_factory=tuple,
        description="Child answer semantics for structured answers such as tuples or vectors.",
    )
    cases: tuple[StrictPhysicsAnswerCaseSemantics, ...] = Field(
        default_factory=tuple,
        description="Piecewise branches pairing expressions with their activating conditions.",
    )
    subject_to: tuple[StrictPhysicsAnswerSemantics, ...] = Field(
        default_factory=tuple,
        description="Global side conditions or constraints attached conjunctively to this answer.",
    )
    choice_label: str | None = Field(
        default=None,
        description="Canonical label for multiple-choice answers.",
    )
    boolean_value: bool | None = Field(
        default=None,
        description="Boolean interpretation when the answer is truth-valued.",
    )
    sign_value: str | None = Field(
        default=None,
        description="Canonical sign or direction label when applicable.",
    )
    coordinate_frame: str | None = Field(
        default=None,
        description="Coordinate-frame metadata attached to the answer.",
    )
    sign_convention: str | None = Field(
        default=None,
        description="Sign-convention metadata attached to the answer.",
    )
    diagnostics: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Non-fatal normalization diagnostics attached to the answer.",
    )

    def to_canonical(self) -> PhysicsAnswerSemantics:
        """Return the public canonical answer semantics model."""

        return PhysicsAnswerSemantics.model_validate(self.model_dump(mode="python"))


class StrictReferenceSemanticsResponse(_StrictInferenceModel):
    """Provider-facing response model for reference-semantics generation."""

    question_semantics: StrictPhysicsQuestionSemantics = Field(
        description="Question-side semantics that define acceptable reference answers.",
    )
    reference_answer_semantics: StrictPhysicsAnswerSemantics = Field(
        description="Canonical reference answer semantics for the ground-truth final answer.",
    )


class StrictPredictionSemanticsResponse(_StrictInferenceModel):
    """Provider-facing response model for prediction-semantics generation."""

    reasoning: str = Field(
        description="Concise reasoning summary used to produce the final answer.",
    )
    final_answer: str = Field(
        description="Final answer surface form only.",
    )
    question_semantics: StrictPhysicsQuestionSemantics = Field(
        description="Question-side semantics inferred while solving the problem.",
    )
    prediction_answer_semantics: StrictPhysicsAnswerSemantics = Field(
        description="Canonical semantics for the predicted final answer.",
    )


class StrictPredictionFinalAnswerResponse(_StrictInferenceModel):
    """Compact provider-facing response for models with strict schema limits."""

    reasoning: str = Field(
        description="Concise reasoning summary used to produce the final answer.",
    )
    final_answer: str = Field(
        description="Final answer surface form only.",
    )


StrictPhysicsAnswerCaseSemantics.model_rebuild()
StrictPhysicsAnswerSemantics.model_rebuild()
StrictReferenceSemanticsResponse.model_rebuild()
StrictPredictionSemanticsResponse.model_rebuild()
StrictPredictionFinalAnswerResponse.model_rebuild()


__all__ = [
    "StrictPhysicsAnswerCaseSemantics",
    "StrictPhysicsAnswerSemantics",
    "StrictPredictionFinalAnswerResponse",
    "StrictPhysicsQuestionSemantics",
    "StrictPredictionSemanticsResponse",
    "StrictReferenceSemanticsResponse",
]
