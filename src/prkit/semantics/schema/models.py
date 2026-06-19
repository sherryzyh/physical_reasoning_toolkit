"""Pydantic models for physics semantics protocols."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .enums import (
    AnswerObjectKind,
    AnswerStructure,
    BridgeTier,
    ComparisonPolicyMode,
    ContractValidationStatus,
    OrderingPolicy,
    QuestionSymbolicMode,
    QuestionUnitPolicy,
    SymbolAssumption,
)

DEFAULT_NUMERIC_TOLERANCE = 1e-10


class _SemanticsModel(BaseModel):
    """Shared strict base model for public semantics schemas."""

    model_config = ConfigDict(extra="forbid")


class PhysicsQuestionSemantics(_SemanticsModel):
    """Question-conditioned semantics that constrain answer interpretation."""

    target_variable: str | None = Field(
        default=None,
        description="Primary target variable the question asks the solver to determine.",
    )
    symbol_aliases: tuple[PhysicsSymbolAliasSemantics, ...] = Field(
        default_factory=tuple,
        description="Question-conditioned alias groups that map alternate symbol names onto a canonical symbol for comparison.",
    )
    symbol_assumptions: tuple[PhysicsSymbolAssumptionSemantics, ...] = Field(
        default_factory=tuple,
        description="Question-conditioned real-domain declarations for free symbols, used to decide symbolic equivalence over the intended physical domain.",
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
        default=DEFAULT_NUMERIC_TOLERANCE,
        description="Default numeric comparison tolerance for semantic equivalence.",
    )
    choice_space: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Allowed multiple-choice labels when the question is choice-based.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional question-side semantic metadata.",
        json_schema_extra={"additionalProperties": False},
    )

    def merged(
        self, overrides: dict[str, Any] | None = None
    ) -> PhysicsQuestionSemantics:
        """Return a copy updated with ``overrides``."""
        if not overrides:
            return self
        return self.model_copy(update=overrides)


class PhysicsEvaluationContract(_SemanticsModel):
    """Reference-assisted contract used during evaluation."""

    contract_version: str = Field(
        default="pasec-base-v1",
        description="Version tag for the evaluation contract schema and policy.",
    )
    question_semantics: PhysicsQuestionSemantics = Field(
        description="Question-only semantics used as the contract base.",
    )
    question_text: str | None = Field(
        default=None,
        description="Question text copied into the contract for guarded bridge heuristics.",
    )
    problem_type: str | None = Field(
        default=None,
        description="Problem type copied into the contract when available.",
    )
    expected_object_kind: AnswerObjectKind = Field(
        description="Expected top-level atomic object kind derived from the reference answer.",
    )
    expected_structure: AnswerStructure = Field(
        description="Expected top-level answer structure derived from the reference answer.",
    )
    target_variable: str | None = Field(
        default=None,
        description="Target variable used during evaluation, usually from the reference answer when available.",
    )
    enabled_bridge_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Named bridges that the contract makes available to the evaluator.",
    )
    enabled_bridge_tiers: tuple[BridgeTier, ...] = Field(
        default_factory=lambda: (BridgeTier.TIER1, BridgeTier.TIER2),
        description="Bridge tiers enabled by default for this contract.",
    )

    @property
    def comparison_context(self) -> PhysicsQuestionSemantics:
        """Return the question semantics augmented with contract-time target metadata."""

        if (
            self.target_variable
            and self.question_semantics.target_variable != self.target_variable
        ):
            return self.question_semantics.merged(
                {"target_variable": self.target_variable}
            )
        return self.question_semantics


class ContractValidationResult(_SemanticsModel):
    """Validation result for one answer under a physics evaluation contract."""

    status: ContractValidationStatus = Field(
        description="Whether the answer is admitted, coercible, or violating under the contract.",
    )
    diagnostics: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Diagnostic notes explaining the validation outcome.",
    )
    bridge_candidates: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Bridge ids that could potentially rescue a non-admitted answer.",
    )


class PhysicsAnswerCaseSemantics(_SemanticsModel):
    """One branch of a piecewise answer semantics object."""

    expression: PhysicsAnswerSemantics = Field(
        description="The answer expression produced when the condition holds.",
    )
    condition: PhysicsAnswerSemantics = Field(
        description="The condition that activates the paired expression.",
    )

    def __iter__(self) -> Any:
        """Expose backward-compatible tuple-style iteration."""

        yield self.expression
        yield self.condition

    def __getitem__(self, index: int) -> PhysicsAnswerSemantics:
        """Expose backward-compatible tuple-style indexing."""

        return (self.expression, self.condition)[index]

    def __len__(self) -> int:
        """Expose backward-compatible tuple-like length."""

        return 2


class PhysicalQuantitySnapshot(_SemanticsModel):
    """Deterministic numeric/unit snapshot for one parsed quantity surface."""

    numeric_value: float = Field(
        description="Parsed numeric magnitude associated with the snapshot unit.",
    )
    numeric_text: str = Field(
        description="Deterministic numeric text associated with the snapshot value.",
    )
    unit: str = Field(
        description="Unit string associated with the snapshot value.",
    )
    canonical_text: str = Field(
        description="Stable ``numeric_text unit`` rendering for the snapshot.",
    )


class PhysicalQuantityView(_SemanticsModel):
    """Source and canonical deterministic views for a physical quantity answer."""

    source: PhysicalQuantitySnapshot | None = Field(
        default=None,
        description="Parsed source quantity snapshot before canonical unit remapping.",
    )
    canonical: PhysicalQuantitySnapshot | None = Field(
        default=None,
        description="Canonical normalized quantity snapshot used for comparison.",
    )
    diagnostics: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Non-fatal deterministic normalization diagnostics for the quantity view.",
    )


class PhysicsSymbolAliasSemantics(_SemanticsModel):
    """One question-conditioned symbol alias group."""

    canonical_symbol: str = Field(
        description="Canonical symbol name that equivalent aliases should map to during symbolic comparison.",
    )
    aliases: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Alternative symbol names that should be treated as equivalent to the canonical symbol in this question.",
    )


class PhysicsSymbolAssumptionSemantics(_SemanticsModel):
    """One question-conditioned real-domain assumption for a symbol.

    The ``symbol`` is the canonical (post-alias) token; ``assumption`` constrains the
    values it ranges over (real / nonzero / nonnegative / positive / complex) so symbolic
    comparison is decided over the intended real-physical domain rather than the generic
    complex default. See ``SymbolAssumption``.
    """

    symbol: str = Field(
        description="Canonical (post-alias) symbol token the assumption applies to.",
    )
    assumption: SymbolAssumption = Field(
        description="Real-domain assumption the symbol ranges over during symbolic comparison.",
    )


class PhysicsAnswerSemantics(_SemanticsModel):
    """Normalized physics-aware final answer semantics."""

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
        description="Named slot label for a child of a multi-part answer when the question defines distinct parts.",
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
    children: tuple[PhysicsAnswerSemantics, ...] = Field(
        default_factory=tuple,
        description="Child answer semantics for structured answers such as tuples or vectors.",
    )
    cases: tuple[PhysicsAnswerCaseSemantics, ...] = Field(
        default_factory=tuple,
        description="Piecewise branches pairing expressions with their activating conditions.",
    )
    subject_to: tuple[PhysicsAnswerSemantics, ...] = Field(
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
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Trace metadata describing where the answer semantics came from.",
        json_schema_extra={"additionalProperties": False},
    )
    diagnostics: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Non-fatal normalization or comparison diagnostics attached to the answer.",
    )
    quantity_view: PhysicalQuantityView | None = Field(
        default=None,
        description="Deterministic parsed and canonical views for physical quantities.",
    )

    def with_diagnostic(self, *messages: str) -> PhysicsAnswerSemantics:
        """Return a copy extended with diagnostic messages."""
        return self.model_copy(
            update={"diagnostics": self.diagnostics + tuple(messages)}
        )

    @property
    def is_atomic(self) -> bool:
        """Whether the answer has no higher-order structure."""
        return self.structure == AnswerStructure.ATOMIC


class AnswerComparison(_SemanticsModel):
    """Comparison result for two normalized answers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    equivalent: bool = Field(
        description="Whether the compared answers are semantically equivalent.",
    )
    comparison_mode: str = Field(
        description="Comparison path or matcher that produced the result.",
    )
    diagnostics: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Diagnostic notes explaining mismatches or fallback paths.",
    )
    validation_status: ContractValidationStatus | None = Field(
        default=None,
        description="Contract validation status that governed the final comparison path.",
    )
    bridge_id: str | None = Field(
        default=None,
        description="Named bridge that accepted the comparison when one was used.",
    )
    bridge_tier: BridgeTier | None = Field(
        default=None,
        description="Risk tier of the bridge that accepted the comparison when one was used.",
    )
    bridge_evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured evidence explaining why a bridge was allowed.",
        json_schema_extra={"additionalProperties": False},
    )
    policy_mode: ComparisonPolicyMode | None = Field(
        default=None,
        description="Comparison policy mode in force for the verdict.",
    )
    surface_shortcut_used: bool = Field(
        default=False,
        description="Whether a same-kind surface equality shortcut decided the verdict.",
    )

    def __init__(
        self,
        equivalent: bool,
        comparison_mode: str,
        diagnostics: tuple[str, ...] = (),
        validation_status: ContractValidationStatus | None = None,
        bridge_id: str | None = None,
        bridge_tier: BridgeTier | None = None,
        bridge_evidence: dict[str, Any] | None = None,
        policy_mode: ComparisonPolicyMode | None = None,
        surface_shortcut_used: bool = False,
    ) -> None:
        data: dict[str, Any] = {
            "equivalent": equivalent,
            "comparison_mode": comparison_mode,
            "diagnostics": diagnostics,
            "validation_status": validation_status,
            "bridge_id": bridge_id,
            "bridge_tier": bridge_tier,
            "bridge_evidence": bridge_evidence or {},
            "policy_mode": policy_mode,
            "surface_shortcut_used": surface_shortcut_used,
        }
        super().__init__(**data)


PhysicsAnswerCaseSemantics.model_rebuild()
PhysicalQuantityView.model_rebuild()
PhysicsAnswerSemantics.model_rebuild()
PhysicsQuestionSemantics.model_rebuild()
PhysicsEvaluationContract.model_rebuild()
