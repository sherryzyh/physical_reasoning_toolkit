"""Inference calls for reference semantics, prediction semantics, and saved comparison."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ValidationError

from prkit.core.domain import PhysicsProblem
from prkit.core.model_clients import BaseModelClient
from prkit.core.model_clients.structured_output import (
    StructuredCallResult,
    StructuredOutputPolicy,
    build_json_schema_prompt_suffix,
    extract_json_payload,
    normalize_response_format,
)
from prkit.core.model_clients.structured_output import (
    extract_json_object as extract_structured_json_object,
)

from ..comparison import (
    build_evaluation_contract,
    compare_protocol_answers,
)
from ..comparison.contract import coerce_policy_mode, validate_answer_against_contract
from ..comparison.structure_canonicalization import canonicalize_structure
from ..normalization import (
    enrich_answer_quantity_views,
    infer_prediction_question_semantics,
    infer_reference_question_semantics,
    normalize_physics_answer,
)
from ..schema import (
    AnswerComparison,
    AnswerObjectKind,
    AnswerStructure,
    ComparisonPolicyMode,
    ContractValidationStatus,
    PhysicsAnswerSemantics,
    PhysicsQuestionSemantics,
    SymbolAssumption,
)
from .artifacts import (
    PredictionSemanticsArtifact,
    ProblemSemanticsArtifact,
    ReferenceSemanticsArtifact,
    SemanticsBuildReport,
    SemanticsComparisonInputs,
    SemanticsEvaluationRecord,
    SemanticsGeneratorInfo,
    SemanticsProblemRecord,
    SymbolAssumptionProvenance,
    load_prediction_semantics_artifact,
    load_reference_semantics_artifact,
)
from .prompts import (
    PREDICTION_PROMPT_NAME,
    PREDICTION_PROMPT_VERSION,
    PROBLEM_PROMPT_NAME,
    PROBLEM_PROMPT_VERSION,
    REFERENCE_PROMPT_NAME,
    REFERENCE_PROMPT_VERSION,
    answer_like_to_text,
    build_answer_surface_cleanup_prompt,
    build_prediction_semantics_prompt,
    build_question_policy_prompt,
    build_symbol_assumptions_prompt,
)
from .semantics_build import (
    alias_source_violations,
    assumptions_from_subject_to,
    assumptions_to_semantics,
    build_alias_map,
    extract_candidate_symbols,
    infer_answer_tolerance,
    meet_assumptions,
    merge_symbol_assumptions,
    reconcile_allowed_sets,
    reference_pair_consistency,
    resolve_to_canonical,
)
from .strict_models import (
    StrictPhysicsAnswerCaseSemantics,
    StrictPhysicsAnswerSemantics,
    StrictPhysicsQuestionSemantics,
    StrictPredictionFinalAnswerResponse,
    StrictPredictionIsolatedResponse,
    StrictPredictionSemanticsResponse,
    StrictReferenceSemanticsResponse,
    StrictSymbolAssumptionsResponse,
)

logger = logging.getLogger(__name__)
_STRICT_PREDICTION_RESPONSE_FIELDS = frozenset(
    StrictPredictionSemanticsResponse.model_fields
)
_STRICT_QUESTION_FIELDS = frozenset(StrictPhysicsQuestionSemantics.model_fields)
_STRICT_ANSWER_FIELDS = frozenset(StrictPhysicsAnswerSemantics.model_fields)
_STRICT_CASE_FIELDS = frozenset(StrictPhysicsAnswerCaseSemantics.model_fields)
_VALID_ALLOWED_OBJECT_KINDS = frozenset(kind.value for kind in AnswerObjectKind)
_VALID_ALLOWED_STRUCTURES = frozenset(kind.value for kind in AnswerStructure)
ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


@dataclass(frozen=True)
class PredictionSemanticsInferenceSpec:
    """Reusable prompt/schema bundle for prediction-semantics inference."""

    prompt: str
    image_paths: tuple[str, ...]
    draft_question_semantics: PhysicsQuestionSemantics
    response_model: type[BaseModel]
    response_format: dict[str, Any]


def _semantics_should_require_native_json_schema(
    model_client: BaseModelClient,
    response_model: type[BaseModel],
) -> bool:
    """Return whether semantics inference should require native schema enforcement."""

    plan = model_client.resolve_structured_output_plan(
        response_model,
        structured_policy="best_effort",
    )
    if plan.native_schema_enforced:
        return True
    if getattr(model_client, "provider", None) == "anthropic":
        logger.warning(
            "Model %s (%s) cannot use Anthropic native structured output for this semantics schema; "
            "falling back to %s parsing.",
            getattr(model_client, "model", "unknown"),
            getattr(model_client, "provider", "unknown"),
            plan.mode,
        )
        return False
    ensure_semantics_native_structured_output_support(model_client, response_model)
    return True


def resolve_prediction_response_model(
    model_client: BaseModelClient,
) -> type[BaseModel]:
    """Pick a provider-facing prediction response model that the provider can enforce."""

    full_plan = model_client.resolve_structured_output_plan(
        StrictPredictionSemanticsResponse,
        structured_policy="best_effort",
    )
    if full_plan.native_schema_enforced:
        return StrictPredictionSemanticsResponse

    compact_plan = model_client.resolve_structured_output_plan(
        StrictPredictionFinalAnswerResponse,
        structured_policy="best_effort",
    )
    if compact_plan.native_schema_enforced:
        logger.warning(
            "Model %s (%s) cannot natively enforce the full prediction semantics schema; "
            "using compact final-answer response schema with deterministic semantics reconstruction.",
            getattr(model_client, "model", "unknown"),
            getattr(model_client, "provider", "unknown"),
        )
        return StrictPredictionFinalAnswerResponse

    return StrictPredictionSemanticsResponse


def resolve_isolated_prediction_response_model(
    model_client: BaseModelClient,
) -> type[BaseModel]:
    """Pick the provider-facing model for the ISOLATED problem-only solve (a_pred_llm).

    Prefers :class:`StrictPredictionIsolatedResponse` (reasoning + final_answer +
    prediction_answer_semantics, no question_semantics). Providers that cannot enforce it
    fall back to the compact :class:`StrictPredictionFinalAnswerResponse`, in which case only
    ``a_pred_ext`` is available (the deterministic reconstruction).
    """

    isolated_plan = model_client.resolve_structured_output_plan(
        StrictPredictionIsolatedResponse,
        structured_policy="best_effort",
    )
    if isolated_plan.native_schema_enforced:
        return StrictPredictionIsolatedResponse

    compact_plan = model_client.resolve_structured_output_plan(
        StrictPredictionFinalAnswerResponse,
        structured_policy="best_effort",
    )
    if compact_plan.native_schema_enforced:
        logger.warning(
            "Model %s (%s) cannot natively enforce the isolated prediction semantics schema; "
            "using compact final-answer response schema (a_pred_ext only).",
            getattr(model_client, "model", "unknown"),
            getattr(model_client, "provider", "unknown"),
        )
        return StrictPredictionFinalAnswerResponse

    return StrictPredictionIsolatedResponse


# ----------------------------------------------------------------------------------------
# DEPRECATED single-call reference build (kept commented for manual review/comparison
# against the staged `build_reference_semantics` that replaced it). The old path made ONE
# fused LLM call returning both question and answer semantics with no determinism control,
# no structure/kind pinning, and no cross-checks. See `build_reference_semantics` below.
# ----------------------------------------------------------------------------------------
# def infer_reference_semantics(
#     problem: PhysicsProblem,
#     model_client: BaseModelClient,
#     *,
#     max_output_tokens: int | None = None,
#     **chat_kwargs: Any,
# ) -> ReferenceSemanticsArtifact:
#     """Infer and package reference semantics for a problem's ground-truth answer."""
#
#     if problem.answer is None:
#         raise ValueError(
#             f"Problem {problem.problem_id} does not provide `problem.answer`."
#         )
#
#     require_native_json_schema = _semantics_should_require_native_json_schema(
#         model_client,
#         StrictReferenceSemanticsResponse,
#     )
#     ground_truth_answer_text = answer_like_to_text(problem.answer)
#     draft_question_semantics = infer_reference_question_semantics(problem)
#     prompt = build_reference_semantics_prompt(
#         problem,
#         draft_question_semantics=draft_question_semantics,
#     )
#     response, structured_result = _run_structured_inference(
#         model_client,
#         prompt=prompt,
#         response_model=StrictReferenceSemanticsResponse,
#         image_paths=tuple(problem.image_path or ()),
#         max_output_tokens=max_output_tokens,
#         require_native_json_schema=require_native_json_schema,
#         **chat_kwargs,
#     )
#
#     merged_question_semantics = _merge_question_semantics_fallbacks(
#         response.question_semantics.to_canonical(),
#         draft_question_semantics,
#     )
#     return ReferenceSemanticsArtifact(
#         problem=_problem_record_from_problem(problem),
#         ground_truth_answer=ground_truth_answer_text,
#         question_semantics=merged_question_semantics,
#         reference_answer_semantics=enrich_answer_quantity_views(
#             response.reference_answer_semantics.to_canonical(),
#             context=merged_question_semantics,
#         ),
#         generator=_generator_info(
#             model_client,
#             prompt_name=REFERENCE_PROMPT_NAME,
#             prompt_version=REFERENCE_PROMPT_VERSION,
#             structured_output_mode=structured_result.structured_output_mode,
#             structured_output_strategy=structured_result.structured_output_strategy,
#         ),
#     )


def create_reference_semantics(
    problem: PhysicsProblem,
    model_client: BaseModelClient | None = None,
    *,
    max_output_tokens: int | None = None,
    **chat_kwargs: Any,
) -> ReferenceSemanticsArtifact:
    """Create reference semantics (q_ref + a_ref) for a problem's golden answer.

    The public entry to the reference-build step; delegates to the staged
    :func:`build_reference_semantics`. **Deterministic** when ``model_client is None``
    (the three advisory LLM calls are skipped and the build records the same
    ``*_call_unavailable`` flags as a degraded call, so ``review_required`` is set);
    **LLM-assisted** otherwise.

    ``model_client`` is positional-or-keyword (default ``None``) so the deprecated
    ``infer_reference_semantics`` alias keeps working for callers that pass it
    positionally.
    """

    return build_reference_semantics(
        problem,
        model_client,
        max_output_tokens=max_output_tokens,
        **chat_kwargs,
    )


#: Deprecated alias for :func:`create_reference_semantics` (renamed in the build/API
#: restructure). Retained for one release; prefer ``create_reference_semantics``.
infer_reference_semantics = create_reference_semantics


# ----------------------------------------------------------------------------------------
# Staged objective semantics build (WS A). Deterministic backbone is authoritative for
# structure/object_kind/tolerance/allowed_*; three advisory LLM calls (temperature 0) clean
# the answer surface, fill question policy, and declare justified symbol assumptions. See
# `semantics_build.py` for the deterministic methodology and the build's compatibility rules.
# ----------------------------------------------------------------------------------------
_ANSWER_SURFACE_FILL_FIELDS = (
    "canonical_latex",
    "unit",
    "dimension",
    "choice_label",
    "boolean_value",
    "sign_value",
    "coordinate_frame",
    "sign_convention",
)


def _advisory_inference(
    model_client: BaseModelClient | None,
    *,
    prompt: str,
    response_model: type[ResponseModelT],
    image_paths: tuple[str, ...],
    max_output_tokens: int | None,
    **chat_kwargs: Any,
) -> ResponseModelT | None:
    """Run one advisory build call, returning ``None`` on any failure.

    Advisory calls only refine fields the deterministic backbone has already decided, so a
    failed call degrades to the deterministic value rather than crashing the build. When
    ``model_client is None`` (the deterministic build mode) the call is skipped entirely and
    ``None`` is returned, so the same "unavailable" degradation path runs with no LLM call.

    These calls run **best-effort** (``require_native_json_schema=False``): native structured
    output is used when the provider supports it, otherwise the LLM still enriches via the
    plain-text / ``json_object`` route and the result is parsed back. So lacking native
    structured output does **not** reduce the reference build (Step 1) to deterministic-only —
    native structured output is a Step-2 (answer-generation output-form) concern, not a Step-1
    one. Only a genuine inference/parse failure degrades to the deterministic value.
    """

    if model_client is None:
        return None

    try:
        response, _ = _run_structured_inference(
            model_client,
            prompt=prompt,
            response_model=response_model,
            image_paths=image_paths,
            max_output_tokens=max_output_tokens,
            require_native_json_schema=False,
            **chat_kwargs,
        )
        return response
    except Exception as exc:  # noqa: BLE001 - advisory; the deterministic value stands
        logger.warning(
            "Advisory build call (%s) failed; using the deterministic value instead: %s",
            response_model.__name__,
            exc,
        )
        return None


def _adopt_answer_cleanup(
    deterministic: PhysicsAnswerSemantics,
    cleaned: PhysicsAnswerSemantics,
) -> tuple[PhysicsAnswerSemantics, list[str], dict[str, str]]:
    """Fill empty surface fields from the cleanup call; structure/kind stay pinned.

    Only presentational/gap fields are adopted (and only when the deterministic value is
    empty); ``canonical_text``/``numeric_text``/``numeric_value`` stay deterministic so the
    golden's printed precision is preserved (compat #1).
    """

    flags: list[str] = []
    if (
        cleaned.structure != deterministic.structure
        or cleaned.object_kind != deterministic.object_kind
    ):
        flags.append(
            "answer_cleanup_structure_disagreement:"
            f"{cleaned.structure.value}/{cleaned.object_kind.value}"
        )
    updates: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    for field in _ANSWER_SURFACE_FILL_FIELDS:
        det_value = getattr(deterministic, field)
        llm_value = getattr(cleaned, field)
        if not det_value and llm_value:
            updates[field] = llm_value
            provenance[field] = "llm_declared"
    adopted = deterministic.model_copy(update=updates) if updates else deterministic
    return adopted, flags, provenance


def _adopt_question_policy(
    draft: PhysicsQuestionSemantics,
    policy: PhysicsQuestionSemantics,
) -> tuple[PhysicsQuestionSemantics, dict[str, str]]:
    """Adopt LLM question-policy fields onto the draft.

    ``allowed_*``, ``tolerance``, and ``symbol_assumptions`` from the policy call are ignored
    here; the build sets them deterministically.
    """

    updates: dict[str, Any] = {}
    provenance: dict[str, str] = {}

    # Enum policy fields are decisions; adopt them as-is.
    for field in ("question_symbolic_mode", "question_unit_policy", "ordering"):
        updates[field] = getattr(policy, field)
        provenance[field] = "llm_declared"

    # Optional / collection fields: adopt only when the LLM provided a value.
    if policy.target_variable:
        updates["target_variable"] = policy.target_variable
        provenance["target_variable"] = "llm_declared"
    if policy.symbol_aliases:
        updates["symbol_aliases"] = policy.symbol_aliases
        provenance["symbol_aliases"] = "llm_declared"
    if policy.question_unit:
        updates["question_unit"] = policy.question_unit
        provenance["question_unit"] = "llm_declared"
    if policy.dimension:
        updates["dimension"] = policy.dimension
        provenance["dimension"] = "llm_declared"
    if policy.required_parts:
        updates["required_parts"] = policy.required_parts
        provenance["required_parts"] = "llm_declared"
    if policy.coordinate_frame:
        updates["coordinate_frame"] = policy.coordinate_frame
        provenance["coordinate_frame"] = "llm_declared"
    if policy.sign_convention:
        updates["sign_convention"] = policy.sign_convention
        provenance["sign_convention"] = "llm_declared"
    if policy.choice_space:
        updates["choice_space"] = policy.choice_space
        provenance["choice_space"] = "llm_declared"

    return draft.merged(updates), provenance


def _collect_declared_assumptions(
    response: StrictSymbolAssumptionsResponse | None,
    alias_map: dict[str, str],
) -> tuple[dict[str, SymbolAssumption], dict[str, str]]:
    """Resolve declared assumptions to canonical tokens, returning (map, justifications)."""

    declared: dict[str, SymbolAssumption] = {}
    justifications: dict[str, str] = {}
    if response is None:
        return declared, justifications
    for entry in response.assumptions:
        canonical = resolve_to_canonical(entry.symbol, alias_map)
        if canonical in declared:
            declared[canonical] = meet_assumptions(
                declared[canonical], entry.assumption
            )
        else:
            declared[canonical] = entry.assumption
        if entry.justification:
            justifications[canonical] = entry.justification
    return declared, justifications


def _assumption_provenance(
    merged: dict[str, SymbolAssumption],
    subject_to: dict[str, SymbolAssumption],
    declared: dict[str, SymbolAssumption],
    justifications: dict[str, str],
) -> tuple[SymbolAssumptionProvenance, ...]:
    """Tag each adopted assumption with its source for the build report."""

    provenance: list[SymbolAssumptionProvenance] = []
    for symbol in sorted(merged):
        in_st = symbol in subject_to
        in_llm = symbol in declared
        source = (
            "merged"
            if (in_st and in_llm)
            else ("subject_to" if in_st else "llm_declared")
        )
        provenance.append(
            SymbolAssumptionProvenance(
                symbol=symbol,
                assumption=merged[symbol],
                source=source,
                justification=justifications.get(symbol),
            )
        )
    return tuple(provenance)


def build_reference_semantics(
    problem: PhysicsProblem,
    model_client: BaseModelClient | None = None,
    *,
    golden: str | None = None,
    max_output_tokens: int | None = None,
    temperature: float = 0.0,
    **chat_kwargs: Any,
) -> ReferenceSemanticsArtifact:
    """Build reference semantics (q_ref + a_ref) from a problem and its golden answer.

    Deterministic backbone (authoritative): ``normalize_physics_answer`` +
    ``canonicalize_structure`` pin ``a_ref``'s structure/object_kind; ``subject_to``
    constraints seed ``symbol_assumptions``; tolerance and ``allowed_*`` are deterministic.
    Three advisory LLM calls (temperature 0) then clean the answer surface, fill question
    policy, and declare justified symbol assumptions; their outputs are validated and never
    override the deterministic decisions. A build report records provenance and cross-checks.
    When ``model_client is None`` the advisory calls are skipped and each records its
    ``*_call_unavailable`` flag, yielding a fully deterministic ``(q_ref, a_ref)`` bundle
    (``review_required``) with no LLM call.
    """

    resolved_golden = (
        golden
        if golden is not None
        else (
            answer_like_to_text(problem.answer) if problem.answer is not None else None
        )
    )
    if not resolved_golden:
        raise ValueError(
            f"Problem {problem.problem_id} has no golden answer to build reference semantics."
        )

    call_kwargs = dict(chat_kwargs)
    call_kwargs.setdefault("temperature", temperature)
    image_paths = tuple(problem.image_path or ())
    flags: list[str] = []
    provenance: dict[str, str] = {
        "structure": "deterministic",
        "object_kind": "deterministic",
        "tolerance": "deterministic",
    }

    # Stage 0 - deterministic backbone (authoritative for structure/kind).
    q_draft = infer_reference_question_semantics(problem)
    a_det = canonicalize_structure(
        normalize_physics_answer(resolved_golden, context=q_draft),
        context=q_draft,
    )

    # Stage 1a - answer-surface cleanup (advisory; structure/kind pinned).
    a_ref = a_det
    cleanup = _advisory_inference(
        model_client,
        prompt=build_answer_surface_cleanup_prompt(
            problem, golden_text=resolved_golden, draft_answer=a_det
        ),
        response_model=StrictPhysicsAnswerSemantics,
        image_paths=image_paths,
        max_output_tokens=max_output_tokens,
        **call_kwargs,
    )
    if cleanup is not None:
        a_ref, cleanup_flags, cleanup_provenance = _adopt_answer_cleanup(
            a_det, cleanup.to_canonical()
        )
        flags.extend(cleanup_flags)
        provenance.update(cleanup_provenance)
    else:
        flags.append("answer_cleanup_call_unavailable")

    # Stage 1b - question policy (advisory).
    policy = _advisory_inference(
        model_client,
        prompt=build_question_policy_prompt(problem, answer_draft=a_ref),
        response_model=StrictPhysicsQuestionSemantics,
        image_paths=image_paths,
        max_output_tokens=max_output_tokens,
        **call_kwargs,
    )
    if policy is not None:
        q_ref, policy_provenance = _adopt_question_policy(
            q_draft, policy.to_canonical()
        )
        provenance.update(policy_provenance)
    else:
        q_ref = q_draft
        flags.append("question_policy_call_unavailable")

    alias_map = build_alias_map(q_ref)

    # Stage 1c - justified symbol-assumption declaration (advisory).
    assumptions_response = _advisory_inference(
        model_client,
        prompt=build_symbol_assumptions_prompt(
            problem,
            candidate_symbols=extract_candidate_symbols(
                a_ref.canonical_text, alias_map=alias_map
            ),
            answer_draft=a_ref,
        ),
        response_model=StrictSymbolAssumptionsResponse,
        image_paths=image_paths,
        max_output_tokens=max_output_tokens,
        **call_kwargs,
    )
    if assumptions_response is None:
        flags.append("symbol_assumptions_call_unavailable")
    declared, justifications = _collect_declared_assumptions(
        assumptions_response, alias_map
    )

    # Stage 2 - deterministic synthesis & reconciliation.
    subject_to_assumptions = assumptions_from_subject_to(
        a_ref.subject_to, alias_map=alias_map
    )
    merged_assumptions, merge_flags = merge_symbol_assumptions(
        subject_to_assumptions, declared
    )
    flags.extend(merge_flags)
    for symbol in alias_source_violations(merged_assumptions, alias_map):
        merged_assumptions.pop(symbol, None)
        flags.append(f"dropped_alias_source_assumption:{symbol}")

    q_ref = q_ref.merged(
        {
            "symbol_assumptions": assumptions_to_semantics(merged_assumptions),
            "tolerance": infer_answer_tolerance(instruction_text=problem.question),
        }
    )
    q_ref = reconcile_allowed_sets(q_ref, a_ref)
    a_ref = enrich_answer_quantity_views(a_ref, context=q_ref)

    # Stage 3 - cross-checks (validate authority; never rescue).
    cross_ok = True
    try:
        round_trip = normalize_physics_answer(a_ref.canonical_text, context=q_ref)
        if (
            round_trip.structure != a_ref.structure
            or round_trip.object_kind != a_ref.object_kind
        ):
            flags.append(
                "roundtrip_drift:"
                f"{round_trip.structure.value}/{round_trip.object_kind.value}"
            )
            cross_ok = False
    except Exception as exc:  # noqa: BLE001 - cross-check is best-effort, never fatal
        flags.append(f"roundtrip_error:{exc.__class__.__name__}")
        cross_ok = False

    contract = build_evaluation_contract(
        question_semantics=q_ref, reference_answer_semantics=a_ref
    )
    if (
        validate_answer_against_contract(a_ref, contract).status
        == ContractValidationStatus.VIOLATING
    ):
        flags.append("reference_self_contract_violation")
        cross_ok = False

    pair_issues = reference_pair_consistency(q_ref, a_ref)
    if pair_issues:
        flags.extend(f"pair:{issue}" for issue in pair_issues)
        cross_ok = False
        if (
            any(issue.startswith("target_variable_mismatch") for issue in pair_issues)
            and a_ref.target_variable
        ):
            q_ref = q_ref.merged({"target_variable": a_ref.target_variable})
            provenance["target_variable"] = "deterministic"
            flags.append("reverted_target_variable_to_deterministic")

    report = SemanticsBuildReport(
        build_method="reference_3call",
        temperature=temperature,
        field_provenance=provenance,
        assumption_provenance=_assumption_provenance(
            merged_assumptions, subject_to_assumptions, declared, justifications
        ),
        flags=tuple(flags),
        cross_checks_passed=cross_ok,
        # Only a genuine cross-check failure warrants review. An advisory call being
        # unavailable (e.g. no native structured output) is a normal route, not a defect —
        # it is recorded as an informational flag but does not force review.
        review_required=not cross_ok,
    )

    return ReferenceSemanticsArtifact(
        problem=_problem_record_from_problem(problem),
        ground_truth_answer=resolved_golden,
        question_semantics=q_ref,
        reference_answer_semantics=a_ref,
        generator=_generator_info(
            model_client,
            prompt_name=REFERENCE_PROMPT_NAME,
            prompt_version=REFERENCE_PROMPT_VERSION,
            structured_output_mode="staged_3call",
        ),
        build_report=report,
    )


def build_problem_semantics(
    problem: PhysicsProblem,
    model_client: BaseModelClient,
    *,
    max_output_tokens: int | None = None,
    temperature: float = 0.0,
    **chat_kwargs: Any,
) -> ProblemSemanticsArtifact:
    """Build problem-only question semantics (q_prob), answer-blind.

    Used as the contract for reference-free clustering. There is no golden answer, so no
    answer record and no structure/kind realization: ``allowed_*`` stay permissive and
    symbol assumptions come only from LLM declarations (cross-checked for canonical tokens).
    """

    call_kwargs = dict(chat_kwargs)
    call_kwargs.setdefault("temperature", temperature)
    image_paths = tuple(problem.image_path or ())
    flags: list[str] = []
    provenance: dict[str, str] = {"tolerance": "deterministic"}

    q_draft = infer_prediction_question_semantics(problem)

    policy = _advisory_inference(
        model_client,
        prompt=build_question_policy_prompt(problem, answer_draft=None),
        response_model=StrictPhysicsQuestionSemantics,
        image_paths=image_paths,
        max_output_tokens=max_output_tokens,
        **call_kwargs,
    )
    if policy is not None:
        q_prob, policy_provenance = _adopt_question_policy(
            q_draft, policy.to_canonical()
        )
        provenance.update(policy_provenance)
    else:
        q_prob = q_draft
        flags.append("question_policy_call_unavailable")

    alias_map = build_alias_map(q_prob)

    assumptions_response = _advisory_inference(
        model_client,
        prompt=build_symbol_assumptions_prompt(
            problem,
            candidate_symbols=extract_candidate_symbols(
                problem.question, alias_map=alias_map
            ),
            answer_draft=None,
        ),
        response_model=StrictSymbolAssumptionsResponse,
        image_paths=image_paths,
        max_output_tokens=max_output_tokens,
        **call_kwargs,
    )
    declared, justifications = _collect_declared_assumptions(
        assumptions_response, alias_map
    )
    for symbol in alias_source_violations(declared, alias_map):
        declared.pop(symbol, None)
        flags.append(f"dropped_alias_source_assumption:{symbol}")
    if any(value != SymbolAssumption.REAL for value in declared.values()):
        # Stronger-than-real assumptions have no golden/subject_to to cross-check here.
        flags.append("problem_only_assumptions_unverified")

    q_prob = q_prob.merged(
        {
            "symbol_assumptions": assumptions_to_semantics(declared),
            "tolerance": infer_answer_tolerance(instruction_text=problem.question),
        }
    )

    report = SemanticsBuildReport(
        build_method="problem_3call",
        temperature=temperature,
        field_provenance=provenance,
        assumption_provenance=_assumption_provenance(
            declared, {}, declared, justifications
        ),
        flags=tuple(flags),
        cross_checks_passed=True,
        review_required=bool(flags),
    )

    return ProblemSemanticsArtifact(
        problem=_problem_record_from_problem(problem),
        question_semantics=q_prob,
        generator=_generator_info(
            model_client,
            prompt_name=PROBLEM_PROMPT_NAME,
            prompt_version=PROBLEM_PROMPT_VERSION,
            structured_output_mode="staged_3call",
        ),
        build_report=report,
    )


PredictionAnswerForm = Literal["structured", "extracted", "auto"]
_PREDICTION_ANSWER_FORMS: tuple[PredictionAnswerForm, ...] = (
    "structured",
    "extracted",
    "auto",
)


def generate_prediction_semantics(
    problem: PhysicsProblem,
    model_client: BaseModelClient,
    *,
    isolated_solve: bool = True,
    answer_semantics: PredictionAnswerForm = "auto",
    max_output_tokens: int | None = None,
    allow_non_native_structured_output: bool = False,
    **chat_kwargs: Any,
) -> PredictionSemanticsArtifact:
    """Let a model solve a problem and package the predicted answer semantics.

    By default (``isolated_solve=True``) the model solves the problem **answer-blind and
    contract-blind**: the solve prompt is problem + options + context only, with no embedded
    question-semantics draft (the leakage guard). The artifact's ``prediction_answer_semantics``
    is ``a_pred_llm`` (the LLM-structured record, canonicalized), and the build report carries
    the ``a_pred_llm``-vs-``a_pred_ext`` structure-disagreement audit. Set
    ``isolated_solve=False`` to keep the legacy fused path (the draft is injected and the
    model authors a prediction-side ``question_semantics``).

    ``answer_semantics`` selects which prediction record the isolated solve yields — the
    **consumer's** choice, not a toolkit decision (the toolkit is neutral and does exactly
    what is asked):

    * ``"structured"`` — return ``a_pred_llm`` (native provider-enforced structured output);
      raises if the provider cannot enforce it (no silent substitution).
    * ``"extracted"`` — return ``a_pred_ext`` (plain-text solve, then deterministic
      ``canonicalize_structure(normalize_physics_answer(...))`` extraction); needs no native
      structured output.
    * ``"auto"`` (default) — pick by provider capability (``a_pred_llm`` when the provider
      supports native structured output, otherwise ``a_pred_ext``). The only capability-driven
      mode, and only because the consumer left the choice to the toolkit.

    ``answer_semantics`` governs the isolated path only; passing a non-``"auto"`` value with
    ``isolated_solve=False`` raises.
    """

    if answer_semantics not in _PREDICTION_ANSWER_FORMS:
        raise ValueError(
            f"answer_semantics must be one of {list(_PREDICTION_ANSWER_FORMS)}, "
            f"got {answer_semantics!r}."
        )
    if not isolated_solve and answer_semantics != "auto":
        raise ValueError(
            "answer_semantics is only supported on the isolated solve path "
            "(isolated_solve=True). The legacy fused path always returns a_pred_llm with "
            "deterministic a_pred_ext reconstruction for compact providers."
        )

    if isolated_solve:
        return _infer_isolated_prediction_semantics(
            problem,
            model_client,
            answer_semantics=answer_semantics,
            max_output_tokens=max_output_tokens,
            **chat_kwargs,
        )

    response_model = resolve_prediction_response_model(model_client)
    require_native_json_schema = _resolve_prediction_native_requirement(
        model_client,
        response_model,
        allow_non_native_structured_output=allow_non_native_structured_output,
    )
    spec = prepare_prediction_semantics_inference_spec(
        problem,
        response_model=response_model,
    )
    response, structured_result = _run_structured_inference(
        model_client,
        prompt=spec.prompt,
        response_model=spec.response_model,
        image_paths=spec.image_paths,
        max_output_tokens=max_output_tokens,
        require_native_json_schema=require_native_json_schema,
        **chat_kwargs,
    )
    strict_response = _coerce_prediction_response_to_strict(
        response,
        draft_question_semantics=spec.draft_question_semantics,
    )

    return build_prediction_semantics_artifact(
        problem,
        strict_response,
        provider=getattr(model_client, "provider", None),
        model_name=getattr(model_client, "model", None),
        structured_output_mode=structured_result.structured_output_mode,
        structured_output_strategy=structured_result.structured_output_strategy,
        draft_question_semantics=spec.draft_question_semantics,
    )


#: Deprecated alias for :func:`generate_prediction_semantics` (renamed in the build/API
#: restructure). Retained for one release; prefer ``generate_prediction_semantics``.
infer_prediction_semantics = generate_prediction_semantics


def _resolve_prediction_native_requirement(
    model_client: BaseModelClient,
    response_model: type[BaseModel],
    *,
    allow_non_native_structured_output: bool,
) -> bool:
    """Decide whether native structured output is required for a prediction call."""

    if allow_non_native_structured_output:
        return model_client.resolve_structured_output_plan(
            response_model,
            structured_policy="best_effort",
        ).native_schema_enforced
    return _semantics_should_require_native_json_schema(model_client, response_model)


def _infer_isolated_prediction_semantics(
    problem: PhysicsProblem,
    model_client: BaseModelClient,
    *,
    answer_semantics: PredictionAnswerForm = "auto",
    max_output_tokens: int | None,
    **chat_kwargs: Any,
) -> PredictionSemanticsArtifact:
    """Problem-only isolated solve producing the consumer-selected prediction record.

    ``answer_semantics`` decides the form (see :func:`infer_prediction_semantics`):
    ``"structured"`` forces ``a_pred_llm`` and **raises** if the provider cannot enforce native
    structured output; ``"extracted"`` forces the plain-text ``a_pred_ext`` route; ``"auto"``
    picks by provider capability (``a_pred_llm`` when supported, else ``a_pred_ext``). The
    toolkit does exactly what is asked — it never silently substitutes a different form.
    """

    response_model: type[BaseModel]
    if answer_semantics == "structured":
        # Honor the explicit request: require native structured output; _run_structured_inference
        # raises a clear error (ensure_semantics_native_structured_output_support) if unavailable.
        response_model = StrictPredictionIsolatedResponse
        require_native_json_schema = True
    elif answer_semantics == "extracted":
        # Plain-text solve, then deterministic extraction. Never needs native structured output.
        response_model = StrictPredictionFinalAnswerResponse
        require_native_json_schema = model_client.resolve_structured_output_plan(
            response_model,
            structured_policy="best_effort",
        ).native_schema_enforced
    else:
        # "auto": the only capability-driven mode (the consumer left the choice to the toolkit).
        response_model = resolve_isolated_prediction_response_model(model_client)
        require_native_json_schema = model_client.resolve_structured_output_plan(
            response_model,
            structured_policy="best_effort",
        ).native_schema_enforced
    spec = prepare_isolated_prediction_semantics_inference_spec(
        problem,
        response_model=response_model,
    )
    response, structured_result = _run_structured_inference(
        model_client,
        prompt=spec.prompt,
        response_model=spec.response_model,
        image_paths=spec.image_paths,
        max_output_tokens=max_output_tokens,
        require_native_json_schema=require_native_json_schema,
        **chat_kwargs,
    )
    if not isinstance(
        response,
        (StrictPredictionIsolatedResponse, StrictPredictionFinalAnswerResponse),
    ):
        raise TypeError(
            "Isolated prediction response must be either StrictPredictionIsolatedResponse "
            f"or StrictPredictionFinalAnswerResponse. Got {type(response)!r}."
        )

    # a_pred_ext: deterministic extraction from the final-answer surface (same authority as
    # a_ref). Always computable, so it is the A/B baseline and the disagreement reference.
    a_pred_ext = extract_prediction_answer_semantics(response.final_answer)

    flags: list[str] = []
    if isinstance(response, StrictPredictionIsolatedResponse):
        # a_pred_llm: the LLM-structured record, canonicalized for symmetric structure.
        a_pred_llm = canonicalize_structure(
            response.prediction_answer_semantics.to_canonical(),
            context=PhysicsQuestionSemantics(),
        )
        adopted = a_pred_llm
        build_method = "prediction_isolated_llm"
        if (
            a_pred_llm.structure != a_pred_ext.structure
            or a_pred_llm.object_kind != a_pred_ext.object_kind
        ):
            # Disagreement audit (compat: do NOT reconcile; feeds the A/B comparison).
            flags.append(
                "a_pred_llm_vs_ext_disagreement:"
                f"{a_pred_llm.structure.value}/{a_pred_llm.object_kind.value}"
                f"!={a_pred_ext.structure.value}/{a_pred_ext.object_kind.value}"
            )
    else:
        # a_pred_ext is the produced record. Under "extracted" it is the requested form (no
        # flag); under "auto" it means the provider could not enforce the structured schema.
        adopted = a_pred_ext
        build_method = "prediction_isolated_extracted"
        if answer_semantics == "auto":
            flags.append("a_pred_llm_unavailable")

    report = SemanticsBuildReport(
        build_method=build_method,
        field_provenance={
            "structure": "deterministic" if adopted is a_pred_ext else "llm_declared",
            "object_kind": "deterministic" if adopted is a_pred_ext else "llm_declared",
        },
        flags=tuple(flags),
        cross_checks_passed=True,
        review_required=bool(flags),
    )

    return PredictionSemanticsArtifact(
        problem=_problem_record_from_problem(problem),
        reasoning=response.reasoning,
        final_answer=response.final_answer,
        question_semantics=PhysicsQuestionSemantics(),
        prediction_answer_semantics=enrich_answer_quantity_views(
            adopted, context=PhysicsQuestionSemantics()
        ),
        generator=_generator_info_from_metadata(
            provider=getattr(model_client, "provider", None),
            model_name=getattr(model_client, "model", None),
            prompt_name=PREDICTION_PROMPT_NAME,
            prompt_version=PREDICTION_PROMPT_VERSION,
            structured_output_mode=structured_result.structured_output_mode,
            structured_output_strategy=structured_result.structured_output_strategy,
        ),
        build_report=report,
    )


def prepare_prediction_semantics_inference_spec(
    problem: PhysicsProblem,
    *,
    response_model: type[BaseModel] = StrictPredictionSemanticsResponse,
) -> PredictionSemanticsInferenceSpec:
    """Build the prompt/schema bundle for the legacy fused prediction-semantics inference."""

    draft_question_semantics = infer_prediction_question_semantics(problem)
    return PredictionSemanticsInferenceSpec(
        prompt=build_prediction_semantics_prompt(
            problem,
            draft_question_semantics=draft_question_semantics,
            include_prediction_answer_semantics=(
                response_model is StrictPredictionSemanticsResponse
            ),
        ),
        image_paths=tuple(problem.image_path or ()),
        draft_question_semantics=draft_question_semantics,
        response_model=response_model,
        response_format=normalize_response_format(response_model),
    )


def prepare_isolated_prediction_semantics_inference_spec(
    problem: PhysicsProblem,
    *,
    response_model: type[BaseModel] = StrictPredictionIsolatedResponse,
) -> PredictionSemanticsInferenceSpec:
    """Build the prompt/schema bundle for the ISOLATED problem-only solve.

    The prompt suppresses the embedded question-semantics draft (the leakage guard): it is
    problem + options + context only. ``draft_question_semantics`` is left at the empty default
    because nothing reference/contract-side reaches the solver here.
    """

    return PredictionSemanticsInferenceSpec(
        prompt=build_prediction_semantics_prompt(
            problem,
            include_prediction_answer_semantics=(
                response_model is StrictPredictionIsolatedResponse
            ),
            suppress_question_semantics_draft=True,
        ),
        image_paths=tuple(problem.image_path or ()),
        draft_question_semantics=PhysicsQuestionSemantics(),
        response_model=response_model,
        response_format=normalize_response_format(response_model),
    )


def parse_prediction_semantics_response_text(
    raw_response: str,
    *,
    draft_question_semantics: PhysicsQuestionSemantics | None = None,
    response_model: type[BaseModel] = StrictPredictionSemanticsResponse,
) -> StrictPredictionSemanticsResponse:
    """Validate provider output text as a strict prediction-semantics response."""

    response = _parse_response_model(response_model, raw_response)
    return _coerce_prediction_response_to_strict(
        response,
        draft_question_semantics=draft_question_semantics,
    )


def parse_reference_semantics_response_text(
    raw_response: str,
) -> StrictReferenceSemanticsResponse:
    """Validate provider output text as a strict reference-semantics response."""

    return _parse_response_model(StrictReferenceSemanticsResponse, raw_response)


def extract_prediction_answer_semantics(
    answer_text: str,
    *,
    context: PhysicsQuestionSemantics | None = None,
) -> PhysicsAnswerSemantics:
    """Build the deterministic ``a_pred_ext`` record from a plain-text answer surface.

    ``a_pred_ext = canonicalize_structure(normalize_physics_answer(final_answer))`` — the
    same deterministic authority used for ``a_ref``, so the two classify identically (the
    structure-mismatch defense). No generation happens here, so this is also the entry point
    for externally-supplied answers and the A/B baseline.
    """

    resolved_context = context or PhysicsQuestionSemantics()
    return canonicalize_structure(
        normalize_physics_answer(answer_text, context=resolved_context),
        context=resolved_context,
    )


def build_extracted_prediction_semantics_artifact(
    problem: PhysicsProblem,
    answer_text: str,
    *,
    context: PhysicsQuestionSemantics | None = None,
    provider: str | None = None,
    model_name: str | None = None,
    reasoning: str = "",
) -> PredictionSemanticsArtifact:
    """Package ``a_pred_ext`` for an externally-supplied plain-text answer (no generation).

    This is the standalone, generation-free path: it deterministically extracts the
    prediction answer semantics from ``answer_text`` and is also the A/B baseline against the
    LLM-structured ``a_pred_llm``. The artifact carries no question semantics (the prediction
    side never authors a contract); ``question_semantics`` stays the empty default.
    """

    a_pred_ext = enrich_answer_quantity_views(
        extract_prediction_answer_semantics(answer_text, context=context),
        context=context or PhysicsQuestionSemantics(),
    )
    return PredictionSemanticsArtifact(
        problem=_problem_record_from_problem(problem),
        reasoning=reasoning,
        final_answer=answer_text,
        question_semantics=PhysicsQuestionSemantics(),
        prediction_answer_semantics=a_pred_ext,
        generator=_generator_info_from_metadata(
            provider=provider,
            model_name=model_name,
            prompt_name=PREDICTION_PROMPT_NAME,
            prompt_version=PREDICTION_PROMPT_VERSION,
            structured_output_mode="extracted",
        ),
        build_report=SemanticsBuildReport(
            build_method="prediction_extracted",
            field_provenance={
                "structure": "deterministic",
                "object_kind": "deterministic",
            },
            cross_checks_passed=True,
            review_required=False,
        ),
    )


def _coerce_prediction_response_to_strict(
    response: BaseModel,
    *,
    draft_question_semantics: PhysicsQuestionSemantics | None = None,
) -> StrictPredictionSemanticsResponse:
    """Lift compact provider-facing responses into the full strict response model.

    The compact response carries only a ``final_answer`` surface, so the prediction answer
    semantics are reconstructed deterministically as ``a_pred_ext`` — the same
    ``canonicalize_structure(normalize_physics_answer(...))`` authority used for ``a_ref``.
    """

    if isinstance(response, StrictPredictionSemanticsResponse):
        return response
    if not isinstance(response, StrictPredictionFinalAnswerResponse):
        raise TypeError(
            "Prediction response must be either StrictPredictionSemanticsResponse "
            f"or StrictPredictionFinalAnswerResponse. Got {type(response)!r}."
        )

    resolved_question_semantics = draft_question_semantics or PhysicsQuestionSemantics()
    strict_answer_payload = _strict_answer_payload(
        extract_prediction_answer_semantics(
            response.final_answer,
            context=resolved_question_semantics,
        )
    )
    return StrictPredictionSemanticsResponse.model_validate(
        {
            "reasoning": response.reasoning,
            "final_answer": response.final_answer,
            "question_semantics": {
                key: value
                for key, value in resolved_question_semantics.model_dump(
                    mode="python"
                ).items()
                if key != "metadata"
            },
            "prediction_answer_semantics": strict_answer_payload,
        }
    )


def build_prediction_semantics_artifact(
    problem: PhysicsProblem,
    response: StrictPredictionSemanticsResponse,
    *,
    provider: str | None,
    model_name: str | None,
    structured_output_mode: str = "json_schema",
    structured_output_strategy: str | None = None,
    draft_question_semantics: PhysicsQuestionSemantics | None = None,
) -> PredictionSemanticsArtifact:
    """Construct a prediction-semantics artifact from validated response JSON."""

    resolved_draft_question_semantics = (
        draft_question_semantics or infer_prediction_question_semantics(problem)
    )
    merged_question_semantics = _merge_question_semantics_fallbacks(
        response.question_semantics.to_canonical(),
        resolved_draft_question_semantics,
    )
    return PredictionSemanticsArtifact(
        problem=_problem_record_from_problem(problem),
        reasoning=response.reasoning,
        final_answer=response.final_answer,
        question_semantics=merged_question_semantics,
        prediction_answer_semantics=enrich_answer_quantity_views(
            response.prediction_answer_semantics.to_canonical(),
            context=merged_question_semantics,
        ),
        generator=_generator_info_from_metadata(
            provider=provider,
            model_name=model_name,
            prompt_name=PREDICTION_PROMPT_NAME,
            prompt_version=PREDICTION_PROMPT_VERSION,
            structured_output_mode=structured_output_mode,
            structured_output_strategy=structured_output_strategy,
        ),
    )


def prepare_semantics_comparison(
    reference_artifact: ReferenceSemanticsArtifact | str | Path,
    prediction_artifact: PredictionSemanticsArtifact | str | Path,
) -> SemanticsComparisonInputs:
    """Load or coerce two artifacts into comparison-ready semantics inputs."""

    reference = _coerce_reference_artifact(reference_artifact)
    prediction = _coerce_prediction_artifact(prediction_artifact)

    if reference.problem.problem_id != prediction.problem.problem_id:
        raise ValueError(
            "Reference and prediction artifacts must refer to the same problem_id. "
            f"Got {reference.problem.problem_id!r} and {prediction.problem.problem_id!r}."
        )

    evaluation_contract = build_evaluation_contract(
        question_semantics=reference.question_semantics,
        reference_answer_semantics=reference.reference_answer_semantics,
        problem=reference.problem,
    )

    return SemanticsComparisonInputs(
        problem=reference.problem,
        question_semantics=reference.question_semantics,
        evaluation_contract=evaluation_contract,
        reference_answer_semantics=enrich_answer_quantity_views(
            reference.reference_answer_semantics,
            context=evaluation_contract.comparison_context,
        ),
        prediction_answer_semantics=enrich_answer_quantity_views(
            prediction.prediction_answer_semantics,
            context=evaluation_contract.comparison_context,
        ),
    )


def evaluate_saved_semantics(
    reference_artifact: ReferenceSemanticsArtifact | str | Path,
    prediction_artifact: PredictionSemanticsArtifact | str | Path,
    *,
    policy_mode: ComparisonPolicyMode | str | None = None,
) -> SemanticsEvaluationRecord:
    """Evaluate saved reference and prediction artifacts."""

    resolved_policy = coerce_policy_mode(policy_mode or ComparisonPolicyMode.AUDITED)
    comparison_inputs = prepare_semantics_comparison(
        reference_artifact,
        prediction_artifact,
    )
    comparison = compare_protocol_answers(
        comparison_inputs.prediction_answer_semantics,
        comparison_inputs.reference_answer_semantics,
        contract=comparison_inputs.evaluation_contract,
        context=comparison_inputs.question_semantics,
        policy_mode=resolved_policy,
    )

    return SemanticsEvaluationRecord(
        problem=comparison_inputs.problem,
        question_semantics=comparison_inputs.question_semantics,
        evaluation_contract=comparison_inputs.evaluation_contract,
        policy_mode=resolved_policy,
        reference_answer_semantics=comparison_inputs.reference_answer_semantics,
        prediction_answer_semantics=comparison_inputs.prediction_answer_semantics,
        comparison=comparison,
    )


def compare_saved_semantics(
    reference_artifact: ReferenceSemanticsArtifact | str | Path,
    prediction_artifact: PredictionSemanticsArtifact | str | Path,
    *,
    policy_mode: ComparisonPolicyMode | str | None = None,
) -> AnswerComparison:
    """Compare two saved semantics artifacts and return only the verdict."""

    return evaluate_saved_semantics(
        reference_artifact,
        prediction_artifact,
        policy_mode=policy_mode,
    ).comparison


def _run_structured_inference(
    model_client: BaseModelClient,
    *,
    prompt: str,
    response_model: type[ResponseModelT],
    image_paths: tuple[str, ...],
    max_output_tokens: int | None = None,
    require_native_json_schema: bool = False,
    **chat_kwargs: Any,
) -> tuple[ResponseModelT, StructuredCallResult[ResponseModelT]]:
    """Run one semantics-generation request through the typed structured-output API."""

    structured_policy: StructuredOutputPolicy = (
        "native_required" if require_native_json_schema else "best_effort"
    )
    if require_native_json_schema:
        ensure_semantics_native_structured_output_support(model_client, response_model)

    request_kwargs = dict(chat_kwargs)
    resolved_max_output_tokens = _resolve_max_output_tokens(
        model_client,
        explicit=max_output_tokens,
    )
    if resolved_max_output_tokens is not None:
        request_kwargs["max_output_tokens"] = resolved_max_output_tokens

    result = model_client.parse(
        input=prompt,
        response_format=response_model,
        image_paths=list(image_paths) or None,
        structured_policy=structured_policy,
        **request_kwargs,
    )
    if result.parsed is not None:
        return result.parsed, result

    if result.raw_text is None:
        raise ValueError(
            result.validation_error
            or f"{response_model.__name__} inference returned no response text."
        )

    try:
        repaired = _parse_response_model(response_model, result.raw_text)
        return repaired, result
    except ValueError:
        if not require_native_json_schema and result.structured_output_mode in {
            "prompt_only",
            "json_object",
        }:
            retry_raw_text = _retry_non_native_json_completion(
                model_client,
                prompt=prompt,
                response_model=response_model,
                image_paths=image_paths,
                previous_raw_text=result.raw_text,
                max_output_tokens=request_kwargs.get("max_output_tokens"),
                **chat_kwargs,
            )
            repaired = _parse_response_model(response_model, retry_raw_text)
            return (
                repaired,
                StructuredCallResult(
                    parsed=repaired,
                    raw_text=retry_raw_text,
                    raw_payload=extract_json_payload(retry_raw_text),
                    validation_error=None,
                    structured_output_mode=result.structured_output_mode,
                    structured_output_strategy=result.structured_output_strategy,
                    native_schema_enforced=False,
                    provider=getattr(model_client, "provider", None) or "unknown",
                    model_name=getattr(model_client, "model", None) or "unknown",
                ),
            )
        raise


def ensure_semantics_native_structured_output_support(
    model_client: BaseModelClient,
    response_model: type[BaseModel],
) -> None:
    """Require native provider-enforced structured output for semantics inference."""

    plan = model_client.resolve_structured_output_plan(
        response_model,
        structured_policy="best_effort",
    )
    if plan.native_schema_enforced:
        return
    provider = getattr(model_client, "provider", None) or "unknown"
    model_name = getattr(model_client, "model", None) or "unknown"
    raise ValueError(
        "Semantics inference requires native provider-enforced structured output support. "
        f"Got provider={provider!r} model={model_name!r} strategy={plan.strategy!r}."
    )


def ensure_semantics_native_json_schema_support(model_client: BaseModelClient) -> None:
    """Backward-compatible wrapper for call sites that still use the old name."""

    ensure_semantics_native_structured_output_support(
        model_client,
        StrictPredictionSemanticsResponse,
    )


def _parse_response_model(
    response_model: type[ResponseModelT], raw_response: str
) -> ResponseModelT:
    """Validate a raw model response against the expected Pydantic schema."""

    if raw_response is None:
        raise ValueError(
            f"{response_model.__name__} inference returned no response text."
        )

    text = raw_response.strip()
    if not text:
        raise ValueError(f"{response_model.__name__} inference returned empty text.")

    try:
        return response_model.model_validate_json(text)
    except ValidationError:
        payload = extract_structured_json_object(text)
        if payload is None:
            raise ValueError(
                f"Could not parse {response_model.__name__} response as JSON.\n"
                f"Raw response:\n{text}"
            ) from None
        normalized_payload = _normalize_response_payload(response_model, payload)
        return response_model.model_validate(normalized_payload)


def _normalize_response_payload(
    response_model: type[BaseModel],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Repair known provider-output variations before strict validation."""

    normalized = dict(payload)

    if response_model is StrictPredictionSemanticsResponse:
        if _looks_like_prediction_answer_semantics_payload(normalized):
            logger.debug(
                "Prompt-only prediction parsing wrapped a standalone answer-semantics object."
            )
            normalized = {
                "prediction_answer_semantics": normalized,
                "question_semantics": {},
            }
        if "question_semantics" not in normalized:
            lifted_question_payload = {
                key: normalized.pop(key)
                for key in tuple(normalized)
                if key in _STRICT_QUESTION_FIELDS
            }
            if lifted_question_payload:
                normalized["question_semantics"] = lifted_question_payload
        normalized.pop("reference_answer_semantics", None)
        reasoning_summary = normalized.pop("reasoning_summary", None)
        if "reasoning" not in normalized and isinstance(reasoning_summary, str):
            normalized["reasoning"] = reasoning_summary
        normalized.setdefault("reasoning", "")
        normalized.setdefault("question_semantics", {})
        normalized["question_semantics"] = _normalize_question_semantics_payload(
            normalized["question_semantics"]
        )
        normalized_question_semantics = PhysicsQuestionSemantics.model_validate(
            normalized["question_semantics"]
        )
        answer_semantics = normalized.get("prediction_answer_semantics")
        if isinstance(answer_semantics, dict):
            normalized["prediction_answer_semantics"] = (
                _normalize_answer_semantics_payload(
                    answer_semantics,
                    path="prediction_answer_semantics",
                )
            )
            if "final_answer" not in normalized:
                final_answer = _infer_final_answer_from_answer_semantics(
                    normalized["prediction_answer_semantics"]
                )
                if final_answer is not None:
                    normalized["final_answer"] = final_answer
        elif isinstance(normalized.get("final_answer"), str):
            normalized["prediction_answer_semantics"] = _strict_answer_payload(
                normalize_physics_answer(
                    normalized["final_answer"],
                    context=normalized_question_semantics,
                )
            )
        dropped_top_level = sorted(
            set(payload)
            - set(normalized)
            - {"reference_answer_semantics", "reasoning_summary"}
        )
        if dropped_top_level:
            logger.debug(
                "Prompt-only prediction parsing dropped top-level fields: %s",
                ", ".join(dropped_top_level),
            )
    elif response_model is StrictPredictionIsolatedResponse:
        # Isolated solve: no question_semantics; keep only reasoning/final_answer/answer.
        if _looks_like_prediction_answer_semantics_payload(normalized):
            normalized = {"prediction_answer_semantics": normalized}
        normalized.pop("question_semantics", None)
        normalized.pop("reference_answer_semantics", None)
        reasoning_summary = normalized.pop("reasoning_summary", None)
        if "reasoning" not in normalized and isinstance(reasoning_summary, str):
            normalized["reasoning"] = reasoning_summary
        normalized.setdefault("reasoning", "")
        answer_semantics = normalized.get("prediction_answer_semantics")
        if isinstance(answer_semantics, dict):
            normalized["prediction_answer_semantics"] = (
                _normalize_answer_semantics_payload(
                    answer_semantics,
                    path="prediction_answer_semantics",
                )
            )
            if "final_answer" not in normalized:
                final_answer = _infer_final_answer_from_answer_semantics(
                    normalized["prediction_answer_semantics"]
                )
                if final_answer is not None:
                    normalized["final_answer"] = final_answer
        elif isinstance(normalized.get("final_answer"), str):
            normalized["prediction_answer_semantics"] = _strict_answer_payload(
                normalize_physics_answer(
                    normalized["final_answer"],
                    context=PhysicsQuestionSemantics(),
                )
            )
        dropped_top_level = sorted(
            set(payload)
            - set(normalized)
            - {"question_semantics", "reference_answer_semantics", "reasoning_summary"}
        )
        if dropped_top_level:
            logger.debug(
                "Prompt-only isolated prediction parsing dropped top-level fields: %s",
                ", ".join(dropped_top_level),
            )
    elif response_model is StrictPredictionFinalAnswerResponse:
        normalized = {
            key: value
            for key, value in normalized.items()
            if key in StrictPredictionFinalAnswerResponse.model_fields
            or key == "reasoning_summary"
        }
        reasoning_summary = normalized.pop("reasoning_summary", None)
        if "reasoning" not in normalized and isinstance(reasoning_summary, str):
            normalized["reasoning"] = reasoning_summary
        normalized.setdefault("reasoning", "")
    elif response_model is StrictReferenceSemanticsResponse:
        normalized.pop("prediction_answer_semantics", None)
        normalized.setdefault("question_semantics", {})
        normalized["question_semantics"] = _normalize_question_semantics_payload(
            normalized["question_semantics"]
        )
        answer_semantics = normalized.get("reference_answer_semantics")
        if isinstance(answer_semantics, dict):
            normalized["reference_answer_semantics"] = (
                _normalize_answer_semantics_payload(
                    answer_semantics,
                    path="reference_answer_semantics",
                )
            )
        dropped_top_level = sorted(
            set(payload) - set(normalized) - {"prediction_answer_semantics"}
        )
        if dropped_top_level:
            logger.debug(
                "Prompt-only reference parsing dropped top-level fields: %s",
                ", ".join(dropped_top_level),
            )

    return normalized


def _looks_like_prediction_answer_semantics_payload(payload: dict[str, Any]) -> bool:
    """Whether a parsed object looks like only the nested answer-semantics block."""

    return (
        "prediction_answer_semantics" not in payload
        and "reference_answer_semantics" not in payload
        and (
            "object_kind" in payload
            or "canonical_text" in payload
            or "numeric_value" in payload
        )
    )


def _normalize_question_semantics_payload(payload: Any) -> dict[str, Any]:
    """Repair prompt-only question-semantics payloads into strict shape."""

    if not isinstance(payload, dict):
        return {}

    normalized = dict(payload)
    normalized.pop("metadata", None)
    allowed_object_kinds = normalized.get("allowed_object_kinds")
    if isinstance(allowed_object_kinds, (list, tuple)):
        filtered = [
            value
            for value in allowed_object_kinds
            if isinstance(value, str) and value in _VALID_ALLOWED_OBJECT_KINDS
        ]
        if filtered:
            normalized["allowed_object_kinds"] = filtered
        else:
            normalized.pop("allowed_object_kinds", None)
    allowed_structures = normalized.get("allowed_structures")
    if isinstance(allowed_structures, (list, tuple)):
        filtered = [
            value
            for value in allowed_structures
            if isinstance(value, str) and value in _VALID_ALLOWED_STRUCTURES
        ]
        if filtered:
            normalized["allowed_structures"] = filtered
        else:
            normalized.pop("allowed_structures", None)
    if normalized.get("question_symbolic_mode") in {"numeric", "symbolic"}:
        normalized["question_symbolic_mode"] = "either"
    if normalized.get("question_unit_policy") == "dimensionless":
        normalized["question_unit_policy"] = "not_applicable"
    if normalized.get("ordering") is None:
        normalized.pop("ordering", None)
    return normalized


def _normalize_answer_semantics_payload(payload: Any, *, path: str) -> Any:
    """Repair prompt-only answer-semantics payloads into strict shape."""

    if not isinstance(payload, dict):
        return payload

    normalized = {
        key: value for key, value in payload.items() if key in _STRICT_ANSWER_FIELDS
    }
    dropped_keys = sorted(set(payload) - set(normalized))
    if dropped_keys:
        logger.debug(
            "Prompt-only prediction parsing dropped unsupported answer fields at %s: %s",
            path,
            ", ".join(dropped_keys),
        )
    diagnostics = normalized.get("diagnostics")
    if isinstance(diagnostics, dict):
        normalized["diagnostics"] = tuple(
            f"{key}={value}" for key, value in diagnostics.items()
        )
    elif isinstance(diagnostics, str):
        normalized["diagnostics"] = () if not diagnostics.strip() else (diagnostics,)

    for key in ("children", "subject_to"):
        value = normalized.get(key)
        if isinstance(value, list):
            normalized_items: list[Any] = []
            for index, item in enumerate(value):
                if key == "subject_to" and isinstance(item, str):
                    cleaned = item.strip()
                    normalized_items.append(
                        None
                        if not cleaned
                        else {
                            "canonical_text": cleaned,
                            "raw_text": cleaned,
                            "object_kind": "relation",
                        }
                    )
                    continue
                normalized_items.append(
                    _normalize_answer_semantics_payload(
                        item,
                        path=f"{path}.{key}[{index}]",
                    )
                )
            normalized[key] = tuple(
                item for item in normalized_items if item is not None
            )
        elif key == "subject_to" and isinstance(value, str):
            cleaned = value.strip()
            normalized[key] = (
                ()
                if not cleaned
                else (
                    {
                        "canonical_text": cleaned,
                        "raw_text": cleaned,
                        "object_kind": "relation",
                    },
                )
            )

    cases = normalized.get("cases")
    if isinstance(cases, list):
        normalized["cases"] = tuple(
            _normalize_answer_case_payload(case, path=f"{path}.cases[{index}]")
            for index, case in enumerate(cases)
            if isinstance(case, dict)
        )

    structure = normalized.get("structure")
    object_kind = normalized.get("object_kind")
    if object_kind == structure and structure in {
        "tuple",
        "set",
        "interval",
        "vector",
        "matrix",
        "tensor",
        "piecewise",
        "multi_part",
    }:
        inferred_kind = _infer_object_kind_from_children(normalized)
        if inferred_kind is not None:
            normalized["object_kind"] = inferred_kind

    return normalized


def _normalize_answer_case_payload(payload: Any, *, path: str) -> dict[str, Any]:
    """Repair prompt-only piecewise-case payloads into strict shape."""

    if not isinstance(payload, dict):
        return {}

    normalized = {
        key: value for key, value in payload.items() if key in _STRICT_CASE_FIELDS
    }
    dropped_keys = sorted(set(payload) - set(normalized))
    if dropped_keys:
        logger.debug(
            "Prompt-only prediction parsing dropped unsupported case fields at %s: %s",
            path,
            ", ".join(dropped_keys),
        )

    normalized["expression"] = _normalize_answer_semantics_payload(
        normalized.get("expression"),
        path=f"{path}.expression",
    )
    normalized["condition"] = _normalize_answer_semantics_payload(
        normalized.get("condition"),
        path=f"{path}.condition",
    )
    return normalized


def _infer_object_kind_from_children(payload: dict[str, Any]) -> str | None:
    """Infer a valid atomic object kind for structured answers."""

    children = payload.get("children")
    if not isinstance(children, tuple) or not children:
        return None
    child_kinds = {
        child.get("object_kind")
        for child in children
        if isinstance(child, dict) and isinstance(child.get("object_kind"), str)
    }
    if len(child_kinds) == 1:
        return next(iter(child_kinds))
    return "expression"


def _infer_final_answer_from_answer_semantics(payload: dict[str, Any]) -> str | None:
    """Derive a final answer surface when prompt-only output omits it."""

    for key in ("raw_text", "canonical_text", "canonical_latex", "numeric_text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    numeric_value = payload.get("numeric_value")
    if numeric_value is not None:
        return str(numeric_value)
    return None


def _strict_answer_payload(
    answer_semantics: PhysicsAnswerSemantics | StrictPhysicsAnswerSemantics,
) -> dict[str, Any]:
    """Project a canonical answer semantics object into the strict provider payload shape."""

    normalized = _normalize_answer_semantics_payload(
        answer_semantics.model_dump(mode="python"),
        path="prediction_answer_semantics",
    )
    if not isinstance(normalized, dict):
        raise TypeError("Strict answer payload normalization must return a dict")
    return normalized


def _merge_question_semantics_fallbacks(
    response: PhysicsQuestionSemantics,
    draft: PhysicsQuestionSemantics,
) -> PhysicsQuestionSemantics:
    """Preserve heuristic symbol aliases when the model omits them."""

    if response == PhysicsQuestionSemantics() and draft != PhysicsQuestionSemantics():
        return draft
    if response.symbol_aliases or not draft.symbol_aliases:
        return response
    return response.model_copy(update={"symbol_aliases": draft.symbol_aliases})


def _generator_info(
    model_client: BaseModelClient | None,
    *,
    prompt_name: str,
    prompt_version: str,
    structured_output_mode: str,
    structured_output_strategy: str | None = None,
) -> SemanticsGeneratorInfo:
    """Capture lightweight provenance for one inference call.

    ``model_client is None`` (the deterministic build) yields a generator record with
    no provider/model — the ``getattr`` fallbacks resolve to ``None``.
    """

    return _generator_info_from_metadata(
        provider=getattr(model_client, "provider", None),
        model_name=getattr(model_client, "model", None),
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        structured_output_mode=structured_output_mode,
        structured_output_strategy=structured_output_strategy,
    )


def _generator_info_from_metadata(
    *,
    provider: str | None,
    model_name: str | None,
    prompt_name: str,
    prompt_version: str,
    structured_output_mode: str,
    structured_output_strategy: str | None = None,
) -> SemanticsGeneratorInfo:
    return SemanticsGeneratorInfo(
        provider=provider,
        model_name=model_name,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        structured_output_mode=structured_output_mode,
        structured_output_strategy=structured_output_strategy,
    )


def _response_schema_has_open_objects(response_model: type[BaseModel]) -> bool:
    """Whether a response schema uses object fields unsafe for native JSON-schema mode."""

    return _schema_has_open_objects(response_model.model_json_schema())


def _schema_has_open_objects(schema: Any) -> bool:
    """Recursively detect object schemas that providers reject in strict mode.

    Some model providers reject JSON-schema response formats when nested
    objects leave ``additionalProperties`` open. They can also reject
    dict-like fields that declare an object but no explicit properties,
    even when ``additionalProperties`` is ``false``. The semantics response
    models use this shape for fields like ``provenance`` and ``metadata``,
    so those schemas must fall back to prompt-only JSON.
    """

    if isinstance(schema, dict):
        if schema.get("type") == "object":
            properties = schema.get("properties")
            additional_properties = schema.get("additionalProperties")
            if additional_properties not in (False, None):
                return True
            if not isinstance(properties, dict) or not properties:
                return True

        for value in schema.values():
            if _schema_has_open_objects(value):
                return True
        return False

    if isinstance(schema, list):
        return any(_schema_has_open_objects(item) for item in schema)

    return False


def _resolve_max_output_tokens(
    model_client: BaseModelClient,
    *,
    explicit: int | None,
) -> int | None:
    """Choose a provider-specific default token cap when the caller omits one."""

    if explicit is not None:
        return explicit
    if getattr(model_client, "provider", None) == "google":
        return 65535
    if getattr(model_client, "provider", None) == "anthropic":
        return 4096
    return None


def _retry_non_native_json_completion(
    model_client: BaseModelClient,
    *,
    prompt: str,
    response_model: type[BaseModel],
    image_paths: tuple[str, ...],
    previous_raw_text: str,
    max_output_tokens: int | None,
    **chat_kwargs: Any,
) -> str:
    """Retry one non-native structured-output call with stricter JSON-only instructions."""

    del previous_raw_text
    retry_prompt = _build_non_native_json_retry_prompt(prompt, response_model)
    retry_kwargs = dict(chat_kwargs)
    retry_max_output_tokens = _resolve_retry_max_output_tokens(
        model_client,
        current=max_output_tokens,
    )
    if retry_max_output_tokens is not None:
        retry_kwargs["max_output_tokens"] = retry_max_output_tokens
    logger.warning(
        "Structured-output fallback from %s (%s) returned invalid or incomplete JSON; "
        "retrying once with stricter JSON-only instructions.",
        getattr(model_client, "model", "unknown"),
        getattr(model_client, "provider", "unknown"),
    )
    return model_client.response(
        input=retry_prompt,
        image_paths=list(image_paths) or None,
        response_format=None,
        **retry_kwargs,
    )


def _build_non_native_json_retry_prompt(
    prompt: str,
    response_model: type[BaseModel],
) -> str:
    """Append stricter JSON-only repair instructions for prompt-only/json_object retries."""

    schema = normalize_response_format(response_model)["schema"]
    extra_lines = [
        "Return exactly one complete JSON object and nothing else.",
        "Do not include any prose, analysis, markdown fences, or comments before or after the JSON.",
        "Keep every string field concise.",
    ]
    if response_model in {
        StrictPredictionSemanticsResponse,
        StrictPredictionIsolatedResponse,
    }:
        extra_lines.append(
            "Keep `reasoning` to a brief 1-3 sentence summary, not a full derivation."
        )
    return (
        prompt + build_json_schema_prompt_suffix(schema) + "\n" + "\n".join(extra_lines)
    )


def _resolve_retry_max_output_tokens(
    model_client: BaseModelClient,
    *,
    current: int | None,
) -> int | None:
    """Increase token budget for one repair retry when prompt-only output was truncated."""

    if current is None:
        return _resolve_max_output_tokens(model_client, explicit=None)
    if getattr(model_client, "provider", None) == "anthropic":
        return max(current, 8192)
    return current


def _problem_record_from_problem(problem: PhysicsProblem) -> SemanticsProblemRecord:
    """Project a runtime ``PhysicsProblem`` into a serializable record."""

    domain = problem.get_domain_name() if problem.domain is not None else None
    return SemanticsProblemRecord(
        problem_id=problem.problem_id,
        question=problem.question,
        problem_type=problem.problem_type,
        domain=domain,
        language=problem.language,
        options=tuple(problem.options or ()),
        correct_option=problem.correct_option,
        image_paths=tuple(problem.image_path or ()),
    )


def _coerce_reference_artifact(
    value: ReferenceSemanticsArtifact | str | Path,
) -> ReferenceSemanticsArtifact:
    """Accept either an already-loaded reference artifact or a JSON path."""

    if isinstance(value, ReferenceSemanticsArtifact):
        return value
    return load_reference_semantics_artifact(value)


def _coerce_prediction_artifact(
    value: PredictionSemanticsArtifact | str | Path,
) -> PredictionSemanticsArtifact:
    """Accept either an already-loaded prediction artifact or a JSON path."""

    if isinstance(value, PredictionSemanticsArtifact):
        return value
    return load_prediction_semantics_artifact(value)


__all__ = [
    "PredictionSemanticsInferenceSpec",
    "build_extracted_prediction_semantics_artifact",
    "build_prediction_semantics_artifact",
    "build_problem_semantics",
    "build_reference_semantics",
    "compare_saved_semantics",
    "create_reference_semantics",
    "ensure_semantics_native_structured_output_support",
    "ensure_semantics_native_json_schema_support",
    "evaluate_saved_semantics",
    "extract_prediction_answer_semantics",
    "generate_prediction_semantics",
    "infer_prediction_semantics",
    "infer_reference_semantics",
    "parse_prediction_semantics_response_text",
    "parse_reference_semantics_response_text",
    "prepare_isolated_prediction_semantics_inference_spec",
    "prepare_prediction_semantics_inference_spec",
    "prepare_semantics_comparison",
    "resolve_isolated_prediction_response_model",
    "resolve_prediction_response_model",
]
