"""Inference calls for reference semantics, prediction semantics, and saved comparison."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from prkit.prkit_core.domain import PhysicsProblem
from prkit.prkit_core.model_clients import BaseModelClient
from prkit.prkit_core.model_clients.structured_output import (
    StructuredCallResult,
    StructuredOutputPolicy,
    build_json_schema_prompt_suffix,
    extract_json_object as extract_structured_json_object,
    extract_json_payload,
    normalize_response_format,
)

from ..comparison import (
    build_evaluation_contract,
    compare_protocol_answers,
)
from ..comparison.contract import coerce_policy_mode
from ..normalization import (
    enrich_answer_quantity_views,
    infer_prediction_question_semantics,
    infer_reference_question_semantics,
    normalize_physics_answer,
)
from ..schema import (
    AnswerObjectKind,
    AnswerStructure,
    ComparisonPolicyMode,
    PhysicsQuestionSemantics,
)
from .artifacts import (
    PredictionSemanticsArtifact,
    ReferenceSemanticsArtifact,
    SemanticsComparisonInputs,
    SemanticsEvaluationRecord,
    SemanticsGeneratorInfo,
    SemanticsProblemRecord,
    load_prediction_semantics_artifact,
    load_reference_semantics_artifact,
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
from .strict_models import (
    StrictPhysicsAnswerCaseSemantics,
    StrictPhysicsAnswerSemantics,
    StrictPredictionFinalAnswerResponse,
    StrictPredictionSemanticsResponse,
    StrictPhysicsQuestionSemantics,
    StrictReferenceSemanticsResponse,
)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
logger = logging.getLogger(__name__)
_STRICT_PREDICTION_RESPONSE_FIELDS = frozenset(
    StrictPredictionSemanticsResponse.model_fields
)
_STRICT_QUESTION_FIELDS = frozenset(StrictPhysicsQuestionSemantics.model_fields)
_STRICT_ANSWER_FIELDS = frozenset(StrictPhysicsAnswerSemantics.model_fields)
_STRICT_CASE_FIELDS = frozenset(StrictPhysicsAnswerCaseSemantics.model_fields)
_VALID_ALLOWED_OBJECT_KINDS = frozenset(kind.value for kind in AnswerObjectKind)
_VALID_ALLOWED_STRUCTURES = frozenset(kind.value for kind in AnswerStructure)


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


def infer_reference_semantics(
    problem: PhysicsProblem,
    model_client: BaseModelClient,
    *,
    max_output_tokens: int | None = None,
    **chat_kwargs: Any,
) -> ReferenceSemanticsArtifact:
    """Infer and package reference semantics for a problem's ground-truth answer."""

    if problem.answer is None:
        raise ValueError(
            f"Problem {problem.problem_id} does not provide `problem.answer`."
        )

    require_native_json_schema = _semantics_should_require_native_json_schema(
        model_client,
        StrictReferenceSemanticsResponse,
    )
    ground_truth_answer_text = answer_like_to_text(problem.answer)
    draft_question_semantics = infer_reference_question_semantics(problem)
    prompt = build_reference_semantics_prompt(
        problem,
        draft_question_semantics=draft_question_semantics,
    )
    response, structured_result = _run_structured_inference(
        model_client,
        prompt=prompt,
        response_model=StrictReferenceSemanticsResponse,
        image_paths=tuple(problem.image_path or ()),
        max_output_tokens=max_output_tokens,
        require_native_json_schema=require_native_json_schema,
        **chat_kwargs,
    )

    merged_question_semantics = _merge_question_semantics_fallbacks(
        response.question_semantics.to_canonical(),
        draft_question_semantics,
    )
    return ReferenceSemanticsArtifact(
        problem=_problem_record_from_problem(problem),
        ground_truth_answer=ground_truth_answer_text,
        question_semantics=merged_question_semantics,
        reference_answer_semantics=enrich_answer_quantity_views(
            response.reference_answer_semantics.to_canonical(),
            context=merged_question_semantics,
        ),
        generator=_generator_info(
            model_client,
            prompt_name=REFERENCE_PROMPT_NAME,
            prompt_version=REFERENCE_PROMPT_VERSION,
            structured_output_mode=structured_result.structured_output_mode,
            structured_output_strategy=structured_result.structured_output_strategy,
        ),
    )


def infer_prediction_semantics(
    problem: PhysicsProblem,
    model_client: BaseModelClient,
    *,
    max_output_tokens: int | None = None,
    allow_non_native_structured_output: bool = False,
    **chat_kwargs: Any,
) -> PredictionSemanticsArtifact:
    """Let a model solve a problem and package the predicted answer semantics."""

    response_model = resolve_prediction_response_model(model_client)
    require_native_json_schema = (
        _semantics_should_require_native_json_schema(
            model_client,
            response_model,
        )
        if not allow_non_native_structured_output
        else model_client.resolve_structured_output_plan(
            response_model,
            structured_policy="best_effort",
        ).native_schema_enforced
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


def prepare_prediction_semantics_inference_spec(
    problem: PhysicsProblem,
    *,
    response_model: type[BaseModel] = StrictPredictionSemanticsResponse,
) -> PredictionSemanticsInferenceSpec:
    """Build the prompt/schema bundle used for prediction-semantics inference."""

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


def _coerce_prediction_response_to_strict(
    response: BaseModel,
    *,
    draft_question_semantics: PhysicsQuestionSemantics | None = None,
) -> StrictPredictionSemanticsResponse:
    """Lift compact provider-facing responses into the full strict response model."""

    if isinstance(response, StrictPredictionSemanticsResponse):
        return response
    if not isinstance(response, StrictPredictionFinalAnswerResponse):
        raise TypeError(
            "Prediction response must be either StrictPredictionSemanticsResponse "
            f"or StrictPredictionFinalAnswerResponse. Got {type(response)!r}."
        )

    resolved_question_semantics = draft_question_semantics or PhysicsQuestionSemantics()
    strict_answer_payload = _strict_answer_payload(
        normalize_physics_answer(
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
):
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
    response_model: type[BaseModel],
    image_paths: tuple[str, ...],
    max_output_tokens: int | None = None,
    require_native_json_schema: bool = False,
    **chat_kwargs: Any,
):
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

    result = model_client.chat_structured(
        user_prompt=prompt,
        response_model=response_model,
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
        if (
            not require_native_json_schema
            and result.structured_output_mode in {"prompt_only", "json_object"}
        ):
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


def _parse_response_model(response_model: type[BaseModel], raw_response: str):
    """Validate a raw model response against the expected Pydantic schema."""

    if raw_response is None:
        raise ValueError(f"{response_model.__name__} inference returned no response text.")

    text = raw_response.strip()
    if not text:
        raise ValueError(f"{response_model.__name__} inference returned empty text.")

    try:
        return response_model.model_validate_json(text)
    except ValidationError:
        payload = _extract_json_object(text)
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
            normalized["prediction_answer_semantics"] = _normalize_answer_semantics_payload(
                answer_semantics,
                path="prediction_answer_semantics",
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
            set(payload) - set(normalized) - {"reference_answer_semantics", "reasoning_summary"}
        )
        if dropped_top_level:
            logger.debug(
                "Prompt-only prediction parsing dropped top-level fields: %s",
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
            normalized["reference_answer_semantics"] = _normalize_answer_semantics_payload(
                answer_semantics,
                path="reference_answer_semantics",
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


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a model response, if any."""
    return extract_structured_json_object(text)


def _iter_braced_json_candidates(text: str):
    """Yield balanced brace substrings that could be JSON objects."""

    start_indices = [index for index, char in enumerate(text) if char == "{"]
    for start in start_indices:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : index + 1]
                    break


def _try_parse_json_object(candidate: str) -> dict[str, Any] | None:
    """Parse one JSON-object candidate and ignore nested-shape false positives."""

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if _looks_like_prediction_answer_semantics_payload(parsed):
        return {
            "prediction_answer_semantics": parsed,
            "question_semantics": {},
        }
    return parsed


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
        normalized["diagnostics"] = (() if not diagnostics.strip() else (diagnostics,))

    for key in ("children", "subject_to"):
        value = normalized.get(key)
        if isinstance(value, list):
            normalized_items: list[Any] = []
            for index, item in enumerate(value):
                if key == "subject_to" and isinstance(item, str):
                    cleaned = item.strip()
                    normalized_items.append(
                        (
                            None
                            if not cleaned
                            else {
                                "canonical_text": cleaned,
                                "raw_text": cleaned,
                                "object_kind": "relation",
                            }
                        )
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

    normalized = {key: value for key, value in payload.items() if key in _STRICT_CASE_FIELDS}
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


def _strict_answer_payload(answer_semantics: Any) -> dict[str, Any]:
    """Project a canonical answer semantics object into the strict provider payload shape."""

    return _normalize_answer_semantics_payload(
        answer_semantics.model_dump(mode="python"),
        path="prediction_answer_semantics",
    )


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
    model_client: BaseModelClient,
    *,
    prompt_name: str,
    prompt_version: str,
    structured_output_mode: str,
    structured_output_strategy: str | None = None,
) -> SemanticsGeneratorInfo:
    """Capture lightweight provenance for one inference call."""

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
    return model_client.chat(
        user_prompt=retry_prompt,
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
    if response_model is StrictPredictionSemanticsResponse:
        extra_lines.append(
            "Keep `reasoning` to a brief 1-3 sentence summary, not a full derivation."
        )
    return (
        prompt
        + build_json_schema_prompt_suffix(schema)
        + "\n"
        + "\n".join(extra_lines)
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
    "build_prediction_semantics_artifact",
    "compare_saved_semantics",
    "ensure_semantics_native_structured_output_support",
    "ensure_semantics_native_json_schema_support",
    "evaluate_saved_semantics",
    "infer_prediction_semantics",
    "infer_reference_semantics",
    "parse_prediction_semantics_response_text",
    "parse_reference_semantics_response_text",
    "prepare_prediction_semantics_inference_spec",
    "prepare_semantics_comparison",
]
