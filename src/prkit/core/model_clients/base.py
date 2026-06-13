"""
Base model client interface for multiple AI model providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..logging_config import PRKitLogger
from ..project_env import load_project_dotenv
from .structured_output import (
    StructuredCallResult,
    StructuredOutputPlan,
    StructuredOutputPolicy,
    StructuredOutputSpec,
    build_json_schema_prompt_suffix,
    coerce_structured_output_spec,
    extract_json_payload,
    normalize_response_format,
)

T = TypeVar("T", bound=BaseModel)


class BaseModelClient(ABC):
    """Abstract base class for all model client implementations."""

    supports_response_format_json_schema = False
    supports_response_format_json_object = False

    def __init__(self, model: str, logger=None):
        load_project_dotenv(__file__)
        self.model = model
        self.client = None
        self.provider = None
        self.logger = logger if logger else PRKitLogger.get_logger(__name__)

    @property
    def supports_native_structured_output(self) -> bool:
        return bool(
            self.supports_response_format_json_schema
            or self.supports_response_format_json_object
        )

    @abstractmethod
    def chat(
        self,
        user_prompt: str,
        image_paths: list[str] | None = None,
        response_format: dict[str, Any] | type | None = None,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError("Subclasses must implement .chat()")

    def resolve_structured_output_plan(
        self,
        response_model: type[T],
        *,
        structured_policy: StructuredOutputPolicy = "best_effort",
    ) -> StructuredOutputPlan:
        spec = coerce_structured_output_spec(response_model)
        plan = self._resolve_structured_output_plan(
            spec,
            structured_policy=structured_policy,
        )
        if structured_policy == "native_required" and not plan.native_schema_enforced:
            provider = self.provider or "unknown"
            raise ValueError(
                "Structured output requires native provider-enforced schema support. "
                f"Got provider={provider!r} model={self.model!r} "
                f"strategy={plan.strategy!r}."
            )
        return plan

    def chat_structured(
        self,
        user_prompt: str,
        *,
        response_model: type[T],
        image_paths: Sequence[str] | None = None,
        max_output_tokens: int | None = None,
        structured_policy: StructuredOutputPolicy = "best_effort",
        **kwargs: Any,
    ) -> StructuredCallResult[T]:
        plan = self.resolve_structured_output_plan(
            response_model,
            structured_policy=structured_policy,
        )
        prompt = user_prompt + (plan.prompt_suffix or "")
        request_kwargs = dict(kwargs)
        if max_output_tokens is not None:
            request_kwargs["max_output_tokens"] = max_output_tokens

        raw_text = self.chat(
            user_prompt=prompt,
            image_paths=list(image_paths) if image_paths else None,
            response_format=plan.response_format,
            **request_kwargs,
        )
        return self._build_structured_call_result(
            response_model=response_model,
            raw_text=raw_text,
            plan=plan,
        )

    def build_batch_structured_request(
        self,
        *,
        request_id: str,
        user_prompt: str,
        response_model: type[T],
        image_paths: Sequence[str] | None = None,
        max_output_tokens: int | None = None,
        structured_policy: StructuredOutputPolicy = "native_required",
        **kwargs: Any,
    ) -> dict[str, Any]:
        plan = self.resolve_structured_output_plan(
            response_model,
            structured_policy=structured_policy,
        )
        return self._build_batch_structured_request(
            request_id=request_id,
            user_prompt=user_prompt + (plan.prompt_suffix or ""),
            response_model=response_model,
            image_paths=tuple(image_paths or ()),
            max_output_tokens=max_output_tokens,
            plan=plan,
            **kwargs,
        )

    def parse_batch_structured_response(
        self,
        *,
        response_model: type[T],
        raw_response_text: str,
        structured_policy: StructuredOutputPolicy = "native_required",
        structured_output_strategy: str | None = None,
    ) -> StructuredCallResult[T]:
        plan = self.resolve_structured_output_plan(
            response_model,
            structured_policy=structured_policy,
        )
        if structured_output_strategy is not None:
            plan = StructuredOutputPlan(
                mode=plan.mode,
                strategy=structured_output_strategy,
                native_schema_enforced=plan.native_schema_enforced,
                accepted_artifact_modes=plan.accepted_artifact_modes,
                accepted_artifact_strategies=plan.accepted_artifact_strategies,
                response_format=plan.response_format,
                prompt_suffix=plan.prompt_suffix,
            )
        return self._build_structured_call_result(
            response_model=response_model,
            raw_text=raw_response_text,
            plan=plan,
        )

    def _resolve_structured_output_plan(
        self,
        spec: StructuredOutputSpec,
        *,
        structured_policy: StructuredOutputPolicy,
    ) -> StructuredOutputPlan:
        if self.supports_response_format_json_schema:
            return StructuredOutputPlan(
                mode="json_schema",
                strategy=f"{(self.provider or 'unknown')}_json_schema",
                native_schema_enforced=True,
                accepted_artifact_modes=("json_schema",),
                accepted_artifact_strategies=(
                    f"{(self.provider or 'unknown')}_json_schema",
                ),
                response_format=normalize_response_format(
                    spec.source_model
                    or {
                        "type": "json_schema",
                        "name": spec.name,
                        "schema": spec.schema,
                        "strict": spec.strict,
                        "description": spec.description,
                    }
                ),
            )
        if self.supports_response_format_json_object:
            return StructuredOutputPlan(
                mode="json_object",
                strategy=f"{(self.provider or 'unknown')}_json_object",
                native_schema_enforced=False,
                accepted_artifact_modes=("json_object", "prompt_only"),
                accepted_artifact_strategies=(
                    f"{(self.provider or 'unknown')}_json_object",
                ),
                response_format={"type": "json_object"},
                prompt_suffix=build_json_schema_prompt_suffix(spec.schema),
            )
        return StructuredOutputPlan(
            mode="prompt_only",
            strategy=f"{(self.provider or 'unknown')}_prompt_only",
            native_schema_enforced=False,
            accepted_artifact_modes=("prompt_only",),
            accepted_artifact_strategies=(
                f"{(self.provider or 'unknown')}_prompt_only",
            ),
            response_format=None,
            prompt_suffix=build_json_schema_prompt_suffix(spec.schema),
        )

    def _build_structured_call_result(
        self,
        *,
        response_model: type[T],
        raw_text: str | None,
        plan: StructuredOutputPlan,
    ) -> StructuredCallResult[T]:
        parsed: T | None = None
        raw_payload: dict[str, Any] | list[Any] | None = None
        validation_error: str | None = None

        text = None if raw_text is None else raw_text.strip()
        if not text:
            validation_error = (
                f"{response_model.__name__} inference returned no structured text."
            )
        else:
            try:
                parsed = response_model.model_validate_json(text)
                raw_payload = extract_json_payload(text)
            except ValidationError as exc:
                raw_payload = self._extract_structured_payload(raw_text=text, plan=plan)
                if raw_payload is None:
                    validation_error = str(exc)
                else:
                    try:
                        parsed = response_model.model_validate(raw_payload)
                    except ValidationError as nested_exc:
                        validation_error = str(nested_exc)

        return StructuredCallResult(
            parsed=parsed,
            raw_text=raw_text,
            raw_payload=raw_payload,
            validation_error=validation_error,
            structured_output_mode=plan.mode,
            structured_output_strategy=plan.strategy,
            native_schema_enforced=plan.native_schema_enforced,
            provider=self.provider or "unknown",
            model_name=self.model,
        )

    def _extract_structured_payload(
        self,
        *,
        raw_text: str,
        plan: StructuredOutputPlan,
    ) -> dict[str, Any] | list[Any] | None:
        del plan
        return extract_json_payload(raw_text)

    def _build_batch_structured_request(
        self,
        *,
        request_id: str,
        user_prompt: str,
        response_model: type[T],
        image_paths: tuple[str, ...],
        max_output_tokens: int | None,
        plan: StructuredOutputPlan,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del (
            request_id,
            user_prompt,
            response_model,
            image_paths,
            max_output_tokens,
            plan,
            kwargs,
        )
        raise NotImplementedError(
            f"Batch structured requests are not implemented for provider={self.provider!r}."
        )
