"""Abstract base class and shared structured-output logic for all model clients."""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..domain import PhysicsDataset, PhysicsProblem
from ..logging_config import PRKitLogger
from ..project_env import load_project_dotenv
from .batch_types import BatchResult, BatchStatus
from .modes import PhysicsOutputMode
from .prompts import build_plain_question_prompt
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

if TYPE_CHECKING:
    from prkit.semantics.schema import PhysicsQuestionSemantics

T = TypeVar("T", bound=BaseModel)

# Default system prompt sent to every provider except OpenAI when the caller
# does not supply their own ``instructions``. Kept short and free of any
# response-format directives — structured output is handled separately via the
# per-provider ``response_format`` / prompt-suffix machinery.
DEFAULT_INSTRUCTIONS = (
    "You are a physics expert. Provide accurate, detailed analysis of physics problems."
)


class BaseModelClient(ABC):
    """Abstract base class for all model client implementations."""

    supports_response_format_json_schema = False
    supports_response_format_json_object = False

    def __init__(self, model: str, logger: logging.Logger | None = None) -> None:
        load_project_dotenv(__file__)
        self.model = model
        self.provider: str | None = None
        self.logger = logger if logger else PRKitLogger.get_logger(__name__)

    def __getattr__(self, name: str) -> None:
        if name == "client":
            return None
        raise AttributeError(name)

    def _provider_name(self) -> str:
        """Return the provider identifier string, or ``'unknown'`` when unset."""
        return self.provider or "unknown"

    def _resolve_instructions(self, instructions: str | None) -> str | None:
        """Resolve the effective system prompt (``instructions``) to send.

        Returns *instructions* when the caller supplied one (including an empty
        string, which suppresses the system prompt), otherwise falls back to
        :data:`DEFAULT_INSTRUCTIONS`. Providers that do not require a system
        prompt (e.g. OpenAI) override this to return *instructions* unchanged.
        """
        return instructions if instructions is not None else DEFAULT_INSTRUCTIONS

    @property
    def supports_native_structured_output(self) -> bool:
        """True when this provider supports at least one form of native structured output."""
        return bool(
            self.supports_response_format_json_schema
            or self.supports_response_format_json_object
        )

    @abstractmethod
    def response(
        self,
        input: str,
        image_paths: list[str] | None = None,
        response_format: dict[str, Any] | type | None = None,
        *,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a single request to the provider and return the response text.

        Mirrors OpenAI's ``client.responses.create``: *input* is the user prompt
        and *instructions* is the (optional) system prompt. When *instructions*
        is ``None``, non-OpenAI providers fall back to
        :data:`DEFAULT_INSTRUCTIONS` (see :meth:`_resolve_instructions`).
        """
        raise NotImplementedError("Subclasses must implement .response()")

    def chat(
        self,
        user_prompt: str,
        image_paths: list[str] | None = None,
        response_format: dict[str, Any] | type | None = None,
        **kwargs: Any,
    ) -> str:
        """Deprecated alias for :meth:`response`; use ``response()`` instead."""
        warnings.warn(
            "BaseModelClient.chat() is deprecated; use .response() instead "
            "(the first parameter is now `input` rather than `user_prompt`).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.response(
            input=user_prompt,
            image_paths=image_paths,
            response_format=response_format,
            **kwargs,
        )

    def resolve_structured_output_plan(
        self,
        response_model: type[T],
        *,
        structured_policy: StructuredOutputPolicy = "best_effort",
    ) -> StructuredOutputPlan:
        """Build the structured-output plan for *response_model* under *structured_policy*.

        Raises:
            ValueError: If *structured_policy* is ``'native_required'`` and the provider
                cannot enforce schema-validated output.
        """
        spec = coerce_structured_output_spec(response_model)
        plan = self._resolve_structured_output_plan(
            spec,
            structured_policy=structured_policy,
        )
        if structured_policy == "native_required" and not plan.native_schema_enforced:
            provider = self._provider_name()
            raise ValueError(
                "Structured output requires native provider-enforced schema support. "
                f"Got provider={provider!r} model={self.model!r} "
                f"strategy={plan.strategy!r}."
            )
        return plan

    def parse(
        self,
        input: str,
        *,
        response_format: type[T],
        image_paths: Sequence[str] | None = None,
        structured_policy: StructuredOutputPolicy = "best_effort",
        instructions: str | None = None,
        **kwargs: Any,
    ) -> StructuredCallResult[T]:
        """Send a request and parse the response into a Pydantic model instance.

        Mirrors the SDK ``.parse()`` idiom (OpenAI ``client.responses.parse``,
        Anthropic ``client.messages.parse``): unlike :meth:`response`, which
        returns the raw text, ``parse`` returns a typed
        :class:`~prkit.core.model_clients.structured_output.StructuredCallResult`
        whose ``.parsed`` holds the validated *response_format* model (or ``None``
        with a ``.validation_error`` on failure). Call ``.require_parsed()`` for
        the model-or-raise path. ``max_output_tokens`` (when needed) flows through
        ``**kwargs`` exactly as in :meth:`response`.

        Raises:
            ValueError: If *structured_policy* is ``'native_required'`` and the
                provider cannot enforce schema-validated output.
        """
        plan = self.resolve_structured_output_plan(
            response_format,
            structured_policy=structured_policy,
        )
        prompt = input + (plan.prompt_suffix or "")
        raw_text = self.response(
            input=prompt,
            image_paths=list(image_paths) if image_paths else None,
            response_format=plan.response_format,
            instructions=instructions,
            **kwargs,
        )
        return self._build_structured_call_result(
            response_model=response_format,
            raw_text=raw_text,
            plan=plan,
        )

    def chat_structured(
        self,
        user_prompt: str,
        *,
        response_model: type[T],
        image_paths: Sequence[str] | None = None,
        max_output_tokens: int | None = None,
        structured_policy: StructuredOutputPolicy = "best_effort",
        instructions: str | None = None,
        **kwargs: Any,
    ) -> StructuredCallResult[T]:
        """Deprecated alias for :meth:`parse`; use ``parse()`` instead."""
        warnings.warn(
            "BaseModelClient.chat_structured() is deprecated; use .parse() instead "
            "(the first parameter is now `input` rather than `user_prompt`, and the "
            "schema parameter is now `response_format` rather than `response_model`).",
            DeprecationWarning,
            stacklevel=2,
        )
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        return self.parse(
            input=user_prompt,
            response_format=response_model,
            image_paths=image_paths,
            structured_policy=structured_policy,
            instructions=instructions,
            **kwargs,
        )

    def solve_physics_problem(
        self,
        problem: PhysicsProblem | PhysicsQuestionSemantics | str,
        *,
        output_mode: PhysicsOutputMode = PhysicsOutputMode.ANSWER_TEXT,
        instructions: str | None = None,
        response_format: dict[str, Any] | type | None = None,
        **kwargs: Any,
    ) -> str:
        """Ask the model to solve a physics problem and return its answer.

        The *input kind* is dispatched on the runtime type of *problem*:

        - ``str``: treated directly as the question text (no images).
        - :class:`~prkit.core.domain.PhysicsProblem`: parsed into a prompt and its
          attached ``image_path`` images.
        - :class:`~prkit.semantics.schema.PhysicsQuestionSemantics`: physics
          question semantics input — not yet supported (raises
          ``NotImplementedError``).
        - any other type: unsupported (raises ``TypeError``).

        *output_mode* selects what the model returns. Only
        :attr:`PhysicsOutputMode.ANSWER_TEXT` is implemented; answer-semantics
        output raises ``NotImplementedError``. The return type is ``str`` today
        and will broaden to a structured result once answer semantics lands.

        This builds the prompt and prepares images, then delegates to
        :meth:`response`.
        """
        if output_mode is not PhysicsOutputMode.ANSWER_TEXT:
            # TODO(answer-semantics-output): build the prediction-answer-semantics
            # request via prkit.semantics and return a structured result.
            raise NotImplementedError(
                f"output_mode={output_mode!r} is not implemented yet; only "
                f"{PhysicsOutputMode.ANSWER_TEXT!r} is supported."
            )

        if isinstance(problem, str):
            prompt = problem.strip()
            image_paths: list[str] | None = None
        elif isinstance(problem, PhysicsProblem):
            prompt = build_plain_question_prompt(problem)
            image_paths = problem.image_path or None
        else:
            # Question-semantics input is a TODO; everything else is unsupported.
            try:
                from prkit.semantics.schema import PhysicsQuestionSemantics
            except ImportError:
                question_semantics_types: tuple[type, ...] = ()
            else:
                question_semantics_types = (PhysicsQuestionSemantics,)

            if question_semantics_types and isinstance(
                problem, question_semantics_types
            ):
                # TODO(question-semantics-input): render question semantics into a
                # prompt via prkit.semantics.
                raise NotImplementedError(
                    "Question-semantics input is not supported yet; pass a "
                    "PhysicsProblem or a plain question string."
                )
            raise TypeError(
                f"Unsupported problem input type: {type(problem)!r}. Expected a "
                "PhysicsProblem, a question string, or a PhysicsQuestionSemantics."
            )

        return self.response(
            input=prompt,
            image_paths=image_paths,
            response_format=response_format,
            instructions=instructions,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Free-text batch requests + the asynchronous job lifecycle
    # ------------------------------------------------------------------
    def build_batch_request(
        self,
        *,
        request_id: str,
        input: str,
        instructions: str | None = None,
        image_paths: Sequence[str] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a provider-specific FREE-TEXT batch request line (no structured output).

        Mirrors :meth:`response`: the same *input* / *instructions* handling and
        no ``response_format``. Passing ``instructions=""`` suppresses the system
        prompt on every provider (see :meth:`_resolve_instructions`), so the
        resulting request matches a synchronous
        ``response(input=..., instructions="")`` call.
        """
        return self._build_batch_request(
            request_id=request_id,
            input=input,
            instructions=self._resolve_instructions(instructions),
            image_paths=tuple(image_paths or ()),
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            **kwargs,
        )

    def build_problem_batch_request(
        self,
        problem: PhysicsProblem | str,
        *,
        request_id: str | None = None,
        instructions: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a free-text batch request line for *problem*.

        Batch analogue of :meth:`solve_physics_problem`, dispatching on the
        runtime type of *problem* the same way: a ``str`` is treated as the
        question text (no images, and *request_id* is required since there is no
        ``problem_id``); a :class:`~prkit.core.domain.PhysicsProblem` is rendered
        with :func:`build_plain_question_prompt` and its ``image_path`` images,
        defaulting *request_id* to ``problem.problem_id``. Reusing the same prompt
        builder and image path as the synchronous solver guarantees batch ≡ sync
        prompts. Free-text only, matching the ANSWER_TEXT-only sync path; delegates
        to :meth:`build_batch_request`.
        """
        if isinstance(problem, str):
            if request_id is None:
                raise ValueError(
                    "request_id is required when problem is a raw question string."
                )
            prompt = problem.strip()
            image_paths: list[str] | None = None
        elif isinstance(problem, PhysicsProblem):
            prompt = build_plain_question_prompt(problem)
            image_paths = problem.image_path or None
            if request_id is None:
                request_id = problem.problem_id
        else:
            raise TypeError(
                f"Unsupported problem input type: {type(problem)!r}. Expected a "
                "PhysicsProblem or a question string."
            )

        return self.build_batch_request(
            request_id=request_id,
            input=prompt,
            instructions=instructions,
            image_paths=image_paths,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            **kwargs,
        )

    def submit_batch_physics_reasoning(
        self,
        problems: PhysicsDataset | Sequence[PhysicsProblem],
        **kwargs: Any,
    ) -> str:
        """Submit *problems* as provider batch jobs; return the run-folder path.

        Thin one-line facade mirroring :meth:`solve_physics_problem`: lazily imports
        :mod:`prkit.batch` (kept off the import-light path, like the lazy
        ``prkit.semantics`` import in :meth:`solve_physics_problem`) and delegates to
        :func:`prkit.batch.submit_batch_physics_reasoning`. See that function for the
        keyword arguments (``output_dir`` / ``run_name`` / ``minibatch_size`` / …).
        The ledger is saved to ``<run_dir>/metadata.json``; reconstruct it later via
        :meth:`fetch_batch_physics_reasoning` (or ``BatchSubmission.load(run_dir)``).
        """
        from prkit.batch import submit_batch_physics_reasoning as _submit_batch

        return _submit_batch(self, problems, **kwargs)

    def _build_batch_request(
        self,
        *,
        request_id: str,
        input: str,
        instructions: str | None,
        image_paths: tuple[str, ...],
        max_output_tokens: int | None,
        temperature: float | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Override per provider. Default raises ``NotImplementedError``."""
        del (
            request_id,
            input,
            instructions,
            image_paths,
            max_output_tokens,
            temperature,
            kwargs,
        )
        raise NotImplementedError(
            f"Free-text batch requests are not implemented for provider={self._provider_name()!r}."
        )

    def submit_batch(
        self,
        requests: Sequence[dict[str, Any]],
        *,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Submit a list of batch request dicts; return the provider batch/job id."""
        del requests, metadata
        raise NotImplementedError(
            f"Batch submission is not implemented for provider={self._provider_name()!r}."
        )

    def poll_batch(self, batch_id: str) -> BatchStatus:
        """Return a single snapshot of the batch job's state (no sleeping/looping)."""
        del batch_id
        raise NotImplementedError(
            f"Batch polling is not implemented for provider={self._provider_name()!r}."
        )

    def retrieve_batch_results(self, batch_id: str) -> Iterator[BatchResult]:
        """Yield per-request results for a terminal batch, correlated by ``custom_id``."""
        del batch_id
        raise NotImplementedError(
            f"Batch result retrieval is not implemented for provider={self._provider_name()!r}."
        )

    def _resolve_structured_output_plan(
        self,
        spec: StructuredOutputSpec,
        *,
        structured_policy: StructuredOutputPolicy,
    ) -> StructuredOutputPlan:
        """Map a ``StructuredOutputSpec`` to the best plan supported by this provider."""
        if self.supports_response_format_json_schema:
            return StructuredOutputPlan(
                mode="json_schema",
                strategy=f"{self._provider_name()}_json_schema",
                native_schema_enforced=True,
                accepted_artifact_modes=("json_schema",),
                accepted_artifact_strategies=(f"{self._provider_name()}_json_schema",),
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
                strategy=f"{self._provider_name()}_json_object",
                native_schema_enforced=False,
                accepted_artifact_modes=("json_object", "prompt_only"),
                accepted_artifact_strategies=(f"{self._provider_name()}_json_object",),
                response_format={"type": "json_object"},
                prompt_suffix=build_json_schema_prompt_suffix(spec.schema),
            )
        return StructuredOutputPlan(
            mode="prompt_only",
            strategy=f"{self._provider_name()}_prompt_only",
            native_schema_enforced=False,
            accepted_artifact_modes=("prompt_only",),
            accepted_artifact_strategies=(f"{self._provider_name()}_prompt_only",),
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
        """Parse *raw_text* into a ``StructuredCallResult``, capturing validation errors."""
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
            provider=self._provider_name(),
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
