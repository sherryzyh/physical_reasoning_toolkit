"""
Ollama API client implementation.

This module provides a client implementation for models hosted locally via Ollama.
It supports both text and vision models (like qwen3-vl, qwen2-vl, or llava).

Prerequisites:
- Ollama must be installed and running locally (default: http://localhost:11434)
- The model must be pulled first: `ollama pull qwen3-vl`
"""

import logging
import os
from typing import Any
from urllib.parse import urlparse

import ollama

from .base import BaseModelClient
from .structured_output import (
    StructuredOutputPlan,
    StructuredOutputPolicy,
    StructuredOutputSpec,
    normalize_response_format,
)


def _is_remote_host(url: str | None) -> bool:
    """Return True when *url* points to a non-local host."""
    if not url:
        return False
    hostname = (urlparse(url).hostname or "").lower()
    return hostname not in ("localhost", "127.0.0.1", "0.0.0.0")


def normalize_ollama_model_name(model: str) -> str:
    """
    Normalize Ollama model identifiers.

    Accepts either raw Ollama model names (e.g., "qwen3-vl:8b")
    or provider-prefixed form (e.g., "ollama/qwen3-vl:8b").
    """
    if model.lower().startswith("ollama/"):
        return model.split("/", 1)[1]
    return model


class OllamaModel(BaseModelClient):
    """Ollama model client implementation."""

    supports_response_format_json_schema = True
    supports_response_format_json_object = True

    @staticmethod
    def check_ollama_running(base_url: str | None = None) -> bool:
        """
        Check if Ollama service is running and accessible.

        Args:
            base_url: Optional base URL for Ollama API (defaults to http://localhost:11434)

        Returns:
            True if Ollama is running and accessible, False otherwise
        """
        try:
            if base_url:
                client = ollama.Client(host=base_url)
            else:
                client = ollama.Client()
            client.list()
            return True
        except Exception:
            return False

    def __init__(
        self,
        model: str,
        logger: logging.Logger | None = None,
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        api_key_env: str | None = None,
    ) -> None:
        """
        Initialize Ollama model client.

        Args:
            model: Ollama model identifier. Supports either:
                - Raw model name: 'qwen3-vl', 'qwen2.5', 'llava'
                - Prefixed form: 'ollama/qwen3-vl:8b'
            logger: Optional logger instance
            base_url: Optional base URL for Ollama API (defaults to http://localhost:11434,
                      or ``OLLAMA_HOST`` env var). Use ``https://ollama.com`` for cloud.
            api_key: Explicit API key forwarded as an ``Authorization: Bearer`` header.
                     Takes precedence over ``api_key_env``. When omitted the library falls
                     back to the ``OLLAMA_API_KEY`` environment variable automatically.
            api_key_env: Name of an environment variable to read the API key from.
                         Used when ``api_key`` is not supplied.

        Raises:
            ConnectionError: If Ollama service is not running or unreachable (local hosts only;
                             remote hosts emit a warning and continue).
        """
        super().__init__(normalize_ollama_model_name(model), logger)
        self.provider = "ollama"
        self.base_url = base_url

        # Resolve explicit auth — lib auto-injects OLLAMA_API_KEY when not set here.
        if api_key is not None:
            self._auth_header: str | None = f"Bearer {api_key}"
        elif api_key_env is not None:
            env_val = os.environ.get(api_key_env)
            self._auth_header = f"Bearer {env_val}" if env_val else None
        else:
            self._auth_header = None

        # Check if Ollama is running during initialization
        self._check_ollama_running()

    def _client_options(self) -> dict[str, Any]:
        """Build keyword args for ``ollama.Client`` from resolved instance state."""
        opts: dict[str, Any] = {}
        if self.base_url:
            opts["host"] = self.base_url
        if self._auth_header is not None:
            # Use lowercase key so the lib's env-injection guard (which checks
            # headers.get('authorization')) suppresses duplicate auth injection.
            opts["headers"] = {"authorization": self._auth_header}
        return opts

    def _check_ollama_running(self) -> None:
        """
        Check if Ollama service is running and accessible.

        For remote hosts a failed preflight emits a warning instead of raising,
        since network conditions differ and chat() will surface precise errors
        at call time.

        Raises:
            ConnectionError: If Ollama service is not running on a local host.
        """
        effective_host = self.base_url or os.environ.get("OLLAMA_HOST")
        try:
            client = ollama.Client(**self._client_options())
            client.list()
        except Exception as e:
            if _is_remote_host(effective_host):
                self.logger.warning(
                    "Remote Ollama host %s preflight failed (%s). "
                    "Continuing — errors will surface at inference time.",
                    effective_host,
                    e,
                )
                return
            error_msg = (
                f"Ollama service is not running or unreachable at "
                f"{effective_host or 'http://localhost:11434'}. "
                f"Please ensure Ollama is installed and running.\n"
                f"To start Ollama:\n"
                f"  1. Install Ollama from https://ollama.com/download\n"
                f"  2. Start the Ollama service (usually starts automatically after installation)\n"
                f"  3. Verify it's running: `ollama list` or visit http://localhost:11434\n"
                f"Original error: {str(e)}"
            )
            self.logger.error(error_msg)
            raise ConnectionError(error_msg) from e

    def chat(
        self,
        user_prompt: str,
        image_paths: list[str] | None = None,
        response_format: dict[str, Any] | type | None = None,
        max_output_tokens: int = 65535,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response using the local Ollama service.

        Args:
            user_prompt: The user's prompt text.
            image_paths: Optional list of file paths to images.
                         Ollama accepts local file paths.
            response_format: Optional structured output format. Supports either:
                - {"type": "json_schema", "name": str, "schema": dict, "strict": bool}
                - A Pydantic BaseModel class
                - {"type": "json_object"} for generic JSON mode
            *args: Additional positional arguments (ignored, kept for compatibility)
            **kwargs: Additional keyword arguments for request parameters
                     (e.g., max_tokens, etc.)

        Returns:
            Response text from the model.

        Raises:
            FileNotFoundError: If any image file path doesn't exist
            ConnectionError: If Ollama service becomes unreachable during inference
            ValueError: If the model is not found (not pulled in Ollama)
        """
        message: dict[str, Any] = {
            "role": "user",
            "content": user_prompt,
        }

        request_format: str | dict[str, Any] | None = None
        if response_format is not None:
            if (
                isinstance(response_format, dict)
                and response_format.get("type") == "json_object"
            ):
                request_format = "json"
            else:
                normalized = normalize_response_format(response_format)
                request_format = normalized["schema"]

        if image_paths:
            # Ollama's python SDK can handle paths, but for consistency with your
            # OpenAI implementation, we will ensure they exist.
            valid_images: list[str] = []
            for path in image_paths:
                if not os.path.exists(path):
                    self.logger.error(f"Image not found: {path}")
                    raise FileNotFoundError(f"Image file not found: {path}")
                valid_images.append(path)

            message["images"] = valid_images
        options: dict[str, Any] = {
            "temperature": 0,
            "num_predict": max_output_tokens,
            **kwargs,
        }
        # Note: Do NOT set num_gpu=99. It causes "memory layout cannot be allocated"
        # errors. Let Ollama auto-determine GPU layer allocation based on VRAM.

        try:
            request_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [message],
                "options": options,
            }
            if request_format is not None:
                request_kwargs["format"] = request_format

            # Use Client when explicit options are set; fall back to module-level otherwise.
            opts = self._client_options()
            if opts:
                response = ollama.Client(**opts).chat(**request_kwargs)
            else:
                response = ollama.chat(**request_kwargs)

            # Handle both dict-like and object-like response access
            if hasattr(response, "message"):
                content = response.message.content
            else:
                content = response["message"]["content"]
            return content if isinstance(content, str) else str(content)

        except Exception as e:
            # Check if it's a model not found error (ResponseError with 404 or model not found message)
            error_str = str(e).lower()
            error_type = type(e).__name__

            # Handle model not found errors
            if (
                ("model" in error_str and "not found" in error_str)
                or (hasattr(e, "status_code") and getattr(e, "status_code") == 404)
                or (error_type == "ResponseError" and "404" in error_str)
            ):
                error_msg = (
                    f"Model '{self.model}' not found in Ollama. "
                    f"Please pull the model first: `ollama pull {self.model}`\n"
                    f"To list available models: `ollama list`\n"
                    f"Original error: {str(e)}"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg) from e

            # Handle connection errors
            if (
                "connection" in error_str
                or "refused" in error_str
                or "unreachable" in error_str
                or (hasattr(e, "status_code") and getattr(e, "status_code", 0) >= 500)
            ):
                error_msg = (
                    f"Ollama service is not running or unreachable at "
                    f"{self.base_url or 'http://localhost:11434'}. "
                    f"Please ensure Ollama is running.\n"
                    f"To check: `ollama list` or visit http://localhost:11434\n"
                    f"Original error: {str(e)}"
                )
                self.logger.error(error_msg)
                raise ConnectionError(error_msg) from e

            # Re-raise other exceptions as-is
            self.logger.error(f"Ollama inference failed: {str(e)}")
            raise

    def _resolve_structured_output_plan(
        self,
        spec: StructuredOutputSpec,
        *,
        structured_policy: StructuredOutputPolicy,
    ) -> StructuredOutputPlan:
        del structured_policy
        return StructuredOutputPlan(
            mode="json_schema",
            strategy="ollama_format_schema",
            native_schema_enforced=True,
            accepted_artifact_modes=("json_schema", "json_object"),
            accepted_artifact_strategies=("ollama_format_schema", "ollama_json_object"),
            response_format=spec.source_model
            or normalize_response_format(
                {
                    "type": "json_schema",
                    "name": spec.name,
                    "schema": spec.schema,
                    "strict": spec.strict,
                    "description": spec.description,
                }
            ),
        )
