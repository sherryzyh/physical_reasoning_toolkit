"""Tests for ``BaseModelClient.build_problem_batch_request`` (batch ≡ sync prompts).

Provider SDKs are faked with ``MagicMock`` so these run fully offline, mirroring
the pattern in ``tests/prkit/core/model_clients/test_batch.py``. The parity tests
assert that the batch line carries the SAME prompt the synchronous
``format_problem_context`` builds, across the OpenAI / Anthropic / Gemini line shapes.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from prkit.core.domain import PhysicsProblem
from prkit.core.model_clients import format_problem_context


def _openai_client(model: str = "gpt-5.1"):
    with patch("prkit.core.model_clients.openai.OpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        from prkit.core.model_clients.openai import OpenAIModel

        return OpenAIModel(model)


def _anthropic_client(model: str = "claude-opus-4-8"):
    with patch("prkit.core.model_clients.anthropic.Anthropic") as mock_cls:
        mock_cls.return_value = MagicMock()
        from prkit.core.model_clients.anthropic import AnthropicModel

        return AnthropicModel(model)


def _gemini_client(model: str = "gemini-3.5-flash"):
    with patch("prkit.core.model_clients.gemini.genai.Client") as mock_cls:
        mock_cls.return_value = MagicMock()
        from prkit.core.model_clients.gemini import GeminiModel

        return GeminiModel(model)


def _problem() -> PhysicsProblem:
    return PhysicsProblem(
        problem_id="prob-1",
        question="A block slides down a frictionless incline. Find its acceleration.",
        problem_type="OE",
        domain="mechanics",
    )


class TestPromptParity:
    def test_openai_prompt_equals_format_problem_context(self):
        problem = _problem()
        req = _openai_client().build_problem_batch_request(problem, instructions="")
        text = req["body"]["input"][0]["content"][0]["text"]
        assert text == format_problem_context(problem)

    def test_anthropic_prompt_equals_format_problem_context(self):
        problem = _problem()
        req = _anthropic_client().build_problem_batch_request(problem, instructions="")
        text = req["params"]["messages"][0]["content"][0]["text"]
        assert text == format_problem_context(problem)

    def test_gemini_prompt_equals_format_problem_context(self):
        problem = _problem()
        req = _gemini_client().build_problem_batch_request(problem, instructions="")
        text = req["request"]["contents"][0]["parts"][0]["text"]
        assert text == format_problem_context(problem)


class TestCorrelationAndImages:
    def test_request_id_defaults_to_problem_id(self):
        problem = _problem()
        req = _openai_client().build_problem_batch_request(problem, instructions="")
        assert req["custom_id"] == "prob-1"

    def test_explicit_request_id_overrides(self):
        problem = _problem()
        req = _openai_client().build_problem_batch_request(
            problem, request_id="surrogate-9", instructions=""
        )
        assert req["custom_id"] == "surrogate-9"

    def test_image_paths_forwarded_from_problem(self):
        # Patch the delegate so we assert the forwarded image_paths without any
        # file I/O (the provider image encoders would otherwise read the file).
        client = _openai_client()
        client.build_batch_request = MagicMock(return_value={})
        problem = PhysicsProblem(
            problem_id="img-1", question="Q", image_path=["/tmp/a.png", "/tmp/b.png"]
        )
        client.build_problem_batch_request(problem)
        _, kwargs = client.build_batch_request.call_args
        assert kwargs["image_paths"] == ["/tmp/a.png", "/tmp/b.png"]
        assert kwargs["input"] == format_problem_context(problem)

    def test_no_images_forwards_none(self):
        client = _openai_client()
        client.build_batch_request = MagicMock(return_value={})
        client.build_problem_batch_request(_problem())
        _, kwargs = client.build_batch_request.call_args
        assert kwargs["image_paths"] is None


class TestStringAndUnsupported:
    def test_str_input_requires_request_id(self):
        with pytest.raises(ValueError, match="request_id is required"):
            _openai_client().build_problem_batch_request("just a question")

    def test_str_input_uses_stripped_text(self):
        req = _openai_client().build_problem_batch_request(
            "  hello world  ", request_id="r1", instructions=""
        )
        assert req["custom_id"] == "r1"
        assert req["body"]["input"][0]["content"][0]["text"] == "hello world"

    def test_unsupported_type_raises_type_error(self):
        with pytest.raises(TypeError, match="Unsupported problem input type"):
            _openai_client().build_problem_batch_request(123, request_id="r1")
