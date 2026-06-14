"""
Tests for BaseModelClient abstract base class.
"""

from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from prkit.core.domain import PhysicsProblem
from prkit.core.model_clients.base import DEFAULT_INSTRUCTIONS, BaseModelClient
from prkit.core.model_clients.modes import PhysicsOutputMode


class TestBaseModelClient:
    """Test cases for BaseModelClient."""

    def test_cannot_instantiate_base_class(self):
        """Test that BaseModelClient cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModelClient("test-model")

    def test_base_class_has_abstract_response_method(self):
        """Test that BaseModelClient defines an abstract response method."""
        assert hasattr(BaseModelClient, "response")
        # A concrete implementation without response() stays abstract.
        with pytest.raises(TypeError):

            class IncompleteModel(BaseModelClient):
                pass

            IncompleteModel("test-model")

    def test_base_class_initialization_attributes(self):
        """Test that BaseModelClient initializes with correct attributes."""

        class ConcreteModel(BaseModelClient):
            def response(self, input, image_paths=None, **kwargs: Any):
                return "response"

        with patch("prkit.core.model_clients.base.load_project_dotenv"):
            client = ConcreteModel("test-model")
            assert client.model == "test-model"
            assert client.client is None
            assert client.provider is None
            assert client.logger is not None

    def test_base_class_with_custom_logger(self):
        """Test BaseModelClient initialization with custom logger."""
        import logging

        custom_logger = logging.getLogger("custom")

        class ConcreteModel(BaseModelClient):
            def response(self, input, image_paths=None, **kwargs: Any):
                return "response"

        with patch("prkit.core.model_clients.base.load_project_dotenv"):
            client = ConcreteModel("test-model", logger=custom_logger)
            assert client.logger == custom_logger

    def test_base_class_loads_project_dotenv(self):
        """Test that BaseModelClient loads environment variables."""

        class ConcreteModel(BaseModelClient):
            def response(self, input, image_paths=None, **kwargs: Any):
                return "response"

        with patch("prkit.core.model_clients.base.load_project_dotenv") as mock_load:
            ConcreteModel("test-model")
            mock_load.assert_called_once()

    def test_concrete_implementation_must_implement_response(self):
        """Test that concrete implementations can call response()."""

        class ConcreteModel(BaseModelClient):
            def response(self, input, image_paths=None, **kwargs: Any):
                return "response"

        with patch("prkit.core.model_clients.base.load_project_dotenv"):
            client = ConcreteModel("test-model")
            result = client.response("test prompt")
            assert result == "response"

    def test_response_method_signature(self):
        """Test that response() accepts the expected parameters."""

        class ConcreteModel(BaseModelClient):
            def response(self, input, image_paths=None, **kwargs: Any):
                return f"Response to: {input}"

        with patch("prkit.core.model_clients.base.load_project_dotenv"):
            client = ConcreteModel("test-model")
            assert "Hello" in client.response("Hello")
            assert "Hello" in client.response("Hello", image_paths=["image.jpg"])

    def test_deprecated_chat_alias_warns_and_forwards(self):
        """The legacy chat() alias should warn and forward to response()."""

        class ConcreteModel(BaseModelClient):
            def response(
                self, input, image_paths=None, response_format=None, **kwargs: Any
            ):
                return f"resp:{input}:{image_paths}"

        with patch("prkit.core.model_clients.base.load_project_dotenv"):
            client = ConcreteModel("test-model")
            with pytest.warns(DeprecationWarning, match="use .response"):
                result = client.chat("Hello", image_paths=["a.jpg"])
            assert result == "resp:Hello:['a.jpg']"

    def test_resolve_instructions_default_and_overrides(self):
        """_resolve_instructions falls back to the default but honors explicit values."""

        class ConcreteModel(BaseModelClient):
            def response(self, input, image_paths=None, **kwargs: Any):
                return "response"

        with patch("prkit.core.model_clients.base.load_project_dotenv"):
            client = ConcreteModel("test-model")
            assert client._resolve_instructions(None) == DEFAULT_INSTRUCTIONS
            assert client._resolve_instructions("custom") == "custom"
            # An explicit empty string is preserved (suppresses the system prompt).
            assert client._resolve_instructions("") == ""

    def test_chat_structured_returns_parsed_model(self):
        class ResponseModel(BaseModel):
            answer: str

        class ConcreteModel(BaseModelClient):
            supports_response_format_json_schema = True

            def response(
                self, input, image_paths=None, response_format=None, **kwargs: Any
            ):
                assert response_format["type"] == "json_schema"
                assert response_format["name"] == "ResponseModel"
                return '{"answer":"ok"}'

        with patch("prkit.core.model_clients.base.load_project_dotenv"):
            client = ConcreteModel("test-model")
            client.provider = "dummy"
            result = client.chat_structured(
                "Hello",
                response_model=ResponseModel,
                structured_policy="best_effort",
            )

        assert result.parsed == ResponseModel(answer="ok")
        assert result.structured_output_mode == "json_schema"
        assert result.structured_output_strategy == "dummy_json_schema"

    def test_chat_structured_propagates_instructions(self):
        """chat_structured should forward `instructions` to response()."""

        class ResponseModel(BaseModel):
            answer: str

        seen: dict[str, Any] = {}

        class ConcreteModel(BaseModelClient):
            supports_response_format_json_schema = True

            def response(
                self,
                input,
                image_paths=None,
                response_format=None,
                *,
                instructions=None,
                **kwargs: Any,
            ):
                seen["instructions"] = instructions
                return '{"answer":"ok"}'

        with patch("prkit.core.model_clients.base.load_project_dotenv"):
            client = ConcreteModel("test-model")
            client.provider = "dummy"
            client.chat_structured(
                "Hello",
                response_model=ResponseModel,
                structured_policy="best_effort",
                instructions="be terse",
            )
            assert seen["instructions"] == "be terse"

            client.chat_structured(
                "Hello",
                response_model=ResponseModel,
                structured_policy="best_effort",
            )
            assert seen["instructions"] is None

    def test_chat_structured_native_required_rejects_non_native_provider(self):
        class ResponseModel(BaseModel):
            answer: str

        class ConcreteModel(BaseModelClient):
            supports_response_format_json_schema = False
            supports_response_format_json_object = True

            def response(
                self, input, image_paths=None, response_format=None, **kwargs: Any
            ):
                return '{"answer":"ok"}'

        with patch("prkit.core.model_clients.base.load_project_dotenv"):
            client = ConcreteModel("test-model")
            client.provider = "dummy"
            with pytest.raises(
                ValueError, match="native provider-enforced schema support"
            ):
                client.chat_structured(
                    "Hello",
                    response_model=ResponseModel,
                    structured_policy="native_required",
                )


class _RecordingModel(BaseModelClient):
    """Concrete client that records the arguments passed to response()."""

    def __init__(self, model: str = "test-model"):
        with patch("prkit.core.model_clients.base.load_project_dotenv"):
            super().__init__(model)
        self.calls: list[dict[str, Any]] = []

    def response(
        self,
        input,
        image_paths=None,
        response_format=None,
        *,
        instructions=None,
        **kwargs: Any,
    ):
        self.calls.append(
            {
                "input": input,
                "image_paths": image_paths,
                "response_format": response_format,
                "instructions": instructions,
            }
        )
        return "ANSWER"


class TestSolvePhysicsProblem:
    """Tests for the high-level solve_physics_problem dispatch."""

    def test_plain_string_input_passes_through_without_images(self):
        client = _RecordingModel()
        result = client.solve_physics_problem("  What is g?  ")
        assert result == "ANSWER"
        assert client.calls[0]["input"] == "What is g?"
        assert client.calls[0]["image_paths"] is None

    def test_physics_problem_input_builds_prompt_and_attaches_images(self, tmp_path):
        image = tmp_path / "fig.png"
        image.write_bytes(b"x")
        problem = PhysicsProblem(
            problem_id="p1",
            question="Find the net force.",
            problem_type="MC",
            options=["1 N", "2 N"],
            image_path=[str(image)],
        )
        client = _RecordingModel()
        client.solve_physics_problem(problem)

        sent = client.calls[0]
        assert "p1" in sent["input"]
        assert "Find the net force." in sent["input"]
        assert "A. 1 N" in sent["input"] and "B. 2 N" in sent["input"]
        assert "Attached images: 1 image" in sent["input"]
        assert sent["image_paths"] == [str(image)]

    def test_physics_problem_without_images_sends_none(self):
        problem = PhysicsProblem(problem_id="p2", question="Define momentum.")
        client = _RecordingModel()
        client.solve_physics_problem(problem)
        assert client.calls[0]["image_paths"] is None

    def test_instructions_forwarded(self):
        client = _RecordingModel()
        client.solve_physics_problem("Q?", instructions="Answer in one word.")
        assert client.calls[0]["instructions"] == "Answer in one word."

    def test_answer_semantics_output_not_implemented(self):
        client = _RecordingModel()
        with pytest.raises(NotImplementedError):
            client.solve_physics_problem(
                "Q?", output_mode=PhysicsOutputMode.ANSWER_SEMANTICS
            )

    def test_question_semantics_input_not_implemented(self):
        from prkit.semantics.schema import PhysicsQuestionSemantics

        client = _RecordingModel()
        qs = PhysicsQuestionSemantics()
        with pytest.raises(NotImplementedError, match="Question-semantics"):
            client.solve_physics_problem(qs)

    def test_unsupported_input_type_raises_type_error(self):
        client = _RecordingModel()
        with pytest.raises(TypeError, match="Unsupported problem input type"):
            client.solve_physics_problem(123)
