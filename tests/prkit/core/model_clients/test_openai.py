"""
Tests for OpenAI model client.
"""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from prkit.core.model_clients.openai import (
    OpenAIModel,
    _is_o_family_model,
    _is_supported_openai_model,
    prepare_image_url_from_image_path,
)
from prkit.core.model_clients.structured_output import coerce_structured_output_spec

OPENAI_TEST_MODEL = "gpt-5.4-mini"


class TestOpenAIModelValidation:
    """Test cases for OpenAI model validation functions."""

    def test_is_supported_openai_model_gpt_4_1(self):
        """Test gpt-4.1 is supported."""
        assert _is_supported_openai_model("gpt-4.1") is True
        assert _is_supported_openai_model("gpt-4.1-mini") is True
        assert _is_supported_openai_model("gpt-4.1-nano") is True

    def test_is_supported_openai_model_gpt_5(self):
        """Test gpt-5xxxx models are supported."""
        assert _is_supported_openai_model("gpt-5") is True
        assert _is_supported_openai_model("gpt-5.1") is True
        assert _is_supported_openai_model("gpt-5.2") is True
        assert _is_supported_openai_model("gpt-5.1-mini") is True

    def test_is_supported_openai_model_o_family(self):
        """Test o-family models are supported."""
        assert _is_supported_openai_model("o3") is True
        assert _is_supported_openai_model("o4") is True
        assert _is_supported_openai_model("o4-mini") is True

    def test_is_supported_openai_model_unsupported(self):
        """Test unsupported models return False."""
        assert _is_supported_openai_model("gpt-3.5-turbo") is False
        assert _is_supported_openai_model("gpt-4") is False
        assert _is_supported_openai_model("unknown") is False

    def test_is_o_family_model(self):
        """Test o-family model detection."""
        assert _is_o_family_model("o3") is True
        assert _is_o_family_model("o4") is True
        assert _is_o_family_model("o4-mini") is True
        assert _is_o_family_model("o1") is True

    def test_is_o_family_model_false(self):
        """Test non-o-family models return False."""
        assert _is_o_family_model("gpt-5.1") is False
        assert _is_o_family_model("gpt-4.1") is False
        assert _is_o_family_model("gemini-pro") is False
        assert (
            _is_o_family_model("openai") is False
        )  # starts with 'o' but not followed by digit


class TestOpenAIModel:
    """Test cases for OpenAIModel class."""

    @patch("prkit.core.model_clients.openai.OpenAI")
    def test_init_supported_model(self, mock_openai_class):
        """Test initializing with supported model."""
        mock_openai_class.return_value = MagicMock()
        client = OpenAIModel(OPENAI_TEST_MODEL)
        assert client.model == OPENAI_TEST_MODEL
        assert client.provider == "openai"
        assert client.is_o_family is False

    @patch("prkit.core.model_clients.openai.OpenAI")
    def test_init_o_family_model(self, mock_openai_class):
        """Test initializing with o-family model."""
        mock_openai_class.return_value = MagicMock()
        client = OpenAIModel("o3")
        assert client.model == "o3"
        assert client.is_o_family is True

    def test_init_unsupported_model(self):
        """Test initializing with unsupported model raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported OpenAI model"):
            OpenAIModel("gpt-3.5-turbo")

    @patch("prkit.core.model_clients.openai.OpenAI")
    def test_chat_text_only(self, mock_openai_class):
        """Test chat with text-only input."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.output_text = "Test response"
        mock_client.responses.create.return_value = mock_response

        client = OpenAIModel(OPENAI_TEST_MODEL)
        response = client.response("Hello, world!")

        assert response == "Test response"
        mock_client.responses.create.assert_called_once()
        call_kwargs = mock_client.responses.create.call_args[1]
        assert call_kwargs["model"] == OPENAI_TEST_MODEL
        assert len(call_kwargs["input"]) == 1
        assert call_kwargs["input"][0]["role"] == "user"
        assert len(call_kwargs["input"][0]["content"]) == 1
        assert call_kwargs["input"][0]["content"][0]["type"] == "input_text"
        assert call_kwargs["input"][0]["content"][0]["text"] == "Hello, world!"
        # OpenAI works with input alone — no default system prompt.
        assert "instructions" not in call_kwargs

    @patch("prkit.core.model_clients.openai.OpenAI")
    def test_response_instructions_only_when_provided(self, mock_openai_class):
        """OpenAI sends `instructions` only when the caller supplies it."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.output_text = "ok"
        mock_client.responses.create.return_value = mock_response

        client = OpenAIModel(OPENAI_TEST_MODEL)

        client.response("Hi")
        assert "instructions" not in mock_client.responses.create.call_args[1]

        client.response("Hi", instructions="Answer tersely.")
        assert (
            mock_client.responses.create.call_args[1]["instructions"]
            == "Answer tersely."
        )

    @patch("prkit.core.model_clients.openai.OpenAI")
    def test_chat_with_images(self, mock_openai_class, tmp_path):
        """Test chat with image inputs."""
        # Create a temporary image file
        image_file = tmp_path / "test_image.png"
        image_file.write_bytes(b"fake image data")

        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.output_text = "Image description"
        mock_client.responses.create.return_value = mock_response

        client = OpenAIModel(OPENAI_TEST_MODEL)
        response = client.response("Describe this image", image_paths=[str(image_file)])

        assert response == "Image description"
        call_kwargs = mock_client.responses.create.call_args[1]
        assert len(call_kwargs["input"]) == 1  # One message with role and content
        assert call_kwargs["input"][0]["role"] == "user"
        assert len(call_kwargs["input"][0]["content"]) == 2  # text + image
        assert call_kwargs["input"][0]["content"][0]["type"] == "input_text"
        assert call_kwargs["input"][0]["content"][1]["type"] == "input_image"
        assert "image_url" in call_kwargs["input"][0]["content"][1]
        assert call_kwargs["input"][0]["content"][1]["image_url"].startswith(
            "data:image/png;base64,"
        )

    @patch("prkit.core.model_clients.openai.OpenAI")
    def test_chat_with_http_url(self, mock_openai_class):
        """Test chat with HTTP URL image."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.output_text = "URL image description"
        mock_client.responses.create.return_value = mock_response

        client = OpenAIModel(OPENAI_TEST_MODEL)
        response = client.response(
            "Describe this image", image_paths=["https://example.com/image.jpg"]
        )

        assert response == "URL image description"
        call_kwargs = mock_client.responses.create.call_args[1]
        assert len(call_kwargs["input"][0]["content"]) == 2  # text + image
        assert (
            call_kwargs["input"][0]["content"][1]["image_url"]
            == "https://example.com/image.jpg"
        )

    @patch("prkit.core.model_clients.openai.OpenAI")
    def test_chat_with_base64_data_url(self, mock_openai_class):
        """Test chat with base64 data URL."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.output_text = "Base64 image description"
        mock_client.responses.create.return_value = mock_response

        client = OpenAIModel(OPENAI_TEST_MODEL)
        data_url = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        response = client.response("Describe this image", image_paths=[data_url])

        assert response == "Base64 image description"
        call_kwargs = mock_client.responses.create.call_args[1]
        assert len(call_kwargs["input"][0]["content"]) == 2  # text + image
        assert call_kwargs["input"][0]["content"][1]["image_url"] == data_url

    @patch("prkit.core.model_clients.openai.OpenAI")
    def test_chat_o_family_with_reasoning(self, mock_openai_class):
        """Test o-family models include reasoning parameter."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.output_text = "Reasoned response"
        mock_client.responses.create.return_value = mock_response

        client = OpenAIModel("o3")
        client.response("Solve this problem")

        call_kwargs = mock_client.responses.create.call_args[1]
        assert "reasoning" in call_kwargs
        assert call_kwargs["reasoning"] == {"effort": "medium"}

    @patch("prkit.core.model_clients.openai.OpenAI")
    def test_chat_non_o_family_no_reasoning(self, mock_openai_class):
        """Test non-o-family models don't include reasoning parameter."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.output_text = "Regular response"
        mock_client.responses.create.return_value = mock_response

        client = OpenAIModel(OPENAI_TEST_MODEL)
        client.response("Hello")

        call_kwargs = mock_client.responses.create.call_args[1]
        assert "reasoning" not in call_kwargs

    @patch("prkit.core.model_clients.openai.OpenAI")
    def test_chat_forwards_max_output_tokens(self, mock_openai_class):
        """Test max_output_tokens is forwarded to the Responses API."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.output_text = "Budgeted response"
        mock_client.responses.create.return_value = mock_response

        client = OpenAIModel(OPENAI_TEST_MODEL)
        client.response("Hello", max_output_tokens=321)

        call_kwargs = mock_client.responses.create.call_args[1]
        assert call_kwargs["max_output_tokens"] == 321

    @patch("prkit.core.model_clients.openai.OpenAI")
    def test_chat_structured_output_uses_openai_strict_schema(self, mock_openai_class):
        """Test structured outputs mark every declared field as required for OpenAI."""

        class ExampleResponse(BaseModel):
            answer: str
            notes: str | None = None

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.output_text = '{"answer":"x","notes":null}'
        mock_client.responses.create.return_value = mock_response

        client = OpenAIModel(OPENAI_TEST_MODEL)
        client.response("Hello", response_format=ExampleResponse)

        schema = mock_client.responses.create.call_args[1]["text"]["format"]["schema"]
        assert set(schema["required"]) == set(schema["properties"].keys())
        assert "notes" in schema["required"]

    def test_resolve_structured_output_plan_falls_back_when_schema_uses_allof(self):
        client = object.__new__(OpenAIModel)
        client.model = OPENAI_TEST_MODEL
        client.provider = "openai"
        client.logger = MagicMock()
        spec = coerce_structured_output_spec(
            {
                "type": "json_schema",
                "name": "AllOfResponse",
                "schema": {
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        }
                    ]
                },
            }
        )

        plan = client._resolve_structured_output_plan(
            spec,
            structured_policy="best_effort",
        )

        assert plan.mode == "prompt_only"
        assert plan.strategy == "openai_prompt_only_unsupported_schema"


class TestPrepareImageURL:
    """Test cases for prepare_image_url_from_image_path function."""

    def test_prepare_image_url_from_file_path(self, tmp_path):
        """Test preparing image URL from file path."""
        image_file = tmp_path / "test.jpg"
        image_file.write_bytes(b"fake image data")

        image_url = prepare_image_url_from_image_path(str(image_file))

        assert image_url.startswith("data:image/jpeg;base64,")
        assert len(image_url) > len("data:image/jpeg;base64,")

    def test_prepare_image_url_from_png(self, tmp_path):
        """Test preparing PNG image URL."""
        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"fake png data")

        image_url = prepare_image_url_from_image_path(str(image_file))

        assert image_url.startswith("data:image/png;base64,")

    def test_prepare_image_url_from_http_url(self):
        """Test HTTP URL is returned as-is."""
        url = "http://example.com/image.jpg"
        result = prepare_image_url_from_image_path(url)
        assert result == url

    def test_prepare_image_url_from_https_url(self):
        """Test HTTPS URL is returned as-is."""
        url = "https://example.com/image.jpg"
        result = prepare_image_url_from_image_path(url)
        assert result == url

    def test_prepare_image_url_from_data_url(self):
        """Test base64 data URL is returned as-is."""
        data_url = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        result = prepare_image_url_from_image_path(data_url)
        assert result == data_url

    def test_prepare_image_url_file_not_found(self):
        """Test FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            prepare_image_url_from_image_path("/nonexistent/image.jpg")

    def test_prepare_image_url_different_mime_types(self, tmp_path):
        """Test preparing URLs for different image MIME types."""
        mime_tests = [
            (".jpg", "image/jpeg"),
            (".jpeg", "image/jpeg"),
            (".png", "image/png"),
            (".gif", "image/gif"),
            (".webp", "image/webp"),
        ]

        for ext, expected_mime in mime_tests:
            image_file = tmp_path / f"test{ext}"
            image_file.write_bytes(b"fake image")
            url = prepare_image_url_from_image_path(str(image_file))
            assert url.startswith(f"data:{expected_mime};base64,")

    def test_prepare_image_url_unknown_extension(self, tmp_path):
        """Test preparing URL for file with unknown extension defaults to jpeg."""
        image_file = tmp_path / "test.unknown"
        image_file.write_bytes(b"fake image")
        url = prepare_image_url_from_image_path(str(image_file))
        assert url.startswith("data:image/jpeg;base64,")

    def test_prepare_image_url_case_insensitive_extension(self, tmp_path):
        """Test that file extension matching is case insensitive."""
        image_file = tmp_path / "test.PNG"
        image_file.write_bytes(b"fake image")
        url = prepare_image_url_from_image_path(str(image_file))
        assert url.startswith("data:image/png;base64,")

    @patch("prkit.core.model_clients.openai.OpenAI")
    def test_chat_with_multiple_images(self, mock_openai_class, tmp_path):
        """Test chat with multiple images."""
        images = []
        for i in range(3):
            img_file = tmp_path / f"image{i}.jpg"
            img_file.write_bytes(b"fake image data")
            images.append(str(img_file))

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.output_text = "Multi-image response"
        mock_client.responses.create.return_value = mock_response

        client = OpenAIModel(OPENAI_TEST_MODEL)
        response = client.response("Describe these images", image_paths=images)

        assert response == "Multi-image response"
        call_kwargs = mock_client.responses.create.call_args[1]
        # Should have 1 text + 3 images = 4 content items
        assert len(call_kwargs["input"]) == 1  # One message
        assert len(call_kwargs["input"][0]["content"]) == 4  # 1 text + 3 images

    @patch("prkit.core.model_clients.openai.OpenAI")
    def test_chat_with_empty_string_prompt(self, mock_openai_class):
        """Test chat with empty string prompt."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.output_text = "Response"
        mock_client.responses.create.return_value = mock_response

        client = OpenAIModel(OPENAI_TEST_MODEL)
        response = client.response("")

        assert response == "Response"
        call_kwargs = mock_client.responses.create.call_args[1]
        assert call_kwargs["input"][0]["content"][0]["text"] == ""

    @patch("prkit.core.model_clients.openai.OpenAI")
    def test_chat_with_none_image_paths(self, mock_openai_class):
        """Test chat with None image_paths."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.output_text = "Response"
        mock_client.responses.create.return_value = mock_response

        client = OpenAIModel(OPENAI_TEST_MODEL)
        response = client.response("Hello", image_paths=None)

        assert response == "Response"
        call_kwargs = mock_client.responses.create.call_args[1]
        # Should only have text content
        assert len(call_kwargs["input"]) == 1  # One message
        assert len(call_kwargs["input"][0]["content"]) == 1  # Only text

    def test_is_supported_openai_model_case_insensitive(self):
        """Test model validation is case insensitive."""
        assert _is_supported_openai_model("GPT-5.1") is True
        assert _is_supported_openai_model("O3") is True
        assert _is_supported_openai_model("GPT-4.1-MINI") is True

    def test_is_o_family_model_edge_cases(self):
        """Test o-family detection with edge cases."""
        assert _is_o_family_model("o1") is True
        assert _is_o_family_model("o10") is True
        assert _is_o_family_model("o1-mini") is True
        assert _is_o_family_model("openai") is False  # 'o' but not followed by digit
        assert _is_o_family_model("o") is False  # Too short
        assert _is_o_family_model("oa") is False  # 'o' followed by letter


class TestOpenAIModelCustomEndpoint:
    """Test custom endpoint / API key params on OpenAIModel."""

    @patch("prkit.core.model_clients.openai.OpenAI")
    def test_explicit_base_url_and_api_key_forwarded(self, mock_openai_class):
        """Explicit base_url and api_key are passed directly to the OpenAI SDK client."""
        mock_openai_class.return_value = MagicMock()
        OpenAIModel(
            OPENAI_TEST_MODEL,
            base_url="https://gw.example/v1",
            api_key="explicit-key",
        )
        _, kwargs = mock_openai_class.call_args
        assert kwargs["base_url"] == "https://gw.example/v1"
        assert kwargs["api_key"] == "explicit-key"

    @patch("prkit.core.model_clients.openai.OpenAI")
    def test_api_key_env_resolves_named_var(self, mock_openai_class):
        """api_key_env reads the key from the named environment variable."""
        mock_openai_class.return_value = MagicMock()
        with patch.dict(os.environ, {"MY_CUSTOM_KEY": "env-key-value"}):
            OpenAIModel(OPENAI_TEST_MODEL, api_key_env="MY_CUSTOM_KEY")
        _, kwargs = mock_openai_class.call_args
        assert kwargs["api_key"] == "env-key-value"

    @patch("prkit.core.model_clients.openai.OpenAI")
    def test_omitting_base_url_does_not_forward_it(self, mock_openai_class):
        """Omitting base_url must not pass base_url= to the SDK (backward-compat guard)."""
        mock_openai_class.return_value = MagicMock()
        OpenAIModel(OPENAI_TEST_MODEL)
        _, kwargs = mock_openai_class.call_args
        assert "base_url" not in kwargs


class TestOmitTemperature:
    """o-family models 400 on an explicit temperature; both paths must drop it.

    The guard used to live only in the batch builder, so ``response()`` sent the
    parameter to exactly the models it existed to protect.
    """

    @staticmethod
    def _client(model):
        with patch("prkit.core.model_clients.openai.OpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            return OpenAIModel(model)

    def test_sync_body_drops_temperature_for_o_family(self):
        client = self._client("o4-mini")

        body = client._build_responses_body(
            input="q",
            instructions=None,
            image_paths=None,
            max_output_tokens=64,
            extra={"temperature": 0.0},
        )

        assert "temperature" not in body

    def test_batch_body_drops_temperature_for_o_family(self):
        client = self._client("o4-mini")

        request = client.build_batch_request(
            request_id="r", input="q", instructions="", temperature=0.0
        )

        assert "temperature" not in request["body"]

    def test_non_o_family_keeps_temperature_on_both_paths(self):
        client = self._client("gpt-5.4-mini")

        body = client._build_responses_body(
            input="q",
            instructions=None,
            image_paths=None,
            max_output_tokens=64,
            extra={"temperature": 0.3},
        )
        request = client.build_batch_request(
            request_id="r", input="q", instructions="", temperature=0.3
        )

        assert body["temperature"] == 0.3
        assert request["body"]["temperature"] == 0.3
