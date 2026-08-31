"""
Tests for the Moonshot AI (Kimi) model client.

The OpenAI SDK constructor is patched, so these run fully offline.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from prkit.core.model_clients.base import DEFAULT_INSTRUCTIONS
from prkit.core.model_clients.moonshot import MoonshotModel
from prkit.core.model_clients.retry import DEFAULT_MAX_RETRIES
from prkit.core.model_clients.structured_output import coerce_structured_output_spec

MOONSHOT_TEST_MODEL = "kimi-k3"
SYSTEM_MESSAGE = {"role": "system", "content": DEFAULT_INSTRUCTIONS}


class ExampleResponse(BaseModel):
    answer: str


def _stub_client(model: str = MOONSHOT_TEST_MODEL) -> MoonshotModel:
    client = object.__new__(MoonshotModel)
    client.model = model
    client.provider = "moonshot"
    client.logger = MagicMock()
    return client


class TestMoonshotInit:
    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.core.model_clients.base.load_project_dotenv")
    def test_defaults_to_the_global_endpoint(self, _dotenv, mock_openai_class):
        mock_openai_class.return_value = MagicMock()

        with patch.dict("os.environ", {"MOONSHOT_API_KEY": "k"}, clear=True):
            client = MoonshotModel(MOONSHOT_TEST_MODEL)

        assert client.model == MOONSHOT_TEST_MODEL
        assert client.provider == "moonshot"
        assert client.base_url == "https://api.moonshot.ai/v1"
        mock_openai_class.assert_called_once_with(
            api_key="k",
            base_url="https://api.moonshot.ai/v1",
            max_retries=DEFAULT_MAX_RETRIES,
        )

    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.core.model_clients.base.load_project_dotenv")
    def test_region_cn_selects_the_china_endpoint(self, _dotenv, mock_openai_class):
        mock_openai_class.return_value = MagicMock()

        with patch.dict(
            "os.environ",
            {"MOONSHOT_API_KEY": "k", "MOONSHOT_REGION": "cn"},
            clear=True,
        ):
            client = MoonshotModel(MOONSHOT_TEST_MODEL)

        assert client.base_url == "https://api.moonshot.cn/v1"

    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.core.model_clients.base.load_project_dotenv")
    def test_explicit_base_url_wins_over_region(self, _dotenv, mock_openai_class):
        mock_openai_class.return_value = MagicMock()

        with patch.dict(
            "os.environ",
            {
                "MOONSHOT_API_KEY": "k",
                "MOONSHOT_REGION": "cn",
                "MOONSHOT_BASE_URL": "https://proxy.example/v1",
            },
            clear=True,
        ):
            client = MoonshotModel(MOONSHOT_TEST_MODEL)

        assert client.base_url == "https://proxy.example/v1"

    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.core.model_clients.base.load_project_dotenv")
    def test_unknown_region_falls_back_to_global(self, _dotenv, mock_openai_class):
        mock_openai_class.return_value = MagicMock()

        with patch.dict(
            "os.environ",
            {"MOONSHOT_API_KEY": "k", "MOONSHOT_REGION": "atlantis"},
            clear=True,
        ):
            client = MoonshotModel(MOONSHOT_TEST_MODEL)

        assert client.base_url == "https://api.moonshot.ai/v1"

    def test_normalize_model_name_strips_the_moonshot_prefix(self):
        assert MoonshotModel.normalize_model_name("moonshot/kimi-k3") == "kimi-k3"

    def test_moonshot_v1_model_names_are_left_alone(self):
        assert MoonshotModel.normalize_model_name("moonshot-v1-8k") == "moonshot-v1-8k"


class TestMoonshotChat:
    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.core.model_clients.base.load_project_dotenv")
    def test_chat_text_only(self, _dotenv, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        message = Mock()
        message.content = "Test response"
        choice = Mock()
        choice.message = message
        completion = Mock()
        completion.choices = [choice]
        mock_client.chat.completions.create.return_value = completion

        with patch.dict("os.environ", {"MOONSHOT_API_KEY": "k"}, clear=True):
            client = MoonshotModel(MOONSHOT_TEST_MODEL)
        response = client.response("Hello, world!")

        assert response == "Test response"
        mock_client.chat.completions.create.assert_called_once_with(
            model=MOONSHOT_TEST_MODEL,
            messages=[SYSTEM_MESSAGE, {"role": "user", "content": "Hello, world!"}],
        )

    @patch("prkit.core.model_clients.openai_compatible_chat.OpenAI")
    @patch("prkit.core.model_clients.base.load_project_dotenv")
    def test_sends_max_completion_tokens_not_max_tokens(
        self, _dotenv, mock_openai_class
    ):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        message = Mock()
        message.content = "ok"
        choice = Mock()
        choice.message = message
        completion = Mock()
        completion.choices = [choice]
        mock_client.chat.completions.create.return_value = completion

        with patch.dict("os.environ", {"MOONSHOT_API_KEY": "k"}, clear=True):
            client = MoonshotModel("kimi-k2.6")
        client.response("hi", max_output_tokens=123)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_completion_tokens"] == 123
        assert "max_tokens" not in call_kwargs

    @patch(
        "prkit.core.model_clients.openai_compatible_chat.prepare_image_url_from_image_path"
    )
    def test_images_use_the_openai_block_shape(self, mock_prepare):
        """Moonshot follows OpenAI here and must not inherit xAI's override."""
        mock_prepare.side_effect = ["data:image/png;base64,abc"]
        client = _stub_client()

        content = client._build_message_content("describe", ["a.png"])

        assert content == [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]


class TestMoonshotParamOmissions:
    def test_k3_drops_its_server_pinned_params(self):
        client = _stub_client("kimi-k3")

        result = client._apply_param_omissions(
            {
                "model": "kimi-k3",
                "temperature": 0.2,
                "top_p": 0.9,
                "n": 1,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "messages": [],
            }
        )

        assert result == {"model": "kimi-k3", "messages": []}

    def test_k2_keeps_temperature(self):
        client = _stub_client("kimi-k2.6")

        result = client._apply_param_omissions(
            {"model": "kimi-k2.6", "temperature": 0.2}
        )

        assert result == {"model": "kimi-k2.6", "temperature": 0.2}


class TestMoonshotStructuredOutput:
    def test_k3_resolves_to_native_json_schema(self):
        plan = _stub_client("kimi-k3").resolve_structured_output_plan(ExampleResponse)

        assert plan.mode == "json_schema"
        assert plan.strategy == "moonshot_chat_json_schema"
        assert plan.native_schema_enforced is True

    def test_k2_7_code_resolves_to_native_json_schema(self):
        plan = _stub_client("kimi-k2.7-code").resolve_structured_output_plan(
            ExampleResponse
        )

        assert plan.mode == "json_schema"
        assert plan.native_schema_enforced is True

    def test_k2_6_demotes_to_json_object_with_a_prompt_suffix(self):
        """Moonshot documents json_schema as unstable here, so do not promise it."""
        plan = _stub_client("kimi-k2.6").resolve_structured_output_plan(ExampleResponse)

        assert plan.mode == "json_object"
        assert plan.strategy == "moonshot_json_object"
        assert plan.native_schema_enforced is False
        assert plan.response_format == {"type": "json_object"}
        assert plan.prompt_suffix and "JSON Schema" in plan.prompt_suffix

    def test_k2_6_raises_under_native_required(self):
        with pytest.raises(ValueError, match="unreliable"):
            _stub_client("kimi-k2.6").resolve_structured_output_plan(
                ExampleResponse, structured_policy="native_required"
            )

    def test_k3_carries_the_schema_and_strict_flag(self):
        spec = coerce_structured_output_spec(ExampleResponse)
        plan = _stub_client("kimi-k3")._resolve_structured_output_plan(
            spec, structured_policy="best_effort"
        )

        assert plan.response_format["strict"] is True
        assert "answer" in str(plan.response_format["schema"])


class TestMoonshotHasNoBatchSurface:
    """Sync only, by contract: Moonshot's batch API excludes kimi-k3."""

    def test_client_batch_methods_raise(self):
        client = _stub_client()

        with pytest.raises(NotImplementedError):
            client.build_batch_request(request_id="r", input="q")
        with pytest.raises(NotImplementedError):
            client.submit_batch([])
        with pytest.raises(NotImplementedError):
            client.poll_batch("batch_1")
        with pytest.raises(NotImplementedError):
            list(client.retrieve_batch_results("batch_1"))

    def test_batch_submit_is_refused_up_front(self):
        from prkit.batch import batch_submit_supported

        assert batch_submit_supported(_stub_client()) is False
