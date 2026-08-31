"""Tests for provider-neutral retry configuration.

The point of the module is that one workload sees the same retry behaviour
whichever provider ran it, so the cross-provider test at the bottom is the one
that matters most: it is what would have caught the original asymmetry.
"""

from unittest.mock import MagicMock, patch

import pytest

from prkit.core.model_clients.retry import (
    DEFAULT_MAX_RETRIES,
    attempts_from_retries,
    resolve_max_retries,
)


class TestResolveMaxRetries:
    def test_defaults_when_nothing_is_set(self):
        with patch.dict("os.environ", {}, clear=True):
            assert resolve_max_retries("XAI_MAX_RETRIES") == DEFAULT_MAX_RETRIES

    def test_provider_variable_wins_over_the_global_one(self):
        with patch.dict(
            "os.environ",
            {"XAI_MAX_RETRIES": "5", "PRKIT_MAX_RETRIES": "1"},
            clear=True,
        ):
            assert resolve_max_retries("XAI_MAX_RETRIES") == 5

    def test_global_variable_applies_when_no_provider_one_is_set(self):
        with patch.dict("os.environ", {"PRKIT_MAX_RETRIES": "7"}, clear=True):
            assert resolve_max_retries("XAI_MAX_RETRIES") == 7

    def test_zero_disables_retries(self):
        with patch.dict("os.environ", {"PRKIT_MAX_RETRIES": "0"}, clear=True):
            assert resolve_max_retries(None) == 0

    def test_negative_is_clamped_rather_than_passed_to_an_sdk(self):
        with patch.dict("os.environ", {"PRKIT_MAX_RETRIES": "-3"}, clear=True):
            assert resolve_max_retries(None) == 0

    def test_blank_value_falls_through(self):
        with patch.dict(
            "os.environ",
            {"XAI_MAX_RETRIES": "   ", "PRKIT_MAX_RETRIES": "4"},
            clear=True,
        ):
            assert resolve_max_retries("XAI_MAX_RETRIES") == 4

    def test_non_integer_raises_rather_than_being_ignored(self):
        """Silently ignoring it would leave a retry policy that was never applied."""
        with patch.dict("os.environ", {"PRKIT_MAX_RETRIES": "lots"}, clear=True):
            with pytest.raises(ValueError, match="must be an integer"):
                resolve_max_retries(None)


class TestAttemptsFromRetries:
    @pytest.mark.parametrize(("retries", "attempts"), [(0, 1), (1, 2), (2, 3), (5, 6)])
    def test_attempts_include_the_initial_request(self, retries, attempts):
        assert attempts_from_retries(retries) == attempts

    def test_never_returns_zero(self):
        """google-genai treats 0 as 1; returning 0 would be a silent no-op."""
        assert attempts_from_retries(-1) == 1


class TestProvidersAgreeOnRetries:
    """The asymmetry this module exists to remove.

    Before it, the same workload retried twice on OpenAI and Anthropic, not at
    all on Gemini (google-genai does not retry unless retry options are passed)
    and not at all on DashScope (prkit passed 0). A provider that silently
    retried transient failures looked more reliable than one that did not.
    """

    @staticmethod
    def _openai_style_retries(cls_path, factory):
        with (
            patch(cls_path) as mock_cls,
            patch("prkit.core.model_clients.base.load_project_dotenv"),
        ):
            mock_cls.return_value = MagicMock()
            factory()
            return mock_cls.call_args.kwargs["max_retries"]

    def test_every_openai_compatible_provider_uses_the_same_count(self):
        from prkit.core.model_clients.dashscope import DashscopeModel
        from prkit.core.model_clients.deepseek import DeepseekModel
        from prkit.core.model_clients.moonshot import MoonshotModel
        from prkit.core.model_clients.xai import XAIModel

        path = "prkit.core.model_clients.openai_compatible_chat.OpenAI"
        with patch.dict("os.environ", {}, clear=True):
            counts = {
                cls.__name__: self._openai_style_retries(path, lambda c=cls: c("m"))
                for cls in (XAIModel, MoonshotModel, DeepseekModel, DashscopeModel)
            }

        assert set(counts.values()) == {DEFAULT_MAX_RETRIES}, counts

    def test_openai_and_anthropic_use_the_same_count(self):
        from prkit.core.model_clients.anthropic import AnthropicModel
        from prkit.core.model_clients.openai import OpenAIModel

        with patch.dict("os.environ", {"OPENAI_API_KEY": "k"}, clear=True):
            openai_retries = self._openai_style_retries(
                "prkit.core.model_clients.openai.OpenAI",
                lambda: OpenAIModel("gpt-5.4-mini"),
            )
            anthropic_retries = self._openai_style_retries(
                "prkit.core.model_clients.anthropic.Anthropic",
                lambda: AnthropicModel("claude-sonnet-4-6"),
            )

        assert openai_retries == anthropic_retries == DEFAULT_MAX_RETRIES

    def test_gemini_gets_the_equivalent_attempt_count(self):
        """Gemini counts attempts, not retries — the off-by-one this converts."""
        from prkit.core.model_clients.gemini import GeminiModel

        with (
            patch("prkit.core.model_clients.gemini.genai") as mock_genai,
            patch("prkit.core.model_clients.base.load_project_dotenv"),
            patch.dict("os.environ", {"GEMINI_API_KEY": "k"}, clear=True),
        ):
            GeminiModel("gemini-3-pro")
            options = mock_genai.Client.call_args.kwargs["http_options"]

        assert options.retry_options.attempts == attempts_from_retries(
            DEFAULT_MAX_RETRIES
        )

    def test_a_global_override_reaches_every_provider(self):
        from prkit.core.model_clients.moonshot import MoonshotModel
        from prkit.core.model_clients.xai import XAIModel

        path = "prkit.core.model_clients.openai_compatible_chat.OpenAI"
        with patch.dict("os.environ", {"PRKIT_MAX_RETRIES": "4"}, clear=True):
            assert self._openai_style_retries(path, lambda: XAIModel("m")) == 4
            assert self._openai_style_retries(path, lambda: MoonshotModel("m")) == 4
