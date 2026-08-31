"""Tests for the N6 cost-meter (``prkit.cost``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from prkit.cost import (
    CallRecord,
    CostMeter,
    ModelPrice,
    PriceTable,
    TokenUsage,
    extract_token_usage,
)


def _openai_response(
    input_tokens: int, output_tokens: int, cached: int = 0, reasoning: int = 0
) -> SimpleNamespace:
    return SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_tokens_details=SimpleNamespace(cached_tokens=cached),
            output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        )
    )


def _anthropic_response(
    input_tokens: int, output_tokens: int, cache_read: int = 0
) -> SimpleNamespace:
    return SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
        )
    )


def _chat_completions_response() -> dict:
    """The OpenAI Chat Completions usage shape every compatible provider emits."""
    return {
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 40,
            "prompt_tokens_details": {"cached_tokens": 100},
            "completion_tokens_details": {"reasoning_tokens": 25},
        }
    }


def _gemini_response(
    prompt: int, candidates: int, cached: int = 0, thoughts: int = 0
) -> SimpleNamespace:
    return SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt,
            candidates_token_count=candidates,
            cached_content_token_count=cached,
            thoughts_token_count=thoughts,
        )
    )


class TestTokenUsage:
    def test_total_and_add(self) -> None:
        a = TokenUsage(input_tokens=10, output_tokens=5, cached_input_tokens=4)
        b = TokenUsage(input_tokens=3, output_tokens=7, reasoning_tokens=2)
        assert a.total_tokens == 15
        summed = a + b
        assert summed.input_tokens == 13
        assert summed.output_tokens == 12
        assert summed.cached_input_tokens == 4
        assert summed.reasoning_tokens == 2


class TestModelPrice:
    def test_cost_with_cached_fallback_to_input(self) -> None:
        price = ModelPrice(input_per_token=5e-6, output_per_token=25e-6)
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        assert price.cost_of(usage) == pytest.approx(30.0)

    def test_cached_priced_lower(self) -> None:
        price = ModelPrice(
            input_per_token=5e-6, output_per_token=25e-6, cached_input_per_token=0.5e-6
        )
        # 1M input of which 400k cached, 0 output.
        usage = TokenUsage(
            input_tokens=1_000_000, output_tokens=0, cached_input_tokens=400_000
        )
        expected = 600_000 * 5e-6 + 400_000 * 0.5e-6
        assert price.cost_of(usage) == pytest.approx(expected)


class TestPriceTable:
    def test_strict_miss_raises(self) -> None:
        table = PriceTable(strict=True)
        with pytest.raises(KeyError, match="No price"):
            table.cost_of("openai", "mystery-model", TokenUsage(input_tokens=10))

    def test_lenient_miss_returns_none(self) -> None:
        table = PriceTable(strict=False)
        assert (
            table.cost_of("openai", "mystery-model", TokenUsage(input_tokens=10))
            is None
        )

    def test_default_table_prices_known_model(self) -> None:
        table = PriceTable.default()
        price = table.price_for("anthropic", "claude-opus-4-8")
        assert price is not None
        assert price.input_per_token == pytest.approx(5e-6)
        assert price.output_per_token == pytest.approx(25e-6)
        assert price.as_of == "2026-06"

    def test_provider_lookup_is_case_insensitive(self) -> None:
        table = PriceTable.default()
        assert table.price_for("OpenAI", "gpt-5.1") is not None


class TestNewProviderPrices:
    def test_xai_and_moonshot_are_priced(self) -> None:
        table = PriceTable.default()

        assert table.price_for("xai", "grok-4.6") is not None
        assert table.price_for("moonshot", "kimi-k3") is not None

    def test_published_cache_rates_are_not_assumed_to_be_a_tenth(self) -> None:
        """xAI publishes its cache rate; it is a quarter of input, not a tenth."""
        price = PriceTable.default().price_for("xai", "grok-4.6")

        assert price is not None
        assert price.cached_input_per_token == pytest.approx(0.50 / 1_000_000)
        assert price.input_per_token == pytest.approx(2.00 / 1_000_000)

    def test_unknown_model_still_raises_rather_than_reporting_zero(self) -> None:
        """A guessed price would be worse than a loud miss."""
        with pytest.raises(KeyError):
            PriceTable.default().cost_of(
                "deepseek", "deepseek-not-a-model", TokenUsage(input_tokens=10)
            )

    def test_deepseek_and_dashscope_are_priced(self) -> None:
        table = PriceTable.default()

        assert table.price_for("deepseek", "deepseek-v4-flash") is not None
        assert table.price_for("deepseek", "deepseek-v4-pro") is not None
        assert table.price_for("dashscope", "qwen3.6-plus") is not None

    def test_retired_deepseek_names_are_not_priced(self) -> None:
        """deepseek-chat / deepseek-reasoner were discontinued 2026-07-24.

        Pricing a name the provider no longer serves would put a plausible
        number on a call that cannot succeed.
        """
        table = PriceTable.default()

        assert table.price_for("deepseek", "deepseek-chat") is None
        assert table.price_for("deepseek", "deepseek-reasoner") is None

    def test_rows_checked_later_carry_their_own_date(self) -> None:
        """A row stamped with a date nobody verified it on is worse than none."""
        table = PriceTable.default()

        assert table.price_for("deepseek", "deepseek-v4-flash").as_of == "2026-08"
        assert table.price_for("openai", "gpt-5.1").as_of == "2026-06"


class TestExtractTokenUsage:
    def test_openai_object_and_dict(self) -> None:
        usage = extract_token_usage(
            "openai", _openai_response(100, 50, cached=20, reasoning=10)
        )
        assert usage == TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cached_input_tokens=20,
            reasoning_tokens=10,
        )
        # dict-shaped (batch lines)
        as_dict = {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "input_tokens_details": {"cached_tokens": 20},
                "output_tokens_details": {"reasoning_tokens": 10},
            }
        }
        assert extract_token_usage("openai", as_dict).cached_input_tokens == 20

    def test_anthropic(self) -> None:
        usage = extract_token_usage(
            "anthropic", _anthropic_response(200, 80, cache_read=50)
        )
        assert usage.input_tokens == 200
        assert usage.output_tokens == 80
        assert usage.cached_input_tokens == 50

    def test_gemini_folds_thoughts_into_output(self) -> None:
        usage = extract_token_usage(
            "google", _gemini_response(300, 100, cached=40, thoughts=25)
        )
        assert usage.input_tokens == 300
        assert usage.output_tokens == 125  # candidates + thoughts
        assert usage.reasoning_tokens == 25
        assert usage.cached_input_tokens == 40

    def test_unrecognizable_response_returns_zeros(self) -> None:
        assert extract_token_usage("mystery", object()) == TokenUsage()

    @pytest.mark.parametrize(
        "provider", ["xai", "moonshot", "deepseek", "dashscope", "some-custom-client"]
    )
    def test_chat_completions_providers_report_real_counts(self, provider: str) -> None:
        """These returned zeros silently before, which read as a free call."""
        usage = extract_token_usage(provider, _chat_completions_response())

        assert usage.input_tokens == 120
        assert usage.output_tokens == 40
        assert usage.cached_input_tokens == 100
        assert usage.reasoning_tokens == 25

    def test_deepseek_flat_cache_hit_field_is_read(self) -> None:
        """DeepSeek reports cache reads flat, not under prompt_tokens_details."""
        response = {
            "usage": {
                "prompt_tokens": 90,
                "completion_tokens": 10,
                "prompt_cache_hit_tokens": 64,
            }
        }

        assert extract_token_usage("deepseek", response).cached_input_tokens == 64

    def test_chat_completions_shape_does_not_disturb_the_named_providers(self) -> None:
        """A Chat Completions body must not be read for a Responses-API provider."""
        assert (
            extract_token_usage("openai", _chat_completions_response()) == TokenUsage()
        )

    def test_missing_fields_never_raise(self) -> None:
        assert extract_token_usage("openai", SimpleNamespace()) == TokenUsage()
        assert extract_token_usage("anthropic", None) == TokenUsage()


class TestCostMeter:
    def test_records_and_totals(self) -> None:
        meter = CostMeter()
        rec = meter.record(
            provider="anthropic",
            model="claude-opus-4-8",
            raw_response=_anthropic_response(1_000_000, 1_000_000),
        )
        assert isinstance(rec, CallRecord)
        assert rec.cost == pytest.approx(30.0)
        assert meter.total_cost == pytest.approx(30.0)
        assert meter.total_usage.total_tokens == 2_000_000
        assert len(meter.records) == 1

    def test_mixed_providers_accumulate(self) -> None:
        meter = CostMeter()
        meter.record(
            provider="openai",
            model="gpt-5.1",
            raw_response=_openai_response(1_000_000, 0),
        )  # 1.25
        meter.record(
            provider="google",
            model="gemini-3-flash",
            raw_response=_gemini_response(1_000_000, 0),
        )  # 0.50
        assert meter.total_cost == pytest.approx(1.75)
        assert meter.total_usage.input_tokens == 2_000_000

    def test_budget_predicates(self) -> None:
        meter = CostMeter(budget=10.0)
        assert meter.over_budget is False
        assert meter.remaining() == pytest.approx(10.0)
        assert meter.would_exceed(9.0) is False
        assert meter.would_exceed(11.0) is True
        meter.record(
            provider="anthropic",
            model="claude-opus-4-8",
            raw_response=_anthropic_response(0, 1_000_000),
        )  # 25.0
        assert meter.over_budget is True
        assert meter.remaining() == pytest.approx(-15.0)

    def test_no_budget_never_over(self) -> None:
        meter = CostMeter()
        meter.add_usage(
            provider="anthropic",
            model="claude-opus-4-8",
            usage=TokenUsage(output_tokens=10**9),
        )
        assert meter.over_budget is False
        assert meter.remaining() is None
        assert meter.would_exceed(10**9) is False

    def test_unknown_model_lenient_records_none_cost(self) -> None:
        meter = CostMeter(prices=PriceTable(strict=False))
        rec = meter.add_usage(
            provider="openai", model="mystery", usage=TokenUsage(input_tokens=100)
        )
        assert rec.cost is None
        assert meter.total_cost == pytest.approx(0.0)
        assert meter.total_usage.input_tokens == 100
