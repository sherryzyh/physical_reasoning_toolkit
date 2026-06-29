"""Seeded ``$/token`` price table for PRKit's factory-supported providers.

Kept separate from the cost logic so price drift is a one-file data edit. Each
entry is stamped ``as_of`` (rates change — re-verify before relying on a cost
number). Per-token = published $/1M ÷ 1e6. ``cached_input_per_token`` is the
prompt-cache read rate (~0.1× input for all three providers).

Sources (as_of 2026-06):
- Anthropic: the ``claude-api`` skill model/pricing table.
- OpenAI: https://developers.openai.com/api/docs/pricing
- Google Gemini: https://ai.google.dev/gemini-api/docs/pricing

This table is intentionally small and stale-able. A caller can override it by
constructing ``PriceTable(prices=...)`` or extend it via ``DEFAULT_PRICES``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prkit.cost import ModelPrice

_M = 1_000_000.0
_AS_OF = "2026-06"


def _price(input_per_m: float, output_per_m: float) -> ModelPrice:
    """Build a ModelPrice from published $/1M rates; cached input = 0.1× input."""
    from prkit.cost import ModelPrice

    return ModelPrice(
        input_per_token=input_per_m / _M,
        output_per_token=output_per_m / _M,
        cached_input_per_token=(input_per_m * 0.1) / _M,
        as_of=_AS_OF,
    )


# (provider, model) -> ModelPrice. Provider keys are lowercase (PriceTable
# lowercases the provider on lookup); model strings match what the factory submits.
DEFAULT_PRICES: dict[tuple[str, str], ModelPrice] = {
    # --- Anthropic (claude-api skill, 2026-06) ---
    ("anthropic", "claude-opus-4-8"): _price(5.00, 25.00),
    ("anthropic", "claude-opus-4-7"): _price(5.00, 25.00),
    ("anthropic", "claude-opus-4-6"): _price(5.00, 25.00),
    ("anthropic", "claude-sonnet-4-6"): _price(3.00, 15.00),
    ("anthropic", "claude-haiku-4-5"): _price(1.00, 5.00),
    ("anthropic", "claude-fable-5"): _price(10.00, 50.00),
    # --- OpenAI (developers.openai.com/api/docs/pricing, 2026-06) ---
    ("openai", "gpt-5.5"): _price(5.00, 30.00),
    ("openai", "gpt-5.4"): _price(2.50, 15.00),
    ("openai", "gpt-5.1"): _price(1.25, 10.00),
    ("openai", "gpt-5-mini"): _price(0.25, 2.00),
    ("openai", "gpt-5-nano"): _price(0.05, 0.40),
    # --- Google Gemini (ai.google.dev/gemini-api/docs/pricing, 2026-06; <200K context) ---
    ("google", "gemini-3.5-flash"): _price(1.50, 9.00),
    ("google", "gemini-3-pro"): _price(2.00, 12.00),
    ("google", "gemini-3-flash"): _price(0.50, 3.00),
    ("google", "gemini-3.1-flash-lite"): _price(0.25, 1.50),
}
