"""Seeded ``$/token`` price table for PRKit's factory-supported providers.

Kept separate from the cost logic so price drift is a one-file data edit. Each
entry is stamped ``as_of`` (rates change — re-verify before relying on a cost
number). Per-token = published $/1M ÷ 1e6. ``cached_input_per_token`` is the
prompt-cache read rate, ~0.1× input for most providers but published explicitly
by xAI and Moonshot, so those pass it directly.

Sources (as_of 2026-06):
- Anthropic: the ``claude-api`` skill model/pricing table.
- OpenAI: https://developers.openai.com/api/docs/pricing
- Google Gemini: https://ai.google.dev/gemini-api/docs/pricing
- xAI: https://docs.x.ai/developers/pricing (standard tier; xAI bills every
  token in a request at a higher rate once the prompt reaches 200k tokens, and
  this table carries the standard rate only)
- Moonshot: https://platform.kimi.ai/docs/pricing/chat-k3 (input rate is the
  cache-miss rate; the hit rate is the cached rate)

- DeepSeek: https://api-docs.deepseek.com/quick_start/pricing
- DashScope: https://www.alibabacloud.com/help/en/model-studio

Two rows carry a caveat the table cannot express. DeepSeek publishes peak and
off-peak rates a factor of two apart and nothing records call time, so its rows
are the peak (upper-bound) figure. DashScope prices differ by region and these
are the US/Global endpoint, matching ``DASHSCOPE_REGION``'s default.

This table is intentionally small and stale-able. A caller can override it by
constructing ``PriceTable(prices=...)`` or extend it via ``DEFAULT_PRICES``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prkit.cost import ModelPrice

_M = 1_000_000.0
_AS_OF = "2026-06"


def _price(
    input_per_m: float,
    output_per_m: float,
    cached_per_m: float | None = None,
    *,
    as_of: str = _AS_OF,
) -> ModelPrice:
    """Build a ModelPrice from published $/1M rates.

    Cached input defaults to 0.1× input, the common ratio. Pass *cached_per_m*
    for providers that publish their cache-read rate explicitly, and *as_of*
    when a row was checked on a different date than the rest of the table — a
    row stamped with a date nobody verified it on is worse than no stamp.
    """
    from prkit.cost import ModelPrice

    return ModelPrice(
        input_per_token=input_per_m / _M,
        output_per_token=output_per_m / _M,
        cached_input_per_token=(
            cached_per_m / _M if cached_per_m is not None else (input_per_m * 0.1) / _M
        ),
        as_of=as_of,
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
    # --- xAI (docs.x.ai/developers/pricing, 2026-06; standard tier) ---
    ("xai", "grok-4.6"): _price(2.00, 6.00, 0.50),
    ("xai", "grok-4.5"): _price(2.00, 6.00, 0.30),
    ("xai", "grok-4.3"): _price(1.25, 2.50, 0.20),
    # --- Moonshot (platform.kimi.ai/docs/pricing, 2026-06; input = cache miss) ---
    ("moonshot", "kimi-k3"): _price(3.00, 15.00, 0.30),
    ("moonshot", "kimi-k2.6"): _price(0.95, 4.00, 0.16),
    # --- DeepSeek (api-docs.deepseek.com/quick_start/pricing, 2026-08) ---
    # PEAK rates. DeepSeek halves them off-peak (peak is 01:00-04:00 and
    # 06:00-10:00 UTC, Mon-Fri) and this table has one rate per model, so a cost
    # here is an upper bound. Seeded peak deliberately: overstating is
    # recoverable, and a batch scheduled to run overnight from the US lands
    # squarely inside the peak window. Input is the cache-miss rate.
    ("deepseek", "deepseek-v4-flash"): _price(0.44, 1.32, 0.014, as_of="2026-08"),
    ("deepseek", "deepseek-v4-pro"): _price(1.32, 3.96, 0.044, as_of="2026-08"),
    # --- DashScope (alibabacloud.com/help/en/model-studio, 2026-08) ---
    # US/Global endpoint, which is what DASHSCOPE_REGION defaults to and so what
    # an unconfigured run actually bills; the Singapore endpoint is ~1.8x these
    # and prices.py has no region dimension. Input tier 0-256K.
    ("dashscope", "qwen3.6-plus"): _price(0.276, 1.651, 0.0552, as_of="2026-08"),
}
