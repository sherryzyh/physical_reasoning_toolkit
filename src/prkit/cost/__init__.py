"""Cost-meter building block (roadmap N6).

A small, **optional, composable** component a caller wraps around its *own*
provider calls. PRKit does not run the loop; the meter only (a) extracts token
usage from a raw provider response, (b) prices it, and (c) accumulates a running
total with an optional advisory budget.

This is the surviving sliver of the rescoped runner: **no runner, no response
cache, no orchestration**. The package is a leaf — it imports no hub, scorers,
batch lifecycle, or provider SDKs, and makes no network calls.

    meter = CostMeter(budget=20.00)
    client = create_model_client("gpt-5.1")
    for problem in dataset:
        if meter.over_budget:          # caller decides to stop
            break
        text, raw = client.response_with_raw(problem.question)
        meter.record(model=client.model, provider="openai", raw_response=raw)
    print(meter.total_cost, meter.total_usage.output_tokens)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "TokenUsage",
    "ModelPrice",
    "CallRecord",
    "PriceTable",
    "CostMeter",
    "extract_token_usage",
]


@dataclass(frozen=True)
class TokenUsage:
    """Token counts for a single call. ``cached_input`` ⊆ ``input``; ``reasoning`` ⊆ ``output``."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0  # prompt-cache reads (priced lower; subset of input)
    reasoning_tokens: int = 0  # o-family / thinking tokens (billed as output)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass(frozen=True)
class ModelPrice:
    """$/token rates for one model. ``cached_input`` falls back to ``input`` when None."""

    input_per_token: float
    output_per_token: float
    cached_input_per_token: float | None = None
    as_of: str | None = None  # e.g. "2026-06" — rates drift; stamp the source date

    def cost_of(self, usage: TokenUsage) -> float:
        cached_rate = (
            self.cached_input_per_token
            if self.cached_input_per_token is not None
            else self.input_per_token
        )
        uncached_input = max(usage.input_tokens - usage.cached_input_tokens, 0)
        return (
            uncached_input * self.input_per_token
            + usage.cached_input_tokens * cached_rate
            + usage.output_tokens * self.output_per_token
        )


@dataclass(frozen=True)
class CallRecord:
    """One priced call. ``cost`` is None when the price is unknown and ``strict=False``."""

    provider: str
    model: str
    usage: TokenUsage
    cost: float | None


class PriceTable:
    """A ``(provider, model) -> ModelPrice`` lookup; data, not code (rates drift)."""

    def __init__(
        self,
        prices: dict[tuple[str, str], ModelPrice] | None = None,
        *,
        strict: bool = True,
    ) -> None:
        self._prices: dict[tuple[str, str], ModelPrice] = dict(prices or {})
        self._strict = strict

    def price_for(self, provider: str, model: str) -> ModelPrice | None:
        return self._prices.get((provider.lower(), model))

    def cost_of(self, provider: str, model: str, usage: TokenUsage) -> float | None:
        price = self.price_for(provider, model)
        if price is None:
            if self._strict:
                raise KeyError(
                    f"No price for ({provider!r}, {model!r}). Pass a PriceTable with "
                    f"this entry, or use strict=False to record usage with cost=None."
                )
            return None
        return price.cost_of(usage)

    @classmethod
    def default(cls, *, strict: bool = True) -> PriceTable:
        """A seeded, stale-able table for PRKit's factory-supported providers."""
        from prkit.cost.prices import DEFAULT_PRICES  # lazy: avoids an import cycle

        return cls(DEFAULT_PRICES, strict=strict)


def _attr(obj: Any, name: str) -> Any:
    """Tolerant dict-or-object attribute read (mirrors model_clients' ``_block_attr``)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _int(value: Any) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _chat_completions_usage(raw_response: Any) -> TokenUsage:
    """Read token counts off an OpenAI **Chat Completions** shaped response.

    Distinct from the Responses API shape: the counts are ``prompt_tokens`` /
    ``completion_tokens``, not ``input_tokens`` / ``output_tokens``. Cache reads
    are reported two ways in this family — a nested
    ``prompt_tokens_details.cached_tokens`` and DeepSeek's flat
    ``prompt_cache_hit_tokens`` — so both are consulted.
    """
    usage = _attr(raw_response, "usage")
    prompt_details = _attr(usage, "prompt_tokens_details")
    completion_details = _attr(usage, "completion_tokens_details")
    return TokenUsage(
        input_tokens=_int(_attr(usage, "prompt_tokens")),
        output_tokens=_int(_attr(usage, "completion_tokens")),
        cached_input_tokens=(
            _int(_attr(prompt_details, "cached_tokens"))
            or _int(_attr(usage, "prompt_cache_hit_tokens"))
        ),
        reasoning_tokens=_int(_attr(completion_details, "reasoning_tokens")),
    )


def extract_token_usage(provider: str, raw_response: Any) -> TokenUsage:
    """Read token counts off a raw provider response (object or dict). Missing → 0."""
    p = (provider or "").lower()
    if p == "openai":
        usage = _attr(raw_response, "usage")
        input_details = _attr(usage, "input_tokens_details")
        output_details = _attr(usage, "output_tokens_details")
        return TokenUsage(
            input_tokens=_int(_attr(usage, "input_tokens")),
            output_tokens=_int(_attr(usage, "output_tokens")),
            cached_input_tokens=_int(_attr(input_details, "cached_tokens")),
            reasoning_tokens=_int(_attr(output_details, "reasoning_tokens")),
        )
    if p == "anthropic":
        usage = _attr(raw_response, "usage")
        return TokenUsage(
            input_tokens=_int(_attr(usage, "input_tokens")),
            output_tokens=_int(_attr(usage, "output_tokens")),
            cached_input_tokens=_int(_attr(usage, "cache_read_input_tokens")),
        )
    if p in ("google", "gemini"):
        usage = _attr(raw_response, "usage_metadata")
        candidates = _int(_attr(usage, "candidates_token_count"))
        thoughts = _int(_attr(usage, "thoughts_token_count"))
        return TokenUsage(
            input_tokens=_int(_attr(usage, "prompt_token_count")),
            output_tokens=candidates + thoughts,  # thoughts billed as output
            cached_input_tokens=_int(_attr(usage, "cached_content_token_count")),
            reasoning_tokens=thoughts,
        )
    # Everything else: xAI, Moonshot, DeepSeek and DashScope all speak the
    # OpenAI Chat Completions shape, and so will any client registered through
    # ``register_model_client``. Reading it is strictly better than returning
    # zeros for a response that plainly carries the counts; a response that does
    # not carry them still yields zeros, as documented.
    return _chat_completions_usage(raw_response)


class CostMeter:
    """Accumulates token usage + cost across a caller's own loop. Advisory only."""

    def __init__(
        self, *, prices: PriceTable | None = None, budget: float | None = None
    ) -> None:
        self._prices = prices if prices is not None else PriceTable.default()
        self._budget = budget
        self._records: list[CallRecord] = []
        self._total_usage = TokenUsage()
        self._total_cost = 0.0

    def record(self, *, provider: str, model: str, raw_response: Any) -> CallRecord:
        """Extract usage from *raw_response*, price it, and accumulate."""
        usage = extract_token_usage(provider, raw_response)
        return self.add_usage(provider=provider, model=model, usage=usage)

    def add_usage(self, *, provider: str, model: str, usage: TokenUsage) -> CallRecord:
        """Price a pre-extracted :class:`TokenUsage` and accumulate."""
        cost = self._prices.cost_of(provider, model, usage)
        record = CallRecord(provider=provider, model=model, usage=usage, cost=cost)
        self._records.append(record)
        self._total_usage = self._total_usage + usage
        if cost is not None:
            self._total_cost += cost
        return record

    @property
    def records(self) -> tuple[CallRecord, ...]:
        return tuple(self._records)

    @property
    def total_usage(self) -> TokenUsage:
        return self._total_usage

    @property
    def total_cost(self) -> float:
        """Sum of known per-call costs; unknown (None) costs count as 0."""
        return self._total_cost

    @property
    def over_budget(self) -> bool:
        """True when a budget is set and spend has reached it. Check *before* each call."""
        return self._budget is not None and self._total_cost >= self._budget

    def remaining(self) -> float | None:
        return None if self._budget is None else self._budget - self._total_cost

    def would_exceed(self, next_estimate: float) -> bool:
        if self._budget is None:
            return False
        return (self._total_cost + next_estimate) > self._budget
