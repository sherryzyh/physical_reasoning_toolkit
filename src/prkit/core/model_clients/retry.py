"""
Provider-neutral retry configuration for the model-client layer.

prkit writes no retry loop of its own, and should not: every provider SDK ships
one with exponential backoff, jitter and a considered set of retryable status
codes. What differed was whether it was switched *on*.

Before this module, the same workload had a different effective failure rate per
provider — OpenAI and Anthropic retried twice by SDK default, google-genai did
not retry at all unless retry options were passed explicitly, and prkit was
passing ``max_retries=0`` for DashScope. For a toolkit whose purpose is
cross-provider comparison, that is a confound in the measurement, not a tuning
detail: a provider that silently retried transient failures looks more reliable
than one that did not.

So one number is resolved the same way everywhere, and each client hands it to
its own SDK. Two things this deliberately does **not** do:

- **It does not replace SDK backoff.** The SDKs' retry implementations are
  better tested than anything written here would be.
- **It does not classify errors.** Notably, Moonshot returns three
  semantically different 429s, and ``exceeded_current_quota_error`` means the
  account is out of credit, where retrying can never succeed. The OpenAI SDK
  retries every 429 and offers no hook to discriminate, so such a call costs
  the retry budget before failing. That is slow, not wrong, and buying it back
  would mean hand-rolling the whole retry loop.
"""

from __future__ import annotations

import os

#: Retries *after* the initial request. Matches the OpenAI and Anthropic SDK
#: defaults, which is the closest thing to a cross-provider convention.
DEFAULT_MAX_RETRIES = 2

#: Consulted when no provider-specific variable is set.
GLOBAL_MAX_RETRIES_ENV_VAR = "PRKIT_MAX_RETRIES"


def resolve_max_retries(
    provider_env_var: str | None = None,
    *,
    default: int = DEFAULT_MAX_RETRIES,
) -> int:
    """Resolve the retry count, provider-specific variable winning over the global one.

    Returns *default* when neither is set. A negative value is clamped to zero,
    so a stray ``-1`` disables retries rather than meaning something undefined
    to whichever SDK receives it.

    Raises:
        ValueError: If a variable is set to something that is not an integer.
            Silently ignoring it would leave the caller believing they had
            configured a retry policy they had not.
    """
    for name in (provider_env_var, GLOBAL_MAX_RETRIES_ENV_VAR):
        if not name:
            continue
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        try:
            return max(0, int(raw))
        except ValueError as exc:
            raise ValueError(
                f"Environment variable {name} must be an integer; got {raw!r}."
            ) from exc
    return default


def attempts_from_retries(max_retries: int) -> int:
    """Convert a retry count into a total-attempt count.

    OpenAI and Anthropic count retries *after* the initial request; google-genai
    counts attempts *including* it, and treats 0 or 1 as "no retries". Getting
    this off by one gives Gemini one try fewer or more than every other
    provider, which is precisely the asymmetry this module exists to remove.
    """
    return max(1, max_retries + 1)
