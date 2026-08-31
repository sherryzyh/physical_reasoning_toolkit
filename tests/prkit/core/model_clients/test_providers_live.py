"""Live, gated checks for the provider surfaces the offline suite cannot see.

Every other test in this directory patches the provider SDK, so it validates the
dict prkit *builds* and never the request a provider *accepts*. That gap has hidden
real defects more than once:

- xAI vision was sent as a Responses-API ``input_image`` block on a
  chat-completions endpoint. The offline suite was green; the API returned 422
  and the body never reached a model.
- DashScope structured output failed two ways at once — a 400 unless the word
  "json" appeared in the messages, and a ``json_schema`` response format that
  was accepted and then silently ignored, so prkit recorded native enforcement
  it never received.

Neither is observable through a mock. These tests cost a handful of tiny
requests and are the cheapest way to notice the next one.

These are **opt-in twice over**: ``PRKIT_LIVE_PROVIDER_TESTS=1`` must be set,
*and* that provider's key must be present. Key presence alone is not a safe
gate here — ``BaseModelClient.__init__`` calls ``load_project_dotenv``, so the
moment any other test constructs a client the whole process's environment gains
every key in ``.env``, and a gate reading only the key would arm itself
mid-suite and start spending money on a plain ``pytest`` run.

Run explicitly with::

    PRKIT_LIVE_PROVIDER_TESTS=1 .venv/bin/pytest \
        tests/prkit/core/model_clients/test_providers_live.py -m integration -v

Models come from the provider matrix in DEVELOPER.md and are overridable with
``PRKIT_LIVE_<PROVIDER>_MODEL``.
"""

from __future__ import annotations

import base64
import io
import json
import os

import pytest
from pydantic import BaseModel

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class _Answer(BaseModel):
    """Deliberately minimal: two required scalars, no nesting, no free-form dict.

    A schema every provider should manage, so a failure indicts prkit's request
    rather than the model's capability.
    """

    number: int
    unit: str


_PROMPT = "How many meters are in one kilometre? Give the number and the unit."

# provider -> (default model, key env var, output cap). The cap is generous for
# Gemini because thinking tokens are billed against it and a tight cap returns
# an empty response rather than an error.
_PROVIDERS = {
    "openai": ("gpt-5.4-mini", "OPENAI_API_KEY", 2048),
    "anthropic": ("claude-sonnet-4-6", "ANTHROPIC_API_KEY", 2048),
    "gemini": ("gemini-2.5-pro", "GEMINI_API_KEY", 4096),
    "xai": ("grok-4.6", "XAI_API_KEY", 2048),
    "dashscope": ("qwen3.6-plus", "DASHSCOPE_API_KEY", 2048),
    "deepseek": ("deepseek-v4-flash", "DEEPSEEK_API_KEY", 2048),
    "moonshot": ("kimi-k3", "MOONSHOT_API_KEY", 2048),
}


def _model_for(provider: str) -> str:
    default, _, _ = _PROVIDERS[provider]
    return os.environ.get(f"PRKIT_LIVE_{provider.upper()}_MODEL", default)


#: Explicit opt-in. See the module docstring: a key-presence check alone is
#: armed by any other test loading the project's ``.env``.
_OPT_IN_ENV_VAR = "PRKIT_LIVE_PROVIDER_TESTS"


def _requires(provider: str) -> None:
    if not os.environ.get(_OPT_IN_ENV_VAR):
        pytest.skip(f"{_OPT_IN_ENV_VAR} not set (these make billable requests)")
    _, env_var, _ = _PROVIDERS[provider]
    if not os.environ.get(env_var):
        pytest.skip(f"{env_var} not set")


@pytest.mark.parametrize("provider", sorted(_PROVIDERS))
def test_structured_output_round_trips_on_the_wire(provider: str) -> None:
    """The schema reaches the provider and a conforming object comes back.

    Asserts the parsed result, not merely that a response arrived: a provider
    that accepts the schema and ignores it still returns text, and that is the
    failure mode worth catching.
    """
    _requires(provider)
    from prkit.core.model_clients import create_model_client

    _, _, cap = _PROVIDERS[provider]
    client = create_model_client(_model_for(provider))

    result = client.parse(_PROMPT, response_format=_Answer, max_output_tokens=cap)

    assert result.parsed is not None, (
        f"{provider} plan={result.structured_output_strategy} "
        f"native={result.native_schema_enforced} "
        f"error={result.validation_error} raw={(result.raw_text or '')[:120]!r}"
    )
    assert result.parsed.number == 1000
    assert result.parsed.unit


@pytest.mark.parametrize("provider", sorted(_PROVIDERS))
def test_native_enforcement_claims_are_true(provider: str) -> None:
    """A provider prkit calls schema-enforcing must actually honour the schema.

    ``native_schema_enforced`` gates the ``native_required`` raise and is written
    into saved artifacts, so a provider that accepts a schema and ignores it must
    not be recorded as enforcing one. DashScope did exactly that.
    """
    _requires(provider)
    from prkit.core.model_clients import create_model_client

    _, _, cap = _PROVIDERS[provider]
    client = create_model_client(_model_for(provider))
    plan = client.resolve_structured_output_plan(_Answer)
    if not plan.native_schema_enforced:
        pytest.skip(f"{provider} does not claim native enforcement")

    result = client.parse(_PROMPT, response_format=_Answer, max_output_tokens=cap)

    assert result.raw_text is not None
    payload = json.loads(result.raw_text)
    assert sorted(payload) == ["number", "unit"], (
        f"{provider} claims native_schema_enforced but returned {sorted(payload)}; "
        f"strategy={plan.strategy}"
    )


def test_deepseek_still_rejects_a_native_json_schema() -> None:
    """Sentinel for ``supports_response_format_json_schema = False`` on DeepSeek.

    V4 answers a ``json_schema`` response format with "This response_format type
    is unavailable now", so the flag is correct rather than understating the
    provider. If DeepSeek ever ships it, this test fails and says to flip the
    flag — which is the only signal that would otherwise arrive.
    """
    _requires("deepseek")
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"
    )

    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - SDK error class varies
        client.chat.completions.create(
            model=_model_for("deepseek"),
            messages=[{"role": "user", "content": _PROMPT}],
            max_tokens=256,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "Answer",
                    "strict": True,
                    "schema": _Answer.model_json_schema(),
                },
            },
        )

    assert "response_format" in str(excinfo.value), (
        "DeepSeek may now accept a native json_schema; if so, flip "
        "DeepseekModel.supports_response_format_json_schema to True. "
        f"Got: {excinfo.value}"
    )


def test_xai_vision_block_shape_is_accepted() -> None:
    """xAI's documented ``input_image`` block belongs to its Responses API.

    Sending it to chat-completions fails deserialization with a 422 before the
    request reaches a model. This asserts the inherited OpenAI nested form is
    what actually works.
    """
    _requires("xai")
    from PIL import Image

    from prkit.core.model_clients.xai import XAIModel

    buffer = io.BytesIO()
    Image.new("RGB", (48, 48), (220, 20, 20)).save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    text = XAIModel(_model_for("xai")).response(
        "What colour is this image? One word.",
        image_paths=[data_url],
        max_output_tokens=512,
    )

    assert "red" in text.lower(), f"unexpected answer: {text[:80]!r}"
