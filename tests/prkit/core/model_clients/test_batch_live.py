"""Live, gated smoke test for the Gemini file-based batch path.

This is the check the offline suite *cannot* make: it submits a real batch and
confirms each answer lands under the **correct** ``custom_id`` (identity), not
merely that some text came back. It is the difference between a working
file-based correlation and a silent positional misalignment.

Opt-in twice over, like ``test_providers_live.py``: ``PRKIT_LIVE_PROVIDER_TESTS``
must be set *and* the provider's key present. Key presence alone is not a safe
gate — ``BaseModelClient.__init__`` loads the project ``.env``, so any other test
constructing a client puts every key into the process environment. Run with::

    PRKIT_LIVE_PROVIDER_TESTS=1 .venv/bin/pytest \
        tests/prkit/core/model_clients/test_batch_live.py -m integration -v -s

Polling timeout is configurable via ``PRKIT_BATCH_TIMEOUT_SECONDS`` (default
1800s); batch jobs are asynchronous and may take minutes to complete.
"""

from __future__ import annotations

import json
import os
import time

import pytest
from pydantic import BaseModel

from prkit.core.model_clients.batch_types import BatchItemStatus, BatchState

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_OPT_IN_ENV_VAR = "PRKIT_LIVE_PROVIDER_TESTS"


def _requires(*key_vars: str) -> None:
    """Skip unless live tests are opted into and the provider's key is present."""
    if not os.environ.get(_OPT_IN_ENV_VAR):
        pytest.skip(f"{_OPT_IN_ENV_VAR} not set (these make billable requests)")
    if not any(os.environ.get(var) for var in key_vars):
        pytest.skip(f"none of {', '.join(key_vars)} set")


# request_id -> (prompt, unique expected answer). Distinct arithmetic gives
# deterministic, non-overlapping answers that any model returns reliably, so the
# test validates *correlation* (the right answer under the right id) rather than
# the model's instruction-following. Answers are chosen so none is a substring of
# another, which is what makes the "leak" check below meaningful.
_CASES = {
    "q-alpha": ("What is 40 + 2? Reply with only the number.", "42"),
    "q-bravo": ("What is 50 + 3? Reply with only the number.", "53"),
    "q-charlie": ("What is 60 + 4? Reply with only the number.", "64"),
}


def test_gemini_batch_correlates_each_answer_to_its_request():
    _requires("GEMINI_API_KEY", "GOOGLE_API_KEY")
    from prkit.core.model_clients.gemini import GeminiModel

    client = GeminiModel(os.environ.get("PRKIT_BATCH_MODEL", "gemini-3.5-flash"))

    requests = [
        client.build_batch_request(
            request_id=request_id,
            input=prompt,
            instructions="",
            # gemini-3.5-flash is a thinking model: the budget must cover hidden
            # reasoning tokens (observed ~100-500) plus the short answer.
            max_output_tokens=2048,
            temperature=0.0,
        )
        for request_id, (prompt, _expected) in _CASES.items()
    ]

    batch_id = client.submit_batch(
        requests, metadata={"display_name": "prkit-live-smoke"}
    )
    assert batch_id

    timeout = float(os.environ.get("PRKIT_BATCH_TIMEOUT_SECONDS", "1800"))
    deadline = time.monotonic() + timeout
    status = client.poll_batch(batch_id)
    while not status.is_terminal:
        if time.monotonic() > deadline:
            pytest.fail(
                f"Batch {batch_id} not terminal after {timeout}s (last={status.raw_status})"
            )
        time.sleep(20)
        status = client.poll_batch(batch_id)

    results = {r.custom_id: r for r in client.retrieve_batch_results(batch_id)}

    # Every request must come back, keyed by exactly the id we submitted.
    assert set(results) == set(
        _CASES
    ), f"custom_id mismatch: {set(results)} != {set(_CASES)}"

    for request_id, (_prompt, expected) in _CASES.items():
        result = results[request_id]
        assert (
            result.status is BatchItemStatus.SUCCEEDED
        ), f"{request_id}: {result.error}"
        text = (result.text or "").upper()
        # The answer for this id must contain ITS word and none of the others' —
        # this is what catches a positional misalignment.
        assert (
            expected in text
        ), f"{request_id}: expected {expected!r} in {result.text!r}"
        for other_id, (_p, other_word) in _CASES.items():
            if other_id != request_id:
                assert (
                    other_word not in text
                ), f"{request_id} answer leaked {other_word!r}: {result.text!r}"


# --------------------------------------------------------------------------- #
# Structured-output batch submission                                          #
# --------------------------------------------------------------------------- #
# The batch lane can now request a provider-enforced schema, and the risk in
# that lives entirely at submit time: whether the provider accepts a request
# body prkit assembled. Gemini's is the sharpest case — it hand-builds
# snake_case wire keys into a dict prkit serializes itself, so no SDK type
# checks it and only the backend can refuse it.
#
# These submit and then cancel rather than waiting. Acceptance is the assertion;
# completion would cost minutes and money for nothing extra.
_BATCH_PROVIDERS = {
    "openai": ("gpt-5.4-mini", ("OPENAI_API_KEY",)),
    "anthropic": ("claude-sonnet-4-6", ("ANTHROPIC_API_KEY",)),
    "gemini": ("gemini-2.5-pro", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
}


class _BatchAnswer(BaseModel):
    number: int
    unit: str


def _cancel_quietly(client, batch_id: str) -> None:
    """Best-effort cancel so an accepted probe does not run to completion."""
    for attempt in (
        lambda: client.client.batches.cancel(batch_id),
        lambda: client.client.messages.batches.cancel(batch_id),
        lambda: client.genai_client.batches.cancel(name=batch_id),
    ):
        try:
            attempt()
            return
        except Exception:  # noqa: BLE001, S110 - cleanup only
            continue


@pytest.mark.parametrize("provider", sorted(_BATCH_PROVIDERS))
def test_structured_batch_submission_is_accepted(provider: str) -> None:
    """The provider accepts a batch line carrying a native schema.

    Offline tests prove prkit builds the dict it intends to. Only the provider
    can say whether that dict is a request — which is the half that has broken
    before, on the synchronous path, while the suite stayed green.
    """
    model, key_vars = _BATCH_PROVIDERS[provider]
    _requires(*key_vars)
    from prkit.core.model_clients import create_model_client

    client = create_model_client(
        os.environ.get(f"PRKIT_LIVE_{provider.upper()}_MODEL", model)
    )
    plan = client.resolve_structured_output_plan(_BatchAnswer)

    requests = [
        client.build_batch_request(
            request_id=f"structured-{index}",
            input="How many meters are in one kilometre?",
            instructions="",
            max_output_tokens=4096,
            response_format=_BatchAnswer,
        )
        for index in range(2)
    ]

    batch_id = client.submit_batch(
        requests, metadata={"display_name": "prkit-live-structured"}
    )
    try:
        assert batch_id, (
            f"{provider} returned no batch id for a "
            f"{plan.strategy} request (native={plan.native_schema_enforced})"
        )
        status = client.poll_batch(batch_id)
        assert status.batch_id == batch_id
        assert status.provider == client.provider
    finally:
        _cancel_quietly(client, batch_id)


@pytest.mark.parametrize("provider", sorted(_BATCH_PROVIDERS))
def test_free_text_batch_submission_is_still_accepted(provider: str) -> None:
    """The default path must stay byte-identical in effect, not just in shape."""
    model, key_vars = _BATCH_PROVIDERS[provider]
    _requires(*key_vars)
    from prkit.core.model_clients import create_model_client

    client = create_model_client(
        os.environ.get(f"PRKIT_LIVE_{provider.upper()}_MODEL", model)
    )

    batch_id = client.submit_batch(
        [
            client.build_batch_request(
                request_id="free-text-0",
                input="What is 40 + 2? Reply with only the number.",
                instructions="",
                max_output_tokens=2048,
            )
        ],
        metadata={"display_name": "prkit-live-free-text"},
    )
    try:
        assert batch_id
    finally:
        _cancel_quietly(client, batch_id)


# --------------------------------------------------------------------------- #
# Structured-output read-back                                                 #
# --------------------------------------------------------------------------- #
# The last unverified half of the batch lane. Submission proves the provider
# accepts the request; this proves prkit can read a *structured* result back
# out. Each provider's parser was written against free text and is exercised
# offline only through mocks that return whatever the test author typed — so a
# renamed field or a payload arriving in an unexpected block would surface here
# and nowhere else.
#
# Gated separately from the rest: this one waits for real jobs to finish, which
# is minutes at best, so it should not fire on a routine live run.
_READBACK_ENV_VAR = "PRKIT_LIVE_BATCH_READBACK"


@pytest.mark.parametrize("provider", sorted(_BATCH_PROVIDERS))
def test_structured_batch_results_read_back_as_valid_objects(provider: str) -> None:
    """A structured batch round-trips: submitted, completed, parsed, correlated.

    Asserts the payload validates against the schema rather than that some text
    came back. An empty string tagged SUCCEEDED is the failure this exists to
    catch, and it is indistinguishable from success by any weaker assertion.
    """
    model, key_vars = _BATCH_PROVIDERS[provider]
    _requires(*key_vars)
    if not os.environ.get(_READBACK_ENV_VAR):
        pytest.skip(f"{_READBACK_ENV_VAR} not set (this one waits for real jobs)")
    from prkit.core.model_clients import create_model_client

    client = create_model_client(
        os.environ.get(f"PRKIT_LIVE_{provider.upper()}_MODEL", model)
    )
    expected = {"rb-alpha": 1000, "rb-bravo": 100}
    requests = [
        client.build_batch_request(
            request_id=request_id,
            input=(
                "How many meters are in one kilometre?"
                if request_id == "rb-alpha"
                else "How many centimetres are in one metre?"
            ),
            instructions="",
            max_output_tokens=4096,
            response_format=_BatchAnswer,
        )
        for request_id in expected
    ]

    batch_id = client.submit_batch(
        requests, metadata={"display_name": "prkit-live-readback"}
    )
    assert batch_id

    timeout = float(os.environ.get("PRKIT_BATCH_TIMEOUT_SECONDS", "1800"))
    deadline = time.monotonic() + timeout
    status = client.poll_batch(batch_id)
    while not status.is_terminal:
        if time.monotonic() > deadline:
            _cancel_quietly(client, batch_id)
            pytest.fail(
                f"{provider} batch {batch_id} not terminal after {timeout}s "
                f"(last={status.raw_status})"
            )
        time.sleep(20)
        status = client.poll_batch(batch_id)

    assert status.state is BatchState.COMPLETED, (
        f"{provider} batch {batch_id} ended {status.state} "
        f"(raw={status.raw_status!r}, counts={status.counts}). "
        "A batch that never ran says nothing about prkit's read-back path — "
        "check the provider account before reading this as a code defect."
    )

    results = {r.custom_id: r for r in client.retrieve_batch_results(batch_id)}

    assert set(results) == set(
        expected
    ), f"{provider} returned ids {sorted(results)}, expected {sorted(expected)}"
    for request_id, want in expected.items():
        result = results[request_id]
        assert (
            result.status is BatchItemStatus.SUCCEEDED
        ), f"{provider}/{request_id}: {result.status} error={result.error}"
        assert result.text, (
            f"{provider}/{request_id} succeeded with empty text — the silent-empty "
            "failure this assertion exists for"
        )
        payload = json.loads(result.text)
        parsed = _BatchAnswer.model_validate(payload)
        assert parsed.number == want, (
            f"{provider}/{request_id} answered {parsed.number}, expected {want} "
            "— a correlation failure, not a wrong answer"
        )
