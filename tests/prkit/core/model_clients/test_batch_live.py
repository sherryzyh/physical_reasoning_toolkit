"""Live, gated smoke test for the Gemini file-based batch path.

This is the check the offline suite *cannot* make: it submits a real batch and
confirms each answer lands under the **correct** ``custom_id`` (identity), not
merely that some text came back. It is the difference between a working
file-based correlation and a silent positional misalignment.

Skipped unless a Gemini API key is present, so default ``pytest`` runs ignore
it. Run explicitly with::

    GEMINI_API_KEY=... .venv/bin/pytest \
        tests/prkit/core/model_clients/test_batch_live.py -m integration -v -s

Polling timeout is configurable via ``PRKIT_BATCH_TIMEOUT_SECONDS`` (default
1800s); batch jobs are asynchronous and may take minutes to complete.
"""

from __future__ import annotations

import os
import time

import pytest

from prkit.core.model_clients.batch_types import BatchItemStatus

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_HAS_KEY = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

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


@pytest.mark.skipif(not _HAS_KEY, reason="No GEMINI_API_KEY/GOOGLE_API_KEY set")
def test_gemini_batch_correlates_each_answer_to_its_request():
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
