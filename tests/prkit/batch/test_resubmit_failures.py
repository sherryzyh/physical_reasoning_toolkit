"""Tests for the finalize half: ``resubmit_failures`` whole-minibatch path (Stage 3/4).

Fully offline. Runs are built with an SDK-patched submit client then driven to
terminal statuses by the duck-typed ``_FetchClient``; resubmit is then driven by
a tiny duck-typed ``_Client`` whose ``submit_batch`` is a ``MagicMock``. Covers
the happy path (requests re-read from the persisted input file + merged
metadata, entries reset to SUBMITTED, ``attempt`` bumped), CANCELLED now
resubmitted (Stage 4 D1) + no-op, the terminal and capability preconditions,
per-item failure + continue, submit-first ordering, the client facade, and the
next-command prompt. The record-drain half of ``resubmit_failures`` is covered in
``test_resubmit_records.py``.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from prkit.batch import (
    FETCHED,
    SUBMIT_ERROR,
    SUBMITTED,
    BatchFetchUnsupportedError,
    BatchNotTerminalError,
    BatchSubmission,
    fetch_batch,
    resubmit_failures,
    submit_batch_physics_reasoning,
)
from prkit.core.domain import PhysicsDataset, PhysicsProblem
from prkit.core.model_clients.batch_types import (
    BatchItemStatus,
    BatchResult,
    BatchState,
)


# --------------------------------------------------------------------------- #
# Offline builders                                                             #
# --------------------------------------------------------------------------- #
def _openai_submit_client(model: str = "gpt-5.1"):
    with patch("prkit.core.model_clients.openai.OpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        from prkit.core.model_clients.openai import OpenAIModel

        return OpenAIModel(model)


def _dataset(n: int):
    problems = [PhysicsProblem(problem_id=f"p{i}", question=f"Q{i}") for i in range(n)]
    return PhysicsDataset(problems, info={"name": "physreason", "version": "1.0"})


class _FetchClient:
    def __init__(self, provider="openai", *, states=None, results=None):
        self.provider = provider
        self.model = "model-x"
        self._states = states or {}
        self._results = results or {}

    def poll_batch(self, batch_id: str):
        from prkit.core.model_clients.batch_types import BatchStatus

        seq = self._states[batch_id]
        state = seq.pop(0) if len(seq) > 1 else seq[0]
        return BatchStatus(
            batch_id=batch_id,
            state=state,
            provider=self.provider,
            raw_status=str(state),
            counts={"total": 1},
        )

    def retrieve_batch_results(self, batch_id: str):
        return iter(self._results.get(batch_id, []))


class _Client:
    """A duck-typed resubmit client: only ``provider`` + ``submit_batch`` are used."""

    def __init__(self, provider="openai"):
        self.provider = provider
        self.model = "model-x"


_STATE_MAP = {
    "fetched": BatchState.COMPLETED,
    "failed": BatchState.FAILED,
    "expired": BatchState.EXPIRED,
    "cancelled": BatchState.CANCELLED,
}


def _build_terminal_run(tmp_path, *, statuses, run_name="run"):
    """One minibatch per status, driven to a terminal ledger (1 problem each)."""
    n = len(statuses)
    bids = [f"b{i}" for i in range(n)]
    client = _openai_submit_client()
    client.submit_batch = MagicMock(side_effect=bids)
    run_dir = submit_batch_physics_reasoning(
        client, _dataset(n), output_dir=tmp_path, run_name=run_name, minibatch_size=1
    )
    sub0 = BatchSubmission.load(run_dir)
    cid_of = {mb["batch_id"]: next(iter(mb["id_map"])) for mb in sub0.minibatches}
    states = {bids[i]: [_STATE_MAP[s]] for i, s in enumerate(statuses)}
    results = {
        bids[i]: [
            BatchResult(cid_of[bids[i]], BatchItemStatus.SUCCEEDED, text=f"ans-{i}")
        ]
        for i, s in enumerate(statuses)
        if s == "fetched"
    }
    fetch_batch(_FetchClient(states=states, results=results), run_dir, progress=False)
    return run_dir


def _build_running_run(tmp_path):
    """A non-terminal run (one minibatch left RUNNING)."""
    client = _openai_submit_client()
    client.submit_batch = MagicMock(side_effect=["b0"])
    run_dir = submit_batch_physics_reasoning(
        client, _dataset(1), output_dir=tmp_path, run_name="run", minibatch_size=1
    )
    fetch_batch(
        _FetchClient(states={"b0": [BatchState.IN_PROGRESS]}),
        run_dir,
        progress=False,
    )
    return run_dir


# --------------------------------------------------------------------------- #
class TestHappyPath:
    def test_resubmits_failed_and_expired_resetting_each_entry(self, tmp_path):
        run_dir = _build_terminal_run(
            tmp_path, statuses=["fetched", "failed", "expired"]
        )
        sub0 = BatchSubmission.load(run_dir)
        client = _Client()
        client.submit_batch = MagicMock(side_effect=["new1", "new2"])

        sub = resubmit_failures(client, run_dir)

        assert client.submit_batch.call_count == 2  # FAILED + EXPIRED only
        for call in client.submit_batch.call_args_list:
            requests = call.args[0]
            # Requests are parsed from the persisted input file (real request dicts).
            assert requests and all("custom_id" in r for r in requests)
            assert call.kwargs["metadata"] == {"display_name": sub0.display_name}

        mb1, mb2 = sub.minibatches[1], sub.minibatches[2]
        assert mb1["status"] == SUBMITTED and mb1["batch_id"] == "new1"
        assert mb2["status"] == SUBMITTED and mb2["batch_id"] == "new2"
        for mb in (mb1, mb2):
            assert mb["error"] is None
            assert mb["output_path"] is None
            assert mb["fetched_at"] is None
            assert mb["counts"] == {}
        assert sub.minibatches[0]["status"] == FETCHED  # the succeeded one is untouched
        # Whole-minibatch resubmit bumps each entry's attempt (submit => 1, +1 here).
        assert mb1["attempt"] == 2 and mb2["attempt"] == 2
        # Persisted to disk.
        assert BatchSubmission.load(run_dir).minibatches[1]["batch_id"] == "new1"


class TestCancelledResubmitted:
    """Stage 4 D1: CANCELLED is now resubmitted like any other failure (no dead-end)."""

    def test_cancelled_minibatch_is_resubmitted(self, tmp_path):
        run_dir = _build_terminal_run(
            tmp_path, statuses=["fetched", "cancelled", "failed"]
        )
        client = _Client()
        client.submit_batch = MagicMock(side_effect=["new1", "new2"])
        sub = resubmit_failures(client, run_dir)
        assert client.submit_batch.call_count == 2  # CANCELLED + FAILED both
        assert sub.minibatches[1]["status"] == SUBMITTED  # cancelled re-driven
        assert sub.minibatches[1]["batch_id"] == "new1"
        assert sub.minibatches[1]["attempt"] == 2
        assert sub.minibatches[2]["status"] == SUBMITTED

    def test_only_cancelled_is_resubmitted_not_noop(self, tmp_path, caplog):
        run_dir = _build_terminal_run(tmp_path, statuses=["fetched", "cancelled"])
        client = _Client()
        client.submit_batch = MagicMock(side_effect=["new1"])
        with caplog.at_level(logging.INFO, logger="prkit.batch"):
            caplog.clear()
            sub = resubmit_failures(client, run_dir)
        client.submit_batch.assert_called_once()
        assert sub.minibatches[1]["status"] == SUBMITTED
        msgs = [r.getMessage() for r in caplog.records if r.name == "prkit.batch"]
        assert any("Resubmitted 1 minibatches (0 still failed)" in m for m in msgs)


class TestPreconditions:
    def test_non_terminal_ledger_raises_before_any_submit(self, tmp_path):
        run_dir = _build_running_run(tmp_path)
        client = _Client()
        client.submit_batch = MagicMock()
        with pytest.raises(BatchNotTerminalError):
            resubmit_failures(client, run_dir)
        client.submit_batch.assert_not_called()

    def test_non_batch_provider_raises_up_front(self, tmp_path):
        run_dir = _build_terminal_run(tmp_path, statuses=["failed"])
        client = _Client("xai")
        client.submit_batch = MagicMock()
        with pytest.raises(BatchFetchUnsupportedError, match="xai"):
            resubmit_failures(client, run_dir)
        client.submit_batch.assert_not_called()


class TestPerItemFailure:
    def test_one_submit_raises_recorded_and_loop_continues(self, tmp_path, caplog):
        run_dir = _build_terminal_run(tmp_path, statuses=["failed", "failed", "failed"])
        client = _Client()
        client.submit_batch = MagicMock(
            side_effect=["new0", RuntimeError("boom"), "new2"]
        )
        with caplog.at_level(logging.INFO, logger="prkit.batch"):
            caplog.clear()
            sub = resubmit_failures(client, run_dir)

        assert (
            sub.minibatches[0]["status"] == SUBMITTED
            and sub.minibatches[0]["batch_id"] == "new0"
        )
        assert sub.minibatches[1]["status"] == SUBMIT_ERROR
        assert sub.minibatches[1]["batch_id"] == ""
        assert "boom" in sub.minibatches[1]["error"]
        assert (
            sub.minibatches[2]["status"] == SUBMITTED
            and sub.minibatches[2]["batch_id"] == "new2"
        )
        msgs = [r.getMessage() for r in caplog.records if r.name == "prkit.batch"]
        assert any("Resubmitted 2 minibatches (1 still failed)" in m for m in msgs)

    def test_failed_submit_never_leaves_submitted_with_empty_id(self, tmp_path):
        run_dir = _build_terminal_run(tmp_path, statuses=["failed"])
        client = _Client()
        client.submit_batch = MagicMock(side_effect=RuntimeError("boom"))
        sub = resubmit_failures(client, run_dir)
        mb = sub.minibatches[0]
        # Submit-first / mutate-second: a raising submit must NOT yield SUBMITTED+"".
        assert not (mb["status"] == SUBMITTED and mb["batch_id"] == "")
        assert mb["status"] == SUBMIT_ERROR and mb["batch_id"] == ""


class TestFacade:
    def test_resubmit_facade_delegates_to_module(self):
        client = _openai_submit_client()
        with patch("prkit.batch.resubmit_failures") as mock_resub:
            mock_resub.return_value = "LEDGER"
            out = client.resubmit_failures("run-dir")
        mock_resub.assert_called_once()
        args, _ = mock_resub.call_args
        assert args[0] is client and args[1] == "run-dir"
        assert out == "LEDGER"


class TestNextCommand:
    def test_resubmit_logs_fetch_next(self, tmp_path, caplog):
        run_dir = _build_terminal_run(tmp_path, statuses=["failed", "fetched"])
        client = _Client()
        client.submit_batch = MagicMock(side_effect=["new0"])
        with caplog.at_level(logging.INFO, logger="prkit.batch"):
            caplog.clear()
            resubmit_failures(client, run_dir)
        msgs = [r.getMessage() for r in caplog.records if r.name == "prkit.batch"]
        assert any(
            m.startswith("Resubmitted 1 minibatches (0 still failed)")
            and "fetch_batch_physics_reasoning" in m
            for m in msgs
        )
