"""Tests for the Stage-4 record drain inside ``resubmit_failures``.

Fully offline. A terminal run carrying pending siphoned ``failed_records`` is
built by submitting (SDK-patched real client) then fetching with a duck-typed
``_FetchClient`` that injects a per-record failure; the drain is then driven by a
duck-typed ``_Client`` whose ``submit_batch`` is a ``MagicMock``. Covers the
drain mechanics (a fresh retry minibatch with the right shape; ``minibatch_count``
bumped; consumed accumulator), the headline end-to-end recovery loop, the
field-parse correlation regression (``custom_id`` and Gemini ``key``), the
MAX_ATTEMPTS off-by-one across whole-minibatch + record retries, a drain submit
error, and the ledger round-trip of the new minibatch keys.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prkit.batch import (
    CONSOLIDATED,
    SUBMIT_ERROR,
    SUBMITTED,
    BatchSubmission,
    consolidate_batch_results,
    fetch_batch,
    iter_batch_results,
    resubmit_failures,
    submit_batch_physics_reasoning,
)
from prkit.core.domain import PhysicsDataset, PhysicsProblem
from prkit.core.model_clients.batch_types import (
    BatchItemStatus,
    BatchResult,
    BatchState,
    BatchStatus,
)


# --------------------------------------------------------------------------- #
# Offline builders                                                             #
# --------------------------------------------------------------------------- #
def _openai_submit_client(model: str = "gpt-5.1"):
    with patch("prkit.core.model_clients.openai.OpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        from prkit.core.model_clients.openai import OpenAIModel

        return OpenAIModel(model)


def _gemini_submit_client(model: str = "gemini-3.5-flash"):
    with patch("prkit.core.model_clients.gemini.genai.Client") as mock_cls:
        mock_cls.return_value = MagicMock()
        from prkit.core.model_clients.gemini import GeminiModel

        return GeminiModel(model)


_SUBMIT_BUILDERS = {"openai": _openai_submit_client, "google": _gemini_submit_client}
_ID_FIELD = {"openai": "custom_id", "google": "key"}


def _dataset(n: int):
    problems = [PhysicsProblem(problem_id=f"p{i}", question=f"Q{i}") for i in range(n)]
    return PhysicsDataset(problems, info={"name": "physreason", "version": "1.0"})


class _FetchClient:
    def __init__(self, provider="openai", *, states=None, results=None):
        self.provider = provider
        self.model = "model-x"
        self._states = states or {}
        self._results = results or {}

    def poll_batch(self, batch_id: str) -> BatchStatus:
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


def _run_with_siphoned_record(tmp_path, *, provider="openai", run_name="run"):
    """Submit 2 problems in one minibatch, fetch with p0 ERRORED -> p0 siphoned.

    Returns the run_dir; afterwards minibatch 0 is FETCHED with one pending
    ``failed_records`` entry (custom_id ``p0``, attempt 1) and p1 succeeded.
    """
    client = _SUBMIT_BUILDERS[provider]()
    client.submit_batch = MagicMock(side_effect=["b0"])
    run_dir = submit_batch_physics_reasoning(
        client,
        _dataset(2),
        output_dir=tmp_path,
        run_name=run_name,
        minibatch_size=2,
    )
    cids = list(BatchSubmission.load(run_dir).minibatches[0]["id_map"])
    fc = _FetchClient(
        provider,
        states={"b0": [BatchState.COMPLETED]},
        results={
            "b0": [
                BatchResult(cids[0], BatchItemStatus.ERRORED, error="boom"),
                BatchResult(cids[1], BatchItemStatus.SUCCEEDED, text="A1"),
            ]
        },
    )
    fetch_batch(fc, run_dir, progress=False)
    return run_dir


# --------------------------------------------------------------------------- #
class TestDrainMechanics:
    def test_pending_record_drained_into_fresh_minibatch(self, tmp_path):
        run_dir = _run_with_siphoned_record(tmp_path)
        sub0 = BatchSubmission.load(run_dir)
        failed_cid = sub0.minibatches[0]["failed_records"][0]["custom_id"]

        client = _Client()
        client.submit_batch = MagicMock(side_effect=["r0"])
        sub = resubmit_failures(client, run_dir)

        # A single fresh retry minibatch appended with the right shape.
        assert len(sub.minibatches) == 2
        retry = sub.minibatches[1]
        assert retry["index"] == 1  # max(existing index) + 1
        assert retry["status"] == SUBMITTED and retry["batch_id"] == "r0"
        assert retry["is_retry"] is True
        assert retry["attempt"] == 1
        assert retry["retry_sources"] == [0]
        assert retry["record_attempts"] == {failed_cid: 1}
        assert retry["id_map"] == {failed_cid: "p0"}
        assert retry["num_requests"] == len(retry["id_map"]) == 1
        assert retry["endpoint"] is None and retry["completion_window"] is None

        # Ledger bookkeeping: minibatch_count bumped, source accumulator consumed.
        assert sub.minibatch_count == 2
        assert sub.minibatches[0]["failed_records"] == []
        # The derived accumulator file is gone once nothing is pending.
        assert not (Path(run_dir) / "failed-records-batch-input.jsonl").exists()

        # submit_batch got exactly the failed record's input line (by wire id).
        client.submit_batch.assert_called_once()
        sent = client.submit_batch.call_args.args[0]
        assert [r["custom_id"] for r in sent] == [failed_cid]
        # The retry input file was persisted for resubmittability.
        assert (Path(run_dir) / "inputs" / "minibatch_0001.jsonl").is_file()

    def test_drain_submit_error_keeps_record_recoverable(self, tmp_path):
        run_dir = _run_with_siphoned_record(tmp_path)
        client = _Client()
        client.submit_batch = MagicMock(side_effect=RuntimeError("nope"))
        sub = resubmit_failures(client, run_dir)
        retry = sub.minibatches[1]
        # Recorded as SUBMIT_ERROR with an input file -> the whole-minibatch path can
        # resubmit it later (records are not lost).
        assert retry["status"] == SUBMIT_ERROR and retry["batch_id"] == ""
        assert "nope" in retry["error"]
        assert sub.minibatches[0]["failed_records"] == []  # still consumed


class TestEndToEndRecovery:
    def test_full_loop_reaches_fully_consolidated(self, tmp_path):
        run_dir = _run_with_siphoned_record(tmp_path)
        # After the siphon pruned p0, minibatch 0's id_map holds only "p1"; the
        # siphoned (failed) record is "p0".
        assert list(BatchSubmission.load(run_dir).minibatches[0]["id_map"]) == ["p1"]
        failed_cid = "p0"

        consolidate_batch_results(run_dir)
        manifest = json.loads((Path(run_dir) / "results_manifest.json").read_text())
        assert manifest["fully_consolidated"] is False  # p0 still pending
        assert manifest["pending_failed_records"] == 1
        assert not (Path(run_dir) / "results" / "p0.json").exists()

        client = _Client()
        client.submit_batch = MagicMock(side_effect=["r0"])
        resubmit_failures(client, run_dir)

        fc = _FetchClient(
            "openai",
            states={"r0": [BatchState.COMPLETED]},
            results={
                "r0": [BatchResult(failed_cid, BatchItemStatus.SUCCEEDED, text="fixed")]
            },
        )
        fetch_batch(fc, run_dir, progress=False)
        sub = consolidate_batch_results(run_dir)

        manifest = json.loads((Path(run_dir) / "results_manifest.json").read_text())
        assert manifest["fully_consolidated"] is True
        assert manifest["pending_failed_records"] == 0
        assert all(mb["status"] == CONSOLIDATED for mb in sub.minibatches)
        # The recovered result is the succeeded one.
        rec = json.loads((Path(run_dir) / "results" / "p0.json").read_text())
        assert rec["status"] == "succeeded" and rec["text"] == "fixed"


class TestCorrelationFieldParse:
    @pytest.mark.parametrize("provider", ["openai", "google"])
    def test_drained_record_maps_by_field_not_position(self, tmp_path, provider):
        # Reorder the on-disk input lines so a positional zip would mis-correlate;
        # the field-parse must still pick p0's own line for p0's failed record.
        run_dir = _run_with_siphoned_record(tmp_path, provider=provider)
        id_field = _ID_FIELD[provider]
        input_path = Path(run_dir) / "inputs" / "minibatch_0000.jsonl"
        lines = input_path.read_text().splitlines()
        input_path.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")

        client = _Client(provider)
        client.submit_batch = MagicMock(side_effect=["r0"])
        sub = resubmit_failures(client, run_dir)

        sent = client.submit_batch.call_args.args[0]
        assert len(sent) == 1
        # The submitted line is p0's own request (its wire id is p0), not p1's.
        assert sent[0][id_field] == "p0"
        assert sub.minibatches[1]["id_map"] == {"p0": "p0"}


class TestMaxAttemptsOffByOne:
    def test_attempt_history_spans_whole_minibatch_and_record_retries(self, tmp_path):
        # Owner's scenario guarding the off-by-one. Simulate a minibatch already
        # whole-resubmitted once (attempt=2); a dropped record -> submissions 0+2=2
        # (siphoned). After the record drain (attempt=1, record_attempts[cid]=2), a
        # further failure -> submissions 2+1=3 == MAX_ATTEMPTS -> MAX_ATTEMPTED.
        client = _openai_submit_client()
        client.submit_batch = MagicMock(side_effect=["b0"])
        run_dir = submit_batch_physics_reasoning(
            client, _dataset(2), output_dir=tmp_path, run_name="run", minibatch_size=2
        )
        sub0 = BatchSubmission.load(run_dir)
        sub0.minibatches[0]["attempt"] = 2  # one prior whole-minibatch resubmit
        sub0.save()
        cids = list(sub0.minibatches[0]["id_map"])

        fc = _FetchClient(
            "openai",
            states={"b0": [BatchState.COMPLETED]},
            results={
                "b0": [
                    BatchResult(cids[0], BatchItemStatus.ERRORED, error="e1"),
                    BatchResult(cids[1], BatchItemStatus.SUCCEEDED, text="A1"),
                ]
            },
        )
        sub = fetch_batch(fc, run_dir, progress=False)
        assert sub.minibatches[0]["failed_records"][0]["attempt"] == 2  # 0 + 2

        rclient = _Client()
        rclient.submit_batch = MagicMock(side_effect=["r0"])
        sub = resubmit_failures(rclient, run_dir)
        retry = sub.minibatches[1]
        assert retry["record_attempts"] == {cids[0]: 2} and retry["attempt"] == 1

        fc2 = _FetchClient(
            "openai",
            states={"r0": [BatchState.COMPLETED]},
            results={"r0": [BatchResult(cids[0], BatchItemStatus.ERRORED, error="e2")]},
        )
        sub = fetch_batch(fc2, run_dir, progress=False)
        retry = sub.minibatches[1]
        # submissions = 2 + 1 = 3 == MAX_ATTEMPTS -> exhausted, NOT re-siphoned.
        assert retry.get("failed_records", []) == []
        assert cids[0] in retry["id_map"]
        pairs = dict(iter_batch_results(run_dir))
        assert pairs[cids[0]].status == BatchItemStatus.MAX_ATTEMPTED


class TestLedgerRoundTrip:
    def test_stage4_minibatch_keys_survive_to_dict_from_dict(self, tmp_path):
        run_dir = _run_with_siphoned_record(tmp_path)
        client = _Client()
        client.submit_batch = MagicMock(side_effect=["r0"])
        sub = resubmit_failures(client, run_dir)

        restored = BatchSubmission.from_dict(sub.to_dict())
        retry = restored.minibatches[1]
        assert retry["is_retry"] is True
        assert retry["attempt"] == 1
        assert retry["retry_sources"] == [0]
        assert retry["record_attempts"] == {"p0": 1}
        assert restored.minibatches[0]["failed_records"] == []


class TestPreconditions:
    def test_drain_requires_terminal_ledger(self, tmp_path):
        # A run left RUNNING: resubmit refuses before any submit (terminal gate).
        from prkit.batch import BatchNotTerminalError

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
        rclient = _Client()
        rclient.submit_batch = MagicMock()
        with pytest.raises(BatchNotTerminalError):
            resubmit_failures(rclient, run_dir)
        rclient.submit_batch.assert_not_called()


class TestNextCommand:
    def test_consolidate_done_hint_points_at_resubmit_failures(self, tmp_path, caplog):
        run_dir = _run_with_siphoned_record(tmp_path)
        with caplog.at_level(logging.INFO, logger="prkit.batch"):
            caplog.clear()
            consolidate_batch_results(run_dir)
        msgs = [r.getMessage() for r in caplog.records if r.name == "prkit.batch"]
        assert any(
            "records still failed" in m and "resubmit_failures" in m for m in msgs
        )
