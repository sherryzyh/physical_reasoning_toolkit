"""Tests for the ``BatchSubmission`` whole-batch ledger (state + serialization).

These exercise the mutable ledger in isolation — no client, no network: the pure
status helpers (``minibatches_to_fetch`` / ``set_status`` / ``is_complete`` /
``status_counts``) and the ``to_dict`` / ``from_dict`` / ``save`` / ``load``
round-trip, including the timezone-by-equality contract for ``created_at``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from prkit.batch import (
    CANCELLED,
    COMPLETED,
    EXPIRED,
    FAILED,
    FETCH_ERROR,
    FETCHED,
    RUNNING,
    SUBMIT_ERROR,
    SUBMITTED,
    BatchSubmission,
)


def _minibatch(index: int, status: str) -> dict:
    return {
        "index": index,
        "batch_id": f"b{index}",
        "status": status,
        "input_file_path": f"/tmp/run/inputs/minibatch_{index:04d}.jsonl",
        "num_requests": 2,
        "id_map": {f"p{index}a": f"p{index}a", f"p{index}b": f"p{index}b"},
        "submitted_at": "2026-06-22T00:00:00+00:00",
        "error": None,
        "output_path": None,
        "fetched_at": None,
        "counts": {},
        "endpoint": None,
        "completion_window": None,
    }


def _ledger(statuses: list[str], *, run_dir: str = "/tmp/run") -> BatchSubmission:
    minibatches = [_minibatch(i, status) for i, status in enumerate(statuses)]
    return BatchSubmission(
        run_name="run",
        provider="openai",
        model="gpt-5.1",
        created_at=datetime(2026, 6, 22, 1, 2, 3, 456789, tzinfo=UTC),
        minibatch_size=2,
        minibatch_count=len(statuses),
        total_problems=2 * len(statuses),
        dataset={"name": "d", "version": "1.0"},
        request_kind="free_text",
        prkit_api_version="1.0",
        display_name="run",
        run_dir=run_dir,
        metadata={"k": "v"},
        minibatches=minibatches,
    )


class TestStatusHelpers:
    def test_status_counts(self):
        sub = _ledger([SUBMITTED, SUBMITTED, FETCHED, FAILED])
        assert sub.status_counts() == {SUBMITTED: 2, FETCHED: 1, FAILED: 1}

    def test_minibatches_to_fetch_skips_only_terminal_done(self):
        sub = _ledger(
            [
                SUBMITTED,
                RUNNING,
                COMPLETED,
                EXPIRED,
                FETCH_ERROR,  # all of the above are still in the fetch loop
                FETCHED,
                SUBMIT_ERROR,
                FAILED,
                CANCELLED,  # these four are skipped
            ]
        )
        indices = [mb["index"] for mb in sub.minibatches_to_fetch()]
        assert indices == [0, 1, 2, 3, 4]

    def test_set_status_updates_in_place_with_extra_fields(self):
        sub = _ledger([SUBMITTED])
        sub.set_status(0, FETCHED, output_path="/tmp/o.jsonl", fetched_at="T")
        mb = sub.minibatches[0]
        assert mb["status"] == FETCHED
        assert mb["output_path"] == "/tmp/o.jsonl"
        assert mb["fetched_at"] == "T"

    def test_set_status_unknown_index_raises(self):
        sub = _ledger([SUBMITTED])
        with pytest.raises(KeyError):
            sub.set_status(99, FETCHED)

    def test_is_complete_true_when_all_terminal(self):
        sub = _ledger([FETCHED, FAILED, CANCELLED, SUBMIT_ERROR, EXPIRED])
        assert sub.is_complete()

    def test_is_complete_false_with_any_in_flight(self):
        for in_flight in (SUBMITTED, RUNNING, COMPLETED, FETCH_ERROR):
            sub = _ledger([FETCHED, in_flight])
            assert not sub.is_complete()


class TestSerialization:
    def test_to_dict_from_dict_round_trip_equal(self):
        sub = _ledger([SUBMITTED, FETCHED])
        rebuilt = BatchSubmission.from_dict(sub.to_dict())
        assert rebuilt == sub
        # created_at survives by value with tz compared by equality, not identity.
        assert rebuilt.created_at == sub.created_at
        assert rebuilt.created_at.tzinfo == UTC

    def test_save_load_round_trip_equal(self, tmp_path):
        sub = _ledger([SUBMITTED, RUNNING], run_dir=str(tmp_path / "run"))
        path = sub.save()
        assert path == tmp_path / "run" / "metadata.json"
        assert BatchSubmission.load(str(tmp_path / "run")) == sub

    def test_load_accepts_run_dir_or_metadata_path(self, tmp_path):
        sub = _ledger([FETCHED], run_dir=str(tmp_path / "run"))
        sub.save()
        from_dir = BatchSubmission.load(str(tmp_path / "run"))
        from_file = BatchSubmission.load(str(tmp_path / "run" / "metadata.json"))
        assert from_dir == from_file == sub

    def test_save_to_explicit_dir(self, tmp_path):
        sub = _ledger([SUBMITTED], run_dir="/nonexistent/placeholder")
        path = sub.save(tmp_path / "elsewhere")
        assert path == tmp_path / "elsewhere" / "metadata.json"
        assert BatchSubmission.load(tmp_path / "elsewhere") == sub

    def test_from_dict_parses_iso_timestamp_to_utc_equal(self):
        sub = _ledger([SUBMITTED])
        data = sub.to_dict()
        assert isinstance(data["created_at"], str)
        assert data["created_at"].endswith("+00:00")
        rebuilt = BatchSubmission.from_dict(data)
        # fromisoformat yields timezone(timedelta(0)) which == utc but is not it.
        assert rebuilt.created_at.tzinfo == UTC
