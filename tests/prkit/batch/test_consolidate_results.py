"""Tests for the finalize half: ``consolidate_batch_results`` (Stage 3).

Fully offline. Runs are built with an SDK-patched submit client (realistic
``id_map`` + input files) then driven to terminal statuses by the duck-typed
``_FetchClient`` from the fetch tests' pattern, so each minibatch lands FETCHED
(carries a SUCCEEDED result) or terminal-failed (none). Covers the happy path,
the lenient succeeded-subset, incremental resume, streaming (one record at a
time), filename sanitization + collision, crash-safety (before the manifest and
mid-minibatch), the manifest layout, ``iter_batch_results`` after consolidate,
the empty-ledger guard, and the next-command prompt.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import prkit.batch as batch_module
from prkit.batch import (
    CONSOLIDATED,
    FETCHED,
    BatchInputError,
    BatchSubmission,
    consolidate_batch_results,
    fetch_batch,
    iter_batch_results,
    submit_batch_physics_reasoning,
)
from prkit.core.domain import PhysicsDataset, PhysicsProblem
from prkit.core.model_clients.batch_types import (
    BatchItemStatus,
    BatchResult,
    BatchState,
)


# --------------------------------------------------------------------------- #
# Offline builders (mirroring tests/prkit/batch/test_fetch_batch.py)           #
# --------------------------------------------------------------------------- #
def _openai_submit_client(model: str = "gpt-5.1"):
    with patch("prkit.core.model_clients.openai.OpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        from prkit.core.model_clients.openai import OpenAIModel

        return OpenAIModel(model)


def _dataset(n: int):
    problems = [PhysicsProblem(problem_id=f"p{i}", question=f"Q{i}") for i in range(n)]
    return PhysicsDataset(problems, info={"name": "physreason", "version": "1.0"})


def _dataset_ids(ids):
    problems = [PhysicsProblem(problem_id=pid, question=f"Q-{pid}") for pid in ids]
    return PhysicsDataset(problems, info={"name": "physreason", "version": "1.0"})


def _submit_dataset(tmp_path, dataset, *, minibatch_size, batch_ids, run_name="run"):
    client = _openai_submit_client()
    client.submit_batch = MagicMock(side_effect=list(batch_ids))
    return submit_batch_physics_reasoning(
        client,
        dataset,
        output_dir=tmp_path,
        run_name=run_name,
        minibatch_size=minibatch_size,
    )


class _FetchClient:
    """Minimal poll/retrieve client driven by per-``batch_id`` scripted state."""

    def __init__(self, provider="openai", *, states=None, results=None):
        self.provider = provider
        self.model = "model-x"
        self._states = states or {}
        self._results = results or {}
        self.poll_calls: list[str] = []

    def poll_batch(self, batch_id: str):
        self.poll_calls.append(batch_id)
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


_STATE_MAP = {
    "fetched": BatchState.COMPLETED,
    "failed": BatchState.FAILED,
    "expired": BatchState.EXPIRED,
    "cancelled": BatchState.CANCELLED,
}


def _build_run(tmp_path, *, statuses, run_name="run"):
    """One minibatch per status; ``"fetched"`` carries one SUCCEEDED result.

    After this returns, the ledger is terminal and minibatch ``i`` is FETCHED
    (text ``ans-i``) or terminal-failed per ``statuses[i]``.
    """
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


# --------------------------------------------------------------------------- #
class TestHappyPath:
    def test_all_fetched_writes_per_problem_files_and_manifest(self, tmp_path):
        run_dir = _build_run(tmp_path, statuses=["fetched", "fetched", "fetched"])
        sub = consolidate_batch_results(run_dir)

        results_dir = Path(run_dir) / "results"
        assert sorted(p.name for p in results_dir.iterdir()) == [
            "p0.json",
            "p1.json",
            "p2.json",
        ]  # results/ holds ONLY per-problem files
        for i in range(3):
            rec = json.loads((results_dir / f"p{i}.json").read_text())
            assert rec == {
                "problem_id": f"p{i}",
                "custom_id": f"p{i}",
                "status": "succeeded",
                "text": f"ans-{i}",
                "error": None,
            }
        assert all(mb["status"] == CONSOLIDATED for mb in sub.minibatches)

        # Manifest lives at the run-dir ROOT, not inside results/.
        assert (Path(run_dir) / "results_manifest.json").is_file()
        assert not (results_dir / "results_manifest.json").exists()
        manifest = json.loads((Path(run_dir) / "results_manifest.json").read_text())
        assert manifest["fully_consolidated"] is True
        assert manifest["minibatches_consolidated"] == 3
        assert manifest["minibatches_total"] == 3
        assert manifest["results_written"] == 3
        assert manifest["status_counts"] == {"succeeded": 3}
        assert manifest["uncorrelated_total"] == 0
        assert manifest["run_name"] == sub.run_name

    def test_custom_results_dirname(self, tmp_path):
        run_dir = _build_run(tmp_path, statuses=["fetched"])
        consolidate_batch_results(run_dir, results_dirname="answers")
        assert (Path(run_dir) / "answers" / "p0.json").is_file()
        assert not (Path(run_dir) / "results").exists()


class TestLenientSubset:
    def test_mixed_ledger_consolidates_only_fetched_and_warns(self, tmp_path, caplog):
        run_dir = _build_run(tmp_path, statuses=["fetched", "failed", "fetched"])
        with caplog.at_level(logging.WARNING, logger="prkit.batch"):
            sub = consolidate_batch_results(run_dir)

        assert sub.minibatches[0]["status"] == CONSOLIDATED
        assert sub.minibatches[1]["status"] == "failed"  # untouched
        assert sub.minibatches[2]["status"] == CONSOLIDATED
        assert sorted(p.name for p in (Path(run_dir) / "results").iterdir()) == [
            "p0.json",
            "p2.json",
        ]  # the failed minibatch's problem is not written

        manifest = json.loads((Path(run_dir) / "results_manifest.json").read_text())
        assert manifest["fully_consolidated"] is False
        assert manifest["minibatches_consolidated"] == 2

        warnings = [
            r.getMessage()
            for r in caplog.records
            if r.name == "prkit.batch" and r.levelno == logging.WARNING
        ]
        assert any("not yet succeeded" in w for w in warnings)


class TestIncrementalResume:
    def test_rerun_skips_consolidated_and_adds_new(self, tmp_path):
        run_dir = _build_run(tmp_path, statuses=["fetched", "failed"])
        consolidate_batch_results(run_dir)
        p0 = Path(run_dir) / "results" / "p0.json"
        assert p0.is_file()
        mtime0 = p0.stat().st_mtime_ns

        # Simulate the failed minibatch later landing FETCHED (resubmit + fetch).
        sub = BatchSubmission.load(run_dir)
        mb1 = sub.minibatches[1]
        cid1 = next(iter(mb1["id_map"]))
        out_path = Path(run_dir) / "outputs" / "minibatch_0001.jsonl"
        out_path.write_text(
            json.dumps(
                {
                    "custom_id": cid1,
                    "status": "succeeded",
                    "text": "late",
                    "error": None,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        sub.set_status(
            1,
            FETCHED,
            output_path=str(out_path),
            fetched_at="2026-01-01T00:00:00+00:00",
        )
        sub.save()

        sub2 = consolidate_batch_results(run_dir)
        assert all(mb["status"] == CONSOLIDATED for mb in sub2.minibatches)
        assert (Path(run_dir) / "results" / "p1.json").is_file()
        # The already-consolidated p0.json was NOT rewritten (mtime stable).
        assert p0.stat().st_mtime_ns == mtime0


class TestStreaming:
    def test_writes_one_record_at_a_time_manifest_last(self, tmp_path, monkeypatch):
        run_dir = _build_run(tmp_path, statuses=["fetched", "fetched", "fetched"])
        real = batch_module._atomic_write_text
        writes: list[str] = []

        def spy(path, text):
            name = Path(path).name
            if name != "results_manifest.json":
                obj = json.loads(text)
                # One record per file — never a list/aggregate of the whole run.
                assert isinstance(obj, dict) and "problem_id" in obj
            writes.append(name)
            real(path, text)

        monkeypatch.setattr(batch_module, "_atomic_write_text", spy)
        consolidate_batch_results(run_dir)

        per_problem = [w for w in writes if w != "results_manifest.json"]
        assert per_problem == ["p0.json", "p1.json", "p2.json"]  # streamed in order
        assert writes[-1] == "results_manifest.json"  # marker written last


class TestFilenameSafety:
    def test_unsafe_problem_id_is_sanitized(self, tmp_path):
        run_dir = _submit_dataset(
            tmp_path, _dataset_ids(["a/b#1"]), minibatch_size=1, batch_ids=["b0"]
        )
        cid = next(iter(BatchSubmission.load(run_dir).minibatches[0]["id_map"]))
        client = _FetchClient(
            states={"b0": [BatchState.COMPLETED]},
            results={"b0": [BatchResult(cid, BatchItemStatus.SUCCEEDED, text="x")]},
        )
        fetch_batch(client, run_dir, progress=False)
        consolidate_batch_results(run_dir)
        files = [p.name for p in (Path(run_dir) / "results").iterdir()]
        assert files == ["a_b_1.json"]
        rec = json.loads((Path(run_dir) / "results" / "a_b_1.json").read_text())
        assert rec["problem_id"] == "a/b#1"  # original id preserved inside

    def test_filename_collision_raises_no_silent_overwrite(self, tmp_path):
        run_dir = _submit_dataset(
            tmp_path, _dataset_ids(["a/b", "a:b"]), minibatch_size=2, batch_ids=["b0"]
        )
        cids = list(BatchSubmission.load(run_dir).minibatches[0]["id_map"])
        client = _FetchClient(
            states={"b0": [BatchState.COMPLETED]},
            results={
                "b0": [
                    BatchResult(c, BatchItemStatus.SUCCEEDED, text="x") for c in cids
                ]
            },
        )
        fetch_batch(client, run_dir, progress=False)
        with pytest.raises(BatchInputError, match="collision"):
            consolidate_batch_results(run_dir)
        # The first problem's file was written; the second was refused (not clobbered).
        rec = json.loads((Path(run_dir) / "results" / "a_b.json").read_text())
        assert rec["problem_id"] == "a/b"


class TestCrashSafety:
    def test_crash_before_manifest_leaves_valid_files_and_resumes(
        self, tmp_path, monkeypatch
    ):
        run_dir = _build_run(tmp_path, statuses=["fetched", "fetched"])

        def boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(batch_module, "_write_results_manifest", boom)
        with pytest.raises(RuntimeError, match="boom"):
            consolidate_batch_results(run_dir)

        # Per-problem files are complete + valid; the manifest never appeared.
        assert (
            json.loads((Path(run_dir) / "results" / "p0.json").read_text())[
                "problem_id"
            ]
            == "p0"
        )
        assert not (Path(run_dir) / "results_manifest.json").exists()

        monkeypatch.undo()
        sub = consolidate_batch_results(run_dir)
        assert all(mb["status"] == CONSOLIDATED for mb in sub.minibatches)
        assert (Path(run_dir) / "results_manifest.json").is_file()

    def test_crash_mid_minibatch_resumes_without_partial_file(
        self, tmp_path, monkeypatch
    ):
        run_dir = _build_run(tmp_path, statuses=["fetched", "fetched"])
        real = batch_module._atomic_write_text

        def flaky(path, text):
            if Path(path).name == "p1.json":
                raise RuntimeError("disk full")
            real(path, text)

        monkeypatch.setattr(batch_module, "_atomic_write_text", flaky)
        with pytest.raises(RuntimeError, match="disk full"):
            consolidate_batch_results(run_dir)

        sub = BatchSubmission.load(run_dir)
        assert sub.minibatches[0]["status"] == CONSOLIDATED  # saved before the crash
        assert sub.minibatches[1]["status"] == FETCHED  # never reached CONSOLIDATED
        assert (Path(run_dir) / "results" / "p0.json").is_file()
        assert not (
            Path(run_dir) / "results" / "p1.json"
        ).exists()  # atomic: no partial

        monkeypatch.undo()
        sub2 = consolidate_batch_results(run_dir)
        assert all(mb["status"] == CONSOLIDATED for mb in sub2.minibatches)
        assert (Path(run_dir) / "results" / "p1.json").is_file()


class TestPostConsolidateReader:
    def test_iter_batch_results_still_reads_consolidated(self, tmp_path):
        run_dir = _build_run(tmp_path, statuses=["fetched", "fetched"])
        consolidate_batch_results(run_dir)
        assert all(
            mb["status"] == CONSOLIDATED
            for mb in BatchSubmission.load(run_dir).minibatches
        )
        pairs = dict(iter_batch_results(run_dir))
        assert set(pairs) == {"p0", "p1"}
        assert all(r.succeeded for r in pairs.values())


class TestEmptyLedger:
    def test_empty_ledger_raises(self, tmp_path):
        sub = BatchSubmission(
            run_name="r",
            provider="openai",
            model="m",
            created_at=datetime.now(timezone.utc),
            minibatch_size=1,
            minibatch_count=0,
            total_problems=0,
            dataset=None,
            request_kind="free_text",
            prkit_api_version="1.0",
            display_name="r",
            run_dir=str(tmp_path / "r"),
            metadata={},
            minibatches=[],
        )
        sub.save()
        with pytest.raises(BatchInputError, match="no minibatches"):
            consolidate_batch_results(str(tmp_path / "r"))


class TestNextCommand:
    def test_done_line_when_fully_consolidated(self, tmp_path, caplog):
        run_dir = _build_run(tmp_path, statuses=["fetched"])
        with caplog.at_level(logging.INFO, logger="prkit.batch"):
            caplog.clear()
            consolidate_batch_results(run_dir)
        msgs = [r.getMessage() for r in caplog.records if r.name == "prkit.batch"]
        assert any(m.startswith("Done — results in") for m in msgs)

    def test_resume_line_when_partial(self, tmp_path, caplog):
        run_dir = _build_run(tmp_path, statuses=["fetched", "failed"])
        with caplog.at_level(logging.INFO, logger="prkit.batch"):
            caplog.clear()
            consolidate_batch_results(run_dir)
        msgs = [r.getMessage() for r in caplog.records if r.name == "prkit.batch"]
        assert any("Consolidated 1/2 minibatches" in m for m in msgs)
