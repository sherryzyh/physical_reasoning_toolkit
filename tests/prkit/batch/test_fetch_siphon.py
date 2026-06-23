"""Tests for the Stage-4 siphon inside ``fetch_batch`` (record-level recovery).

Fully offline, mirroring ``tests/prkit/batch/test_fetch_batch.py``: runs are
submitted with an SDK-patched real client (realistic ``id_map`` + input files)
then polled/downloaded by a duck-typed ``_FetchClient`` whose
``retrieve_batch_results`` returns scripted ``BatchResult``s. A record-level
failure is injected either as a real ERRORED result for one cid or by omitting a
cid (the synthetic-missing path). Covers: the happy-path siphon (succeeded-only
on the wire; one ``failed_records`` entry; ``id_map`` pruned + ``num_requests``
decremented in lockstep; the derived accumulator), synthetic-missing siphon,
siphon idempotency on a re-fetch, and the MAX_ATTEMPTS exhaustion path
(rewritten as ``max_attempted``, kept in ``id_map``, consolidated terminally).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from prkit.batch import (
    FETCHED,
    MAX_ATTEMPTS,
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
    BatchStatus,
)


# --------------------------------------------------------------------------- #
# Offline builders (mirroring test_fetch_batch.py)                             #
# --------------------------------------------------------------------------- #
def _openai_submit_client(model: str = "gpt-5.1"):
    with patch("prkit.core.model_clients.openai.OpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        from prkit.core.model_clients.openai import OpenAIModel

        return OpenAIModel(model)


def _dataset(n: int):
    problems = [PhysicsProblem(problem_id=f"p{i}", question=f"Q{i}") for i in range(n)]
    return PhysicsDataset(problems, info={"name": "physreason", "version": "1.0"})


def _submit_run(tmp_path, *, n, minibatch_size, batch_ids, run_name="run"):
    client = _openai_submit_client()
    client.submit_batch = MagicMock(side_effect=list(batch_ids))
    return submit_batch_physics_reasoning(
        client,
        _dataset(n),
        output_dir=tmp_path,
        run_name=run_name,
        minibatch_size=minibatch_size,
    )


class _FetchClient:
    def __init__(self, provider="openai", *, states=None, results=None):
        self.provider = provider
        self.model = "model-x"
        self._states = states or {}
        self._results = results or {}
        self.poll_calls: list[str] = []
        self.retrieve_calls: list[str] = []

    def poll_batch(self, batch_id: str) -> BatchStatus:
        self.poll_calls.append(batch_id)
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
        self.retrieve_calls.append(batch_id)
        return iter(self._results.get(batch_id, []))


def _cids(sub, index=0):
    return list(sub.minibatches[index]["id_map"])


# --------------------------------------------------------------------------- #
class TestSiphonHappyPath:
    def test_errored_record_siphoned_outputs_succeeded_only(self, tmp_path):
        run_dir = _submit_run(tmp_path, n=3, minibatch_size=3, batch_ids=["b0"])
        cids = _cids(BatchSubmission.load(run_dir))  # [p0, p1, p2]
        client = _FetchClient(
            "openai",
            states={"b0": [BatchState.COMPLETED]},
            results={
                "b0": [
                    BatchResult(cids[0], BatchItemStatus.SUCCEEDED, text="A0"),
                    BatchResult(cids[1], BatchItemStatus.ERRORED, error="rate limited"),
                    BatchResult(cids[2], BatchItemStatus.SUCCEEDED, text="A2"),
                ]
            },
        )
        sub = fetch_batch(client, run_dir, progress=False)
        mb = sub.minibatches[0]
        assert mb["status"] == FETCHED

        # One siphoned record, recorded on the ledger (attempt=1, the source error).
        assert [e["custom_id"] for e in mb["failed_records"]] == [cids[1]]
        entry = mb["failed_records"][0]
        assert entry == {
            "custom_id": cids[1],
            "problem_id": "p1",
            "error": "rate limited",
            "attempt": 1,
        }
        # id_map pruned and num_requests decremented in lockstep (blocker #2).
        assert cids[1] not in mb["id_map"]
        assert mb["num_requests"] == len(mb["id_map"]) == 2

        # outputs/ holds the succeeded records only (no errored line).
        out_lines = Path(mb["output_path"]).read_text().splitlines()
        assert len(out_lines) == 2
        out_cids = {json.loads(line)["custom_id"] for line in out_lines}
        assert out_cids == {cids[0], cids[2]}

        # The derived accumulator carries the failed record's INPUT line.
        acc = Path(run_dir) / "failed-records-batch-input.jsonl"
        acc_lines = acc.read_text().splitlines()
        assert len(acc_lines) == 1
        assert json.loads(acc_lines[0])["custom_id"] == cids[1]

        # The minibatch now reads as all-success (siphoned id no longer yielded).
        pairs = dict(iter_batch_results(run_dir))
        assert set(pairs) == {"p0", "p2"}
        assert all(r.succeeded for r in pairs.values())

    def test_synthetic_missing_record_is_siphoned_too(self, tmp_path):
        run_dir = _submit_run(tmp_path, n=2, minibatch_size=2, batch_ids=["b0"])
        cids = _cids(BatchSubmission.load(run_dir))
        client = _FetchClient(
            "openai",
            states={"b0": [BatchState.COMPLETED]},
            # cids[1] omitted -> synthetic-missing -> siphoned.
            results={"b0": [BatchResult(cids[0], BatchItemStatus.SUCCEEDED, text="A")]},
        )
        sub = fetch_batch(client, run_dir, progress=False)
        mb = sub.minibatches[0]
        assert [e["problem_id"] for e in mb["failed_records"]] == ["p1"]
        assert mb["num_requests"] == len(mb["id_map"]) == 1
        assert (Path(run_dir) / "failed-records-batch-input.jsonl").is_file()


class TestSiphonIdempotency:
    def test_refetch_does_not_resiphon(self, tmp_path):
        run_dir = _submit_run(tmp_path, n=2, minibatch_size=2, batch_ids=["b0"])
        cids = _cids(BatchSubmission.load(run_dir))
        client = _FetchClient(
            "openai",
            states={"b0": [BatchState.COMPLETED]},
            results={"b0": [BatchResult(cids[0], BatchItemStatus.SUCCEEDED, text="A")]},
        )
        fetch_batch(client, run_dir, progress=False)
        client.poll_calls.clear()
        client.retrieve_calls.clear()
        # A FETCHED minibatch is skipped; the siphon never runs twice.
        sub2 = fetch_batch(client, run_dir, progress=False)
        assert client.poll_calls == [] and client.retrieve_calls == []
        assert len(sub2.minibatches[0]["failed_records"]) == 1  # not doubled


class TestNoFailuresNoArtifact:
    def test_all_succeeded_writes_no_accumulator_file(self, tmp_path):
        run_dir = _submit_run(tmp_path, n=2, minibatch_size=2, batch_ids=["b0"])
        cids = _cids(BatchSubmission.load(run_dir))
        client = _FetchClient(
            "openai",
            states={"b0": [BatchState.COMPLETED]},
            results={
                "b0": [
                    BatchResult(c, BatchItemStatus.SUCCEEDED, text="x") for c in cids
                ]
            },
        )
        sub = fetch_batch(client, run_dir, progress=False)
        assert "failed_records" not in sub.minibatches[0]
        # The Stage-4 artifact appears only when record recovery is in play.
        assert not (Path(run_dir) / "failed-records-batch-input.jsonl").exists()


class TestMaxAttemptsExhaustion:
    def test_exhausted_record_rewritten_max_attempted_and_consolidated(self, tmp_path):
        run_dir = _submit_run(tmp_path, n=2, minibatch_size=2, batch_ids=["b0"])
        # Simulate a minibatch already resubmitted to its bound (attempt == MAX_ATTEMPTS),
        # so a fresh record's submissions = 0 + MAX_ATTEMPTS -> exhausted on this fetch.
        sub0 = BatchSubmission.load(run_dir)
        sub0.minibatches[0]["attempt"] = MAX_ATTEMPTS
        sub0.save()
        cids = _cids(sub0)
        client = _FetchClient(
            "openai",
            states={"b0": [BatchState.COMPLETED]},
            results={
                "b0": [
                    BatchResult(cids[0], BatchItemStatus.ERRORED, error="boom"),
                    BatchResult(cids[1], BatchItemStatus.SUCCEEDED, text="A1"),
                ]
            },
        )
        sub = fetch_batch(client, run_dir, progress=False)
        mb = sub.minibatches[0]
        # NOT siphoned: kept in id_map, num_requests unchanged, no failed_records.
        assert mb.get("failed_records", []) == []
        assert cids[0] in mb["id_map"]
        assert mb["num_requests"] == len(mb["id_map"]) == 2
        assert not (Path(run_dir) / "failed-records-batch-input.jsonl").exists()

        # Rewritten in outputs/ as max_attempted (distinct from a transient errored).
        pairs = dict(iter_batch_results(run_dir))
        assert pairs["p0"].status == BatchItemStatus.MAX_ATTEMPTED
        assert pairs["p1"].succeeded

        # Consolidates terminally; the run is legitimately fully consolidated.
        consolidate_batch_results(run_dir)
        rec = json.loads((Path(run_dir) / "results" / "p0.json").read_text())
        assert rec["status"] == "max_attempted"
        manifest = json.loads((Path(run_dir) / "results_manifest.json").read_text())
        assert manifest["fully_consolidated"] is True
        assert manifest["pending_failed_records"] == 0
