"""Tests for the fetch half: ``fetch_batch`` / ``iter_batch_results`` / capability.

Fully offline. Submission runs are built with SDK-patched real clients (so the
``id_map``s + run folder are realistic), then polled/downloaded with a small
duck-typed ``_FetchClient`` that maps each ``batch_id`` to a ``BatchState`` (or a
sequence of them, for ``wait=True``) and a list of ``BatchResult``s. Covers the
capability gate, the poll→download happy path, resume/skip, non-terminal,
EXPIRED partials, FAILED/CANCELLED, submit-error skip, cross-provider correlation,
completeness, the run_dir/object handle contract, the progress summary, the
facade, and the ``wait=True`` loop + timeout.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prkit.batch import (
    CANCELLED,
    EXPIRED,
    FETCHED,
    RUNNING,
    SUBMIT_ERROR,
    BatchFetchUnsupportedError,
    BatchSubmission,
    batch_fetch_supported,
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
# Offline SUBMIT client builders (SDK constructor patched) + a run helper      #
# --------------------------------------------------------------------------- #
def _openai_submit_client(model: str = "gpt-5.1"):
    with patch("prkit.core.model_clients.openai.OpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        from prkit.core.model_clients.openai import OpenAIModel

        return OpenAIModel(model)


def _anthropic_submit_client(model: str = "claude-opus-4-8"):
    with patch("prkit.core.model_clients.anthropic.Anthropic") as mock_cls:
        mock_cls.return_value = MagicMock()
        from prkit.core.model_clients.anthropic import AnthropicModel

        return AnthropicModel(model)


def _gemini_submit_client(model: str = "gemini-3.5-flash"):
    with patch("prkit.core.model_clients.gemini.genai.Client") as mock_cls:
        mock_cls.return_value = MagicMock()
        from prkit.core.model_clients.gemini import GeminiModel

        return GeminiModel(model)


_SUBMIT_BUILDERS = {
    "openai": _openai_submit_client,
    "anthropic": _anthropic_submit_client,
    "google": _gemini_submit_client,
}


def _dataset(n: int):
    problems = [PhysicsProblem(problem_id=f"p{i}", question=f"Q{i}") for i in range(n)]
    return PhysicsDataset(problems, info={"name": "physreason", "version": "1.0"})


def _submit_run(
    tmp_path,
    *,
    n: int,
    minibatch_size: int,
    batch_ids,
    provider: str = "openai",
    custom_id_fn=None,
    run_name: str = "run",
) -> str:
    """Submit a run offline; ``submit_batch`` returns the given ``batch_ids``."""
    client = _SUBMIT_BUILDERS[provider]()
    client.submit_batch = MagicMock(side_effect=list(batch_ids))
    return submit_batch_physics_reasoning(
        client,
        _dataset(n),
        output_dir=tmp_path,
        run_name=run_name,
        minibatch_size=minibatch_size,
        custom_id_fn=custom_id_fn,
    )


# --------------------------------------------------------------------------- #
# Duck-typed FETCH client                                                      #
# --------------------------------------------------------------------------- #
class _FetchClient:
    """A minimal poll/retrieve client driven by per-``batch_id`` scripted state."""

    def __init__(self, provider: str = "openai", *, states=None, results=None):
        self.provider = provider
        self.model = "model-x"
        self._states: dict[str, list[BatchState]] = states or {}
        self._results: dict[str, list[BatchResult]] = results or {}
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


def _cids(sub: BatchSubmission, index: int = 0) -> list[str]:
    return list(sub.minibatches[index]["id_map"])


# --------------------------------------------------------------------------- #
class TestCapability:
    @pytest.mark.parametrize(
        "provider,ok",
        [
            ("openai", True),
            ("anthropic", True),
            ("google", True),
            ("xai", False),
            ("deepseek", False),
            ("dashscope", False),
            ("ollama", False),
        ],
    )
    def test_batch_fetch_supported(self, provider, ok):
        assert batch_fetch_supported(_FetchClient(provider)) is ok

    def test_unsupported_raises_up_front_without_polling(self, tmp_path):
        run_dir = _submit_run(tmp_path, n=1, minibatch_size=1, batch_ids=["b0"])
        client = _FetchClient("xai", states={"b0": [BatchState.COMPLETED]})
        with pytest.raises(BatchFetchUnsupportedError, match="xai"):
            fetch_batch(client, run_dir)
        assert client.poll_calls == []


class TestHappyPath:
    def test_poll_download_persists_and_correlates(self, tmp_path):
        run_dir = _submit_run(tmp_path, n=2, minibatch_size=2, batch_ids=["b0"])
        cids = _cids(BatchSubmission.load(run_dir))
        client = _FetchClient(
            "openai",
            states={"b0": [BatchState.COMPLETED]},
            results={
                "b0": [
                    BatchResult(cids[0], BatchItemStatus.SUCCEEDED, text="A0"),
                    BatchResult(cids[1], BatchItemStatus.SUCCEEDED, text="A1"),
                ]
            },
        )
        sub = fetch_batch(client, run_dir, progress=False)
        mb = sub.minibatches[0]
        assert mb["status"] == FETCHED
        assert mb["output_path"] and Path(mb["output_path"]).exists()
        assert mb["fetched_at"]
        assert mb["output_path"] == str(
            Path(run_dir) / "outputs" / "minibatch_0000.jsonl"
        )
        assert len(Path(mb["output_path"]).read_text().splitlines()) == 2
        pairs = dict(iter_batch_results(run_dir))
        assert set(pairs) == {"p0", "p1"}
        assert all(r.succeeded for r in pairs.values())
        assert sub.is_complete()


class TestResume:
    def test_fetched_minibatches_skipped_on_second_pass(self, tmp_path):
        bids = [f"b{i}" for i in range(10)]
        run_dir = _submit_run(tmp_path, n=10, minibatch_size=1, batch_ids=bids)
        sub0 = BatchSubmission.load(run_dir)
        cid_of = {mb["batch_id"]: next(iter(mb["id_map"])) for mb in sub0.minibatches}
        states = {
            bid: [BatchState.COMPLETED if i < 3 else BatchState.IN_PROGRESS]
            for i, bid in enumerate(bids)
        }
        results = {
            bid: [BatchResult(cid_of[bid], BatchItemStatus.SUCCEEDED, text="x")]
            for bid in bids[:3]
        }
        client = _FetchClient("openai", states=states, results=results)

        sub = fetch_batch(client, run_dir, progress=False)
        assert sum(mb["status"] == FETCHED for mb in sub.minibatches) == 3
        assert set(client.poll_calls) == set(bids)  # first pass polled all 10

        client.poll_calls.clear()
        client.retrieve_calls.clear()
        fetch_batch(client, run_dir, progress=False)
        # The 3 fetched are skipped (never re-polled); only the 7 running are.
        assert not any(b in client.poll_calls for b in bids[:3])
        assert sorted(client.poll_calls) == sorted(bids[3:])
        assert client.retrieve_calls == []


class TestNonTerminal:
    def test_in_progress_leaves_running_writes_no_output(self, tmp_path):
        run_dir = _submit_run(tmp_path, n=1, minibatch_size=1, batch_ids=["b0"])
        client = _FetchClient("openai", states={"b0": [BatchState.IN_PROGRESS]})
        sub = fetch_batch(client, run_dir, progress=False)
        mb = sub.minibatches[0]
        assert mb["status"] == RUNNING
        assert mb["output_path"] is None
        assert mb["counts"] == {"total": 1}  # poll counts recorded
        assert client.retrieve_calls == []
        # Ledger persisted to disk even on a no-download pass.
        assert BatchSubmission.load(run_dir).minibatches[0]["status"] == RUNNING


class TestExpired:
    def test_expired_partial_subset_persisted_with_synthetic_missing(self, tmp_path):
        run_dir = _submit_run(tmp_path, n=2, minibatch_size=2, batch_ids=["b0"])
        cids = _cids(BatchSubmission.load(run_dir))
        client = _FetchClient(
            "openai",
            states={"b0": [BatchState.EXPIRED]},
            results={
                "b0": [BatchResult(cids[0], BatchItemStatus.SUCCEEDED, text="A0")]
            },
        )
        sub = fetch_batch(client, run_dir, progress=False)
        assert sub.minibatches[0]["status"] == FETCHED  # partial subset -> fetched
        pairs = dict(iter_batch_results(run_dir))
        assert pairs["p0"].succeeded and pairs["p0"].text == "A0"
        assert pairs["p1"].status == BatchItemStatus.ERRORED  # synthetic completeness

    def test_expired_with_nothing_marks_expired_terminal(self, tmp_path):
        run_dir = _submit_run(tmp_path, n=1, minibatch_size=1, batch_ids=["b0"])
        client = _FetchClient(
            "openai", states={"b0": [BatchState.EXPIRED]}, results={"b0": []}
        )
        sub = fetch_batch(client, run_dir, progress=False)
        assert sub.minibatches[0]["status"] == EXPIRED
        assert sub.minibatches[0]["output_path"] is None
        assert sub.is_complete()  # expired-empty counts terminal for wait/completion


class TestTerminalFailures:
    def test_failed_and_cancelled_not_retried(self, tmp_path):
        run_dir = _submit_run(tmp_path, n=2, minibatch_size=1, batch_ids=["b0", "b1"])
        client = _FetchClient(
            "openai",
            states={"b0": [BatchState.FAILED], "b1": [BatchState.CANCELLED]},
        )
        sub = fetch_batch(client, run_dir, progress=False)
        assert sub.minibatches[0]["status"] == "failed"
        assert sub.minibatches[1]["status"] == CANCELLED
        client.poll_calls.clear()
        fetch_batch(client, run_dir, progress=False)
        assert client.poll_calls == []  # both terminal-failed -> skipped
        assert client.retrieve_calls == []


class TestSubmitError:
    def test_submit_error_minibatch_never_polled(self, tmp_path):
        client = _openai_submit_client()
        client.submit_batch = MagicMock(side_effect=["b0", RuntimeError("boom")])
        run_dir = submit_batch_physics_reasoning(
            client, _dataset(2), output_dir=tmp_path, run_name="run", minibatch_size=1
        )
        sub0 = BatchSubmission.load(run_dir)
        assert sub0.minibatches[1]["status"] == SUBMIT_ERROR
        assert sub0.minibatches[1]["batch_id"] == ""
        cid0 = next(iter(sub0.minibatches[0]["id_map"]))
        fc = _FetchClient(
            "openai",
            states={"b0": [BatchState.COMPLETED]},
            results={"b0": [BatchResult(cid0, BatchItemStatus.SUCCEEDED, text="A")]},
        )
        sub = fetch_batch(fc, run_dir, progress=False)
        assert fc.poll_calls == ["b0"]  # the "" submit-error id is never polled
        assert sub.minibatches[0]["status"] == FETCHED
        assert sub.minibatches[1]["status"] == SUBMIT_ERROR


class TestCorrelation:
    @pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
    def test_custom_id_to_problem_id_each_provider(self, tmp_path, provider):
        run_dir = _submit_run(
            tmp_path, n=2, minibatch_size=2, batch_ids=["b0"], provider=provider
        )
        cids = _cids(BatchSubmission.load(run_dir))
        client = _FetchClient(
            provider,
            states={"b0": [BatchState.COMPLETED]},
            results={
                "b0": [
                    BatchResult(c, BatchItemStatus.SUCCEEDED, text=f"ans-{c}")
                    for c in cids
                ]
            },
        )
        fetch_batch(client, run_dir, progress=False)
        pairs = dict(iter_batch_results(run_dir))
        assert set(pairs) == {"p0", "p1"}
        assert all(r.succeeded for r in pairs.values())

    def test_surrogate_ids_recover_problem_id(self, tmp_path):
        run_dir = _submit_run(
            tmp_path,
            n=2,
            minibatch_size=2,
            batch_ids=["b0"],
            custom_id_fn=lambda p: f"sur-{p.problem_id}",
        )
        sub0 = BatchSubmission.load(run_dir)
        assert set(sub0.minibatches[0]["id_map"]) == {"sur-p0", "sur-p1"}
        client = _FetchClient(
            "openai",
            states={"b0": [BatchState.COMPLETED]},
            results={
                "b0": [
                    BatchResult(c, BatchItemStatus.SUCCEEDED, text="x")
                    for c in _cids(sub0)
                ]
            },
        )
        fetch_batch(client, run_dir, progress=False)
        pairs = dict(iter_batch_results(run_dir))
        assert set(pairs) == {"p0", "p1"}  # recovered problem_ids, not surrogates


class TestCompleteness:
    def test_one_result_per_id_in_order_extra_dropped_and_counted(self, tmp_path):
        run_dir = _submit_run(tmp_path, n=3, minibatch_size=3, batch_ids=["b0"])
        cids = _cids(BatchSubmission.load(run_dir))  # [p0, p1, p2] in input order
        client = _FetchClient(
            "openai",
            states={"b0": [BatchState.COMPLETED]},
            results={
                "b0": [
                    BatchResult(cids[0], BatchItemStatus.SUCCEEDED, text="A0"),
                    # cids[1] omitted -> synthetic ERRORED
                    BatchResult(cids[2], BatchItemStatus.SUCCEEDED, text="A2"),
                    BatchResult("ghost", BatchItemStatus.SUCCEEDED, text="X"),  # extra
                ]
            },
        )
        sub = fetch_batch(client, run_dir, progress=False)
        pairs = list(
            iter_batch_results(sub)
        )  # pass the object to read uncorrelated_count
        assert [pid for pid, _ in pairs] == [
            "p0",
            "p1",
            "p2",
        ]  # exactly one each, in order
        by = dict(pairs)
        assert by["p0"].succeeded and by["p2"].succeeded
        assert (
            by["p1"].status == BatchItemStatus.ERRORED
        )  # synthetic for the missing id
        assert sub.minibatches[0]["uncorrelated_count"] == 1  # ghost dropped + counted


class TestHandleContract:
    def test_submit_returns_run_dir_and_fetch_accepts_path_or_object(self, tmp_path):
        run_dir = _submit_run(tmp_path, n=1, minibatch_size=1, batch_ids=["b0"])
        assert isinstance(run_dir, str)
        assert (Path(run_dir) / "metadata.json").is_file()
        cid = next(iter(BatchSubmission.load(run_dir).minibatches[0]["id_map"]))

        # (a) accepts the run_dir string (loads the ledger from disk)
        client = _FetchClient(
            "openai",
            states={"b0": [BatchState.COMPLETED]},
            results={"b0": [BatchResult(cid, BatchItemStatus.SUCCEEDED, text="A")]},
        )
        loaded = fetch_batch(client, run_dir, progress=False)
        assert isinstance(loaded, BatchSubmission)
        assert loaded.minibatches[0]["status"] == FETCHED

        # (b) accepts a BatchSubmission directly and threads the same object back
        obj = BatchSubmission.load(run_dir)
        obj.minibatches[0]["status"] = SUBMIT_ERROR  # force a skip to prove no reload
        returned = fetch_batch(client, obj, progress=False)
        assert returned is obj  # the in-memory object is used as-is (no disk reload)

    def test_iter_batch_results_is_offline(self, tmp_path):
        run_dir = _submit_run(tmp_path, n=2, minibatch_size=2, batch_ids=["b0"])
        cids = _cids(BatchSubmission.load(run_dir))
        client = _FetchClient(
            "openai",
            states={"b0": [BatchState.COMPLETED]},
            results={
                "b0": [
                    BatchResult(c, BatchItemStatus.SUCCEEDED, text="A") for c in cids
                ]
            },
        )
        fetch_batch(client, run_dir, progress=False)
        # Reads purely from disk via the run_dir handle — no client passed.
        pairs = dict(iter_batch_results(run_dir))
        assert set(pairs) == {"p0", "p1"}


class TestProgressReporting:
    def test_summary_line_emitted_with_correct_tallies(self, tmp_path, caplog):
        bids = [f"b{i}" for i in range(10)]
        run_dir = _submit_run(tmp_path, n=10, minibatch_size=1, batch_ids=bids)
        cid_of = {
            mb["batch_id"]: next(iter(mb["id_map"]))
            for mb in BatchSubmission.load(run_dir).minibatches
        }
        states = {
            bid: [BatchState.COMPLETED if i < 3 else BatchState.IN_PROGRESS]
            for i, bid in enumerate(bids)
        }
        results = {
            bid: [BatchResult(cid_of[bid], BatchItemStatus.SUCCEEDED, text="x")]
            for bid in bids[:3]
        }
        client = _FetchClient("openai", states=states, results=results)
        with caplog.at_level(logging.INFO, logger="prkit.batch"):
            fetch_batch(client, run_dir, progress=True)
        summaries = [
            r.getMessage()
            for r in caplog.records
            if r.name == "prkit.batch" and r.getMessage().startswith("Batch ")
        ]
        assert len(summaries) == 1
        line = summaries[0]
        assert "fetched 3/10 minibatches" in line
        assert "(+3 this pass)" in line
        assert "running 7" in line
        assert "failed 0" in line
        assert "not-submitted 0" in line
        assert "results 3/10" in line

    def test_progress_false_suppresses_summary(self, tmp_path, caplog):
        run_dir = _submit_run(tmp_path, n=1, minibatch_size=1, batch_ids=["b0"])
        cid = next(iter(BatchSubmission.load(run_dir).minibatches[0]["id_map"]))
        client = _FetchClient(
            "openai",
            states={"b0": [BatchState.COMPLETED]},
            results={"b0": [BatchResult(cid, BatchItemStatus.SUCCEEDED, text="A")]},
        )
        with caplog.at_level(logging.INFO, logger="prkit.batch"):
            fetch_batch(client, run_dir, progress=False)
        assert [r for r in caplog.records if r.name == "prkit.batch"] == []

    def test_completion_line_only_when_complete(self, tmp_path, caplog):
        run_dir = _submit_run(tmp_path, n=1, minibatch_size=1, batch_ids=["b0"])
        cid = next(iter(BatchSubmission.load(run_dir).minibatches[0]["id_map"]))
        client = _FetchClient(
            "openai",
            states={"b0": [BatchState.COMPLETED]},
            results={"b0": [BatchResult(cid, BatchItemStatus.SUCCEEDED, text="A")]},
        )
        with caplog.at_level(logging.INFO, logger="prkit.batch"):
            fetch_batch(client, run_dir, progress=True)
        msgs = [r.getMessage() for r in caplog.records if r.name == "prkit.batch"]
        assert any(m.startswith("✓ Batch ") and "complete" in m for m in msgs)
        # The non-complete summary line is not emitted when the pass completes.
        assert not any(m.startswith("Batch ") for m in msgs)

    def test_status_counts_tally(self, tmp_path):
        run_dir = _submit_run(
            tmp_path, n=3, minibatch_size=1, batch_ids=["b0", "b1", "b2"]
        )
        cid0 = next(iter(BatchSubmission.load(run_dir).minibatches[0]["id_map"]))
        client = _FetchClient(
            "openai",
            states={
                "b0": [BatchState.COMPLETED],
                "b1": [BatchState.IN_PROGRESS],
                "b2": [BatchState.FAILED],
            },
            results={"b0": [BatchResult(cid0, BatchItemStatus.SUCCEEDED, text="A")]},
        )
        sub = fetch_batch(client, run_dir, progress=False)
        assert sub.status_counts() == {FETCHED: 1, RUNNING: 1, "failed": 1}


class TestFacade:
    def test_fetch_facade_delegates_to_module(self):
        client = _openai_submit_client()
        with patch("prkit.batch.fetch_batch") as mock_fetch:
            mock_fetch.return_value = "LEDGER"
            out = client.fetch_batch_physics_reasoning("run-dir", wait=True)
        mock_fetch.assert_called_once()
        args, kwargs = mock_fetch.call_args
        assert args[0] is client and args[1] == "run-dir"
        assert kwargs["wait"] is True
        assert out == "LEDGER"


class TestWaitLoop:
    def test_wait_loops_until_all_terminal(self, tmp_path):
        run_dir = _submit_run(tmp_path, n=1, minibatch_size=1, batch_ids=["b0"])
        cid = next(iter(BatchSubmission.load(run_dir).minibatches[0]["id_map"]))
        client = _FetchClient(
            "openai",
            states={"b0": [BatchState.IN_PROGRESS, BatchState.COMPLETED]},
            results={"b0": [BatchResult(cid, BatchItemStatus.SUCCEEDED, text="A")]},
        )
        with patch("prkit.batch.time.sleep") as slept:
            sub = fetch_batch(
                client, run_dir, wait=True, poll_interval=0.01, progress=False
            )
        assert sub.is_complete()
        assert client.poll_calls == ["b0", "b0"]  # running, then completed
        slept.assert_called()  # slept between the two passes

    def test_wait_timeout_stops_before_completion(self, tmp_path):
        run_dir = _submit_run(tmp_path, n=1, minibatch_size=1, batch_ids=["b0"])
        client = _FetchClient(
            "openai",
            states={"b0": [BatchState.IN_PROGRESS]},  # never completes
        )
        with patch("prkit.batch.time.sleep") as slept:
            sub = fetch_batch(
                client,
                run_dir,
                wait=True,
                poll_interval=0.01,
                timeout=0,
                progress=False,
            )
        assert not sub.is_complete()
        assert client.poll_calls == ["b0"]  # timeout=0 -> exactly one pass
        slept.assert_not_called()
