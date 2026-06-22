"""Tests for ``prkit.batch.submit_batch_physics_reasoning`` (the submit orchestrator).

Provider SDKs are faked with ``MagicMock`` so these run fully offline (mirroring
``tests/prkit/core/model_clients/test_batch.py``); artifacts are written under
pytest's ``tmp_path``. The tests cover splitting, JSONL artifacts, submission,
validation, surrogate ids, partial failure, the facade, the Anthropic inline
path, the run-folder layout, ``metadata.json`` content, id round-trips, defaults,
and overwrite behavior.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from prkit.batch import (
    BatchInputError,
    BatchSubmission,
    submit_batch_physics_reasoning,
)
from prkit.core.domain import PhysicsDataset, PhysicsProblem


# --------------------------------------------------------------------------- #
# Offline client builders (SDK constructor patched -> no key / network needed) #
# --------------------------------------------------------------------------- #
def _openai_client(model: str = "gpt-5.1"):
    with patch("prkit.core.model_clients.openai.OpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        from prkit.core.model_clients.openai import OpenAIModel

        client = OpenAIModel(model)
    client.client.files.create.return_value = MagicMock(id="file_1")
    client.client.batches.create.return_value = MagicMock(id="batch_abc")
    return client


def _anthropic_client(model: str = "claude-opus-4-8"):
    with patch("prkit.core.model_clients.anthropic.Anthropic") as mock_cls:
        mock_cls.return_value = MagicMock()
        from prkit.core.model_clients.anthropic import AnthropicModel

        client = AnthropicModel(model)
    client.client.messages.batches.create.return_value = MagicMock(id="msgbatch_1")
    return client


def _gemini_client(model: str = "gemini-3.5-flash"):
    with patch("prkit.core.model_clients.gemini.genai.Client") as mock_cls:
        mock_cls.return_value = MagicMock()
        from prkit.core.model_clients.gemini import GeminiModel

        client = GeminiModel(model)
    uploaded = MagicMock()
    uploaded.name = "files/in"
    client.genai_client.files.upload.return_value = uploaded
    job = MagicMock()
    job.name = "batches/xyz"
    client.genai_client.batches.create.return_value = job
    return client


def _dataset(n: int, *, name: str = "physreason", version: str = "1.0"):
    problems = [PhysicsProblem(problem_id=f"p{i}", question=f"Q{i}") for i in range(n)]
    return PhysicsDataset(problems, info={"name": name, "version": version})


def _problems(n: int):
    return [PhysicsProblem(problem_id=f"p{i}", question=f"Q{i}") for i in range(n)]


# --------------------------------------------------------------------------- #
class TestSplitting:
    def test_1200_problems_split_into_three_batches(self, tmp_path):
        client = _openai_client()
        subs = submit_batch_physics_reasoning(
            client, _dataset(1200), output_dir=tmp_path, batch_size=500
        )
        assert [s.num_requests for s in subs] == [500, 500, 200]
        assert [s.batch_index for s in subs] == [0, 1, 2]
        run_dir = tmp_path / subs[0].input_file_path.split("/")[-3]
        files = sorted(p.name for p in (run_dir / "inputs").glob("*.jsonl"))
        assert files == ["batch_0000.jsonl", "batch_0001.jsonl", "batch_0002.jsonl"]


class TestJsonlArtifact:
    def test_one_json_object_per_line_roundtrips(self, tmp_path):
        client = _openai_client()
        subs = submit_batch_physics_reasoning(
            client, _dataset(3), output_dir=tmp_path, batch_size=2
        )
        lines = (
            (tmp_path / subs[0].input_file_path.split("/")[-3] / "inputs")
            .joinpath("batch_0000.jsonl")
            .read_text()
            .splitlines()
        )
        assert len(lines) == 2
        objs = [json.loads(line) for line in lines]
        assert [o["custom_id"] for o in objs] == ["p0", "p1"]
        assert all("body" in o for o in objs)


class TestSubmission:
    def test_submit_called_once_per_batch_with_its_chunk(self, tmp_path):
        client = _openai_client()
        client.submit_batch = MagicMock(side_effect=["b0", "b1", "b2"])
        submit_batch_physics_reasoning(
            client, _dataset(5), output_dir=tmp_path, batch_size=2
        )
        assert client.submit_batch.call_count == 3
        chunk_ids = [
            [r["custom_id"] for r in call.args[0]]
            for call in client.submit_batch.call_args_list
        ]
        assert chunk_ids == [["p0", "p1"], ["p2", "p3"], ["p4"]]

    def test_receipt_carries_stub_fields(self, tmp_path):
        client = _openai_client()
        subs = submit_batch_physics_reasoning(
            client, _dataset(2), output_dir=tmp_path, batch_size=2
        )
        (s,) = subs
        assert s.batch_id == "batch_abc"
        assert s.num_requests == 2
        assert s.provider == "openai"
        assert s.model == "gpt-5.1"
        assert s.error is None
        assert isinstance(s.submitted_at, datetime)
        assert s.submitted_at.tzinfo == timezone.utc
        assert s.id_map == {"p0": "p0", "p1": "p1"}

    def test_display_name_routed_through_submit_metadata(self, tmp_path):
        client = _openai_client()
        client.submit_batch = MagicMock(return_value="b0")
        subs = submit_batch_physics_reasoning(
            client, _dataset(1), output_dir=tmp_path, run_name="my-run"
        )
        _, kwargs = client.submit_batch.call_args
        assert kwargs["metadata"]["display_name"] == "my-run"
        assert subs[0].metadata["display_name"] == "my-run"


class TestValidation:
    def test_duplicate_problem_id_raises_before_submit(self, tmp_path):
        client = _openai_client()
        client.submit_batch = MagicMock()
        problems = [
            PhysicsProblem(problem_id="dup", question="A"),
            PhysicsProblem(problem_id="dup", question="B"),
        ]
        with pytest.raises(BatchInputError, match="Duplicate"):
            submit_batch_physics_reasoning(client, problems, output_dir=tmp_path)
        client.submit_batch.assert_not_called()

    def test_over_64_char_id_openai_raises(self, tmp_path):
        client = _openai_client()
        long_id = "x" * 65
        problems = [PhysicsProblem(problem_id=long_id, question="A")]
        with pytest.raises(BatchInputError, match="64-char"):
            submit_batch_physics_reasoning(client, problems, output_dir=tmp_path)

    def test_illegal_charset_id_anthropic_raises(self, tmp_path):
        client = _anthropic_client()
        problems = [PhysicsProblem(problem_id="has space", question="A")]
        with pytest.raises(BatchInputError, match="Anthropic"):
            submit_batch_physics_reasoning(client, problems, output_dir=tmp_path)

    def test_empty_input_raises(self, tmp_path):
        client = _openai_client()
        with pytest.raises(BatchInputError, match="empty"):
            submit_batch_physics_reasoning(client, [], output_dir=tmp_path)

    def test_non_positive_batch_size_raises(self, tmp_path):
        client = _openai_client()
        with pytest.raises(BatchInputError, match="batch_size must be positive"):
            submit_batch_physics_reasoning(
                client, _dataset(2), output_dir=tmp_path, batch_size=0
            )


class TestBatchSubmitError:
    def test_carries_successes_and_failed_for_resume(self):
        from prkit.batch import BatchSubmitError

        ok = BatchSubmission(
            provider="openai",
            model="gpt-5.1",
            batch_id="batch_0",
            input_file_path="/x/inputs/batch_0000.jsonl",
            submitted_at=datetime.now(timezone.utc),
            num_requests=1,
            batch_index=0,
            id_map={"p0": "p0"},
        )
        failed = BatchSubmission(
            provider="openai",
            model="gpt-5.1",
            batch_id="",
            input_file_path="/x/inputs/batch_0001.jsonl",
            submitted_at=datetime.now(timezone.utc),
            num_requests=1,
            batch_index=1,
            id_map={"p1": "p1"},
            error="RuntimeError: boom",
        )
        err = BatchSubmitError("submit failed", successes=[ok], failed=failed)
        assert err.successes == [ok]
        assert err.failed is failed
        assert isinstance(err, RuntimeError)


class TestSurrogateIds:
    def test_custom_id_fn_surrogates_recovered_via_id_map(self, tmp_path):
        client = _openai_client()
        subs = submit_batch_physics_reasoning(
            client,
            _dataset(2),
            output_dir=tmp_path,
            batch_size=2,
            custom_id_fn=lambda p: f"sur-{p.problem_id}",
        )
        (s,) = subs
        assert s.id_map == {"sur-p0": "p0", "sur-p1": "p1"}
        # The wire custom_id is the surrogate.
        line = json.loads(
            (tmp_path / s.input_file_path.split("/")[-3] / "inputs")
            .joinpath("batch_0000.jsonl")
            .read_text()
            .splitlines()[0]
        )
        assert line["custom_id"] == "sur-p0"


class TestPartialFailure:
    def test_middle_batch_failure_recorded_others_succeed(self, tmp_path):
        client = _openai_client()
        client.client.batches.create.side_effect = [
            MagicMock(id="batch_0"),
            RuntimeError("boom"),
            MagicMock(id="batch_2"),
        ]
        subs = submit_batch_physics_reasoning(
            client, _dataset(5), output_dir=tmp_path, batch_size=2
        )
        assert len(subs) == 3
        assert subs[0].error is None and subs[0].batch_id == "batch_0"
        assert subs[1].error is not None and "boom" in subs[1].error
        assert subs[1].batch_id == ""
        assert subs[2].error is None and subs[2].batch_id == "batch_2"
        # Failed batch's input file still exists for resume.
        for s in subs:
            assert (
                tmp_path / s.input_file_path.split(str(tmp_path) + "/")[-1]
            ).exists()


class TestFacade:
    def test_client_facade_delegates_to_module(self, tmp_path):
        client = _openai_client()
        ds = _dataset(1)
        with patch("prkit.batch.submit_batch_physics_reasoning") as mock_submit:
            mock_submit.return_value = []
            client.submit_batch_physics_reasoning(ds, output_dir=tmp_path)
        mock_submit.assert_called_once()
        args, kwargs = mock_submit.call_args
        assert args[0] is client and args[1] is ds
        assert kwargs["output_dir"] == tmp_path


class TestAnthropicInline:
    def test_artifact_written_and_inline_list_submitted(self, tmp_path):
        client = _anthropic_client()
        subs = submit_batch_physics_reasoning(
            client, _dataset(2), output_dir=tmp_path, batch_size=2
        )
        (s,) = subs
        # The .jsonl artifact is still written for Anthropic (audit/resume).
        artifact = (
            tmp_path / s.input_file_path.split("/")[-3] / "inputs" / "batch_0000.jsonl"
        )
        assert artifact.exists()
        # submit_batch received the inline list of request dicts (no file upload).
        _, kwargs = client.client.messages.batches.create.call_args
        inline = kwargs["requests"]
        assert [r["custom_id"] for r in inline] == ["p0", "p1"]
        client.client.files.create.assert_not_called()


class TestRunFolderLayout:
    def test_layout_and_paths(self, tmp_path):
        client = _openai_client()
        subs = submit_batch_physics_reasoning(
            client, _dataset(3), output_dir=tmp_path, run_name="run-x", batch_size=2
        )
        run_dir = tmp_path / "run-x"
        assert (run_dir / "metadata.json").is_file()
        assert (run_dir / "inputs" / "batch_0000.jsonl").is_file()
        for s in subs:
            assert s.input_file_path.startswith(str(run_dir / "inputs"))
            assert s.metadata_path == str(run_dir / "metadata.json")


class TestMetadataContent:
    def test_metadata_written_before_return_with_header_and_submissions(self, tmp_path):
        client = _openai_client()
        subs = submit_batch_physics_reasoning(
            client, _dataset(3), output_dir=tmp_path, run_name="run-x", batch_size=2
        )
        record = json.loads((tmp_path / "run-x" / "metadata.json").read_text())
        assert record["run_id"] == "run-x"
        assert record["provider"] == "openai"
        assert record["model"] == "gpt-5.1"
        assert record["batch_size"] == 2
        assert record["total_problems"] == 3
        assert record["num_batches"] == 2
        assert record["request_kind"] == "free_text"
        assert record["prkit_api_version"]
        assert record["dataset"] == {"name": "physreason", "version": "1.0"}
        assert record["submissions"] == [s.to_dict() for s in subs]

    def test_bare_list_omits_dataset_field_and_segment(self, tmp_path):
        client = _openai_client()
        subs = submit_batch_physics_reasoning(
            client, _problems(2), output_dir=tmp_path, batch_size=2
        )
        record = json.loads(
            (
                tmp_path / subs[0].metadata_path.split("/")[-2] / "metadata.json"
            ).read_text()
        )
        assert record["dataset"] is None
        # run name omits the <dataset> segment -> starts with the slugified model.
        assert record["run_id"].startswith("gpt-5.1-")


class TestIdNormalizationRoundTrip:
    @pytest.mark.parametrize(
        "builder,expected",
        [
            (_openai_client, "batch_abc"),
            (_anthropic_client, "msgbatch_1"),
            (_gemini_client, "batches/xyz"),
        ],
    )
    def test_wire_id_lands_verbatim(self, builder, expected, tmp_path):
        client = builder()
        subs = submit_batch_physics_reasoning(
            client, _dataset(1), output_dir=tmp_path, batch_size=1
        )
        assert subs[0].batch_id == expected

    def test_openai_id_accepted_by_poll_batch(self, tmp_path):
        client = _openai_client()
        subs = submit_batch_physics_reasoning(
            client, _dataset(1), output_dir=tmp_path, batch_size=1
        )
        client.client.batches.retrieve.return_value = MagicMock(
            status="completed",
            output_file_id=None,
            error_file_id=None,
            request_counts=MagicMock(total=1, completed=1, failed=0),
        )
        status = client.poll_batch(subs[0].batch_id)
        assert status.batch_id == "batch_abc"


class TestDefaultsAndOverrides:
    def test_default_output_dir_and_display_name(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        client = _openai_client()
        subs = submit_batch_physics_reasoning(client, _dataset(1))
        run_id = subs[0].metadata_path.split("/")[-2]
        assert (tmp_path / "batch_runs" / run_id / "metadata.json").is_file()
        assert subs[0].metadata["display_name"] == run_id

    def test_explicit_run_name_is_slugified(self, tmp_path):
        client = _openai_client()
        subs = submit_batch_physics_reasoning(
            client, _dataset(1), output_dir=tmp_path, run_name="My Eval 001!"
        )
        assert (tmp_path / "my-eval-001").is_dir()
        assert subs[0].metadata_path == str(tmp_path / "my-eval-001" / "metadata.json")


class TestOverwrite:
    def test_existing_nonempty_folder_refused_by_default(self, tmp_path):
        client = _openai_client()
        submit_batch_physics_reasoning(
            client, _dataset(1), output_dir=tmp_path, run_name="run-x"
        )
        with pytest.raises(BatchInputError, match="already exists"):
            submit_batch_physics_reasoning(
                client, _dataset(1), output_dir=tmp_path, run_name="run-x"
            )

    def test_overwrite_true_succeeds_and_clears_stale_inputs(self, tmp_path):
        client = _openai_client()
        submit_batch_physics_reasoning(
            client, _dataset(1), output_dir=tmp_path, run_name="run-x"
        )
        stale = tmp_path / "run-x" / "inputs" / "batch_0005.jsonl"
        stale.write_text("stale\n")
        subs = submit_batch_physics_reasoning(
            client, _dataset(1), output_dir=tmp_path, run_name="run-x", overwrite=True
        )
        assert len(subs) == 1
        assert not stale.exists()

    def test_returns_list_of_batch_submission(self, tmp_path):
        client = _openai_client()
        subs = submit_batch_physics_reasoning(client, _dataset(1), output_dir=tmp_path)
        assert all(isinstance(s, BatchSubmission) for s in subs)
