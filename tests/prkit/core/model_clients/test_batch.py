"""Tests for free-text batch request builders and the batch job lifecycle.

Provider SDKs are faked with ``MagicMock`` so these run fully offline. The
lifecycle tests assert the status-enum mapping and per-result text extraction
for each provider; the parity tests assert a batch request body equals the body
the synchronous ``response()`` path would build from the same inputs.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from prkit.core.model_clients.base import BaseModelClient
from prkit.core.model_clients.batch_types import (
    TERMINAL_STATES,
    BatchItemStatus,
    BatchState,
    BatchStatus,
)


def _openai_client(model: str = "gpt-5.4-mini"):
    with patch("prkit.core.model_clients.openai.OpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        from prkit.core.model_clients.openai import OpenAIModel

        return OpenAIModel(model)


def _anthropic_client(model: str = "claude-opus-4-8"):
    with patch("prkit.core.model_clients.anthropic.Anthropic") as mock_cls:
        mock_cls.return_value = MagicMock()
        from prkit.core.model_clients.anthropic import AnthropicModel

        return AnthropicModel(model)


def _gemini_client(model: str = "gemini-3.5-flash"):
    with patch("prkit.core.model_clients.gemini.genai.Client") as mock_cls:
        mock_cls.return_value = MagicMock()
        from prkit.core.model_clients.gemini import GeminiModel

        return GeminiModel(model)


class TestBatchTypes:
    def test_terminal_states(self):
        assert BatchState.COMPLETED in TERMINAL_STATES
        assert BatchState.FAILED in TERMINAL_STATES
        assert BatchState.IN_PROGRESS not in TERMINAL_STATES

    def test_status_is_terminal(self):
        assert BatchStatus("b", BatchState.COMPLETED, "openai", "completed").is_terminal
        assert not BatchStatus(
            "b", BatchState.PENDING, "openai", "validating"
        ).is_terminal


class TestBaseDefaults:
    def test_batch_methods_raise_not_implemented(self):
        class Dummy(BaseModelClient):
            def response(
                self,
                input,
                image_paths=None,
                response_format=None,
                *,
                instructions=None,
                **kwargs,
            ):
                return ""

        d = Dummy("dummy-model")
        with pytest.raises(NotImplementedError):
            d.build_batch_request(request_id="r", input="x")
        with pytest.raises(NotImplementedError):
            d.submit_batch([])
        with pytest.raises(NotImplementedError):
            d.poll_batch("b")
        with pytest.raises(NotImplementedError):
            list(d.retrieve_batch_results("b"))


class TestFreeTextBuilders:
    def test_openai_shape(self):
        client = _openai_client()
        req = client.build_batch_request(
            request_id="req_1",
            input="hello",
            instructions="",
            max_output_tokens=100,
            temperature=0.0,
        )
        assert req["custom_id"] == "req_1"
        assert req["method"] == "POST"
        assert req["url"] == "/v1/responses"
        body = req["body"]
        assert body["model"] == "gpt-5.4-mini"
        assert body["input"][0]["content"][0]["text"] == "hello"
        assert "text" not in body  # no structured response_format
        assert "instructions" not in body  # instructions="" suppressed (parity hinge)
        assert body["max_output_tokens"] == 100
        assert body["temperature"] == 0.0

    def test_openai_o_family_drops_temperature_sets_reasoning(self):
        client = _openai_client("o3")
        body = client.build_batch_request(
            request_id="r",
            input="x",
            instructions="",
            max_output_tokens=50,
            temperature=0.7,
        )["body"]
        assert "temperature" not in body
        assert body["reasoning"] == {"effort": "medium"}

    def test_anthropic_shape(self):
        client = _anthropic_client()
        params = client.build_batch_request(
            request_id="req_2",
            input="hi",
            instructions="",
            max_output_tokens=64,
            temperature=0.2,
        )["params"]
        assert params["model"] == "claude-opus-4-8"
        assert params["max_tokens"] == 64
        assert params["messages"][0]["content"][0]["text"] == "hi"
        assert "output_config" not in params
        assert "system" not in params
        assert params["temperature"] == 0.2

    def test_anthropic_defaults_max_tokens_when_none(self):
        client = _anthropic_client()
        params = client.build_batch_request(request_id="r", input="x", instructions="")[
            "params"
        ]
        assert params["max_tokens"] == 1024

    def test_gemini_shape(self):
        client = _gemini_client()
        req = client.build_batch_request(
            request_id="req_3",
            input="q",
            instructions="",
            max_output_tokens=32,
            temperature=0.5,
        )
        assert req["key"] == "req_3"
        request = req["request"]
        assert request["contents"][0]["parts"][0]["text"] == "q"
        assert request["generation_config"]["max_output_tokens"] == 32
        assert request["generation_config"]["temperature"] == 0.5
        assert "response_json_schema" not in request["generation_config"]
        assert "system_instruction" not in request

    def test_instructions_included_when_provided(self):
        o = _openai_client().build_batch_request(
            request_id="a", input="x", instructions="SYS", max_output_tokens=10
        )
        assert o["body"]["instructions"] == "SYS"
        a = _anthropic_client().build_batch_request(
            request_id="a", input="x", instructions="SYS", max_output_tokens=10
        )
        assert a["params"]["system"] == "SYS"
        g = _gemini_client().build_batch_request(
            request_id="a", input="x", instructions="SYS", max_output_tokens=10
        )
        assert g["request"]["system_instruction"]["parts"][0]["text"] == "SYS"


class TestOpenAILifecycle:
    def test_submit_uploads_then_creates(self):
        client = _openai_client()
        client.client.files.create.return_value = MagicMock(id="file_123")
        client.client.batches.create.return_value = MagicMock(id="batch_abc")
        bid = client.submit_batch(
            [{"custom_id": "r1", "method": "POST", "url": "/v1/responses", "body": {}}]
        )
        assert bid == "batch_abc"
        _, kwargs = client.client.batches.create.call_args
        assert kwargs["input_file_id"] == "file_123"
        assert kwargs["endpoint"] == "/v1/responses"
        assert kwargs["completion_window"] == "24h"

    def test_poll_maps_status_and_counts(self):
        client = _openai_client()
        client.client.batches.retrieve.return_value = MagicMock(
            status="in_progress",
            output_file_id=None,
            error_file_id=None,
            request_counts=MagicMock(total=2, completed=1, failed=0),
        )
        status = client.poll_batch("batch_abc")
        assert status.state == BatchState.IN_PROGRESS
        assert status.counts["completed"] == 1
        assert not status.is_terminal

    def test_retrieve_extracts_output_text(self):
        client = _openai_client()
        line = json.dumps(
            {
                "custom_id": "r1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "output": [
                            {
                                "type": "message",
                                "content": [{"type": "output_text", "text": "ANSWER"}],
                            }
                        ]
                    },
                },
                "error": None,
            }
        )
        client.client.batches.retrieve.return_value = MagicMock(
            output_file_id="out_1", error_file_id=None
        )
        client.client.files.content.return_value = MagicMock(text=line)
        results = list(client.retrieve_batch_results("batch_abc"))
        assert len(results) == 1
        assert results[0].custom_id == "r1"
        assert results[0].status == BatchItemStatus.SUCCEEDED
        assert results[0].text == "ANSWER"

    def test_retrieve_flags_http_error(self):
        client = _openai_client()
        line = json.dumps(
            {
                "custom_id": "r2",
                "response": {"status_code": 500, "body": {}},
                "error": None,
            }
        )
        client.client.batches.retrieve.return_value = MagicMock(
            output_file_id="out_1", error_file_id=None
        )
        client.client.files.content.return_value = MagicMock(text=line)
        results = list(client.retrieve_batch_results("batch_abc"))
        assert results[0].status == BatchItemStatus.ERRORED


class TestAnthropicLifecycle:
    def test_submit(self):
        client = _anthropic_client()
        client.client.messages.batches.create.return_value = MagicMock(id="msgbatch_1")
        assert client.submit_batch([{"custom_id": "r1", "params": {}}]) == "msgbatch_1"

    def test_poll(self):
        client = _anthropic_client()
        client.client.messages.batches.retrieve.return_value = MagicMock(
            processing_status="ended",
            results_url="http://results",
            request_counts=MagicMock(
                processing=0, succeeded=2, errored=0, canceled=0, expired=0
            ),
        )
        status = client.poll_batch("msgbatch_1")
        assert status.state == BatchState.COMPLETED
        assert status.is_terminal
        assert status.output_ref == "http://results"
        assert status.counts["succeeded"] == 2

    def test_retrieve_success(self):
        client = _anthropic_client()
        entry = MagicMock(
            custom_id="r1",
            result=MagicMock(
                type="succeeded",
                message=MagicMock(content=[MagicMock(type="text", text="GRADED")]),
            ),
        )
        client.client.messages.batches.results.return_value = iter([entry])
        results = list(client.retrieve_batch_results("msgbatch_1"))
        assert results[0].custom_id == "r1"
        assert results[0].text == "GRADED"

    def test_retrieve_errored(self):
        client = _anthropic_client()
        entry = MagicMock(
            custom_id="r2", result=MagicMock(type="errored", error="boom")
        )
        client.client.messages.batches.results.return_value = iter([entry])
        results = list(client.retrieve_batch_results("msgbatch_1"))
        assert results[0].status == BatchItemStatus.ERRORED


class TestGeminiLifecycle:
    def test_submit_uploads_keyed_jsonl_file(self):
        client = _gemini_client()
        uploaded = MagicMock()
        uploaded.name = "files/in"
        client.genai_client.files.upload.return_value = uploaded
        job = MagicMock()
        job.name = "batches/xyz"
        client.genai_client.batches.create.return_value = job

        result = client.submit_batch(
            [
                {"key": "r1", "request": {"contents": []}},
                {"key": "r2", "request": {"contents": []}},
            ]
        )

        assert result == "batches/xyz"
        # The batch must be created from the uploaded file, not an inline list.
        _, create_kwargs = client.genai_client.batches.create.call_args
        assert create_kwargs["model"] == "gemini-3.5-flash"
        assert create_kwargs["src"] == "files/in"
        # The uploaded payload must be keyed JSONL so results correlate by key.
        _, upload_kwargs = client.genai_client.files.upload.call_args
        payload = upload_kwargs["file"].getvalue().decode("utf-8")
        lines = [json.loads(line) for line in payload.splitlines()]
        assert [line["key"] for line in lines] == ["r1", "r2"]

    def test_poll(self):
        client = _gemini_client()
        job = MagicMock()
        job.state = MagicMock()
        job.state.name = "JOB_STATE_SUCCEEDED"
        job.dest = MagicMock(file_name="files/out")
        client.genai_client.batches.get.return_value = job
        status = client.poll_batch("batches/xyz")
        assert status.state == BatchState.COMPLETED
        assert status.output_ref == "files/out"

    def test_retrieve_file(self):
        client = _gemini_client()
        job = MagicMock()
        job.dest = MagicMock()
        job.dest.file_name = "files/out"
        client.genai_client.batches.get.return_value = job
        line = json.dumps(
            {
                "key": "r2",
                "response": {
                    "candidates": [{"content": {"parts": [{"text": "FILETEXT"}]}}]
                },
            }
        )
        client.genai_client.files.download.return_value = line.encode("utf-8")
        results = list(client.retrieve_batch_results("batches/xyz"))
        assert results[0].custom_id == "r2"
        assert results[0].text == "FILETEXT"

    def test_retrieve_without_output_file_yields_nothing(self):
        # No destination file => no results, rather than silently miscorrelated ones.
        client = _gemini_client()
        job = MagicMock()
        job.dest = MagicMock()
        job.dest.file_name = None
        client.genai_client.batches.get.return_value = job
        assert list(client.retrieve_batch_results("batches/xyz")) == []
        client.genai_client.files.download.assert_not_called()


class TestSyncBatchParity:
    """A batch request body must equal the body sync ``response()`` would build."""

    def test_openai_body_parity(self):
        client = _openai_client()
        instr = client._resolve_instructions("")
        sync_body = client._build_responses_body(
            input="P",
            instructions=instr,
            image_paths=None,
            max_output_tokens=123,
            response_format=None,
            extra={"temperature": 0.0},
        )
        batch_body = client.build_batch_request(
            request_id="r",
            input="P",
            instructions="",
            max_output_tokens=123,
            temperature=0.0,
        )["body"]
        assert batch_body == sync_body

    def test_gemini_request_shape_is_pinned(self):
        """Gemini has no shared body builder, so pin the whole emitted dict.

        OpenAI and Anthropic get parity for free by sharing one builder between
        ``response()`` and the batch path, so a drift shows up as an inequality.
        Gemini constructs its batch line inline and independently, so the only
        way to notice an unintended change is to compare the entire structure.
        """
        client = _gemini_client()

        request = client.build_batch_request(
            request_id="r",
            input="P",
            instructions="",
            max_output_tokens=123,
            temperature=0.0,
        )

        assert request == {
            "key": "r",
            "request": {
                "contents": [{"parts": [{"text": "P"}], "role": "user"}],
                "generation_config": {"max_output_tokens": 123, "temperature": 0.0},
            },
        }

    def test_gemini_minimal_request_omits_generation_config(self):
        """No cap and no temperature must leave the key out entirely.

        Pins the ``if generation_config:`` guard, which any change that
        unconditionally seeds that dict would silently break.
        """
        client = _gemini_client()

        request = client.build_batch_request(request_id="r", input="P", instructions="")

        assert request == {
            "key": "r",
            "request": {"contents": [{"parts": [{"text": "P"}], "role": "user"}]},
        }

    def test_anthropic_params_parity(self):
        client = _anthropic_client()
        instr = client._resolve_instructions("")
        sync_params = client._build_messages_params(
            input="P",
            instructions=instr,
            image_paths=None,
            max_output_tokens=64,
            response_format=None,
            extra={"temperature": 0.2},
        )
        batch_params = client.build_batch_request(
            request_id="r",
            input="P",
            instructions="",
            max_output_tokens=64,
            temperature=0.2,
        )["params"]
        assert batch_params == sync_params
