"""Tests for single inference answer-tag persistence behavior."""

from __future__ import annotations

import json

from uncertainty_quantification_physical_reasoning.scripts.script_physical_reasoning.inferences import (
    single_inference_with_answer_tags as inference_script,
)


class _ExplodingClient:
    provider = "dashscope"

    def __init__(self, message: str) -> None:
        self._message = message

    def chat(self, **_kwargs):
        raise RuntimeError(self._message)


class _StaticClient:
    provider = "dashscope"

    def __init__(self, response: str) -> None:
        self._response = response

    def chat(self, **_kwargs):
        return self._response


def test_process_single_problem_persists_failure_result(tmp_path, monkeypatch):
    monkeypatch.setattr(
        inference_script,
        "format_problem_question_text_for_batch",
        lambda problem: problem.get("question"),
    )
    success, _elapsed, problem_id, error_message = (
        inference_script.process_single_problem(
            problem={"problem_id": "123", "question": "What is shown?"},
            index=1,
            total_problems=1,
            client=_ExplodingClient("provider timeout"),
            output_dir=tmp_path,
            dataset_name="seephys",
            model_name="qwen3.6-plus",
        )
    )

    assert success is False
    assert problem_id == "123"
    assert "provider timeout" in (error_message or "")

    payload = json.loads((tmp_path / "problem_123.json").read_text(encoding="utf-8"))
    assert payload["request_failed"] is True
    assert payload["request_succeeded"] is False
    assert payload["model_response"] == ""
    assert payload["model_answer"] is None
    assert payload["request_error_message"] == "Inference failed: provider timeout"
    assert payload["visual_input_mode"] == "images"


def test_process_single_problem_marks_max_token_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        inference_script,
        "format_problem_question_text_for_batch",
        lambda problem: problem.get("question"),
    )
    success, _elapsed, problem_id, error_message = (
        inference_script.process_single_problem(
            problem={"problem_id": "456", "question": "Solve it."},
            index=1,
            total_problems=1,
            client=_ExplodingClient("Request failed with finish_reason=length"),
            output_dir=tmp_path,
            dataset_name="seephys",
            model_name="qwen3.6-plus",
        )
    )

    assert success is False
    assert problem_id == "456"
    assert "finish_reason=length" in (error_message or "")

    payload = json.loads((tmp_path / "problem_456.json").read_text(encoding="utf-8"))
    assert payload["request_failed"] is True
    assert payload["request_succeeded"] is False
    assert payload["finish_reason"] == "MAX_TOKEN"


def test_process_single_problem_persists_empty_success(tmp_path, monkeypatch):
    monkeypatch.setattr(
        inference_script,
        "format_problem_question_text_for_batch",
        lambda problem: problem.get("question"),
    )
    success, _elapsed, problem_id, error_message = (
        inference_script.process_single_problem(
            problem={"problem_id": "789", "question": "Answer this."},
            index=1,
            total_problems=1,
            client=_StaticClient(""),
            output_dir=tmp_path,
            dataset_name="seephys",
            model_name="qwen3.6-plus",
        )
    )

    assert success is True
    assert problem_id == "789"
    assert error_message is None

    payload = json.loads((tmp_path / "problem_789.json").read_text(encoding="utf-8"))
    assert payload["request_failed"] is False
    assert payload["request_succeeded"] is True
    assert payload["model_response"] == ""
    assert payload["model_answer"] is None
