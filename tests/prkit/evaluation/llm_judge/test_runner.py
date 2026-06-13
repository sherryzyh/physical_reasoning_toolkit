from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from openai import BadRequestError

from prkit.evaluation.llm_judge.runner import OpenAIJudgeRunner
from prkit.evaluation.llm_judge.types import LLMJudgeResult


def _bad_request_error() -> BadRequestError:
    response = Mock()
    response.status_code = 400
    response.request = Mock()
    return BadRequestError("bad request", response=response, body={})


def test_judge_runner_properties_and_responses_path():
    client = MagicMock()
    client.responses.create.return_value = SimpleNamespace(
        output_text='{"verdict":"correct","confidence":0.9,"expected_answer_type":"other","reasoning":"ok"}'
    )

    with patch(
        "prkit.evaluation.llm_judge.runner.build_responses_api_judge_body",
        return_value={"payload": True},
    ) as mock_body:
        runner = OpenAIJudgeRunner(
            model="gpt-5.4-mini",
            instructions="grade it",
            client=client,
        )
        result = runner.judge_via_responses({"question": "Q"})

    assert runner.model_name == "gpt-5.4-mini"
    assert runner.instructions == "grade it"
    assert result.verdict == "correct"
    mock_body.assert_called_once_with(
        "gpt-5.4-mini",
        {"question": "Q"},
        instructions="grade it",
    )
    client.responses.create.assert_called_once_with(payload=True)


def test_judge_runner_chat_completions_fallback_uses_schema():
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"verdict":"incorrect","confidence":0.2,"expected_answer_type":"other","reasoning":"no"}'
                )
            )
        ]
    )
    runner = OpenAIJudgeRunner(
        model="gpt-5.4-mini",
        instructions="judge carefully",
        client=client,
        fallback_chat_model="gpt-5.4-mini",
    )

    result = runner.judge_via_chat_completions({"question": "Q"})

    assert result.verdict == "incorrect"
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-5.4-mini"
    assert call_kwargs["messages"][0] == {
        "role": "system",
        "content": "judge carefully",
    }


def test_judge_runner_retries_truncated_payload_before_returning():
    client = MagicMock()
    logger = MagicMock()
    runner = OpenAIJudgeRunner(model="gpt-5.4-mini", client=client, logger=logger)
    success = LLMJudgeResult(
        verdict="correct",
        confidence=0.8,
        expected_answer_type="other",
        reasoning="ok",
        raw_response="raw",
        verdict_type="llm_judge",
    )

    with patch.object(
        runner,
        "judge_via_responses",
        side_effect=[_bad_request_error(), success],
    ) as mock_responses, patch(
        "prkit.evaluation.llm_judge.runner.truncate_judge_payload",
        return_value={"question": "trimmed"},
    ) as mock_truncate:
        result = runner.judge({"question": "very long"})

    assert result is success
    assert mock_responses.call_args_list[1].args[0] == {"question": "trimmed"}
    mock_truncate.assert_called_once_with({"question": "very long"})
    logger.warning.assert_called_once()


def test_judge_runner_falls_back_to_chat_after_second_bad_request():
    client = MagicMock()
    logger = MagicMock()
    runner = OpenAIJudgeRunner(model="gpt-5.4-mini", client=client, logger=logger)
    fallback = LLMJudgeResult(
        verdict="incorrect",
        confidence=0.1,
        expected_answer_type="other",
        reasoning="fallback",
        raw_response="raw",
        verdict_type="llm_judge",
    )

    with patch.object(
        runner,
        "judge_via_responses",
        side_effect=[_bad_request_error(), _bad_request_error()],
    ), patch.object(
        runner,
        "judge_via_chat_completions",
        return_value=fallback,
    ) as mock_chat, patch(
        "prkit.evaluation.llm_judge.runner.truncate_judge_payload",
        return_value={"question": "trimmed"},
    ):
        result = runner.judge({"question": "long"})

    assert result is fallback
    mock_chat.assert_called_once_with({"question": "trimmed"})
    assert logger.warning.call_count == 2
