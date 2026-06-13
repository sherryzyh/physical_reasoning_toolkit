from prkit.evaluation.llm_judge.parse import (
    extract_json_object,
    parse_judge_response,
    reasoning_indicates_incorrect,
)


def test_extract_json_object_accepts_full_and_embedded_json():
    assert extract_json_object('{"verdict":"correct"}') == {"verdict": "correct"}
    assert extract_json_object('prefix {"verdict": "incorrect"} suffix') == {
        "verdict": "incorrect"
    }
    assert extract_json_object("not json at all") is None


def test_reasoning_indicates_incorrect_recognizes_red_flags():
    assert reasoning_indicates_incorrect(
        "The answer is incorrect because of a wrong sign."
    )
    assert not reasoning_indicates_incorrect("Equivalent after simplification.")


def test_parse_judge_response_normalizes_invalid_fields_and_fallback_text():
    parsed = parse_judge_response(
        '{"verdict":"maybe","confidence":"oops","expected_answer_type":"mystery","reasoning":"fine"}'
    )
    assert parsed.verdict == "incorrect"
    assert parsed.confidence == 0.5
    assert parsed.expected_answer_type == "other"

    fallback = parse_judge_response("No, this does not match.")
    assert fallback.verdict == "incorrect"
    assert fallback.expected_answer_type == "other"


def test_parse_judge_response_clamps_confidence_and_overrides_contradictory_reasoning():
    parsed = parse_judge_response(
        '{"verdict":"correct","confidence":2.0,"expected_answer_type":"numeric_value","reasoning":"The model answer is incorrect because it is missing factor 2."}'
    )
    assert parsed.verdict == "incorrect"
    assert parsed.confidence == 0.35
