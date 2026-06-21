from prkit.core.domain.answer import Answer
from prkit.evaluation.llm_judge.payload import (
    answer_to_text_and_category,
    build_standard_answer_judge_payload,
    clean_answer_text,
    truncate_judge_payload,
)


def test_answer_to_text_and_category_for_answers_and_plain_strings():
    # Answer with source_type → category is the source_type string
    answer = Answer(value=" 42 ", source_type="NV")
    assert answer_to_text_and_category(answer) == ("42", "NV")

    # Answer without source_type → empty string
    answer_no_type = Answer(value=" 42 ")
    assert answer_to_text_and_category(answer_no_type) == ("42", "")

    # Plain string → empty string category
    assert answer_to_text_and_category("  free text  ") == ("free text", "")


def test_build_standard_answer_judge_payload_cleans_fields():
    payload = build_standard_answer_judge_payload(
        Answer(value=" 10  m/s "),
        Answer(value=" 10\tm/s "),
        "  What is the speed?  ",
    )

    assert payload == {
        "question": "What is the speed?",
        "ground_truth": {"text": "10 m/s", "category": ""},
        "model_answer": {"text": "10 m/s", "category": ""},
    }
    assert clean_answer_text(" a  \t b ") == "a b"


def test_truncate_judge_payload_trims_long_fields_and_preserves_shape():
    payload = {
        "question": "abcdefgh",
        "ground_truth": {"text": "12345678", "category": "number"},
        "model_answer": {"text": "short", "category": "number"},
    }

    truncated = truncate_judge_payload(payload, max_chars=6)

    assert truncated["question"] == "abc..."
    assert truncated["ground_truth"]["text"] == "123..."
    assert truncated["model_answer"]["text"] == "short"
