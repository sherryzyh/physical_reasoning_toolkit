from prkit.prkit_core.domain import Answer, AnswerCategory
from prkit.prkit_evaluation.comparator.normalized_match import (
    NormalizedMatchComparator,
)


def test_normalized_match_comparator_compares_numbers_by_normalized_value():
    comparator = NormalizedMatchComparator()
    assert comparator.compare("4.0", "4") is True
    assert comparator.accuracy_score("4.0", "4") == 1.0


def test_normalized_match_comparator_compares_options_case_insensitively():
    comparator = NormalizedMatchComparator()
    answer1 = Answer(value="a", answer_category=AnswerCategory.OPTION)
    answer2 = Answer(value="A", answer_category=AnswerCategory.OPTION)
    assert comparator.compare(answer1, answer2) is True


def test_normalized_match_comparator_falls_back_to_text_for_mixed_categories():
    comparator = NormalizedMatchComparator()
    assert comparator.compare("Energy", r"\text{Energy}") is False


def test_normalized_match_comparator_returns_false_for_mismatched_numbers():
    comparator = NormalizedMatchComparator()
    assert comparator.compare("4", "5") is False
    assert comparator.accuracy_score("4", "5") == 0.0
