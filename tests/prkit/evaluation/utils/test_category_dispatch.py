from prkit.core.domain.answer_category import AnswerCategory
from prkit.evaluation.utils.category_dispatch import compare_by_category


def test_compare_by_category_normalizes_text_before_dispatch():
    seen = {}

    def compare_text(predicted, ground_truth):
        seen["values"] = (predicted, ground_truth)
        return predicted == ground_truth

    assert (
        compare_by_category(
            AnswerCategory.TEXT,
            "  HELLO,\nWorld  ",
            "HELLO, World",
            {AnswerCategory.TEXT: compare_text},
        )
        is True
    )
    assert seen["values"] == ("HELLO, World", "HELLO, World")


def test_compare_by_category_falls_back_to_plain_text_on_exception():
    class Logger:
        def __init__(self):
            self.messages = []

        def warning(self, message):
            self.messages.append(message)

    logger = Logger()

    def exploding_compare(_predicted, _ground_truth):
        raise RuntimeError("boom")

    assert (
        compare_by_category(
            AnswerCategory.NUMBER,
            "42",
            "42",
            {AnswerCategory.NUMBER: exploding_compare},
            logger,
        )
        is True
    )
    assert logger.messages


def test_compare_by_category_uses_plain_text_for_unknown_categories():
    assert (
        compare_by_category(
            AnswerCategory.EQUATION,
            "Eq(x, 2)",
            "Eq(x, 2)",
            {},
        )
        is True
    )
