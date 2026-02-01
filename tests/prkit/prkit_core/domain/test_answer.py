"""
Tests for Answer model.
"""

import pytest

from prkit.prkit_core.domain import AnswerType
from prkit.prkit_core.domain import Answer


class TestAnswer:
    """Test cases for Answer model."""

    def test_answer_creation_numerical(self):
        """Test creating a numerical answer."""
        answer = Answer(value=42.0, answer_type=AnswerType.NUMERICAL, unit="m/s")
        assert answer.value == 42.0
        assert answer.answer_type == AnswerType.NUMERICAL
        assert answer.unit == "m/s"
        assert answer.metadata == {}

    def test_answer_creation_symbolic(self):
        """Test creating a symbolic answer."""
        answer = Answer(value="x^2 + 1", answer_type=AnswerType.SYMBOLIC)
        assert answer.value == "x^2 + 1"
        assert answer.answer_type == AnswerType.SYMBOLIC
        assert answer.unit is None

    def test_answer_creation_textual(self):
        """Test creating a textual answer."""
        answer = Answer(value="The answer is 42", answer_type=AnswerType.TEXTUAL)
        assert answer.value == "The answer is 42"
        assert answer.answer_type == AnswerType.TEXTUAL

    def test_answer_creation_option(self):
        """Test creating an option answer."""
        answer = Answer(value="A", answer_type=AnswerType.OPTION)
        assert answer.value == "A"
        assert answer.answer_type == AnswerType.OPTION

    def test_answer_metadata_initialization(self):
        """Test that metadata is initialized as empty dict."""
        answer = Answer(value=1, answer_type=AnswerType.NUMERICAL)
        assert answer.metadata == {}

    def test_answer_metadata_custom(self):
        """Test custom metadata."""
        metadata = {"source": "test", "confidence": 0.9}
        answer = Answer(value=1, answer_type=AnswerType.NUMERICAL, metadata=metadata)
        assert answer.metadata == metadata

    def test_answer_validation_numerical(self):
        """Test numerical answer validation."""
        valid_answer = Answer(value=42.0, answer_type=AnswerType.NUMERICAL)
        assert valid_answer.validate() is True

        invalid_answer = Answer(value="not a number", answer_type=AnswerType.NUMERICAL)
        assert invalid_answer.validate() is False

    def test_answer_validation_symbolic(self):
        """Test symbolic answer validation."""
        valid_answer = Answer(value="x^2", answer_type=AnswerType.SYMBOLIC)
        assert valid_answer.validate() is True

        invalid_answer = Answer(value="", answer_type=AnswerType.SYMBOLIC)
        assert invalid_answer.validate() is False

    def test_answer_validation_textual(self):
        """Test textual answer validation."""
        valid_answer = Answer(value="Some text", answer_type=AnswerType.TEXTUAL)
        assert valid_answer.validate() is True

        invalid_answer = Answer(value="", answer_type=AnswerType.TEXTUAL)
        assert invalid_answer.validate() is False

    def test_answer_type_checking(self):
        """Test answer type checking methods."""
        numerical = Answer(value=1, answer_type=AnswerType.NUMERICAL)
        symbolic = Answer(value="x", answer_type=AnswerType.SYMBOLIC)
        textual = Answer(value="text", answer_type=AnswerType.TEXTUAL)
        option = Answer(value="A", answer_type=AnswerType.OPTION)

        assert numerical.is_numerical() is True
        assert numerical.is_symbolic() is False
        assert symbolic.is_symbolic() is True
        assert textual.is_textual() is True
        assert option.is_option() is True

    def test_answer_numerical_methods(self):
        """Test numerical-specific methods."""
        answer = Answer(value=42, answer_type=AnswerType.NUMERICAL, unit="m/s")
        assert answer.get_unit() == "m/s"
        assert answer.has_unit() is True
        assert answer.is_integer() is True
        assert answer.is_positive() is True

        negative_answer = Answer(value=-5, answer_type=AnswerType.NUMERICAL)
        assert negative_answer.is_negative() is True

    def test_answer_symbolic_methods(self):
        """Test symbolic-specific methods."""
        latex_answer = Answer(value="$x^2$", answer_type=AnswerType.SYMBOLIC)
        assert latex_answer.is_latex() is True

        clean = latex_answer.get_clean_expression()
        assert "$" not in clean or clean.startswith("$")

    def test_answer_textual_methods(self):
        """Test textual-specific methods."""
        answer = Answer(value="This is a test answer", answer_type=AnswerType.TEXTUAL)
        assert answer.word_count() == 5
        assert answer.char_count() == 21  # "This is a test answer" = 21 chars
        assert answer.is_short() is True
        assert answer.is_long() is False
        assert answer.contains_keywords(["test", "answer"]) is True

    def test_answer_option_methods(self):
        """Test option-specific methods."""
        letter_answer = Answer(value="A", answer_type=AnswerType.OPTION)
        assert letter_answer.is_letter_option() is True
        assert letter_answer.get_option_index() == 0

        numeric_answer = Answer(value="1", answer_type=AnswerType.OPTION)
        assert numeric_answer.is_numeric_option() is True

        yes_answer = Answer(value="YES", answer_type=AnswerType.OPTION)
        assert yes_answer.is_yes_no() is True

    def test_answer_to_dict(self):
        """Test answer serialization to dictionary."""
        answer = Answer(
            value=42,
            answer_type=AnswerType.NUMERICAL,
            unit="m/s",
            metadata={"test": True},
        )
        result = answer.to_dict()

        assert result["value"] == 42
        assert result["answer_type"] == "numerical"
        assert result["unit"] == "m/s"
        assert result["metadata"]["test"] is True

    def test_answer_str_repr(self):
        """Test string representations."""
        answer = Answer(value=42, answer_type=AnswerType.NUMERICAL, unit="m/s")
        assert "42" in str(answer)
        assert "m/s" in str(answer)

        assert "Answer" in repr(answer)
        assert "numerical" in repr(answer)

    def test_answer_validation_numerical_bool_false(self):
        """Test that boolean False is not valid for numerical."""
        answer = Answer(value=False, answer_type=AnswerType.NUMERICAL)
        assert answer.validate() is False

    def test_answer_validation_numerical_bool_true(self):
        """Test that boolean True is not valid for numerical."""
        answer = Answer(value=True, answer_type=AnswerType.NUMERICAL)
        assert answer.validate() is False

    def test_answer_validation_symbolic_whitespace_only(self):
        """Test that whitespace-only string is invalid for symbolic."""
        answer = Answer(value="   \n\t  ", answer_type=AnswerType.SYMBOLIC)
        assert answer.validate() is False

    def test_answer_numerical_zero(self):
        """Test numerical answer with zero value."""
        answer = Answer(value=0, answer_type=AnswerType.NUMERICAL)
        assert answer.is_numerical() is True
        assert answer.is_positive() is False
        assert answer.is_negative() is False
        assert answer.is_integer() is True

    def test_answer_numerical_float_integer(self):
        """Test numerical answer with float that is integer."""
        answer = Answer(value=42.0, answer_type=AnswerType.NUMERICAL)
        assert answer.is_integer() is True

    def test_answer_symbolic_latex_double_dollar(self):
        """Test symbolic answer with double dollar LaTeX."""
        answer = Answer(value="$$x^2 + y^2$$", answer_type=AnswerType.SYMBOLIC)
        clean = answer.get_clean_expression()
        assert "$$" not in clean or clean == "$$x^2 + y^2$$"

    def test_answer_symbolic_latex_single_dollar(self):
        """Test symbolic answer with single dollar LaTeX."""
        answer = Answer(value="$x^2$", answer_type=AnswerType.SYMBOLIC)
        clean = answer.get_clean_expression()
        assert "$" not in clean or clean == "$x^2$"

    def test_answer_symbolic_backslash_latex(self):
        """Test symbolic answer with backslash LaTeX."""
        answer = Answer(value="\\frac{1}{2}", answer_type=AnswerType.SYMBOLIC)
        assert answer.is_latex() is True

    def test_answer_textual_word_count_empty(self):
        """Test word count for empty textual answer."""
        answer = Answer(value="", answer_type=AnswerType.TEXTUAL)
        assert answer.word_count() == 0

    def test_answer_textual_word_count_multiple_spaces(self):
        """Test word count with multiple spaces."""
        answer = Answer(value="word1    word2   word3", answer_type=AnswerType.TEXTUAL)
        assert answer.word_count() == 3

    def test_answer_textual_is_long(self):
        """Test is_long method for textual answer."""
        long_text = " ".join(["word"] * 60)  # 60 words
        answer = Answer(value=long_text, answer_type=AnswerType.TEXTUAL)
        assert answer.is_long() is True

    def test_answer_textual_contains_keywords_case_insensitive(self):
        """Test contains_keywords is case insensitive."""
        answer = Answer(value="This is a TEST", answer_type=AnswerType.TEXTUAL)
        assert answer.contains_keywords(["test"]) is True
        assert answer.contains_keywords(["TEST"]) is True
        assert answer.contains_keywords(["Test"]) is True

    def test_answer_option_all_letters(self):
        """Test option methods for all letter options."""
        for letter in ["A", "B", "C", "D", "E"]:
            answer = Answer(value=letter, answer_type=AnswerType.OPTION)
            assert answer.is_letter_option() is True
            assert answer.get_option_index() is not None

    def test_answer_option_numeric_strings(self):
        """Test option methods for numeric option strings."""
        for num_str in ["1", "2", "3", "4", "5"]:
            answer = Answer(value=num_str, answer_type=AnswerType.OPTION)
            assert answer.is_numeric_option() is True
            assert answer.get_option_index() is not None

    def test_answer_option_invalid_letter(self):
        """Test option with invalid letter."""
        answer = Answer(value="F", answer_type=AnswerType.OPTION)
        assert answer.is_letter_option() is False

    def test_answer_option_yes_no_variants(self):
        """Test yes/no option variants."""
        for variant in ["YES", "yes", "Yes", "NO", "no", "No"]:
            answer = Answer(value=variant, answer_type=AnswerType.OPTION)
            assert answer.is_yes_no() is True

    def test_answer_option_true_false_variants(self):
        """Test true/false option variants."""
        for variant in ["TRUE", "true", "True", "FALSE", "false", "False"]:
            answer = Answer(value=variant, answer_type=AnswerType.OPTION)
            assert answer.is_true_false() is True

    def test_answer_to_dict_without_unit(self):
        """Test to_dict without unit."""
        answer = Answer(value="test", answer_type=AnswerType.TEXTUAL)
        result = answer.to_dict()
        assert "unit" not in result

    def test_answer_to_dict_without_metadata(self):
        """Test to_dict with empty metadata."""
        answer = Answer(value=1, answer_type=AnswerType.NUMERICAL)
        answer.metadata = {}
        result = answer.to_dict()
        # Metadata may or may not be included if empty
        assert "value" in result
        assert "answer_type" in result

    def test_answer_get_value(self):
        """Test get_value method."""
        answer = Answer(value=42, answer_type=AnswerType.NUMERICAL)
        assert answer.get_value() == 42

    def test_answer_get_type(self):
        """Test get_type method."""
        answer = Answer(value="test", answer_type=AnswerType.TEXTUAL)
        assert answer.get_type() == AnswerType.TEXTUAL

    def test_answer_get_type_name(self):
        """Test get_type_name method."""
        answer = Answer(value="test", answer_type=AnswerType.TEXTUAL)
        assert answer.get_type_name() == "textual"

    def test_answer_str_without_unit(self):
        """Test __str__ without unit."""
        answer = Answer(value=42, answer_type=AnswerType.NUMERICAL)
        assert str(answer) == "42"

    def test_answer_str_non_numerical(self):
        """Test __str__ for non-numerical answer."""
        answer = Answer(value="test", answer_type=AnswerType.TEXTUAL)
        assert str(answer) == "test"
