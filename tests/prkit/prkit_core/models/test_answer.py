"""
Tests for Answer model.
"""

import pytest
from prkit.prkit_core.models import Answer
from prkit.prkit_core.definitions import AnswerType


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
        answer = Answer(value=42, answer_type=AnswerType.NUMERICAL, unit="m/s", metadata={"test": True})
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
