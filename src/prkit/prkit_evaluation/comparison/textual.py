"""
Textual comparator for natural language descriptions and explanations.

This module provides comparison strategies for textual answers
using semantic similarity and fuzzy matching techniques.
"""

from typing import Any, Dict

from prkit.prkit_core.domain.answer_type import AnswerType
from prkit.prkit_core.model_clients import create_model_client
from prkit.prkit_core.domain.answer import Answer

from .base import BaseComparator


class TextualComparator(BaseComparator):
    """Comparator for textual answers using semantic similarity."""

    def __init__(self):
        """Initialize the textual comparator."""
        self.client = create_model_client("gpt-5.1")

    def compare(self, answer1: Answer, answer2: Answer) -> Dict[str, Any]:
        """
        Compare two textual answers for semantic similarity.

        This comparator handles:
        - Exact text matches
        - Semantic similarity using embeddings
        - Keyword matching
        - Phrase-level comparison
        """
        if not self.can_compare(answer1, answer2):
            raise ValueError(
                "Cannot compare non-textual answers with TextualComparator"
            )

        prompt = (
            "You are a physics expert. You are given two textual answers and you need to compare them. "
            "Return TRUE if they are equal, FALSE otherwise.\n\n"
            f"Answer 1: {answer1.value}\nAnswer 2: {answer2.value}"
        )
        response = self.client.chat(prompt)
        return {
            "is_equal": response == "TRUE",
            "details": {
                "method": "llm_comparison",
                "answer1": answer1.value,
                "answer2": answer2.value,
            },
        }

    def can_compare(self, answer1: Answer, answer2: Answer) -> bool:
        """Check if both answers are textual."""
        return (
            isinstance(answer1, Answer)
            and isinstance(answer2, Answer)
            and answer1.answer_type == AnswerType.TEXTUAL
            and answer2.answer_type == AnswerType.TEXTUAL
        )
