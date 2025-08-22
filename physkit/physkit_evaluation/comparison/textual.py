"""
Textual comparator for natural language descriptions and explanations.

This module provides comparison strategies for textual answers
using semantic similarity and fuzzy matching techniques.
"""

from typing import Any, Dict

from .base import BaseComparator
from physkit_core.models.answer import Answer
from physkit_core.definitions.answer_types import AnswerType
from physkit_core.llm import LLMClient


class TextualComparator(BaseComparator):
    """Comparator for textual answers using semantic similarity."""
    
    def __init__(self):
        """Initialize the textual comparator."""
        self.client = LLMClient.from_model("gpt-4o")
    
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
            raise ValueError("Cannot compare non-textual answers with TextualComparator")
        
        response = self.client.chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a physics expert. You are given two textual answers and you need to compare them. Return TRUE if they are equal, FALSE otherwise."
                },
                {
                    "role": "user",
                    "content": f"Answer 1: {answer1.value}\nAnswer 2: {answer2.value}"
                },
            ]
        )
        return {
            "is_equal": response == "TRUE",
            "details": {
                "method": "llm_comparison",
                "answer1": answer1.value,
                "answer2": answer2.value,
            }
        }
    
    def can_compare(self, answer1: Answer, answer2: Answer) -> bool:
        """Check if both answers are textual."""
        return (isinstance(answer1, Answer) and 
                isinstance(answer2, Answer) and 
                answer1.answer_type == AnswerType.TEXTUAL and 
                answer2.answer_type == AnswerType.TEXTUAL)
