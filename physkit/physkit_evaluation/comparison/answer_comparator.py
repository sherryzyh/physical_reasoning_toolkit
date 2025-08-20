"""
Main answer comparator that routes to appropriate comparison strategies.

This module provides the main AnswerComparator class that automatically
selects and uses the appropriate comparator based on answer types.
"""

from typing import Any, Dict
from physkit_core.definitions.answer_types import Answer, AnswerType
from .symbolic import SymbolicComparator
from .numerical import NumericalComparator
from .textual import TextualComparator


class AnswerComparator:
    """Main comparator that routes to appropriate comparison strategy."""
    
    def __init__(self):
        self.comparators = {
            AnswerType.SYMBOLIC: SymbolicComparator(),
            AnswerType.NUMERICAL: NumericalComparator(),
            AnswerType.TEXTUAL: TextualComparator(),
        }
    
    def compare(self, answer1: Answer, answer2: Answer) -> Dict[str, Any]:
        """
        Automatically select and use the appropriate comparator.
        
        Args:
            answer1: First answer to compare
            answer2: Second answer to compare
            
        Returns:
            Comparison results from the appropriate comparator
        """
        # Check if answer types match
        if answer1.answer_type != answer2.answer_type:
            return {
                "is_equal": False,
                "details": {
                    "error": "Answer types do not match",
                    "type1": answer1.answer_type.value,
                    "type2": answer2.answer_type.value
                }
            }
        
        # Get appropriate comparator
        comparator = self.comparators.get(answer1.answer_type)
        if comparator is None:
            return {
                "is_equal": False,
                "details": {
                    "error": f"No comparator available for {answer1.answer_type.value}"
                }
            }
        
        return comparator.compare(answer1, answer2)
