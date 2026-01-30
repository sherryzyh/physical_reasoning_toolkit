"""
Option Comparator for Multiple Choice Questions

This comparator handles both single and multiple choice answers, with support for:
- Case-insensitive comparison
- Order-independent matching for multiple options
- Normalization of option formats
"""

from typing import Any, Dict, List
from prkit.prkit_core.models.answer import Answer
from prkit.prkit_core.definitions.answer_types import AnswerType
from .base import BaseComparator


class OptionComparator(BaseComparator):
    """Comparator for multiple choice questions."""
    
    def can_compare(self, answer1: Answer, answer2: Answer) -> bool:
        """
        Check if this comparator can handle the given answer types.
        
        Args:
            answer1: First answer to check
            answer2: Second answer to check
            
        Returns:
            True if both answers are option type, False otherwise
        """
        # Check if both answers are option type
        return (answer1.answer_type == AnswerType.OPTION and 
                answer2.answer_type == AnswerType.OPTION)
    
    def compare(self, answer1: Answer, answer2: Answer) -> Dict[str, Any]:
        """
        Compare two multiple choice answers.
        
        Args:
            answer1: First answer (predicted/student answer)
            answer2: Second answer (ground truth/correct answer)
            
        Returns:
            Dictionary with comparison results:
            - is_equal: Boolean indicating if answers are equivalent
            - similarity_score: 1.0 if equal, 0.0 if not
            - details: Additional comparison information
        """
        # Extract answer strings
        ans1_str = str(answer1.value).strip()
        ans2_str = str(answer2.value).strip()
        
        # Handle empty answers
        if not ans1_str or not ans2_str:
            return {
                "is_equal": ans1_str == ans2_str,
                "similarity_score": 1.0 if ans1_str == ans2_str else 0.0,
                "details": {
                    "answer1": ans1_str,
                    "answer2": ans2_str,
                    "comparison_method": "exact_match",
                    "normalized_answer1": ans1_str,
                    "normalized_answer2": ans2_str,
                    "is_empty_comparison": True,
                    "is_multiple_choice": False
                }
            }
        
        # Normalize answers for comparison
        normalized_ans1 = self._normalize_answer(ans1_str)
        normalized_ans2 = self._normalize_answer(ans2_str)
        
        # Check if answers are equal after normalization
        is_equal = normalized_ans1 == normalized_ans2
        
        # Calculate similarity score
        similarity_score = 1.0 if is_equal else 0.0
        
        # Prepare detailed comparison information
        details = {
            "answer1": ans1_str,
            "answer2": ans2_str,
            "normalized_answer1": normalized_ans1,
            "normalized_answer2": normalized_ans2,
            "comparison_method": "normalized_exact_match",
            "is_multiple_choice": self._is_multiple_choice(ans1_str) or self._is_multiple_choice(ans2_str),
            "answer1_length": len(ans1_str),
            "answer2_length": len(ans2_str),
            "normalized_length": len(normalized_ans1)
        }
        
        # Add multiple choice specific details
        if details["is_multiple_choice"]:
            details.update({
                "answer1_options": self._extract_options(ans1_str),
                "answer2_options": self._extract_options(ans2_str),
                "normalized_options1": self._extract_options(normalized_ans1),
                "normalized_options2": self._extract_options(normalized_ans2),
                "option_count1": len(self._extract_options(ans1_str)),
                "option_count2": len(self._extract_options(ans2_str))
            })
        
        return {
            "is_equal": is_equal,
            "similarity_score": similarity_score,
            "details": details
        }
    
    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize an answer string for comparison.
        
        Args:
            answer: Raw answer string
            
        Returns:
            Normalized answer string
        """
        if not answer:
            return ""
        
        # Convert to uppercase for case-insensitive comparison
        normalized = answer.upper()
        
        # Remove common separators and whitespace
        normalized = normalized.replace(" ", "").replace(",", "").replace(";", "").replace("-", "")
        
        # Sort characters for order-independent comparison
        if self._is_multiple_choice(normalized):
            # Extract individual options and sort them
            options = self._extract_options(normalized)
            sorted_options = sorted(options)
            normalized = "".join(sorted_options)
        
        return normalized
    
    def _is_multiple_choice(self, answer: str) -> bool:
        """
        Check if an answer represents multiple choices.
        
        Args:
            answer: Answer string
            
        Returns:
            True if multiple choice, False otherwise
        """
        if not answer:
            return False
        
        # Multiple choice indicators:
        # - Multiple characters (e.g., "ABC", "123")
        # - Length > 1 and all characters are letters or digits
        if len(answer) > 1:
            # Check if all characters are letters or digits
            return all(c.isalnum() for c in answer)
        
        return False
    
    def _extract_options(self, answer: str) -> List[str]:
        """
        Extract individual options from a multiple choice answer.
        
        Args:
            answer: Answer string (e.g., "ABC", "123")
            
        Returns:
            List of individual options
        """
        if not answer:
            return []
        
        # Split into individual characters and filter out non-alphanumeric
        options = [c for c in answer if c.isalnum()]
        return options
    
    def compare_with_tolerance(self, answer1: Answer, answer2: Answer, 
                             case_sensitive: bool = False, 
                             order_sensitive: bool = False) -> Dict[str, Any]:
        """
        Compare answers with configurable tolerance options.
        
        Args:
            answer1: First answer
            answer2: Second answer
            case_sensitive: Whether to consider case differences
            order_sensitive: Whether to consider order differences
            
        Returns:
            Dictionary with comparison results
        """
        # Extract answer strings
        ans1_str = str(answer1.value).strip()
        ans2_str = str(answer2.value).strip()
        
        # Handle empty answers
        if not ans1_str or not ans2_str:
            return {
                "is_equal": ans1_str == ans2_str,
                "similarity_score": 1.0 if ans1_str == ans2_str else 0.0,
                "details": {
                    "answer1": ans1_str,
                    "answer2": ans2_str,
                    "comparison_method": "tolerant_exact_match",
                    "case_sensitive": case_sensitive,
                    "order_sensitive": order_sensitive,
                    "is_empty_comparison": True
                }
            }
        
        # Apply tolerance settings
        if not case_sensitive:
            ans1_str = ans1_str.upper()
            ans2_str = ans2_str.upper()
        
        if not order_sensitive and self._is_multiple_choice(ans1_str):
            # Sort options for order-independent comparison
            options1 = self._extract_options(ans1_str)
            options2 = self._extract_options(ans2_str)
            ans1_str = "".join(sorted(options1))
            ans2_str = "".join(sorted(options2))
        
        # Compare normalized answers
        is_equal = ans1_str == ans2_str
        similarity_score = 1.0 if is_equal else 0.0
        
        details = {
            "answer1": str(answer1.value),
            "answer2": str(answer2.value),
            "normalized_answer1": ans1_str,
            "normalized_answer2": ans2_str,
            "comparison_method": "tolerant_exact_match",
            "case_sensitive": case_sensitive,
            "order_sensitive": order_sensitive,
            "is_multiple_choice": self._is_multiple_choice(str(answer1.value)) or self._is_multiple_choice(str(answer2.value))
        }
        
        return {
            "is_equal": is_equal,
            "similarity_score": similarity_score,
            "details": details
        }