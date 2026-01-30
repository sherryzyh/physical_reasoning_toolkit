"""
Numerical comparator for numerical values with units and significant figures.

This module provides comparison strategies for numerical answers
including significant figure handling and unit conversion support.
"""

import math
from typing import Any, Dict, Optional, Tuple


from .base import BaseComparator
from prkit.prkit_core.models.answer import Answer
from prkit.prkit_core.definitions.answer_types import AnswerType
from prkit.prkit_core.llm import LLMClient

class NumericalComparator(BaseComparator):
    """Comparator for numerical values with significant figure support."""
    
    def __init__(self):
        """Initialize the numerical comparator."""
        pass
    
    def compare(self, answer1: Answer, answer2: Answer) -> Dict[str, Any]:
        """
        Compare two numerical values using significant figures.
        
        This comparator handles:
        - Exact numerical matches
        - Significant figure-based rounding
        - Unit conversions (if implemented)
        - Special cases (infinity, NaN, etc.)
        """
        if not self.can_compare(answer1, answer2):
            raise ValueError("Cannot compare non-numerical answers with NumericalComparator")
        
        # Extract values and metadata
        val1, units1 = self._extract_numerical_data(answer1)
        val2, units2 = self._extract_numerical_data(answer2)
        
        # Handle special cases
        if self._is_special_case(val1, val2):
            return self._handle_special_case(val1, val2)
        
        # Check if units are compatible
        if units1 and (not units2):
            units2 = units1
        elif units2 and (not units1):
            units1 = units2
        
        if units1 == units2:
            is_equal, comparison_details = self._compare_with_significant_figures(
                val1,
                val2,
            )
        else:
            is_equal, comparison_details = self._compare_with_units(
                val1,
                units1,
                val2,
                units2,
            )
        
        return {
            "is_equal": is_equal,
            "details": {
                "method": "numerical_comparison",
                "status": "completed",
                "value1": val1,
                "value2": val2,
                "units1": units1,
                "units2": units2,
                "explanation": comparison_details
            }
        }
    
    def _extract_numerical_data(self, answer: Answer) -> Tuple[float, Optional[str]]:
        """Extract numerical value and units from an answer."""
        return answer.value, answer.unit
    
    def _is_special_case(self, val1: float, val2: float) -> bool:
        """Check if values are special cases (infinity, NaN, etc.)."""
        return (math.isnan(val1) or math.isnan(val2) or 
                math.isinf(val1) or math.isinf(val2))
    
    def _handle_special_case(self, val1: float, val2: float) -> Dict[str, Any]:
        """Handle special numerical cases."""
        if math.isnan(val1) and math.isnan(val2):
            return {
                "is_equal": True,
                "details": {
                    "method": "numerical_comparison",
                    "status": "special_case",
                    "case": "both_nan",
                    "note": "Both values are NaN"
                }
            }
        elif math.isnan(val1) or math.isnan(val2):
            return {
                "is_equal": False,
                "details": {
                    "method": "numerical_comparison",
                    "status": "special_case",
                    "case": "one_nan",
                    "note": "One value is NaN, the other is not"
                }
            }
        elif math.isinf(val1) and math.isinf(val2):
            # Check if both infinities have the same sign
            is_equal = (val1 > 0) == (val2 > 0)
            return {
                "is_equal": is_equal,
                "details": {
                    "method": "numerical_comparison",
                    "status": "special_case",
                    "case": "both_infinity",
                    "note": f"Both values are infinity: {val1} and {val2}",
                    "signs_match": is_equal
                }
            }
        else:
            return {
                "is_equal": False,
                "details": {
                    "method": "numerical_comparison",
                    "status": "special_case",
                    "case": "mixed_special",
                    "note": f"Mixed special cases: {val1} and {val2}"
                }
            }
    
    def _count_significant_figures(self, value: float) -> int:
        """Count the number of significant figures in a number."""
        if value == 0:
            return 1
        
        # Convert to string to analyze
        str_val = str(abs(value))
        
        # Remove decimal point and leading zeros
        if '.' in str_val:
            # For decimal numbers
            str_val = str_val.replace('.', '')
            # Remove leading zeros
            str_val = str_val.lstrip('0')
        else:
            # For integers, remove trailing zeros
            str_val = str_val.rstrip('0')
        
        return len(str_val) if str_val else 1
    
    def _round_to_significant_figures(self, value: float, sig_figs: int) -> float:
        """Round a number to a specified number of significant figures."""
        if value == 0:
            return 0.0
        
        # Calculate the magnitude
        magnitude = 10 ** (int(math.floor(math.log10(abs(value))) - sig_figs + 1))
        
        # Round to the nearest multiple of the magnitude
        return round(value / magnitude) * magnitude
    
    def _compare_with_significant_figures(self, val1: float, val2: float) -> Tuple[bool, Dict[str, Any]]:
        sig_figs1 = self._count_significant_figures(val1)
        sig_figs2 = self._count_significant_figures(val2)
        
        # Determine the number of significant figures to use for comparison
        # Use the smaller number of significant figures (more strict)
        comparison_sig_figs = min(sig_figs1, sig_figs2)
        
        # Round both values to the same number of significant figures
        rounded_val1 = self._round_to_significant_figures(val1, comparison_sig_figs)
        rounded_val2 = self._round_to_significant_figures(val2, comparison_sig_figs)
        
        # Compare the rounded values
        is_equal = rounded_val1 == rounded_val2
        
        return is_equal, {
            "original_value1": val1,
            "original_value2": val2,
            "significant_figures1": sig_figs1,
            "significant_figures2": sig_figs2,
            "comparison_significant_figures": comparison_sig_figs,
            "rounded_value1": rounded_val1,
            "rounded_value2": rounded_val2,
            "method": "significant_figures_comparison"
        }
    
    def _compare_with_units(
        self,
        val1: float,
        units1: str,
        val2: float,
        units2: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        sig_figs1 = self._count_significant_figures(val1)
        sig_figs2 = self._count_significant_figures(val2)
        
        # Determine the number of significant figures to use for comparison
        # Use the smaller number of significant figures (more strict)
        comparison_sig_figs = min(sig_figs1, sig_figs2)
        
        # Round both values to the same number of significant figures
        rounded_val1 = self._round_to_significant_figures(val1, comparison_sig_figs)
        rounded_val2 = self._round_to_significant_figures(val2, comparison_sig_figs)
        
        comparison_client = LLMClient.from_model("gpt-4o")
        response = comparison_client.chat(
            messages=[
                {"role": "system", "content": "You are a physics expert. You are given two numerical values with units and you need to compare them."},
                {"role": "user", "content": f"Compare the following two numerical values with units: {rounded_val1} {units1} and {rounded_val2} {units2}. Return TRUE if they are equal, FALSE otherwise."},
            ]
        )
        return response == "TRUE", {
            "original_value1": val1,
            "original_value2": val2,
            "rounded_value1": rounded_val1,
            "rounded_value2": rounded_val2,
            "units1": units1,
            "units2": units2,
            "comparison_sig_figs": comparison_sig_figs,
            "method": "llm_comparison",
        }

    def can_compare(self, answer1: Answer, answer2: Answer) -> bool:
        """Check if both answers are numerical."""
        return (isinstance(answer1, Answer) and 
                isinstance(answer2, Answer) and 
                answer1.answer_type == AnswerType.NUMERICAL and 
                answer2.answer_type == AnswerType.NUMERICAL)
